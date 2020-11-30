from typing import List, Union
import multiprocessing as mp
from pathlib import Path
import pickle
import copy
from itertools import product
import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from shapely.geometry import MultiPoint, Polygon, Point, MultiPolygon
from shapely.affinity import translate
from pvmismatch import pvsystem
import PySAM.Pvwattsv7 as pv

from hybrid.log import flicker_logger as logger
from hybrid.resource import SolarResource
from hybrid.flicker.shadow_flicker import get_sun_pos, get_turbine_shadows_timeseries, create_pv_string_points
from hybrid.flicker.pv_module import *

# global variables
tolerance = 1e-3
xs = []
ys = []
poa = []
n_procs = mp.cpu_count()
lat_range = range(20, 65, 2)
lon_range = range(-161, -68, 18)
func_space = product(lat_range, lon_range)


class FlickerMismatch:
    # model properties
    n_hours = 8760
    steps_per_hour = 1
    # dimensions of the heat map grid in diameters
    diam_mult_nwe = 8
    diam_mult_s = 4
    # arrays of single axis tracking, assuming strings isolated
    modules_per_string = 10
    string_width = module_width
    string_height = modules_per_string * module_height
    periodic = False
    """
    Simulates a wind turbine's flicker over a grid for a given location. The shadow cast by the tower and the three
    blades are calculated for the # of simulation steps: number of blade angles (evenly spaced) per step of the hour. 
    
    The shadow heat map is produced as a loss ratio relative to unshaded areas (0 - 1). This loss ratio is with
    respect to plane-of-array irradiance, as calculated for a single-axis tracking system.
    
    The flicker heat map is another loss ratio (0 - 1), but with respect to power production of an unshaded
    string of panels as modeled by PVMismatch. This is calculated by modeling panels at each grid location, grouped into
    strings, and simulating the power of each string. 
    
    The losses are aggregated over the year.
    """
    def __init__(self,
                 lat: float,
                 lon: float,
                 angles_per_step: int = 1,
                 blade_length: int = 35) -> None:
        """
        Setup file output paths, the solar panel array, and the heat map template.

        Also load irradiance and turbine shadow data.

        :param lat: latitude
        :param lon: longitude
        :param angles_per_step: number of blade angles per step of the hour
        :param blade_length: meters
        """
        self.lat = lat
        self.lon = lon

        self.blade_length = blade_length
        self.angles_per_step = angles_per_step
        self.n_steps = self.n_hours * self.steps_per_hour

        self.filename_base = "{}_{}_{}_{}".format(self.lat, self.lon, self.steps_per_hour, self.angles_per_step)
        self.single_turbine_shadow_file = Path(__file__).parent / "data" / str(self.filename_base + "_shd.pkl")

        self.turb_pos = ((0, 0), )
        self.array = None
        self.site = None

        self.site_points = MultiPoint()
        self.array_string_points = []
        self.heat_map_template = None

        self.elv_ang = None
        self.azi_ang = None
        self.poa = None

        self.setup_irradiance()
        self.turbine_shadow = self.get_turbine_shadows()
        self.setup_array()

        # mp
        self.step_intervals = None

    def create_pool(self,
                    n_procs: int
                    ) -> mp.Pool:
        """
        Initialize a multiprocessing pool where each simulation step can be partitioned (by modulo operator) to
        split up work among different FlickerMismatch instances.
        :param n_procs:
        """
        self.step_intervals = []
        n_steps_per_process = int(self.n_steps / n_procs)
        s = 0
        for i in range(n_procs - 1):
            self.step_intervals.append(range(s, s + n_steps_per_process))
            s += n_steps_per_process
        self.step_intervals.append(range(s, self.n_steps))
        return mp.Pool(processes=n_procs)

    def setup_irradiance(self):
        """
        Compute solar azimuth, elevation angles, and plane-of-array irradiance
        :return:
        """
        self.azi_ang, self.elv_ang = self.get_sun_positions()
        logger.info("get_sun_positions success")

        self.poa = self.get_irradiance()
        logger.info("get_irradiance success")

    def get_irradiance(self
                       ) -> np.ndarray:
        """
        Compute hourly plane-of-array irradiance for a single-axis tracking system using PVwatts
        :return: 8760 array
        """
        filename = str(self.lat) + "_" + str(self.lon) + "_psmv3_60_2012.csv"
        weather_path = Path(__file__).parent.parent.parent / "resource_files" / "solar" / filename
        if not weather_path.is_file():
            SolarResource(self.lat, self.lon, year=2012)
            if not weather_path.is_file():
                raise ValueError("resource file does not exist")
        pv_model = pv.default("PVWattsNone")
        pv_model.SystemDesign.array_type = 2
        pv_model.SystemDesign.gcr = .1
        pv_model.SolarResource.solar_resource_file = str(weather_path)
        pv_model.execute(0)
        return np.array(pv_model.Outputs.poa)

    def get_sun_positions(self
                          ) -> tuple:
        """
        Compute solar azimuth and elevation for each steps_per_hour in simulation
        """
        azi_ang = None
        elv_ang = None
        azi_path = Path(__file__).parent / "data" / str(self.filename_base + "_azi.txt")
        if azi_path.is_file():
            azi_ang = np.loadtxt(azi_path)
            logger.info("loaded azimuth angles from {}".format(azi_path))
        elv_path = Path(__file__).parent / "data" / str(self.filename_base + "_elv.txt")
        if elv_path.is_file():
            elv_ang = np.loadtxt(elv_path)
            logger.info("loaded elevation angles from {}".format(azi_path))
        if azi_ang is not None and elv_ang is not None:
            return azi_ang, elv_ang

        n_steps = 8760 * self.steps_per_hour
        step_to_minute = int(60 / self.steps_per_hour)
        azi_ang, elv_ang, _ = get_sun_pos(self.lat,
                                       self.lon,
                                       n_steps,
                                       step_to_minute)
        np.savetxt(azi_path, azi_ang)
        np.savetxt(elv_path, elv_ang)
        logger.info("exported azimuth and elevation angles with format {}".format(azi_path))
        return azi_ang, elv_ang

    def get_turbine_shadows(self
                            ) -> List[Union[None, Polygon, MultiPolygon]]:
        """
        Calculate turbine shadow polygons for each step in simulation
        :return: list with dimension [step_per_hour, angles_per_step]
        """
        if self.single_turbine_shadow_file.is_file():
            f = open(self.single_turbine_shadow_file, 'rb')
            turbine_polygons_per_hour = pickle.load(f)
            f.close()
            logger.info("get_turbine_shadows: loaded single turbine shadow")
        else:
            turbine_polygons_per_hour = get_turbine_shadows_timeseries(self.blade_length,
                                                                       self.n_steps,
                                                                       self.angles_per_step,
                                                                       self.azi_ang,
                                                                       self.elv_ang)
            f = open(self.single_turbine_shadow_file, 'wb')
            pickle.dump(turbine_polygons_per_hour, f)
            f.close()
            logger.info("get_turbine_shadows: completed turbine shadow calculation")
        return turbine_polygons_per_hour

    @staticmethod
    def get_turb_site(diam: int
                      ) -> Polygon:
        """
        Return a polygon with the dimensions of the grid
        """
        min_x, min_y = -FlickerMismatch.diam_mult_nwe * diam, -FlickerMismatch.diam_mult_s * diam
        max_x = max_y = FlickerMismatch.diam_mult_nwe * diam
        return Polygon(((min_x, min_y),
                        (max_x, min_y),
                        (max_x, max_y),
                        (min_x, max_y)))

    @staticmethod
    def setup_heatmap_template(bounds: list
                               ) -> tuple:
        """
        Create the points where each panel is located and the heat map grid template
        :param bounds: [min x, min y, max x, max y] of the grid
        :return: MultiPoint of panel locations, (heat map grid, x coordinates, y coordinates)
        """
        global xs, ys
        xs = np.arange(bounds[0] + module_width / 2, bounds[2], module_width)
        ys = np.arange(bounds[1] + module_height / 2, bounds[3], module_height)
        xxs, yys = np.meshgrid(xs, ys, sparse=True)
        site_points = MultiPoint(np.transpose([np.tile(xs, len(ys)),
                                               np.repeat(yys, len(xs))]))
        heat_map_template = (xxs * 0 + yys * 0, xs, ys)
        return site_points, heat_map_template

    @staticmethod
    def get_turb_pos_indices(heat_map_template: np.ndarray
                             ) -> tuple:
        """
        Get the indices for the heat map template of the cell where the turbine is located
        """
        x_ratio = 1 / 2
        y_ratio = FlickerMismatch.diam_mult_nwe / 4 / (FlickerMismatch.diam_mult_nwe / 4 + FlickerMismatch.diam_mult_s)
        turb_x_ind = int(len(heat_map_template[1]) * x_ratio)
        turb_y_ind = int(len(heat_map_template[2]) * y_ratio)
        return turb_x_ind, turb_y_ind

    def setup_array(self
                    ) -> None:
        """
        Setup the solar panel array within the grid as a Point per panel
        """
        self.site = FlickerMismatch.get_turb_site(self.blade_length * 2)
        self.site_points, self.heat_map_template = self.setup_heatmap_template(self.site.bounds)

        min_y, max_y = self.site.bounds[1], self.site.bounds[3]
        string, string_points = create_pv_string_points(0, min_y, FlickerMismatch.string_width, max_y - min_y)

        # where solar strings are
        self.array = []
        x_pos = self.site.bounds[0]
        while x_pos < xs[-1]:
            tmp_string = translate(string, x_pos, 0)
            self.array.append(tmp_string)
            x_pos += FlickerMismatch.string_width
        logger.info("setup_turbines_and_arrays success")

        # Create points centered on each module
        self.array_string_points = []
        x_pos = self.site.bounds[0]
        while x_pos < xs[-1]:
            array_points = translate(string_points, x_pos, 0)
            self.array_string_points.append(self.setup_string_points(array_points))
            x_pos += FlickerMismatch.string_width

        logger.info("setup_point_maps success")

    def setup_string_points(self,
                            array_points: Union[Point, MultiPoint]
                            ) -> list:
        """
        Divide up the array of solar panels into strings. If FlickerMismatch.periodic, then a string can continue
        from the bottom edge of the grid back to the top, rather than running off the grid entirely.
        :param array_points:
        :return: a list of which points belong in which string, dim [n_string, FlickerMismatch.modules_per_string]
        """
        if isinstance(array_points, Point):
            array_points = (array_points,)
        n_rows_modules = len(array_points)

        if FlickerMismatch.periodic:
            n_strings = int(np.ceil(n_rows_modules / FlickerMismatch.modules_per_string))
        else:
            n_strings = int(n_rows_modules / FlickerMismatch.modules_per_string)
        string_points = []
        for i in range(n_strings):
            start = i * FlickerMismatch.modules_per_string
            end = min(len(array_points), (i + 1) * FlickerMismatch.modules_per_string)
            pts = [array_points[j] for j in range(start, end)]
            string_points.append(pts)

        if FlickerMismatch.periodic:
            assert (len(array_points) == sum([len(i) for i in string_points]))

            # for the last string, continue across the top of the center grid to the bottom of the next
            i = 0
            while len(string_points[-1]) < FlickerMismatch.modules_per_string:
                pt_x = string_points[-1][-1].x
                pt_y = string_points[-1][-1].y + module_height
                if pt_y > self.site_points.bounds[3] + tolerance:
                    pt_y -= (self.site_points.bounds[3] - self.site_points.bounds[1])
                string_points[-1].append(Point(pt_x, pt_y))
                i += 1

        return string_points

    @staticmethod
    def calculate_shading(poa_weight: float,
                          shadows: list,
                          site_points: MultiPoint,
                          heat_map: np.ndarray
                          ) -> None:
        """
        Update the heat_map with shading losses in POA irradiance
        :param poa_weight: loss to apply to shaded cells
        :param shadows: list of shadow (Multi)Polygons for each blade angle
        :param site_points: points of solar panels
        :param heat_map: array with shading losses
        """
        # tx, ty = turbine_grid_shadow.exterior.xy
        # plt.plot(tx, ty, 'c--')
        if not shadows:
            return
        for shadow in shadows:
            intersecting_points = site_points.intersection(shadow)
            if intersecting_points:
                if isinstance(intersecting_points, Point):
                    intersecting_points = (intersecting_points, )
                for pt in intersecting_points:
                    x_ind = int(round((pt.x - site_points.bounds[0]) / module_width))
                    y_ind = int(round((pt.y - site_points.bounds[1]) / module_height))
                    heat_map[y_ind, x_ind] += poa_weight
        #     if isinstance(shadow, Polygon):
        #         shadow = (shadow, )
        #     for poly in shadow:
        #         x, y = poly.exterior.xy
        #         plt.plot(x, y)
        # plt.show()

    @staticmethod
    def calculate_power_loss(poa: float,
                             elv_ang: float,
                             shadows: list,
                             array_points: list,
                             heat_map_flicker: np.ndarray):
        """
        Update the heat map with flicker losses, using an unshaded string as baseline for normalizing
        :param poa: irradiance
        :param elv_ang: solar elevation degree
        :param shadows: list of shadow (Multi)Polygons for each blade angle
        :param array_points: list of solar panels, [# strands, # strings per strand, FlickerMismatch.modules_per_string]
        :param heat_map_flicker: array with flicker losses
        """
        global xs, ys

        poa_suns = poa/1000
        if elv_ang < 0 or poa_suns < 1e-3:
            return 0, 0

        heat_map_flicker_new = np.zeros(heat_map_flicker.shape)

        # set unshaded string for baseline
        pvsys = pvsystem.PVsystem(numberStrs=1, numberMods=FlickerMismatch.modules_per_string)
        sun_dict_unshaded = dict()
        for index in range(FlickerMismatch.modules_per_string):
            sun_dict_unshaded[index] = [(poa_suns,) * 96, range(0, 96)]
        pvsys.setSuns({0: sun_dict_unshaded})
        kwh_unshaded = pvsys.Pmp

        suns_memo = dict()
        suns_memo['hits'] = 0

        for shadow in shadows:
            ht_map = np.zeros(heat_map_flicker.shape)

            for array in array_points:
                if not array:
                    continue

                for string in array:
                    shaded_module_points = MultiPoint(string).intersection(shadow)

                    if shaded_module_points.is_empty:
                        continue
                    elif isinstance(shaded_module_points, Point):
                        shaded_module_points = (shaded_module_points, )

                    shaded_poa_suns = poa_suns * 0.1
                    shaded_indices = []
                    for mod in shaded_module_points:
                        shaded_indices.append(int(np.argmin([(mod.x - m.x) ** 2 + (mod.y - m.y) ** 2 for m in string])))
                    shaded_indices = tuple(shaded_indices)
                    if shaded_indices in suns_memo.keys():
                        flicker_loss = suns_memo[shaded_indices]
                        suns_memo['hits'] += 1
                    else:
                        sun_dict = copy.deepcopy(sun_dict_unshaded)
                        for index in shaded_indices:
                            sun_dict[index] = [(shaded_poa_suns,) * 96, cell_num_map_flat]
                        pvsys.setSuns({0: sun_dict})
                        flicker_loss = (kwh_unshaded - pvsys.Pmp) / kwh_unshaded
                        suns_memo[shaded_indices] = flicker_loss

                    for pt in string:
                        x_ind = int(round((pt.x - np.min(xs)) / module_width))
                        y_ind = int(round((pt.y - np.min(ys)) / module_height))
                        if FlickerMismatch.periodic:
                            if ht_map[y_ind, x_ind] == 0:
                                ht_map[y_ind, x_ind] = flicker_loss
                            else:
                                # if reusing a module, take the average
                                ht_map[y_ind, x_ind] = (ht_map[y_ind, x_ind] + flicker_loss)/2
                        else:
                            ht_map[y_ind, x_ind] = flicker_loss
                    # print(shaded_module_points)
                # print()
            heat_map_flicker_new += ht_map
        # plt.show()
        # print(suns_memo)
        heat_map_flicker += heat_map_flicker_new

    def create_heat_maps_irradiance(self,
                                    steps: range):
        """
        Create shadow and flicker heat maps for a given range of simulation steps
        :param steps: which steps to run, must be within range calculated by steps_per_hour x angles_per_step
        :return: shadow heat map, flicker heat map
        """
        proc_id = mp.current_process().name
        logger.info("Proc {}: Starting heat maps {}".format(proc_id, steps))

        total_poa = sum(self.poa)

        # shadow calculations
        heat_map_shadow = copy.deepcopy(self.heat_map_template[0])

        # flicker calculations
        heat_map_flicker = copy.deepcopy(self.heat_map_template[0])
        progress_size = int(len(steps) / min(10, len(steps)))

        for i, step in enumerate(steps):
            if i % progress_size == 0:
                logger.info("Proc {} created heat maps for step {}".format(proc_id, int(i / len(steps) * 100)))

            hr = int(step / FlickerMismatch.steps_per_hour)
            poa_weight = self.poa[hr] / total_poa / self.steps_per_hour

            shadows = self.turbine_shadow[step]

            if not shadows:
                continue

            FlickerMismatch.calculate_shading(poa_weight, shadows, self.site_points, heat_map_shadow)

            FlickerMismatch.calculate_power_loss(self.poa[hr], self.elv_ang[step], shadows,
                                                 self.array_string_points, heat_map_flicker)

        logger.info("Finished heat maps")

        return heat_map_shadow, heat_map_flicker

    def run_parallel(self,
                     n_procs: int,
                     intervals: [range] = None
                     ):
        """
        Runs create_heat_maps_irradiance in parallel
        :param n_procs:
        :param intervals: list of ranges to simulate; if none, simulate entire weather file's records
        :return: heat_map_shadow, heat_map_flicker
        """
        heat_map_shadow_path = (Path(__file__).parent / "data" /
                                str(self.filename_base + "_shadow.txt"))
        heat_map_flicker_path = (Path(__file__).parent / "data" /
                                 str(self.filename_base + "_flicker.txt"))

        if heat_map_flicker_path.is_file() and heat_map_shadow_path.is_file():
            logger.info("loaded heat maps from file")
            # return np.loadtxt(heat_map_shadow_path), np.loadtxt(heat_map_flicker_path)

        logger.info("run_parallel with {} processes".format(n_procs))
        pool = self.create_pool(n_procs)
        if intervals is None:
            intervals = self.step_intervals
        results = pool.imap(self.create_heat_maps_irradiance, intervals)

        # shadow calculations
        heat_map_shadow = copy.deepcopy(self.heat_map_template[0])

        # flicker calculations
        heat_map_flicker = copy.deepcopy(self.heat_map_template[0])

        for r in results:
            heat_map_shadow += r[0]
            heat_map_flicker += r[1]

        # normalize
        heat_map_shadow /= self.angles_per_step
        heat_map_flicker /= self.angles_per_step * self.steps_per_hour

        logger.info("Create_heat_map_irradiance success")

        return heat_map_shadow, heat_map_flicker

    def plot_on_site(self,
                     plot_points=True,
                     plot_array=True):
        fig, axs = plt.subplots()
        axs.set_aspect('equal')
        xs, ys = self.site.exterior.xy
        plt.plot(xs, ys)
        for t in self.turb_pos:
            plt.plot(t[0], t[1], 'bo')
        if plot_array:
            for p in self.array:
                x, y = p.exterior.xy
                plt.plot(x, y)
        return axs
