from typing import List, Union, Optional, Sequence
import multiprocessing as mp
from pathlib import Path
import functools
import copy
from itertools import product
import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from shapely.geometry import MultiPoint, Polygon, Point, MultiPolygon, box
from shapely.affinity import translate
from pvmismatch import pvsystem
import PySAM.Pvwattsv8 as pv

from hybrid.log import flicker_logger as logger
from hybrid.resource import SolarResource
from hybrid.layout.shadow_flicker import get_sun_pos, get_turbine_shadows_timeseries, create_pv_string_points
from hybrid.layout.pv_module import *

# global variables
tolerance = 1e-3
poa = []
n_procs = mp.cpu_count()
lat_range = range(20, 65, 2)
lon_range = range(-161, -68, 18)
func_space = product(lat_range, lon_range)


class FlickerMismatch:
    """
    Simulates a wind turbine's flicker over a grid for a given location. The shadow cast by the tower and the three
    blades are calculated for each of the simulation steps: number of blade angles (evenly spaced) per step of the hour.

    The turbine is located at (0, 0) and a set of 2D arrays give the flicker losses at grid cell / coordinate. This
    'heatmap' can have variable length and width, determined by 'diam_mult_nwe' and 'diam_mult_s', and can be normalized
    in several ways:

        The 'poa' heat map is produced as a loss ratio relative to unshaded areas (0 - 1). This loss ratio is with
        respect to plane-of-array irradiance, as calculated for a single-axis tracking system using PVWattsv8.

        The 'power' heat map is another loss ratio (0 - 1), but with respect to power production of an unshaded
        string of panels as modeled by PVMismatch. This is calculated by modeling panels at each grid location, grouped into
        strings, and simulating the power of each string.

        The 'time' heat map is weighted by the number of timesteps each grid cell is shaded over the total timesteps
        simulated.

    All heat maps are normalized by the number of timesteps simulated.

    :var n_hours: number of hours in year
    :var steps_per_hour: number of time steps to run each hour
    :var diam_mult_nwe: in number of turbine diameters, the distance of the heat map's north, west and east end from the
            turbine at (0, 0)
    :var diam_mult_s: similarly, the number of turbine diameters the heatmap extends from (0, 0) south
    :var periodic: if true, then the top of the heatmap continues onto the bottom, and vice versa for the east / west
    :var turbine_tower_shadow: if true, then include the tower shadow

    """
    # model properties
    n_hours: int = 8760
    steps_per_hour: int = 1
    # dimensions of the heat map grid in diameters
    diam_mult_nwe: int = 8
    diam_mult_s: int = 4
    # arrays of single axis tracking, assuming strings isolated
    periodic: bool = False
    # shadow properties
    turbine_tower_shadow: bool = True

    def __init__(self,
                 lat: float,
                 lon: float,
                 angles_per_step: Optional[int] = 1,
                 blade_length: int = 35,
                 solar_resource_data: Optional[dict] = None,
                 wind_dir: Optional[list] = None,
                 gridcell_width: float = module_width,
                 gridcell_height: float = module_height,
                 gridcells_per_string: int = modules_per_string
                 ) -> None:
        """
        Setup file output paths, the solar panel array, and the heat map template.

        Also load irradiance and turbine shadow data.

        :param lat: latitude
        :param lon: longitude
        :param blade_length: meters
        :param angles_per_step: number of blade angles to simulate every timestep
        :param solar_resource_data: PySAM's solar resource data: https://github.com/NREL/pysam/blob/master/files/ResourceTools.py
        :param wind_dir: wind direction degrees, 0 as north, time series of len(8760 * steps_per_hour)
        :param gridcell_width: grid cells of the heat map dimension
        :param gridcell_height: grid cells of the heat map dimension
        :param gridcells_per_string: for 'poa' heatmaps
        """
        self.lat = lat
        self.lon = lon
        self.solar_resource_data = solar_resource_data

        self.blade_length = blade_length
        self.angles_per_step = angles_per_step
        self.n_steps = self.n_hours * self.steps_per_hour

        self.turb_pos = ((0, 0), )
        self.array = None
        self.site = None

        self.site_points = MultiPoint()
        self.array_string_points = []
        self.heat_map_template = None

        self.elv_ang = None
        self.azi_ang = None
        self.poa = None
        self.wind_dir = None
        self.turbine_shadow = None

        self.gridcell_width = gridcell_width
        self.gridcell_height = gridcell_height
        self.modules_per_string = gridcells_per_string

        self._setup_wind_dir(wind_dir)
        self._setup_array()

        self.filename_base = "{}_{}_{}_{}_{}_{}".format(
            self.lat, self.lon, self.steps_per_hour, self.angles_per_step,
            np.average(self.wind_dir if self.wind_dir is not None else 0),
            np.std((self.wind_dir if self.wind_dir is not None else 0)))

        # mp
        self.step_intervals = None

    def _create_pool(self,
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

    def _setup_wind_dir(self,
                        wind_dir_degrees):
        if wind_dir_degrees is None:
            return
        if len(wind_dir_degrees) != self.n_steps:
            raise ValueError("'wind_dir' array must be of length {}".format(self.n_steps))
        self.wind_dir = wind_dir_degrees

    def _setup_irradiance(self):
        """
        Compute solar azimuth and elevation degrees;
        Compute plane-of-array irradiance for a single-axis tracking PVwatts system
        :return:
        """
        pv_model = pv.default("PVWattsNone")
        pv_model.SystemDesign.array_type = 2
        pv_model.SystemDesign.gcr = .1
        if self.solar_resource_data is None:
            filename = str(self.lat) + "_" + str(self.lon) + "_psmv3_60_2012.csv"
            weather_path = Path(__file__).parent.parent.parent / "resource_files" / "solar" / filename
            if not weather_path.is_file():
                SolarResource(self.lat, self.lon, year=2012)
                if not weather_path.is_file():
                    raise ValueError("resource file does not exist")
            pv_model.SolarResource.solar_resource_file = str(weather_path)
        else:
            pv_model.SolarResource.solar_resource_data = self.solar_resource_data
        pv_model.execute(0)
        self.poa = np.array(pv_model.Outputs.poa)

        logger.info("get_irradiance success")

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
    def _setup_heatmap_template(bounds: list,
                                gridcell_width: float = module_width,
                                gridcell_height: float = module_height
                                ) -> tuple:
        """
        Create the points where each panel is located and the heat map grid template
        :param bounds: [min x, min y, max x, max y] of the grid
        :param gridcell_width: width of cells in the heat map
        :param gridcell_height: height of cells in the heat map
        :return: MultiPoint of panel locations, (heat map grid, x coordinates, y coordinates)
        """
        global xs, ys
        xs = np.arange(bounds[0] + gridcell_width / 2, bounds[2], gridcell_width)
        ys = np.arange(bounds[1] + gridcell_height / 2, bounds[3], gridcell_height)
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

    def _setup_array(self
                     ) -> None:
        """
        Setup the solar panel array within the grid as a Point per panel
        """
        self.site = FlickerMismatch.get_turb_site(self.blade_length * 2)
        self.site_points, self.heat_map_template = self._setup_heatmap_template(self.site.bounds, self.gridcell_width,
                                                                                self.gridcell_height)

        min_y, max_y = self.site.bounds[1], self.site.bounds[3]
        string, string_points = create_pv_string_points(0, min_y,
                                                        self.gridcell_width, self.gridcell_height,
                                                        self.gridcell_width, max_y - min_y)

        # where solar strings are
        self.array = []
        x_pos = self.site.bounds[0]
        while x_pos < xs[-1]:
            tmp_string = translate(string, x_pos, 0)
            self.array.append(tmp_string)
            x_pos += self.gridcell_width
        logger.info("setup_turbines_and_arrays success")

        # Create points centered on each module
        self.array_string_points = []
        x_pos = self.site.bounds[0]
        while x_pos < xs[-1]:
            array_points = translate(string_points, x_pos, 0)
            self.array_string_points.append(self._setup_string_points(array_points))
            x_pos += self.gridcell_width

        logger.info("setup_point_maps success")

    def _setup_string_points(self,
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
        else:
            array_points = array_points.geoms
        n_rows_modules = len(array_points)

        if FlickerMismatch.periodic:
            n_strings = int(np.ceil(n_rows_modules / self.modules_per_string))
        else:
            n_strings = int(n_rows_modules / self.modules_per_string)
        string_points = []
        for i in range(n_strings):
            start = i * self.modules_per_string
            end = min(n_rows_modules, (i + 1) * self.modules_per_string)
            pts = [array_points[j] for j in range(start, end)]
            string_points.append(pts)

        if FlickerMismatch.periodic:
            assert (n_rows_modules == sum([len(i) for i in string_points]))

            # for the last string, continue across the top of the center grid to the bottom of the next
            i = 0
            while len(string_points[-1]) < self.modules_per_string:
                pt_x = string_points[-1][-1].x
                pt_y = string_points[-1][-1].y + self.gridcell_height
                if pt_y > self.site_points.bounds[3] + tolerance:
                    pt_y -= (self.site_points.bounds[3] - self.site_points.bounds[1])
                string_points[-1].append(Point(pt_x, pt_y))
                i += 1

        return string_points

    @staticmethod
    def _calculate_shading(weight: float,
                           shadows: list,
                           site_points: MultiPoint,
                           heat_map: np.ndarray,
                           gridcell_width: float,
                           gridcell_height: float,
                           normalize_by_area=False
                           ) -> None:
        """
        Update the heat_map with shading losses in POA irradiance

        :param weight: loss to apply to shaded cells
        :param shadows: list of shadow (Multi)Polygons for each blade angle
        :param site_points: points of solar panels
        :param heat_map: array with shading losses
        :param gridcell_width: width of cells in the heat map
        :param gridcell_height: height of cells in the heat map
        :param normalize_by_area: if True, normalize weight per cell by how much area is shaded
        """
        if not shadows:
            return

        module_width_half = gridcell_width / 2
        module_height_half = gridcell_height / 2
        for shadow in shadows:
            if normalize_by_area:
                intersecting_points = site_points.intersection(shadow.buffer(
                    np.linalg.norm([gridcell_height, gridcell_width])))
            else:
                intersecting_points = site_points.intersection(shadow)
            if intersecting_points:
                if isinstance(intersecting_points, Point):
                    intersecting_points = (intersecting_points, )
                else:
                    intersecting_points = intersecting_points.geoms
                # break up into separate instructions for minor speed up by vectorization
                xs = np.array([pt.x for pt in intersecting_points])
                ys = np.array([pt.y for pt in intersecting_points])
                x_ind = (xs - site_points.bounds[0]) / gridcell_width
                y_ind = (ys - site_points.bounds[1]) / gridcell_height
                x_ind = np.round(x_ind).astype(int)
                y_ind = np.round(y_ind).astype(int)
                for n in range(len(intersecting_points)):
                    x = x_ind[n]
                    y = y_ind[n]
                    pt = intersecting_points[n]
                    if normalize_by_area:
                        cell = box(pt.x - module_width_half, pt.y - module_height_half,
                                   pt.x + module_width_half, pt.y + module_height_half)
                        intersection = cell.intersection(shadow)
                        area_weight = intersection.area / cell.area
                        heat_map[y, x] += weight * area_weight
                    else:
                        heat_map[y, x] += weight
            # if isinstance(shadow, Polygon):
            #     shadow = (shadow, )
            # for poly in shadow:
            #     x, y = poly.exterior.xy
            #     plt.plot(x, y)
        # plt.show()

    @staticmethod
    def _calculate_power_loss(poa: float,
                              elv_ang: float,
                              shadows: list,
                              array_points: list,
                              heat_map_flicker: np.ndarray,
                              gridcell_width: float,
                              gridcell_height: float,
                              xs_min: float,
                              ys_min: float,
                              poa_shading_ratio: float = 0.9
                              ):
        """
        Update the heat map with flicker losses, using an unshaded string as baseline for normalizing

        :param poa: irradiance
        :param elv_ang: solar elevation degree
        :param shadows: list of shadow (Multi)Polygons for each blade angle
        :param array_points: list of solar panels, [# strands, # strings per strand, FlickerMismatch.modules_per_string]
        :param heat_map_flicker: array with flicker losses
        :param gridcell_width: width of cells in the heat map
        :param gridcell_height: height of cells in the heat map
        :param xs_min: min of heat map grid's x coordinates
        :param ys_min: min of heat map grid's y coordinates
        :param poa_shading_ratio: how much of the poa is blocked by the shadow
        """
        poa_suns = poa/1000
        if elv_ang < 0 or poa_suns < 1e-3:
            return 0, 0

        heat_map_flicker_new = np.zeros(heat_map_flicker.shape)

        mods_per_string = len(array_points[0][0])

        # set unshaded string for baseline
        pvsys = pvsystem.PVsystem(numberStrs=1, numberMods=mods_per_string)
        sun_dict_unshaded = dict()
        for index in range(mods_per_string):
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
                    else:
                        shaded_module_points = shaded_module_points.geoms

                    shaded_poa_suns = poa_suns * (1 - poa_shading_ratio)
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
                        x_ind = int(round((pt.x - xs_min) / gridcell_width))
                        y_ind = int(round((pt.y - ys_min) / gridcell_height))
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

    def _calculate_turbine_shadow(self,
                                  ind: int
                                  ) -> List[Union[None, Polygon, MultiPolygon]]:
        return self.turbine_shadow[ind]

    def create_heat_maps(self,
                         steps: range,
                         weight_option: tuple,
                         ) -> tuple:
        """
        Create shadow and flicker heat maps for a given range of simulation steps

        :param weight_option: tuple of selected weighting options, producing a heatmap each
                    - "poa": weight by plane-of-array irradiance
                    - "power": weight by power loss of pvmismatch module
                    - "time": weight by number of timesteps shaded
        :param steps: which steps to run, must be within range calculated by steps_per_hour x angles_per_step
        :return: shadow heat map, flicker heat map
        """
        proc_id = mp.current_process().name
        logger.info("Proc {}: Starting heat maps {}".format(proc_id, steps))

        step_to_minute = 60 / self.steps_per_hour
        self.azi_ang, self.elv_ang, _ = get_sun_pos(self.lat,
                                                    self.lon,
                                                    step_to_minute,
                                                    steps=steps)

        self.turbine_shadow = get_turbine_shadows_timeseries(self.blade_length,
                                                             steps,
                                                             self.angles_per_step,
                                                             self.azi_ang,
                                                             self.elv_ang,
                                                             self.wind_dir,
                                                             FlickerMismatch.turbine_tower_shadow)

        by_poa = by_power = by_time = False

        for i in weight_option:
            if i == "poa":
                by_poa = True
                heat_map_shadow = copy.deepcopy(self.heat_map_template[0])
            elif i == "power":
                by_power = True
                heat_map_flicker = copy.deepcopy(self.heat_map_template[0])
            elif i == "time":
                by_time = True
                heat_map_time = copy.deepcopy(self.heat_map_template[0])
            else:
                raise ValueError("Unrecognized 'weight_option'")

        if not (by_poa or by_power or by_time):
            raise ValueError("No valid 'weight_option' provided. Provide a list of selected ways to weight the shading "
                             "from the set ('poa', 'power', 'time')")

        if by_poa or by_power:
            if not isinstance(self.poa, Sequence):
                self._setup_irradiance()
            total_poa = sum(self.poa[steps])

        progress_size = int(len(steps) / min(10, len(steps)))
        for i, step in enumerate(steps):
            if i % progress_size == 0:
                logger.info("Proc {} created heat maps for {} / 100 steps".format(proc_id, int(i / len(steps) * 100)))

            hr = int(step / FlickerMismatch.steps_per_hour)

            shadows = self._calculate_turbine_shadow(i)

            if not shadows:
                continue

            if by_poa:
                poa_weight = self.poa[hr] / total_poa
                FlickerMismatch._calculate_shading(poa_weight, shadows, self.site_points,
                                                   heat_map_shadow, self.gridcell_width, self.gridcell_height)

            if by_power:
                xs, ys = np.min(self.heat_map_template[1]), np.min(self.heat_map_template[2])
                FlickerMismatch._calculate_power_loss(self.poa[hr], self.elv_ang[i], shadows,
                                                      self.array_string_points,
                                                      heat_map_flicker, self.gridcell_width, self.gridcell_height, xs, ys)

            if by_time:
                FlickerMismatch._calculate_shading(1, shadows, self.site_points,
                                                   heat_map_time, self.gridcell_width, self.gridcell_height,
                                                   normalize_by_area=True)

        # normalize by angles per hour (since each will use the same weight) or by number of hours total
        step_normalize = self.angles_per_step if self.angles_per_step else 1
        if by_poa:
            heat_map_shadow /= step_normalize
        if by_power:
            heat_map_flicker /= step_normalize * len(steps)
        if by_time:
            heat_map_time /= step_normalize * len(steps)

        heat_maps_to_return = []
        for i in weight_option:
            if i == 'poa':
                heat_maps_to_return.append(heat_map_shadow)
            elif i == 'power':
                heat_maps_to_return.append(heat_map_flicker)
            elif i == 'time':
                heat_maps_to_return.append(heat_map_time)

        logger.info("Finished heat maps")
        return tuple(heat_maps_to_return)

    def run_parallel(self,
                     n_procs: int,
                     weight_option: tuple,
                     intervals: Optional[Sequence[range]] = None
                     ):
        """
        Runs create_heat_maps_irradiance in parallel

        :param n_procs:
        :param weight_option: tuple of selected weighting options, producing a heatmap each
            - "poa": weight by plane-of-array irradiance
            - "power": weight by power loss of pvmismatch module
            - "time": weight by number of timesteps shaded
        :param intervals: list of ranges to simulate; if none, simulate entire weather file's records
        :return: heat_map_shadow, heat_map_flicker
        """
        logger.info("run_parallel with {} processes".format(n_procs))
        pool = self._create_pool(n_procs)
        if intervals is None:
            intervals = self.step_intervals

        if 'power' in weight_option or 'poa' in weight_option:
            self._setup_irradiance()

        results = pool.imap(functools.partial(self.create_heat_maps, weight_option=weight_option),
                            intervals)

        # aggregate results and renormalize
        heat_maps_to_return = [copy.deepcopy(self.heat_map_template[0]) for _ in weight_option]

        if 'power' in weight_option:
            subhourly_poa = np.repeat(self.poa, FlickerMismatch.steps_per_hour)
            total_poa = sum([sum(subhourly_poa[i]) for i in intervals])
        total_steps = sum([len(i) for i in intervals])
        for r, i in zip(results, intervals):
            for j, hm in enumerate(heat_maps_to_return):
                if weight_option[j] == 'poa':
                    hm += r[j] * sum(self.poa[i]) / total_poa
                elif weight_option[j] == 'power' or weight_option[j] == 'time':
                    hm += r[j] * len(i) / total_steps

        logger.info("Create_heat_map success")

        return tuple(heat_maps_to_return)

    def plot_on_site(self,
                     plot_array=True,
                     plot_points=True
                     ):
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
        if plot_points:
            for p in self.array_string_points:
                for s in p:
                    xs = [point.x for point in s]
                    ys = [point.y for point in s]
                    plt.scatter(xs, ys)
        return axs
