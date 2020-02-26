from collections import OrderedDict
import multiprocessing
from pathlib import Path
import sys
sys.path.append('.')
import pickle

from shapely.affinity import translate
from pvmismatch import *
import PySAM.Pvwattsv7 as pv

from hybrid.solar_wind.shadow_cast import *


class FlickerMismatchModel:
    # pvmismatch standard module description
    cell_len = 0.124    # dimension of cell in meters
    cell_rows = 12      # n rows of cells in a module
    cell_cols = 8       # n columns of cells in a module
    cell_num_map = [[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                    [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24],
                    [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                    [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48],
                    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                    [83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72],
                    [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]]
    cell_num_map_flat = np.array(cell_num_map).flatten()
    n_hours = 8760

    def __init__(self, lat, lon, turbine_spacing, azi_mod=180, blade_length=35, steps_per_hour=1, angles_per_step=36):
        self.lat = lat
        self.lon = lon
        self.turbine_spacing = turbine_spacing
        self.n_rows = int(turbine_spacing / (0.124 * 12))
        self.azi_mod = azi_mod
        self.turb_pos = []
        for i in range(2):
            for j in range(-2, 2):
                self.turb_pos.append((turbine_spacing * j, turbine_spacing * i))
        self.panel_x = -turbine_spacing / 2
        self.panel_y = 0

        self.blade_length = blade_length
        self.steps_per_hour = steps_per_hour
        self.angles_per_step = angles_per_step
        self.n_steps = self.n_hours * self.steps_per_hour * self.angles_per_step

        self.file_suffix = str(self.lat) + "_" + str(self.lon) + "_" + str(self.steps_per_hour) + "_" + str(self.angles_per_step)

        self.azi_ang, self.elv_ang = self.get_sun_positions()
        self.poa = self.get_irradiance()
        self.shadows = self.get_turbine_shadows()

    def get_irradiance(self):
        filename = str(self.lat) + "_" + str(self.lon) + "_psmv3_60_2012.csv"
        weather_path = Path(__file__).parent.parent.parent / "resource_files" / "solar" / filename
        if not weather_path.exists():
            raise ValueError("resource file does not exist")
        pv_model = pv.default("PVWattsNone")
        pv_model.SystemDesign.array_type = 2
        pv_model.SystemDesign.gcr = .1
        pv_model.SolarResource.solar_resource_file = str(weather_path)
        pv_model.execute()
        return pv_model.Outputs.poa

    def get_sun_positions(self):
        azi_ang = None
        elv_ang = None
        azi_path = Path(__file__).parent / "data" / str(self.file_suffix + "azi.txt")
        if azi_path.exists():
            azi_ang = np.loadtxt(azi_path)
        elv_path = Path(__file__).parent / "data" / str(self.file_suffix + "elv.txt")
        if elv_path.exists():
            elv_ang = np.loadtxt(elv_path)
        if azi_ang is not None and elv_ang is not None:
            return azi_ang, elv_ang

        full_year_steps = range(8760 * self.steps_per_hour * self.angles_per_step)
        start = datetime.datetime(2012, 1, 1, 0, 0, 0, 0, tzinfo=Mountain)
        step_to_minute = 60 / (self.steps_per_hour * self.angles_per_step)
        date_generated = [start + datetime.timedelta(minutes=x * step_to_minute) for x in full_year_steps]
        azi_ang = np.zeros(len(full_year_steps))
        elv_ang = np.zeros(len(full_year_steps))
        for tt, date in enumerate(date_generated):
            azi_ang[tt] = get_azimuth(self.lat, self.lon, date)
            elv_ang[tt] = get_altitude(self.lat, self.lon, date)
        np.savetxt(azi_path, azi_ang)
        np.savetxt(elv_path, elv_ang)
        return azi_ang, elv_ang

    def get_turbine_shadows(self):
        all_shadows_path = Path(__file__).parent / "data" / str(self.file_suffix + "shd_" + str(self.turbine_spacing) + ".pkl")
        if all_shadows_path.exists():
            f = open(all_shadows_path, 'rb')
            all_turbine_shadows = pickle.load(f)
            f.close()
            return all_turbine_shadows

        shadow_polygons = []
        shadow_path = Path(__file__).parent / "data" / str(self.file_suffix + "shd.pkl")
        full_year_steps = range(8760 * self.steps_per_hour * self.angles_per_step)
        if shadow_path.exists():
            f = open(shadow_path, 'rb')
            shadow_polygons = pickle.load(f)
            f.close()
        else:
            step_to_angle = 360 / self.angles_per_step
            for step in full_year_steps:
                if self.elv_ang[step] < 0:
                    shadow_polygons.append(None)
                    continue
                turbine_shadow, shadow_ang = get_turbine_shadow_polygons(self.blade_length, step * step_to_angle % 360,
                                                                         azi_ang=self.azi_ang[step],
                                                                         elv_ang=self.elv_ang[step], wind_dir=None)
                shadow_polygons.append(turbine_shadow)
            f = open(shadow_path, 'wb')
            pickle.dump(shadow_polygons, f)
            f.close()

        all_shadows_polygons = []
        for step in full_year_steps:
            turbine_shadow = shadow_polygons[step]
            if not turbine_shadow:
                all_shadows_polygons.append(None)
                continue
            all_turbine_shadows = Polygon()
            for t, offset in enumerate(self.turb_pos):
                translated_shadow = translate(turbine_shadow, xoff=offset[0], yoff=offset[1])
                all_turbine_shadows = cascaded_union([all_turbine_shadows, translated_shadow])
            all_shadows_polygons.append(all_turbine_shadows)

        f = open(all_shadows_path, 'wb')
        pickle.dump(all_shadows_polygons, f)
        f.close()
        return all_shadows_polygons

    def power_loss_during_steps(self, steps):
        annual_kwh_shaded = 0
        annual_kwh_unshaded = 0
        print(multiprocessing.current_process().name, steps)
        module_meshes = create_module_cells_mesh(self.panel_x, self.panel_y, cell_len * cell_cols, cell_len * cell_rows,
                                                 self.n_rows)
        hours = OrderedDict()
        for step in steps:
            hr = int(step / self.steps_per_hour / self.angles_per_step)
            poa_suns = self.poa[hr] / 1000
            hours[hr] = None
            all_turbine_shadows = self.shadows[step]
            if self.elv_ang[step] < 0 or poa_suns < 1e-3 or not all_turbine_shadows:
                continue

            pvsys = pvsystem.PVsystem(numberStrs=1, numberMods=self.n_rows)
            sun_dict = dict()
            for mod in range(self.n_rows):
                sun_dict[mod] = [(poa_suns,) * 96, range(0, 96)]
            pvsys.setSuns({0: sun_dict})

            annual_kwh_unshaded += pvsys.Pmp

            poa_suns *= 0.1     # shaded
            offset_y = 0
            sun_dict.clear()
            for mod in range(self.n_rows):
                shadow = shadow_over_module_cells(module_meshes[mod], all_turbine_shadows)
                if np.amax(shadow) == 0:
                    continue
                shaded_indices = shadow.flatten().nonzero()[0]
                shaded_cells = [self.cell_num_map_flat[s] for s in shaded_indices]
                sun_dict[mod] = [(poa_suns,) * len(shaded_cells), shaded_cells]
                offset_y += cell_rows
            pvsys.setSuns({0: sun_dict})
            annual_kwh_shaded += pvsys.Pmp
            if step % 100 == 0:
                print(multiprocessing.current_process().name, hours.keys())

        wstep_to_kwh = 0.001 / (self.steps_per_hour * self.angles_per_step)
        annual_kwh_shaded *= wstep_to_kwh
        annual_kwh_unshaded *= wstep_to_kwh
        return annual_kwh_shaded, annual_kwh_unshaded

    # def power_loss_annual(self):
    #     n_procs = self.pool._processes
    #     n_steps_per_process = int(self.n_steps / n_procs)
    #
    #     step_intervals = []
    #     i = 0
    #     for i in range(n_procs - 1):
    #         step_intervals.append(range(i, i + n_steps_per_process))
    #         i += n_steps_per_process
    #     step_intervals.append(range(i, self.n_steps))
    #
    #     results = self.pool.map(self.power_loss_during_steps, step_intervals)
    #     print(results)
    #     for r in results:
    #         print(r)


def power_loss_versus_nrows(steps):
    lat = 39.7555
    lon = -105.2211
    n_rows = 8
    panel_x = -100
    panel_y = 86

    flickerShading = FlickerMismatchModel(lat, lon, 35 * mult)
    return flickerShading.power_loss_during_steps(steps)
    # annual_kwh_shaded, annual_kwh_unshaded = flicker_model.power_loss_annual()

    # print("Annual unshaded: {}, Annual shaded: {}, Loss %: {}".format(annual_kwh_unshaded, annual_kwh_shaded,
    #                                           (annual_kwh_unshaded - annual_kwh_shaded) / annual_kwh_unshaded * 100))



diam_multiples = (3, 5, 7, 9, 11)

for mult in diam_multiples[0:1]:

    n_procs = 6
    n_steps = FlickerMismatchModel.n_hours * 36
    n_steps_per_process = int(FlickerMismatchModel.n_hours * 36 / n_procs)

    step_intervals = []
    s = 0
    for i in range(n_procs - 1):
        step_intervals.append(range(s, s + n_steps_per_process))
        s += n_steps_per_process
    step_intervals.append(range(s, n_steps))

    pool = multiprocessing.Pool(processes=n_procs)
    results = pool.map(power_loss_versus_nrows, step_intervals)
    print("results for ", mult, mult)
    for r in results:
        print(r)