import os
import sys
sys.path.append('.')
from pathlib import Path
import pickle
import pytest
from pytest import approx
import csv
from matplotlib.animation import FuncAnimation

from shapely.affinity import translate
from pvmismatch import *

from hybrid.wind.func_tools import plot_site
from hybrid.solar.layout import calculate_solar_extent
from hybrid.solar_wind.shadow_cast import *
from tests.data.defaults_data import Site



verts = Site['site_boundaries']['verts']

# from Shadow analysis of wind turbines for dual use of land for combined wind and solar photovoltaic power generation,
# the average tower_height=2.5R; average tower_width=R/16; average blade_width=R/16
def turb_info(rotor_rad):
    turb_info_dict = {
        'x': [1426.651162, 731.8163169, 655.9166205, 1283.272483, 312.0997645,
              1111.885174, 766.9593724, 934.1454174, 317.6134104, 59.25907537],
        'y': [1080.395979, 580.5661303, 232.6466948, 822.0345446, 786.5217163,
              212.7902666, 825.6651025, 1080.52743, 607.5484287, 408.7925702],
        'rotor_rad': rotor_rad,
        'tower_height': 2.5*rotor_rad,
        'tower_width': rotor_rad/16
    }
    return turb_info_dict


@pytest.fixture
def shadow_info():
    shadow_info = {
        'tower_shadow_weight': 0.9,
        'rotor_shadow_weight': 0.2
    }
    return shadow_info


def sun_info(n):
    # irrad = [0, 0, 0, 0, 0, 0, 0, 0, 96, 229, 321, 399, 501, 444, 330, 106, 14, 0, 0, 0, 0, 0, 0, 0]

    # get DNI from csv
    irrad = []
    with open('../resource_files/solar/39.7555_-105.2211_psmv3_60_2012.csv') as csv_f:
        reader = csv.reader(csv_f, delimiter=',')
        for row in reader:
            try:
                irrad.append(float(row[7]))
            except:
                irrad.append(0)
    irrad = irrad[3:n+3]

    lat = 39.7555
    lon = -105.2211
    azi_ang, elv_ang = get_sun_pos(lat, lon, n=len(irrad))
    sun_info_dict = {
        'azi_ang': azi_ang,
        'elv_ang': elv_ang,
        'irrad': irrad
    }
    return sun_info_dict


def test_pv_extent():
    pv = pvsam.default("FlatPlatePVSingleowner")
    solar_extent = calculate_solar_extent(pv.export())
    assert(solar_extent == (48, 7307, 2))


def test_weighted_avg_mask_by_poa(shadow_info):
    # test with 24 hours of data
    lxy = [3, 3]
    avg_shadow = weighted_avg_masks_by_poa(turb_info(40), shadow_info, sun_info(24), lxy)
    # print(avg_mask)
    correct_mask = [[0.26598361, 0.1, 0.26598361],
                    [0.1, 0.1, 0.13540984],
                    [0.33827869, 0.1, 0.33827869]]
    assert(pytest.approx(avg_shadow, correct_mask, 1e-3))


# takes 16 min with 10 procs and n=8760
def test_weighted_avg_mask_by_poa_large(shadow_info):
    # test with 24 hours of data
    site_dim = (int(np.max([r[0] for r in verts])), int(np.max([r[1] for r in verts])))
    site_x, site_y = int(site_dim[0]), int(site_dim[1])
    lxy = (int(site_x * 2), site_y)

    for rotorR in [40]:
        avg_shadow = weighted_avg_masks_by_poa(turb_info(rotorR), shadow_info, sun_info(8760), lxy)
        # print(avg_mask)
        correct_mask = [[0.26598361, 0.1, 0.26598361],
                        [0.1, 0.1, 0.13540984],
                        [0.33827869, 0.1, 0.33827869]]
        np.savetxt('shadow_wght_annual_rotorR_' + str(rotorR) + '.txt', avg_shadow, delimiter=',')
    # assert(pytest.approx(avg_shadow, correct_mask, 1e-3))


# takes 32.76 seconds with 1 proc and n=10
# takes 18 min with 10 procs and n=8760...
def test_shadow_cast(shadow_info):
    site_dim = (int(np.max([r[0] for r in verts])), int(np.max([r[1] for r in verts])))
    # site_dim = [10, 10]
    plt.figure()
    plot_site(verts, 'ko-', 'Simplified')
    for i in range(10):
        plt.plot(turb_info(40)['x'][i], turb_info(40)['y'][i], 'go')

    n = 2400
    superposed_mask = turbines_shadow_cast_avg(turb_info(40), site_dim, shadow_info, sun_info(n))
    print(np.min(superposed_mask), np.max(superposed_mask))
    np.savetxt('Turbine_shading_weighted_by_irradiance_for_' + str(n) + '_hours.txt', superposed_mask, delimiter=',')


def test_get_shadow_box_turb_rad(shadow_info):
    site_dim = (int(np.max([r[0] for r in verts])), int(np.max([r[1] for r in verts])))
    site_x, site_y = int(site_dim[0]), int(site_dim[1])
    lxy = (int(site_x * 2), site_y)

    for rotorR in [30, 40, 50, 60, 70, 80]:
        for t in [0.006]:
            if os.path.isfile('shadow_wght_annual_rotorR_' + str(rotorR) + '-'+str(t)+'.txt'):
                avg_shadow = np.loadtxt('shadow_wght_annual_rotorR_' + str(rotorR) + '-'+str(t)+'.txt', delimiter=',')
            else:
                avg_shadow = weighted_avg_masks_by_poa(turb_info(rotorR), shadow_info, sun_info(8760), lxy)
                np.savetxt('shadow_wght_annual_rotorR_' + str(rotorR) + '-'+str(t)+'.txt', avg_shadow, delimiter=',')
            len_east, len_west, len_north, mask = get_unshaded_box(avg_shadow, t)
            print(rotorR, t, len_east, len_west, len_north)

def test_get_unshaded_area():
    turb_shadow = np.loadtxt('/Users/dguittet/Projects/HybridSystems/LayoutOpt/ShadowPercents/shadow_wght_annual_rotorR_60.txt', delimiter=',')

    len_east, len_west, len_north, mask = get_unshaded_box(turb_shadow, 0.)
    assert(len_east == 1068)
    assert(len_west == 1794)
    assert(len_north == 809)


def test_get_unshaded_box_from_data():
    # interpolate along x
    len_east, len_west, len_north = get_unshaded_box_from_data(55, 0)
    assert(len_east == approx(1464.5) and len_west == approx(1794) and len_north == approx(1081))

    # interpolate along x and y
    len_east, len_west, len_north = get_unshaded_box_from_data(55, 0.0005)
    assert(len_east == approx(1170.75) and len_west == approx(1597.75) and len_north == approx(1040.5))


def test_get_unshaded_areas_on_site():
    get_unshaded_areas_on_site(verts, turb_info(40), len_east=100, len_west=50, len_north=100, plot_bool=True)


def test_get_blade_shadow_points():
    # azimuth 180, elv 45, winddir 0
    azimuth = 180
    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625, 7)))
    assert(points['top_right'] == approx((0.0625, 7)))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((0, 5.0625)) and points['bottom_right'] == approx((0, 4.9375)))
    assert(points['top_left'] == approx((2, 5.0625)))
    assert(points['top_right'] == approx((2, 4.9375)))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0442, 5.0442), 0.001) and points['bottom_right'] == approx((0.0442, 4.956), 0.001))
    assert(points['top_left'] == approx((1.37, 6.458), 0.001))
    assert(points['top_right'] == approx((1.458, 6.37), 0.001))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)
    assert(points['bottom_left'] == approx((0, 5.062), 0.001) and points['bottom_right'] == approx((0, 4.9375)), 0.001)
    assert(points['top_left'] == approx((1.414, 6.477), 0.001))
    assert(points['top_right'] == approx((1.414, 6.352), 0.001))

    # azimuth 200, elv 45, winddir 0
    azimuth = 200
    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((1.648, 4.698), 0.001) and points['bottom_right'] == approx((1.773, 4.698), 0.001))
    assert(points['top_left'] == approx((2.332, 6.578), 0.001))
    assert(points['top_right'] == approx((2.457, 6.578), 0.001))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((1.710, 4.761), 0.001) and points['bottom_right'] == approx((1.710, 4.636), 0.001))
    assert(points['top_left'] == approx((3.589, 4.077)))
    assert(points['top_right'] == approx((3.589, 3.952)))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))
    assert(points['top_right'] == approx((0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))

    points = get_turbine_shadow_polygons(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))
    assert(points['top_right'] == approx((0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))


def test_shadow_cast_over_panel_plotting():
    plotting = False
    animating = True

    lat = 39.7555
    lon = -105.2211
    blade_length = 35
    cell_len = 0.124
    cell_rows = 12
    cell_cols = 8

    turbine_spacing = blade_length * 2 * 8

    #create a 3 x 4 grid of turbines
    n_rows = int(4 * turbine_spacing / (0.124 * 12))
    n_cols = 1
    panel_height = cell_len * cell_rows * n_rows
    panel_width = cell_len * cell_cols * n_cols
    turb_pos = []
    for i in range(2):
        for j in range(-2, 2):
            turb_pos.append((turbine_spacing * j, turbine_spacing * i))
    panel_x = -turbine_spacing / 2
    panel_y = 0

    xy = None
    yx = None

    # azi_ang, elv_ang = get_sun_pos(34.6, -117, n=8*36, step_secs=60, start_hr=9)

    # create a pvmismatch model
    pvsys_unshaded = pvsystem.PVsystem(numberStrs=1, numberMods=n_rows)
    module_meshes = create_module_cells_mesh(panel_x, panel_y, cell_len * cell_cols, cell_len * cell_rows,
                                             n_rows)

    cell_num_map = [[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                    [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24],
                    [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                    [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48],
                    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                    [83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72],
                    [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]]
    cell_num_map_flat = np.array(cell_num_map).flatten()

    start = datetime.datetime(2012, 1, 1, 0, 0, 0, 0, tzinfo=Mountain)

    fig = None
    if animating or plotting:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim((-turbine_spacing * 2.5, turbine_spacing * 1.5))
        ax1.set_ylim((0, turbine_spacing * 2))
        plot_shadow = []
        for t in turb_pos:
            plot_shadow.append(ax1.plot([], [], color='#6699cc')[0])

        pts = ((panel_x, panel_y),
               (panel_x, panel_y + panel_height),
               (panel_x + panel_width, panel_y + panel_height),
               (panel_x + panel_width, panel_y))
        module = Polygon(pts)
        xm, ym = module.exterior.xy
        ax1.plot(xm, ym, color='black')
        ax1.set_title("Turbine shadow for module at (" + str(panel_x) + ", " + str(panel_y) + ")")
        ax1.set_xlabel("angle")

        ax2.set_title("Pmp")
        ax2.set_xlabel('System Voltage, V [V]')
        ax2.set_ylabel('System Power, P [kW]')
        ax2.grid()
        ax2.set_xlim(0, pvsys_unshaded.Voc * 1.1)
        ax2.set_ylim(0, pvsys_unshaded.Pmp * 1.1 / 1000)
        plot_pv = ax2.plot([], [])

        fig.tight_layout()


    # run every 15 minutes, 30 blade angles so each step is 15/30 = 0.5 minutes
    hours = 12
    steps_per_hour = 1
    angles_per_step = 36
    steps = np.arange(0, hours * steps_per_hour * angles_per_step)
    step_to_angle = 360 / (steps_per_hour * angles_per_step)
    step_to_minute = 60 / (steps_per_hour * angles_per_step)

    shadow_polygons = []
    shadow_path = Path(__file__).parent.parent.absolute() / "hybrid" / "solar_wind" / "data" / "39.7555_-105.2211_1_36shd.pkl"
    print(shadow_path)
    if shadow_path.exists():
        f = open(shadow_path, 'rb')
        shadow_polygons = pickle.load(f)
        f.close()
    else:
        print("not found")

    # preprocess which steps have positive elevation angle
    anim_steps = []
    for step in steps:
        if shadow_polygons[step]:
            anim_steps.append(step)

    if (len(anim_steps) > 0 and anim_steps[0] > 0) or len(anim_steps) == 0:
        anim_steps.insert(0, 0)
    print(len(anim_steps))

    def update(i):
        print(i)
        i = anim_steps[i]
        minute = int(i * step_to_minute)
        date = start + datetime.timedelta(minutes=minute)

        # loop over all times and calculate elevation angle
        azi_ang = get_azimuth(lat, lon, date)
        elv_ang = get_altitude(lat, lon, date)

        turbine_shadow = shadow_polygons[i]

        pvsys = pvsystem.PVsystem(numberStrs=1, numberMods=n_rows)
        if not turbine_shadow:
            plot_pv[0].set_data(pvsys.Vsys, pvsys.Psys / 1000)
            ax2.set_title("Pmp {}".format(pvsys.Pmp / 1000))
            return

        all_turbine_shadows = Polygon()
        for t, offset in enumerate(turb_pos):
            translated_shadow = translate(turbine_shadow, xoff=offset[0], yoff=offset[1])
            all_turbine_shadows = cascaded_union([all_turbine_shadows, translated_shadow])
            xb, yb = translated_shadow.exterior.xy
            plot_shadow[t].set_data(xb, yb)
        # plt.show()

        ax1.set_xlabel('min {}, blade {}, azi {}, elv {}'.format(minute, i * step_to_angle % 360, round(azi_ang, 1), round(elv_ang, 2)))

        offset_y = 0
        sun_dict = dict()
        for mod in range(n_rows):
            shadow = shadow_over_module_cells(module_meshes[mod], all_turbine_shadows)
            if np.amax(shadow) == 0:
                continue
            shaded_indices = shadow.flatten().nonzero()[0]
            # print(shaded_indices)
            shaded_cells = [cell_num_map_flat[s] for s in shaded_indices]
            sun_dict[mod] = [(0.1,) * len(shaded_cells), shaded_cells]
            offset_y += 12
        pvsys.setSuns({0: sun_dict})
        plot_pv[0].set_data(pvsys.Vsys, pvsys.Psys / 1000)
        ax2.set_title("Pmp {}".format(pvsys.Pmp / 1000))

    # update(0)
    # exit()

    if animating:
        anim = FuncAnimation(fig, update, frames=range(len(anim_steps)), interval=1)
        # anim.save('turbine_shadow_azi' + str(azi) + '_elev' + str(elv) + "_yaw" + str(wind_dir) + '.gif', dpi=80, writer='imagemagick')
        anim.save('turbine_shadow_azi.gif', dpi=80, writer='imagemagick')

        # plt.show()

    if plotting:
        blade_angles = np.arange(0, 361, 10)
        azi_ang = 180
        elv_ang = 45
        for blade_angle in blade_angles:
            xy, yx, shadow_mask = shadow_cast_over_panel(panel_x=panel_x, panel_y=panel_y, n_mod=n_rows, n_cols=n_cols,
                                                         blade_length=blade_length, blade_angle=blade_angle, azi_ang=azi_ang,
                                                         elv_ang=elv_ang, wind_dir=None, plot_obj=plot_shadow[0])
            plt.xlim((-blade_length * 2, blade_length * 2))
            plt.ylim((0, blade_length * 4))
            plt.show()
        print(blade_angle)
        print(shadow_mask)
    print(xy)
    print(yx)
    if plotting:
        plt.grid()
        plt.show()



def test_rotated_ellipse():
    Rx = 1
    Ry = 1
    Cx = 0
    Cy = 0

    colors = ('b', 'g', 'r')
    i = 0
    for theta in (np.radians(135), ):
    # theta = np.pi/5
        angles = np.linspace(0, np.pi/4, 10)
        for angle in angles:
            x = Rx * np.cos(angle)*np.cos(theta) - Ry * np.sin(angle)*np.sin(theta) + Cx
            y = Rx * np.cos(angle)*np.sin(theta) + Ry * np.sin(angle)*np.cos(theta) + Cy
            plt.plot(x, y, colors[i] + 'o')
        i += 1

    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.show()


test_shadow_cast_over_panel_plotting()