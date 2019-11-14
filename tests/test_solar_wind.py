import os
import pytest
from pytest import approx
import csv

from hybrid.wind.func_tools import plot_site
from hybrid.solar.layout import calculate_solar_extent
from hybrid.solar_wind.shadow_cast import *
from tests.data.defaults_data import defaults, Site

verts = Site['site_boundaries']['verts']

# from Shadow analysis of wind turbines for dual use of land for combined wind and solar photovoltaic power generation,
# the average tower_height=2.5R and average tower_width=R/16
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
    with open('../resource_files/solar/39.7555_-105.2211_psmv3_60.csv') as csv_f:
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
    pv = defaults['Solar']['Pvsamv1']
    solar_extent = calculate_solar_extent(pv)
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
    points = get_blade_shadow_points(blade_length=2, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625, 7)))
    assert(points['top_right'] == approx((0.0625, 7)))

    points = get_blade_shadow_points(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((0, 5.0625)) and points['bottom_right'] == approx((0, 4.9375)))
    assert(points['top_left'] == approx((2, 5.0625)))
    assert(points['top_right'] == approx((2, 4.9375)))

    points = get_blade_shadow_points(blade_length=2, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0442, 5.0442), 0.001) and points['bottom_right'] == approx((0.0442, 4.956), 0.001))
    assert(points['top_left'] == approx((1.37, 6.458), 0.001))
    assert(points['top_right'] == approx((1.458, 6.37), 0.001))

    points = get_blade_shadow_points(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)
    assert(points['bottom_left'] == approx((0, 5.062), 0.001) and points['bottom_right'] == approx((0, 4.9375)), 0.001)
    assert(points['top_left'] == approx((1.414, 6.477), 0.001))
    assert(points['top_right'] == approx((1.414, 6.352), 0.001))

    # azimuth 200, elv 45, winddir 0
    azimuth = 200
    points = get_blade_shadow_points(blade_length=2, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((1.648, 4.698), 0.001) and points['bottom_right'] == approx((1.773, 4.698), 0.001))
    assert(points['top_left'] == approx((2.332, 6.578), 0.001))
    assert(points['top_right'] == approx((2.457, 6.578), 0.001))

    points = get_blade_shadow_points(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((1.710, 4.761), 0.001) and points['bottom_right'] == approx((1.710, 4.636), 0.001))
    assert(points['top_left'] == approx((3.589, 4.077)))
    assert(points['top_right'] == approx((3.589, 3.952)))

    points = get_blade_shadow_points(blade_length=2, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))
    assert(points['top_right'] == approx((0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))

    points = get_blade_shadow_points(blade_length=2, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)
    assert(points['bottom_left'] == approx((-0.0625, 5)) and points['bottom_right'] == approx((0.0625, 5)))
    assert(points['top_left'] == approx((-0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))
    assert(points['top_right'] == approx((0.0625 + 2 * np.cos(np.pi/4), 2 * np.sin(np.pi/4) + 5)))


def test_shadow_cast_over_panel():
    blade_angles = (90, 70, 50, 30, 10, 0)

    xyc, yxc, shadow_mask = shadow_cast_over_panel(panel_x=0, panel_y=4.6, blade_length=2, blade_angle=0,
                                                   azi_ang=180, elv_ang=45, wind_dir=0)

    plt.pcolormesh(xyc, yxc, shadow_mask, cmap=cm.gray)
    plt.show()
