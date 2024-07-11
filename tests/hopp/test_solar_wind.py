import sys
from pytest import approx
import csv
from hopp.simulation.technologies.layout.shadow_flicker import *
from hopp.simulation.technologies.sites import flatirons_site
from hopp import ROOT_DIR

sys.path.append('..')
verts = flatirons_site['site_boundaries']['verts']


def sun_info(n):
    # irrad = [0, 0, 0, 0, 0, 0, 0, 0, 96, 229, 321, 399, 501, 444, 330, 106, 14, 0, 0, 0, 0, 0, 0, 0]

    # get DNI from csv
    irrad = []
    with open(f'{ROOT_DIR}/simulation/resource_files/solar/39.7555_-105.2211_psmv3_60_2012.csv') as csv_f:
        reader = csv.reader(csv_f, delimiter=',')
        for row in reader:
            try:
                irrad.append(float(row[7]))
            except:
                irrad.append(0)
    irrad = irrad[3:n + 3]

    lat = 39.7555
    lon = -105.2211
    azi_ang, elv_ang, _ = get_sun_pos(lat, lon, n=len(irrad))
    sun_info_dict = {
        'azi_ang': azi_ang,
        'elv_ang': elv_ang,
        'irrad': irrad
    }
    return sun_info_dict


def test_get_turbine_shadow_polygons():
    # azimuth 180, elv 45, winddir 0
    azimuth = 180
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(135.6958)
    expected_bounds = (-17.63300807568878, 3.827021247335479e-17, 17.633008075688775, 70.0)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(135.2005)
    expected_bounds = (-10.541265877365282, 3.827021247335479e-17, 20.0, 67.63300807568878)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(133.7821)
    expected_bounds = (-19.48027842897044, 3.827021247335479e-17, 14.584077361972545, 64.58407736197255)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_winddir():
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=180, elv_ang=45, wind_dir=45)[0]
    assert shadow.area == approx(123.7373)
    expected_bounds = (-7.6123336892307565, 3.827021247335479e-17, 14.142135623730951, 67.63300807568878)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_2():
    # azimuth 200, elv 45, winddir 0
    azimuth = 200
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(129.1531)
    expected_bounds = (-1.9260877865070292, -0.21376258957854294, 29.287699252560525, 65.7784834550136)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(128.6338)
    expected_bounds = (-0.5873078879911927, -0.21376258957854294, 34.76145159747322, 66.78702271471559)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(127.2281)
    expected_bounds = (-0.5873078879911927, -0.21376258957854294, 34.86766417354703, 58.66139314690842)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)[0]
    assert shadow.area == approx(127.4075)
    expected_bounds = (-0.5873078879911927, -0.21376258957854294, 34.13402195906636, 66.67283985847735)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_solstice():
    shadow, shadow_ang = get_turbine_shadow_polygons(35,
                                                     120,
                                                     azi_ang=275.7824404502053,
                                                     elv_ang=22.521680787154185,
                                                     wind_dir=95.78244045020529)
    assert shadow.area == approx(560.1624)
    expected_bounds = (-0.11019683215840337, -57.175994012818265, 285.3811899455883, 4.062288445548752)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_rotated_ellipse():
    Rx = 1
    Ry = 1
    Cx = 0
    Cy = 0

    colors = ('b', 'g', 'r')
    i = 0
    xs = []
    ys = []
    for theta in (np.radians(135),):
        # theta = np.pi/5
        angles = np.linspace(0, np.pi / 4, 10)
        for angle in angles:
            x = Rx * np.cos(angle) * np.cos(theta) - Ry * np.sin(angle) * np.sin(theta) + Cx
            y = Rx * np.cos(angle) * np.sin(theta) + Ry * np.sin(angle) * np.cos(theta) + Cy
            xs.append(x)
            ys.append(y)
        i += 1

    # plt.scatter(xs, ys)
    # plt.xlim((-2, 2))
    # plt.ylim((-2, 2))
    # plt.show()
