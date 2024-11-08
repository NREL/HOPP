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
    expected_bounds = (-17.63300, 0, 17.63300, 70.0)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(135.2005)
    expected_bounds = (-10.54126, 0, 20.0, 67.63300)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(133.7821)
    expected_bounds = (-19.48027, 0, 14.58407, 64.58407)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_winddir():
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=180, elv_ang=45, wind_dir=45)[0]
    assert shadow.area == approx(123.7373)
    expected_bounds = (-7.61233, 0, 14.14213, 67.633)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    # wind coming from north, widest swept area
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=180, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(135.2004)
    expected_bounds = (-10.54126, 0, 20, 67.633)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    # wind coming from east, thinnest swept area
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=180, elv_ang=45, wind_dir=90)[0]
    assert shadow.area == approx(81.1654)
    expected_bounds = (-0.625, 0, 0.625, 67.633)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_2():
    # azimuth 200, elv 45, winddir 0
    azimuth = 200
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=90, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(129.1531)
    expected_bounds = (-1.926087, -0.21376258, 29.28769, 65.77848)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(128.6338)
    expected_bounds = (-0.5873078, -0.21376258, 34.76145, 66.78702)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    assert shadow.area == approx(127.2281)
    expected_bounds = (-0.5873078, -0.21376258, 34.86766, 58.66139)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])

    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=0, azi_ang=azimuth, elv_ang=45, wind_dir=45)[0]
    assert shadow.area == approx(127.4075)
    expected_bounds = (-0.5873078, -0.21376258, 34.13402, 66.67283)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])


def test_get_turbine_shadow_polygons_solstice():
    shadow, shadow_ang = get_turbine_shadow_polygons(35,
                                                     120,
                                                     azi_ang=275.782440,
                                                     elv_ang=22.521680,
                                                     wind_dir=95.78244)
    assert shadow.area == approx(560.1624)
    expected_bounds = (-0.1101968, -57.17599, 285.38118, 4.062289)
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


def test_swept_area():
    azimuth = 75
    shadow = get_turbine_shadow_polygons(blade_length=20, blade_angle=45, azi_ang=azimuth, elv_ang=45, wind_dir=0)[0]
    expected_bounds = (-63.34583, -19.71403, 0.1617619, 0.6037036)
    for b in range(4):
        assert shadow.bounds[b] == approx(expected_bounds[b])
