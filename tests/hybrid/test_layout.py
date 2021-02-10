import pytest
import numpy as np
import matplotlib.pyplot as plt

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.wind_source import WindPlant
from hybrid.solar_source import SolarPlant
from hybrid.layout.hybrid_layout import HybridLayout, WindBoundaryGridParameters, SolarGridParameters
from hybrid.layout.wind_layout_tools import create_grid


@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


technology = {
    'wind': {
        'num_turbines': 5,
        'turbine_rating_kw': 2000,
        'layout_mode': 'boundarygrid',
        'layout_params': WindBoundaryGridParameters(border_spacing=5,
                                                    border_offset=0.5,
                                                    grid_angle=0.5,
                                                    grid_aspect_power=0.5,
                                                    row_phase_offset=0.5)
    },
    'solar': {
        'system_capacity_kw': 5000,
        'layout_params': SolarGridParameters(x_position=0.25,
                                             y_position=0.5,
                                             aspect_power=0,
                                             gcr=0.5,
                                             s_buffer=0.1,
                                             x_buffer=0.1)
    }
}


def test_create_grid():
    site_info = SiteInfo(flatirons_site)
    bounding_shape = site_info.polygon.buffer(-200)
    site_info.plot()
    turbine_positions = create_grid(bounding_shape,
                                    site_info.polygon.centroid,
                                    np.pi / 4,
                                    200,
                                    200,
                                    .5)
    expected_positions = [
        [242., 497.],
        [383., 355.],
        [312., 709.],
        [454., 568.],
        [595., 426.],
        [525., 780.],
        [666., 638.],
        [737., 850.],
        [878., 709.]
    ]
    for n, t in enumerate(turbine_positions):
        assert(t.x == pytest.approx(expected_positions[n][0], 1e-1))
        assert(t.y == pytest.approx(expected_positions[n][1], 1e-1))


def test_wind_layout(site):
    wind_model = WindPlant(site, technology['wind'])
    xcoords, ycoords = wind_model._layout.turb_pos_x, wind_model._layout.turb_pos_y

    expected_xcoords = [0.7510, 1004.83, 1470.38, 903.06, 658.182]
    expected_ycoords = [888.865, 1084.148, 929.881, 266.4096, 647.169]

    for i in range(len(xcoords)):
        assert xcoords[i] == pytest.approx(expected_xcoords[i], 1e-3)
        assert ycoords[i] == pytest.approx(expected_ycoords[i], 1e-3)

    # wind_model.plot()
    # plt.show()


def test_solar_layout(site):
    solar_model = SolarPlant(site, technology['solar'])
    solar_region, buffer_region = solar_model._layout.solar_region.bounds, solar_model._layout.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)


def test_hybrid_layout(site):
    power_sources = {
        'wind': WindPlant(site, technology['wind']),
        'solar': SolarPlant(site, technology['solar'])
    }

    layout = HybridLayout(site, power_sources)
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y

    print(xcoords, ycoords)

    expected_xcoords = [599.999, 1785.929, 873.547, 872.275, 681.380]
    expected_ycoords = [1084.100, 1068.401, 404.454, 48.832, 664.9001]

    # turbines move from `test_wind_layout` due to the solar exclusion
    for i in range(len(xcoords)):
        assert xcoords[i] == pytest.approx(expected_xcoords[i], 1e-3)
        assert ycoords[i] == pytest.approx(expected_ycoords[i], 1e-3)

    assert(layout.solar.flicker_loss == pytest.approx(1.600e-05, 1e-3))


def test_hybrid_layout_wind_only(site):
    power_sources = {
        'wind': WindPlant(site, technology['wind']),
        # 'solar': SolarPlant(site, technology['solar'])
    }

    layout = HybridLayout(site, power_sources)
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y

    print(xcoords, ycoords)

    expected_xcoords = [0.751, 1004.834, 1470.385, 903.063, 658.181]
    expected_ycoords = [888.865, 1084.148, 929.881, 266.409, 647.169]

    # turbines move from `test_wind_layout` due to the solar exclusion
    for i in range(len(xcoords)):
        assert xcoords[i] == pytest.approx(expected_xcoords[i], 1e-3)
        assert ycoords[i] == pytest.approx(expected_ycoords[i], 1e-3)


def test_hybrid_layout_solar_only(site):
    power_sources = {
        # 'wind': WindPlant(site, technology['wind']),
        'solar': SolarPlant(site, technology['solar'])
    }

    layout = HybridLayout(site, power_sources)
    solar_region, buffer_region = layout.solar.solar_region.bounds, layout.solar.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)