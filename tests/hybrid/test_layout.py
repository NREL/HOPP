import pytest
from pathlib import Path
from timeit import default_timer
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiLineString

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.wind_source import WindPlant
from hybrid.pv_source import PVPlant
from hybrid.layout.hybrid_layout import HybridLayout, WindBoundaryGridParameters, PVGridParameters, get_flicker_loss_multiplier
from hybrid.layout.wind_layout_tools import create_grid
from hybrid.layout.pv_design_utils import size_electrical_parameters, find_modules_per_string


@pytest.fixture
def site():
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)


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
    'pv': {
        'system_capacity_kw': 5000,
        'layout_params': PVGridParameters(x_position=0.25,
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
    solar_model = PVPlant(site, technology['pv'])
    solar_region, buffer_region = solar_model._layout.solar_region.bounds, solar_model._layout.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)


def test_hybrid_layout(site):
    power_sources = {
        'wind': WindPlant(site, technology['wind']),
        'pv': PVPlant(site, technology['pv'])
    }

    layout = HybridLayout(site, power_sources)
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y
    buffer_region = layout.pv.buffer_region

    # turbines move from `test_wind_layout` due to the solar exclusion
    for i in range(len(xcoords)):
        assert not buffer_region.contains(Point(xcoords[i], ycoords[i]))

    assert (layout.pv.flicker_loss > 0.0001)


def test_hybrid_layout_rotated_array(site):
    power_sources = {
        'wind': WindPlant(site, technology['wind']),
        'pv': PVPlant(site, technology['pv'])
    }

    layout = HybridLayout(site, power_sources)
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y
    buffer_region = layout.pv.buffer_region

    # modify strands by rotation of LineStrings into trapezoidal solar_region
    for n in range(len(layout.pv.strands)):
        layout.pv.strands[n] = (layout.pv.strands[n][0], layout.pv.strands[n][1],
            affinity.rotate(layout.pv.strands[n][2], 30, 'center'))
    layout.pv.solar_region = MultiLineString([strand[2] for strand in layout.pv.strands]).convex_hull

    layout.plot()
    for strand in layout.pv.strands:
        plt.plot(*strand[2].xy)

    start = default_timer()
    flicker_loss_1 = get_flicker_loss_multiplier(layout._flicker_data,
                                                 layout.wind.turb_pos_x,
                                                 layout.wind.turb_pos_y,
                                                 layout.wind.rotor_diameter,
                                                 (layout.pv.module_width, layout.pv.module_height),
                                                 primary_strands=layout.pv.strands)
    time_strands = default_timer() - start

    # convert strands from LineString into MultiPoints
    module_points = []
    for strand in layout.pv.strands:
        distances = np.arange(0, strand[2].length, 0.992)
        module_points += [strand[2].interpolate(distance) for distance in distances]
    module_points = unary_union(module_points)

    start = default_timer()
    flicker_loss_2 = get_flicker_loss_multiplier(layout._flicker_data,
                                                 layout.wind.turb_pos_x,
                                                 layout.wind.turb_pos_y,
                                                 layout.wind.rotor_diameter,
                                                 (layout.pv.module_width, layout.pv.module_height),
                                                 module_points=module_points)
    time_points = default_timer() - start
    
    assert flicker_loss_1 == pytest.approx(flicker_loss_2, rel=1e-4)
    assert time_points < time_strands


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
        'pv': PVPlant(site, technology['pv'])
    }

    layout = HybridLayout(site, power_sources)
    solar_region, buffer_region = layout.pv.solar_region.bounds, layout.pv.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)


def test_kml_file_read():
    filepath = Path(__file__).absolute().parent / "layout_example.kml"
    site_data = {'kml_file': filepath}
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    site = SiteInfo(site_data, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)
    site.plot()
    assert np.array_equal(np.round(site.polygon.bounds), [ 681175., 4944970.,  686386., 4949064.])
    assert site.polygon.area * 3.86102e-7 == pytest.approx(2.3393, abs=0.01) # m2 to mi2


def test_kml_file_append():
    filepath = Path(__file__).absolute().parent / "layout_example.kml"
    site_data = {'kml_file': filepath}
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    site = SiteInfo(site_data, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

    x = site.polygon.centroid.x
    y = site.polygon.centroid.y
    turb_coords = [x - 500, y - 500]
    solar_region = Polygon(((x, y), (x, y + 5000), (x + 5000, y), (x + 5000, y + 5000)))

    filepath_new = Path(__file__).absolute().parent / "layout_example2.kml"
    site.kml_write(filepath_new, turb_coords, solar_region)
    assert filepath_new.exists()
    k, valid_region, lat, lon = SiteInfo.kml_read(filepath)
    assert valid_region.area > 0
    os.remove(filepath_new)


def test_system_electrical_sizing(site):
    target_solar_kw = 1e5
    target_dc_ac_ratio = 1.34
    modules_per_string = 12
    module_power = 0.310149     # [kW]
    inverter_power = 753.2      # [kW]
    n_inputs_inverter = 50
    n_inputs_combiner = 16

    n_strings, n_combiners, n_inverters, calculated_system_capacity = size_electrical_parameters(
        target_system_capacity=target_solar_kw,
        target_dc_ac_ratio=target_dc_ac_ratio,
        modules_per_string=modules_per_string,
        module_power=module_power,
        inverter_power=inverter_power,
        n_inputs_inverter=n_inputs_inverter,
        n_inputs_combiner=n_inputs_combiner,
    )
    assert n_strings == 26869
    assert n_combiners == 1680
    assert n_inverters == 99
    assert calculated_system_capacity == pytest.approx(1e5, 1e-3)

    with pytest.raises(Exception) as e_info:
        target_solar_kw_mod = 33
        modules_per_string_mod = 24
        n_strings, n_combiners, n_inverters, calculated_system_capacity = size_electrical_parameters(
            target_system_capacity=target_solar_kw_mod,
            target_dc_ac_ratio=target_dc_ac_ratio,
            modules_per_string=modules_per_string_mod,
            module_power=module_power,
            inverter_power=inverter_power,
            n_inputs_inverter=n_inputs_inverter,
        )
    assert "The specified system capacity" in str(e_info)

    modules_per_string = find_modules_per_string(
        v_mppt_min=545,
        v_mppt_max=820,
        v_mp_module=0.310149,
        v_oc_module=64.4,
        inv_vdcmax=820,
        target_relative_string_voltage=0.5,
    )
    assert modules_per_string == 12
