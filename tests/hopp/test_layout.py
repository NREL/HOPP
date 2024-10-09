import pytest
from pytest import approx
from pathlib import Path
from timeit import default_timer
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.ops import unary_union
from shapely.geometry import Point, MultiLineString

from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.simulation.technologies.pv.pv_plant import PVPlant, PVConfig
from hopp.simulation.technologies.layout.hybrid_layout import HybridLayout, WindBoundaryGridParameters, PVGridParameters, get_flicker_loss_multiplier
from hopp.simulation.technologies.layout.wind_layout_tools import create_grid
from hopp.simulation.technologies.layout.pv_design_utils import size_electrical_parameters, find_modules_per_string
from hopp.simulation.technologies.pv.detailed_pv_plant import DetailedPVPlant, DetailedPVConfig

from hopp.utilities.utils_for_tests import create_default_site_info


@pytest.fixture
def site():
    return create_default_site_info()


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


def test_create_grid(site):
    bounding_shape = site.polygon.buffer(-200)
    site.plot()
    turbine_positions = create_grid(bounding_shape,
                                    site.polygon.centroid,
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
    config = WindConfig.from_dict(technology['wind'])
    wind_model = WindPlant(site, config=config)
    xcoords, ycoords = wind_model._layout.turb_pos_x, wind_model._layout.turb_pos_y

    expected_xcoords = [1498, 867, 525, 3, 658]
    expected_ycoords = [951, 265, 74, 288, 647]

    for i in range(len(xcoords)):
        assert xcoords[i] == pytest.approx(expected_xcoords[i], abs=1)
        assert ycoords[i] == pytest.approx(expected_ycoords[i], abs=1)

    # wind_model.plot()
    # plt.show()


def test_solar_layout(site):
    config = PVConfig.from_dict(technology['pv'])
    solar_model = PVPlant(site, config=config)
    solar_region, buffer_region = solar_model.layout.solar_region.bounds, solar_model.layout.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)


def test_hybrid_layout(site):
    pv_config = PVConfig.from_dict(technology['pv'])
    wind_config = WindConfig.from_dict(technology['wind'])
    power_sources = {
        'wind': WindPlant(site, config=wind_config),
        'pv': PVPlant(site, config=pv_config)
    }

    layout = HybridLayout(site, power_sources)
    assert layout.wind is not None
    assert layout.pv is not None
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y
    buffer_region = layout.pv.buffer_region

    # turbines move from `test_wind_layout` due to the solar exclusion
    for i in range(len(xcoords)):
        assert not buffer_region.contains(Point(xcoords[i], ycoords[i]))

    assert (layout.pv.flicker_loss > 0.0001)


def test_hybrid_layout_rotated_array(site):
    pv_config = PVConfig.from_dict(technology['pv'])
    wind_config = WindConfig.from_dict(technology['wind'])
    power_sources = {
        'wind': WindPlant(site, config=wind_config),
        'pv': PVPlant(site, config=pv_config)
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
    # assert time_points < time_strands


def test_hybrid_layout_wind_only(site):
    config = WindConfig.from_dict(technology['wind'])
    power_sources = {
        'wind': WindPlant(site, config=config),
        # 'solar': PVPlant(site, technology['solar'])
    }

    layout = HybridLayout(site, power_sources)
    xcoords, ycoords = layout.wind.turb_pos_x, layout.wind.turb_pos_y

    print(xcoords, ycoords)

    expected_xcoords = [1498, 867, 525, 3, 658]
    expected_ycoords = [951, 265, 74, 288, 647]

    # turbines move from `test_wind_layout` due to the solar exclusion
    for i in range(len(xcoords)):
        assert xcoords[i] == pytest.approx(expected_xcoords[i], abs=1)
        assert ycoords[i] == pytest.approx(expected_ycoords[i], abs=1)


def test_hybrid_layout_solar_only(site):
    config = PVConfig.from_dict(technology['pv'])
    power_sources = {
        # 'wind': WindPlant(site, technology['wind']),
        'pv': PVPlant(site, config=config)
    }

    layout = HybridLayout(site, power_sources)
    solar_region, buffer_region = layout.pv.solar_region.bounds, layout.pv.buffer_region.bounds

    expected_solar_region = (358.026, 451.623, 539.019, 632.617)
    expected_buffer_region = (248.026, 341.623, 649.019, 632.617)

    for i in range(4):
        assert solar_region[i] == pytest.approx(expected_solar_region[i], 1e-3)
        assert buffer_region[i] == pytest.approx(expected_buffer_region[i], 1e-3)


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


def test_detailed_pv_properties(site):
    SYSTEM_CAPACITY_DEFAULT = 50002.22178
    SUBARRAY1_NSTRINGS_DEFAULT = 13435
    SUBARRAY1_MODULES_PER_STRING_DEFAULT = 12
    INVERTER_COUNT_DEFAULT = 99
    CEC_V_MP_REF_DEFAULT = 54.7
    CEC_I_MP_REF_DEFAULT = 5.67
    INV_SNL_PACO_DEFAULT = 753200
    DC_AC_RATIO_DEFAULT = 0.67057

    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hopp/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

    # Verify the values in the pvsamv1_basic_params.json config file are as expected
    assert tech_config['system_capacity'] == approx(SYSTEM_CAPACITY_DEFAULT, 1e-3)
    assert tech_config['subarray1_nstrings'] == SUBARRAY1_NSTRINGS_DEFAULT
    assert tech_config['subarray1_modules_per_string'] == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    assert tech_config['inverter_count'] == INVERTER_COUNT_DEFAULT
    assert tech_config['cec_v_mp_ref'] == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert tech_config['cec_i_mp_ref'] == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert tech_config['inv_snl_paco'] == approx(INV_SNL_PACO_DEFAULT, 1e-3)

    # Create a detailed PV plant with the pvsamv1_basic_params.json config file
    config = DetailedPVConfig.from_dict({'tech_config': tech_config})
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    # Verify that the detailed PV plant has the same values as in the config file
    def verify_defaults():
        assert detailed_pvplant.value('system_capacity') == approx(SYSTEM_CAPACITY_DEFAULT, 1e-3)
        assert detailed_pvplant.value('subarray1_nstrings') == SUBARRAY1_NSTRINGS_DEFAULT
        assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
        assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT
        assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
        assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
        assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
        assert detailed_pvplant.dc_ac_ratio == approx(DC_AC_RATIO_DEFAULT, 1e-3)
    verify_defaults()

    # Modify system capacity and check that values update correctly
    detailed_pvplant.value('system_capacity', 20000)
    assert detailed_pvplant.value('system_capacity') == approx(20000.889, 1e-6)
    assert detailed_pvplant.value('subarray1_nstrings') == 5374
    assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    assert detailed_pvplant.value('inverter_count') == 40
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
    # The dc_ac_ratio changes because the inverter_count is a function of the system capacity, and it is rounded to an integer.
    # Changes to the inverter count do not influence the system capacity, therefore the dc_ac_ratio does not adjust back to the original value
    assert detailed_pvplant.dc_ac_ratio == approx(0.6639, 1e-3)
    # Reset system capacity back to the default value to verify values update correctly
    detailed_pvplant.value('system_capacity', SYSTEM_CAPACITY_DEFAULT)
    # The dc_ac_ratio is not noticeably affected because the inverter_count, calculated from the prior dc_ac_ratio, barely changed when rounded 
    assert detailed_pvplant.dc_ac_ratio == approx(0.6639, 1e-3)
    assert detailed_pvplant.value('system_capacity') == approx(SYSTEM_CAPACITY_DEFAULT, 1e-3)
    assert detailed_pvplant.value('subarray1_nstrings') == SUBARRAY1_NSTRINGS_DEFAULT
    assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    # The inverter count did not change back to the default value because the dc_ac_ratio did not change back to the default value,
    # and unlike the UI, there is no 'desired' dc_ac_ratio that is used to calculate the inverter count, only the prior dc_ac_ratio
    assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT + 1
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == approx(0.664, 1e-3)

    # Reinstantiate (reset) the detailed PV plant
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    # Modify the number of strings and verify that values update correctly
    detailed_pvplant.value('subarray1_nstrings', 10000)
    assert detailed_pvplant.value('system_capacity') == approx(37217.88, 1e-3)
    assert detailed_pvplant.value('subarray1_nstrings') == 10000
    assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == approx(0.499, 1e-3)
    # Reset the number of strings back to the default value to verify other values reset back to their defaults
    detailed_pvplant.value('subarray1_nstrings', SUBARRAY1_NSTRINGS_DEFAULT)
    verify_defaults()

    # Reinstantiate (reset) the detailed PV plant
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    # Modify the modules per string and verify that values update correctly
    detailed_pvplant.value('subarray1_modules_per_string', 10)
    assert detailed_pvplant.value('system_capacity') == approx(41668.52, 1e-3)
    assert detailed_pvplant.value('subarray1_nstrings') == SUBARRAY1_NSTRINGS_DEFAULT
    assert detailed_pvplant.value('subarray1_modules_per_string') == 10
    assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == approx(0.559, 1e-3)
    # Reset the modules per string back to the default value to verify other values reset back to their defaults
    detailed_pvplant.value('subarray1_modules_per_string', SUBARRAY1_MODULES_PER_STRING_DEFAULT)
    verify_defaults()

    # Reinstantiate (reset) the detailed PV plant
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    # Change the PV module and verify that values update correctly
    # (SunPower PL-SUNP-SPR-215)
    default_pv_module = detailed_pvplant.get_pv_module(only_ref_vals=False)
    module_params = {
        'module_model':         1,      # cec
        'module_aspect_ratio':  1.95363,
        'cec_area':             1.244,
        'cec_a_ref':            1.87559,
        'cec_adjust':           13.0949,
        'cec_alpha_sc':         0.0020822,
        'cec_beta_oc':          -0.134854,
        'cec_gamma_r':          -0.3904,
        'cec_i_l_ref':          5.81,
        'cec_i_mp_ref':         5.4,
        'cec_i_o_ref':          3.698e-11,
        'cec_i_sc_ref':         5.8,
        'cec_n_s':              72,
        'cec_r_s':              0.514452,
        'cec_r_sh_ref':         298.34,
        'cec_t_noct':           44.7,
        'cec_v_mp_ref':         39.8,
        'cec_v_oc_ref':         48.3,
        'cec_temp_corr_mode':   0,
        'cec_is_bifacial':      0,
        'cec_bifacial_transmission_factor':         0,
        'cec_bifaciality':                          0,
        'cec_bifacial_ground_clearance_height':     0,
        'cec_standoff':                             6,
        'cec_height':                               0,
        'cec_transient_thermal_model_unit_mass':    0,
    }
    detailed_pvplant.set_pv_module(module_params)
    assert detailed_pvplant.value('system_capacity') == approx(34649.402, 1e-3)
    assert detailed_pvplant.value('subarray1_nstrings') == SUBARRAY1_NSTRINGS_DEFAULT
    assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(module_params['cec_v_mp_ref'], 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(module_params['cec_i_mp_ref'], 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(INV_SNL_PACO_DEFAULT, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == approx(0.465, 1e-3)
    # Reset the PV module back to the default module to verify other values reset back to their defaults
    detailed_pvplant.set_pv_module(default_pv_module)
    verify_defaults()

    # Reinstantiate (reset) the detailed PV plant
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    # Change the inverter and verify that values update correctly
    # (Yaskawa Solectria Solar: SGI 500XTM)
    default_inverter = detailed_pvplant.get_inverter(only_ref_vals=False)
    inverter_params = {
        'inverter_model':       0,      # cec
        'mppt_low_inverter':    545,
        'mppt_hi_inverter':     820,
        'inv_num_mppt':         1,
        'inv_tdc_cec_db':       ((1300, 50, -0.02, 53, -0.47),),
        'inv_snl_c0':           -1.81149e-8,
        'inv_snl_c1':           1.11794e-5,
        'inv_snl_c2':           0.000884631,
        'inv_snl_c3':           -0.000339117,
        'inv_snl_paco':         507000,
        'inv_snl_pdco':         522637,
        'inv_snl_pnt':          88.78,
        'inv_snl_pso':          2725.47,
        'inv_snl_vdco':         615,
        'inv_snl_vdcmax':       820,
    }
    detailed_pvplant.set_inverter(inverter_params)
    assert detailed_pvplant.value('system_capacity') == approx(SYSTEM_CAPACITY_DEFAULT, 1e-3)
    assert detailed_pvplant.value('subarray1_nstrings') == SUBARRAY1_NSTRINGS_DEFAULT
    assert detailed_pvplant.value('subarray1_modules_per_string') == SUBARRAY1_MODULES_PER_STRING_DEFAULT
    assert detailed_pvplant.value('inverter_count') == INVERTER_COUNT_DEFAULT
    assert detailed_pvplant.value('cec_v_mp_ref') == approx(CEC_V_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('cec_i_mp_ref') == approx(CEC_I_MP_REF_DEFAULT, 1e-3)
    assert detailed_pvplant.value('inv_snl_paco') == approx(507000, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == approx(0.996, 1e-3)
    # Reset the inverter back to the default inverter to verify other values reset back to their defaults
    detailed_pvplant.set_inverter(default_inverter)
    verify_defaults()


def test_detailed_pv_plant_custom_design(site):
    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hopp/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

    # Modify the inputs for a custom design
    target_solar_kw = 3e5
    target_dc_ac_ratio = 1.34
    modules_per_string = 12
    module_power = tech_config['cec_v_mp_ref'] * tech_config['cec_i_mp_ref'] * 1e-3         # [kW]
    inverter_power = tech_config['inv_snl_paco'] * 1e-3                                     # [kW]
    n_inputs_inverter = None
    n_inputs_combiner = None

    n_strings, n_combiners, n_inverters, calculated_system_capacity = size_electrical_parameters(
        target_system_capacity=target_solar_kw,
        target_dc_ac_ratio=target_dc_ac_ratio,
        modules_per_string=modules_per_string,
        module_power=module_power,
        inverter_power=inverter_power,
        n_inputs_inverter=n_inputs_inverter,
        n_inputs_combiner=n_inputs_combiner,
    )

    tech_config['system_capacity'] = calculated_system_capacity
    tech_config['subarray1_nstrings'] = n_strings
    tech_config['subarray1_modules_per_string'] = modules_per_string
    tech_config['n_inverters'] = n_inverters

    # Create a detailed PV plant with the pvsamv1_basic_params.json config file
    config = DetailedPVConfig.from_dict({'tech_config': tech_config})
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    assert detailed_pvplant.system_capacity == pytest.approx(calculated_system_capacity, 1e-3)
    assert detailed_pvplant.dc_ac_ratio == pytest.approx(1.341, 1e-3)

    detailed_pvplant.simulate(target_solar_kw)

    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_clip_loss_percent < 1.3
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_eff_loss_percent < 3
    assert detailed_pvplant._system_model.Outputs.annual_ac_gross / detailed_pvplant._system_model.Outputs.annual_dc_gross > 0.91


def test_detailed_pv_plant_modify_after_init(site):
    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hopp/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

    # Create a detailed PV plant with the pvsamv1_basic_params.json config file
    config = DetailedPVConfig.from_dict({'tech_config': tech_config})
    detailed_pvplant = DetailedPVPlant(
        site=site,
        config=config
    )

    assert detailed_pvplant.system_capacity == pytest.approx(tech_config['system_capacity'], 1e-3)
    assert detailed_pvplant.dc_ac_ratio == pytest.approx(0.671, 1e-3)
    
    detailed_pvplant.simulate(5e5)

    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_clip_loss_percent < 1.2
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_eff_loss_percent < 3
    assert detailed_pvplant._system_model.Outputs.annual_ac_gross / detailed_pvplant._system_model.Outputs.annual_dc_gross > 0.91
    assert detailed_pvplant.annual_energy_kwh * 1e-6 == pytest.approx(108.239, abs=10)

    # modify dc ac ratio
    detailed_pvplant.dc_ac_ratio = 1.341
    detailed_pvplant.simulate(5e5)
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_clip_loss_percent < 1.2
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_eff_loss_percent < 3
    assert detailed_pvplant._system_model.Outputs.annual_ac_gross / detailed_pvplant._system_model.Outputs.annual_dc_gross > 0.91
    assert detailed_pvplant.annual_energy_kwh * 1e-6 == pytest.approx(107.502, abs=10)

    # modify system capacity
    detailed_pvplant.system_capacity_kw *= 2
    detailed_pvplant.simulate(5e5)
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_clip_loss_percent < 1.2
    assert detailed_pvplant._system_model.Outputs.annual_ac_inv_eff_loss_percent < 3
    assert detailed_pvplant._system_model.Outputs.annual_ac_gross / detailed_pvplant._system_model.Outputs.annual_dc_gross > 0.91
    assert detailed_pvplant.annual_energy_kwh * 1e-6 == pytest.approx(215.0, abs=10)
