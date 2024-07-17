from pytest import approx
from pathlib import Path
import numpy as np
import copy
import os

import openmdao.api as om

from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml
from hopp import ROOT_DIR

from greenheart.tools.optimization.openmdao import GreenHeartComponent, HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent
from greenheart.tools.optimization.gc_run_greenheart import run_greenheart
from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
from hopp import ROOT_DIR

solar_resource_file = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
floris_input_filename = Path(__file__).absolute().parent / "inputs" / "floris_input.yaml"
hopp_config_filename = Path(__file__).absolute().parent / "inputs" / "hopp_config.yaml"
hopp_config_steel_ammonia_filename = Path(__file__).absolute().parent / "input_files" / "plant" / "hopp_config.yaml"
greenheart_config_onshore_filename = Path(__file__).absolute().parent / "input_files" / "plant" / "greenheart_config_onshore.yaml"
turbine_config_filename = Path(__file__).absolute().parent / "input_files" / "turbines" / "osw_18MW.yaml"
floris_input_filename_steel_ammonia = Path(__file__).absolute().parent / "input_files" / "floris" / "floris_input_osw_18MW.yaml"


from ORBIT.core.library import initialize_library

dirname = os.path.dirname(__file__)
orbit_library_path = os.path.join(dirname, "input_files/")

initialize_library(orbit_library_path)
offshore_hopp_config_wind_wave_solar_battery = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery.yaml"
)
offshore_greenheart_config = os.path.join(
    orbit_library_path, f"plant/greenheart_config.yaml"
)
offshore_turbine_model = "osw_18MW"
offshore_turbine_config = os.path.join(
    orbit_library_path, f"turbines/{offshore_turbine_model}.yaml"
)
offshore_floris_config = os.path.join(
    orbit_library_path, f"floris/floris_input_{offshore_turbine_model}.yaml"
)
offshore_orbit_config = os.path.join(
    orbit_library_path, f"plant/orbit-config-{offshore_turbine_model}-stripped.yaml"
)

rtol = 1E-5

def setup_hopp():
    turbine_x = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])*100.0
    turbine_y = np.array([2.0, 2.0, 4.0, 4.0, 8.0, 8.0])*100.0

    hybrid_config_dict = load_yaml(hopp_config_filename)

    hybrid_config_dict["site"]["solar_resource_file"] = solar_resource_file
    hybrid_config_dict["site"]["solar"] = "true"

    hybrid_config_dict["site"]["wind_resource_file"] = wind_resource_file
    hybrid_config_dict["technologies"]["wind"]["floris_config"] = floris_input_filename


    hybrid_config_dict["site"]["desired_schedule"] = [80000.0]*8760
    hybrid_config_dict["technologies"]["battery"] = {"system_capacity_kwh": 400,
                                                            "system_capacity_kw": 100,
                                                            "minimum_SOC": 20.0,
                                                            "maximum_SOC": 100.0,
                                                            "initial_SOC": 90.0}
    hybrid_config_dict["config"]["dispatch_options"] = {"battery_dispatch": "load_following_heuristic",
                                                            "solver": "cbc",
                                                            "n_look_ahead_periods": 48,
                                                            "grid_charging": False,
                                                            "pv_charging_only": False,
                                                            "include_lifecycle_count": False}

    design_variables = ["pv_capacity_kw", "turbine_x"]

    technologies = hybrid_config_dict["technologies"]
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    hybrid_config_dict["technologies"] = solar_wind_hybrid
    hi = HoppInterface(hybrid_config_dict)

    model = om.Group()

    model.add_subsystem('hopp', HOPPComponent(hi=hi, verbose=False, turbine_x_init=turbine_x, turbine_y_init=turbine_y, design_variables=design_variables), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()

    return prob, turbine_x, hybrid_config_dict

def setup_greenheart():
    config = GreenHeartSimulationConfig(
    filename_hopp_config=hopp_config_steel_ammonia_filename,
    filename_greenheart_config=greenheart_config_onshore_filename,
    filename_turbine_config=turbine_config_filename,
    filename_floris_config=floris_input_filename_steel_ammonia,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir= str(Path(__file__).absolute().parent / "output"),
    use_profast=True,
    post_processing=False,
    incentive_option=1,
    plant_design_scenario=9,
    output_level=7,
    )
    
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_installed_cost_mw"] = 1434000.0 
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_om_per_kw"] = 29.567
    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][0] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    # set skip_financial to false for onshore wind
    config.hopp_config["config"]["simulation_options"]["wind"]["skip_financial"] = False

    
    config.greenheart_config["opt_options"] = {
            "opt_flag": True,
            "general": {
                "folder_output": "output",
                "fname_output": "test_run_greenheart_optimization",
            },
            "design_variables": {
                "electrolyzer_rating_kw": {
                    "flag": True,
                    "lower": 10000.0,
                    "upper": 200000.0,
                    "units": "kW",
                },
                "pv_capacity_kw": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "wave_capacity_kw": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "battery_capacity_kw": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "battery_capacity_kwh": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW*h",
                },
                "battery_capacity_kwh": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW*h",
                },
                "turbine_x": {
                    "flag": False,
                    "lower": 0.0,
                    "upper": 1500000.0,
                    "units": "m",
                },
                "turbine_y": {
                    "flag": False,
                    "lower": 0.0,
                    "upper": 1500000.0,
                    "units": "m",
                },
            },
            "constraints": {
                "turbine_spacing": {
                    "flag": False,
                    "lower": 0.0,
                },
                "boundary_distance": {
                    "flag": False,
                    "lower": 0.0,
                },
            "pv_to_platform_area_ratio": {
                "flag": False, 
                "upper": 1.0, # relative size of solar pv area to platform area
                },
            "user": {}
            },
            "merit_figure": "lcoh",
            "merit_figure_user": {
                "name": "lcoh",
                "max_flag": False,
                "ref": 1.0, # value of objective that scales to 1.0
            },
            "driver": {
                "optimization": {
                    "flag": True,
                    "solver": "SLSQP",
                    "tol": 1E-6,
                    "max_major_iter": 1,
                    "max_minor_iter": 2,
                    "gradient_method": "openmdao",
                    # "time_limit": 10, # (sec) optional
                    # "hist_file_name": "snopt_history.txt", # optional
                    "verify_level": -1, # optional
                    "step_calc": None,
                    "form": "forward", # type of finite differences to use, can be one of ["forward", "backward", "central"]
                    "debug_print": False,
                },
                "design_of_experiments": {
                    "flag": False,
                    "run_parallel": False,
                    "generator": "FullFact", # [Uniform, FullFact, PlackettBurman, BoxBehnken, LatinHypercube]
                    "num_samples": 1, # Number of samples to evaluate model at (Uniform and LatinHypercube only)
                    "seed": 2,
                    "levels":  50, #  Number of evenly spaced levels between each design variable lower and upper bound (FullFactorial only)
                    "criterion": None, # [None, center, c, maximin, m, centermaximin, cm, correelation, corr]
                    "iterations": 1,
                    "debug_print": False
                },
                "step_size_study": {
                    "flag": False  
                },
            },
            "recorder": {
                "flag": True,
                "file_name": str(Path(__file__).absolute().parent / "output" / "recorder.sql"),
                "includes": False,
            },
        }

    return config
    

def test_boundary_distance_component(subtests):
    turbine_x = np.array([2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0
    
    config_dict = load_yaml(hopp_config_filename)
    config_dict["site"]["wind_resource_file"] = wind_resource_file
    config_dict["technologies"]["wind"]["floris_config"] = floris_input_filename
    hi = HoppInterface(config_dict)

    model = om.Group()

    model.add_subsystem('boundary_constraint', BoundaryDistanceComponent(hopp_interface=hi, turbine_x_init=turbine_x, turbine_y_init=turbine_y), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()
    
    with subtests.test("test distance inside"):
        assert prob["boundary_distance_vec"][0] == 200.0
    
    with subtests.test("test_distance_outside"):

        prob.set_val('boundary_constraint.turbine_x', np.array([-2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0)
        prob.set_val('boundary_constraint.turbine_y', np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0)
        prob.run_model()
        
        assert prob["boundary_distance_vec"][0] == -200.0

    # TODO add analytic derivatives and test

def test_turbine_distance_component(subtests):

    turbine_x = np.array([2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 4.0])*100.0

    model = om.Group()

    model.add_subsystem('boundary_constraint', TurbineDistanceComponent(turbine_x_init=turbine_x, turbine_y_init=turbine_y), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()
    
    expected_distances = np.array([200.0, np.sqrt(400**2 + 200**2), np.sqrt(2*200**2)])
    for i in range(len(turbine_x)):
        with subtests.test(f"for element {i}"):
            assert prob["spacing_vec"][i] == expected_distances[i]

def test_hopp_component(subtests):
    
    prob, turbine_x, hybrid_config_dict = setup_hopp()
    
    with subtests.test("inputs_turbine_x"):
        assert prob.get_val('turbine_x')[0] == approx(turbine_x[0])

    with subtests.test("inputs_pv_capacity_kw"):
        assert prob.get_val('pv_capacity_kw')[0] == approx(hybrid_config_dict["technologies"]["pv"]["system_capacity_kw"])

    with subtests.test("costs_pv_capex"):
        assert prob.get_val('pv_capex')[0] == approx(14400000.0)

    # with subtests.test("costs_pv_opex"):
    #     assert prob.get_val('pv_opex')[0] == approx(0.0)

    with subtests.test("costs_wind_capex"):
        assert prob.get_val('wind_capex')[0] == approx(43620000.0)

    # with subtests.test("cost_wind_opex"):
    #     assert prob.get_val('wind_opex')[0] == approx(0.0)

    with subtests.test("costs_battery_capex"):
        assert prob.get_val('battery_capex')[0] == approx(163100.0)

    # with subtests.test("costs_battery_opex"):
    #     assert prob.get_val('battery_opex')[0] == approx(0.0)

    with subtests.test("costs_hybrid_electrical_generation_capex"):
        assert prob.get_val('hybrid_electrical_generation_capex')[0] == approx(58183100.0)

    with subtests.test("costs_total_capex_equals_sum"):
        assert prob.get_val('hybrid_electrical_generation_capex')[0] == approx(14400000.0 + 43620000.0 + 163100.0)
        
    # with subtests.test("costs_hybrid_electrical_generation_opex"):
    #     assert prob.get_val('hybrid_electrical_generation_opex')[0] == approx(0.0)

    with subtests.test("changes_turbine_x"):
        new_x = copy.deepcopy(turbine_x)
        new_x[0] = 0.0
        prob.set_val("turbine_x", new_x)
        assert prob.get_val('turbine_x')[0] == approx(new_x[0])

    with subtests.test("changes_pv_capacity_kw_new"):
        new_pv_capacity_kw = 50
        prob.set_val("pv_capacity_kw", new_pv_capacity_kw)
        assert prob.get_val('pv_capacity_kw')[0] == new_pv_capacity_kw


def test_greenheart_component(subtests):
    
    config = setup_greenheart()

    model = om.Group()

    model.add_subsystem('greenheart', GreenHeartComponent(config=config, design_variables=["electrolyzer_rating_kw"]),  promotes=["*"])

    prob = om.Problem(model)
    prob.setup()
    prob.run_model()

    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert prob["lcoh"][0] == approx(3.1691092704830357, rel=rtol)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert prob["lcoe"][0] == approx(0.03486193, rel=rtol)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1357.046163641118
        assert prob["lcos"][0]  == approx(lcos_expected, rel=rtol)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.04190962
        assert prob["lcoa"][0]  == approx(lcoa_expected, rel=rtol)

def test_run_greenheart_run_only(subtests):
    config = setup_greenheart()
    prob, config = run_greenheart(config, run_only=True)
    
    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert prob["lcoh"][0] == approx(3.1691092704830357, rel=rtol)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert prob["lcoe"] == approx(0.03486193, rel=rtol)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1357.046163641118
        assert prob["lcos"]  == approx(lcos_expected, rel=rtol)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.04190962
        assert prob["lcoa"]  == approx(lcoa_expected, rel=rtol)

def test_run_greenheart_optimize(subtests):
    
    config = setup_greenheart()

    prob, config = run_greenheart(config, run_only=False)

    cr = om.CaseReader(Path(__file__).absolute().parent / "output" / "recorder.sql")

    # get initial LCOH
    case = cr.get_case(0)
    lcoh_init = case.get_val("lcoh", units='USD/kg')[0]

    # get final LCOH
    case = cr.get_case(-1)
    lcoh_final = case.get_val("lcoh", units='USD/kg')[0]
    
    with subtests.test("lcoh"):
        assert lcoh_final < lcoh_init





