from pytest import approx, raises, fixture
from pathlib import Path
import numpy as np
import copy

import openmdao.api as om

from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml

from greenheart.tools.optimization.openmdao import GreenHeartComponent, HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent
from greenheart.tools.optimization.gc_PoseOptimization import PoseOptimization
from greenheart.tools.optimization.gc_run_greenheart import run_greenheart
from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
import unittest

solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
floris_input_filename = Path(__file__).absolute().parent / "inputs" / "floris_input.yaml"
hopp_config_filename = Path(__file__).absolute().parent / "inputs" / "hopp_config.yaml"
hopp_config_steel_ammonia_filename = Path(__file__).absolute().parent / "test_hydrogen" / "input_files" / "plant" / "hopp_config.yaml"
greenheart_config_onshore_filename = Path(__file__).absolute().parent / "test_hydrogen" / "input_files" / "plant" / "greenheart_config_onshore.yaml"
turbine_config_filename = Path(__file__).absolute().parent / "test_hydrogen" / "input_files" / "turbines" / "osw_18MW.yaml"
floris_input_filename_steel_ammonia = Path(__file__).absolute().parent / "test_hydrogen" / "input_files" / "floris" / "floris_input_osw_18MW.yaml"
rtol = 1E-5

class TestBoundaryDistanceComponent(unittest.TestCase):

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
    
    def test_distance_inside(self):
        
        assert self.prob["boundary_distance_vec"][0] == 200.0

    # def test_derivative_inside(self):

        
    #     assert total_capex == 680590.3412708649
    
    def test_distance_outside(self):

        self.prob.set_val('boundary_constraint.turbine_x', np.array([-2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0)
        self.prob.set_val('boundary_constraint.turbine_y', np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0)
        self.prob.run_model()
        
        assert self.prob["boundary_distance_vec"][0] == -200.0

    # def test_derivative_outside(self):
        
    #     assert total_capex == 680590.3412708649

class TestTurbineDistanceComponent(unittest.TestCase):

    turbine_x = np.array([2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 4.0])*100.0

    model = om.Group()

    model.add_subsystem('boundary_constraint', TurbineDistanceComponent(turbine_x_init=turbine_x, turbine_y_init=turbine_y), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()
    
    def test_distance_between_turbines(self):
        expected_distances = np.array([200.0, np.sqrt(400**2 + 200**2), np.sqrt(2*200**2)])
        for i in range(len(self.turbine_x)):
            with self.subTest(f"for element {i}"):
                assert self.prob["spacing_vec"][i] == expected_distances[i]

class TestHoppComponent(unittest.TestCase):

    def setUp(self):
        self.turbine_x = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])*100.0
        self.turbine_y = np.array([2.0, 2.0, 4.0, 4.0, 8.0, 8.0])*100.0

        self.hybrid_config_dict = load_yaml(hopp_config_filename)

        self.hybrid_config_dict["site"]["solar_resource_file"] = solar_resource_file
        self.hybrid_config_dict["site"]["solar"] = "true"

        self.hybrid_config_dict["site"]["wind_resource_file"] = wind_resource_file
        self.hybrid_config_dict["technologies"]["wind"]["floris_config"] = floris_input_filename


        self.hybrid_config_dict["site"]["desired_schedule"] = [80000.0]*8760
        self.hybrid_config_dict["technologies"]["battery"] = {"system_capacity_kwh": 400,
                                                              "system_capacity_kw": 100,
                                                              "minimum_SOC": 20.0,
                                                              "maximum_SOC": 100.0,
                                                              "initial_SOC": 90.0}
        self.hybrid_config_dict["config"]["dispatch_options"] = {"battery_dispatch": "load_following_heuristic",
                                                                "solver": "cbc",
                                                                "n_look_ahead_periods": 48,
                                                                "grid_charging": False,
                                                                "pv_charging_only": False,
                                                                "include_lifecycle_count": False}

        self.design_variables = ["pv_capacity_kw", "turbine_x"]

        technologies = self.hybrid_config_dict["technologies"]
        self.solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
        self.hybrid_config_dict["technologies"] = self.solar_wind_hybrid
        self.hi = HoppInterface(self.hybrid_config_dict)

        model = om.Group()

        model.add_subsystem('hopp', HOPPComponent(hi=self.hi, verbose=False, turbine_x_init=self.turbine_x, turbine_y_init=self.turbine_y, design_variables=self.design_variables), promotes=["*"])

        self.prob = om.Problem(model)
        self.prob.setup()

        self.prob.run_model()
    
    def test_inputs(self):
        with self.subTest("turbine_x"):
            assert self.prob.get_val('turbine_x')[0] == approx(self.turbine_x[0])
        with self.subTest("pv_capacity_kw"):
            assert self.prob.get_val('pv_capacity_kw')[0] == approx(self.hybrid_config_dict["technologies"]["pv"]["system_capacity_kw"])

    def test_changes(self):
        new_pv_capacity_kw = 50
        new_x = copy.deepcopy(self.turbine_x)
        new_x[0] = 0.0
        self.prob.set_val("pv_capacity_kw", new_pv_capacity_kw)
        self.prob.set_val("turbine_x", new_x)
        self.prob.run_model()
        with self.subTest("turbine_x"):
            assert self.prob.get_val('turbine_x')[0] == approx(new_x[0])
        with self.subTest("pv_capacity_kw_new"):
            assert self.prob.get_val('pv_capacity_kw')[0] == new_pv_capacity_kw

    def test_costs(self):
        with self.subTest("pv_capex"):
            assert self.prob.get_val('pv_capex')[0] == approx(14400000.0)
        # with self.subTest("pv_opex"):
        #     assert self.prob.get_val('pv_opex')[0] == approx(0.0)
        with self.subTest("wind_capex"):
            assert self.prob.get_val('wind_capex')[0] == approx(43620000.0)
        # with self.subTest("wind_opex"):
        #     assert self.prob.get_val('wind_opex')[0] == approx(0.0)
        with self.subTest("battery_capex"):
            assert self.prob.get_val('battery_capex')[0] == approx(163100.0)
        # with self.subTest("battery_opex"):
        #     assert self.prob.get_val('battery_opex')[0] == approx(0.0)
        with self.subTest("hybrid_electrical_generation_capex"):
            assert self.prob.get_val('hybrid_electrical_generation_capex')[0] == approx(58183100.0)
        with self.subTest("total capex equals sum"):
            assert self.prob.get_val('hybrid_electrical_generation_capex')[0] == approx(14400000.0 + 43620000.0 + 163100.0)
        # with self.subTest("hybrid_electrical_generation_opex"):
        #     assert self.prob.get_val('hybrid_electrical_generation_opex')[0] == approx(0.0)

class TestGreenHeartComponent(unittest.TestCase):

    def setUp(self):
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
    
        model = om.Group()

        model.add_subsystem('greenheart', GreenHeartComponent(config=config, design_variables=["electrolyzer_rating_kw"]),  promotes=["*"])

        self.prob = om.Problem(model)
        self.prob.setup()

        self.prob.run_model()
    
    def test_costs(self):
        # TODO base this test value on something
        with self.subTest("lcoh"):
            assert self.prob["lcoh"] == approx(3.040736244214041, rel=rtol)

        # TODO base this test value on something
        with self.subTest("lcoe"):
            assert self.prob["lcoe"] == approx(0.034869649135212274, rel=rtol)

        # TODO base this test value on something
        with self.subTest("steel_finance"):
            lcos_expected = 1348.5863267221866

            assert self.prob["lcos"]  == approx(lcos_expected, rel=rtol)

        # TODO base this test value on something
        with self.subTest("ammonia_finance"):
            lcoa_expected = 1.0419316870652462

            assert self.prob["lcoa"]  == approx(lcoa_expected, rel=rtol)

class TestRunGreenHeartRunOnly(unittest.TestCase):

    def setUp(self):
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
    
        self.prob, self.config = run_greenheart(config, run_only=True)
    
    def test_costs_run_only(self):
        # TODO base this test value on something
        with self.subTest("lcoh"):
            assert self.prob["lcoh"] == approx(3.040736244214041, rel=rtol)

        # TODO base this test value on something
        with self.subTest("lcoe"):
            assert self.prob["lcoe"] == approx(0.034869649135212274, rel=rtol)

        # TODO base this test value on something
        with self.subTest("steel_finance"):
            lcos_expected = 1348.5863267221866

            assert self.prob["lcos"]  == approx(lcos_expected, rel=rtol)

        # TODO base this test value on something
        with self.subTest("ammonia_finance"):
            lcoa_expected = 1.0419316870652462

            assert self.prob["lcoa"]  == approx(lcoa_expected, rel=rtol)

class TestRunGreenHeartOptimize(unittest.TestCase):

    def setUp(self):
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
                    "lower": 150000.0,
                    "upper": 200000.0,
                    "units": "kW",
                }
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
    
        self.prob, self.config = run_greenheart(config, run_only=False)

        cr = om.CaseReader(Path(__file__).absolute().parent / "output" / "recorder.sql")

        # get initial LCOH
        case = cr.get_case(0)
        self.lcoh_init = case.get_val("lcoh", units='USD/kg')[0]
        # get final LCOH
        case = cr.get_case(-1)
        self.lcoh_final = case.get_val("lcoh", units='USD/kg')[0]
    
    def test_costs_optimize(self):
        # TODO base this test value on something
        with self.subTest("lcoh"):
            assert self.lcoh_final < self.lcoh_init
