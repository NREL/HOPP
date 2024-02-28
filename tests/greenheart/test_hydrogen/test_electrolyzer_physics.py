from pytest import approx
import pandas as pd
import greenheart.tools.eco.electrolysis as he_elec
from hopp.utilities import load_yaml
import os
import unittest
import numpy as np

dirname = os.path.dirname(__file__)
input_library_path = os.path.join(dirname, "input_files","plant")
project_life_years = 30

TOL = 1e-3


# greenheart_config_filename = os.path.join(input_library_path,"GS_greenheart_config.yaml")
# greenheart_config = load_yaml(greenheart_config_filename)
# electrolyzer_capacity_BOL_MW = he_elec.size_electrolyzer_for_hydrogen_demand(greenheart_config)
# greenheart_config["electrolyzer"]["rating"] = electrolyzer_capacity_BOL_MW
# gh_new = he_elec.check_capacity_based_on_clusters(greenheart_config)
# gh_new["electrolyzer"]["rating"]
[]

class TestOffGridSimulation(unittest.TestCase):
    """
    Test is based on land-based off-grid test case for TX
    - Input power profile 'GS_offgrid_power_signal.csv' is wind farm generation profile
        - uses PySAM as the system model
        - Uses NREL 7MW reference wind turbine Cp curve
        - Includes [] turbines
    """
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestOffGridSimulation, self).setUpClass()
        electrolyzer_size_mw = 960
        grid_connected = False
        cluster_rating_MW = 40

        power_profile_filename = 'GS_offgrid_power_signal.csv'
        offgrid_power_profile_filename = os.path.join(input_library_path,power_profile_filename)
        offgrid_power_profile = pd.read_csv(offgrid_power_profile_filename)

        greenheart_config_filename = os.path.join(input_library_path,"GS_greenheart_config.yaml")
        greenheart_config = load_yaml(greenheart_config_filename)

        greenheart_config['electrolyzer']['rating'] = electrolyzer_size_mw
        greenheart_config['project_parameters']['grid_connection'] = grid_connected
        greenheart_config['electrolyzer']['cluster_rating_MW'] = cluster_rating_MW

        electrolyzer_physics_results = he_elec.run_electrolyzer_physics(offgrid_power_profile, project_life_years, greenheart_config, wind_resource = None, design_scenario='off-grid', show_plots=False, save_plots=False, verbose=False)
        self.H2_Res = electrolyzer_physics_results["H2_Results"]["new_H2_Results"]
        self.power_profile = electrolyzer_physics_results["electrical_generation_timeseries"]
    
    def test_AEP_input_power(self):
        assert self.H2_Res['Sim: Total Input Power [kWh]'] == approx(3383382801.267635,TOL)
    
    def test_electrolyzer_rated_BOL_H2_production_rate(self):
        assert self.H2_Res['Rated BOL: H2 Production [kg/hr]'] == approx(17579.2991094574,TOL)
        
    def test_electrolyzer_rated_BOL_power_consumption(self):
        assert self.H2_Res['Rated BOL: Power Consumed [kWh]'] == approx(960018.4015366472,TOL)
        
    def test_electrolyzer_rated_BOL_efficiency(self):
        assert self.H2_Res['Rated BOL: Efficiency [kWh/kg]'] == approx(54.61073251891887, TOL)
        
    def test_simulation_capacity_factor(self):
        assert self.H2_Res['Sim: Capacity Factor'] == approx(0.4104064586084918, TOL)
    
    def test_simulation_off_cycles(self):
        assert self.H2_Res['Sim: Total Stack Off-Cycles'] == 9483.0
        
    def test_simulation_operation_time(self):
        assert self.H2_Res['Sim: Active Time / Sim Time'] == approx(0.8112157534246575,TOL)

    def test_simulation_H2_production(self):
        assert self.H2_Res['Sim: Total H2 Produced [kg]'] == approx(63200403.136826776,TOL)

    def test_simulation_warmup_losses(self):
        assert self.H2_Res['Sim: H2 Warm-Up Losses [kg]'] == approx(240892.61322440396,TOL)
    
    def test_stack_life(self):
        assert self.H2_Res['Stack Life [hrs]'] == approx(23633.651537254333,TOL)
        
    def test_time_between_replacement(self):
        assert self.H2_Res['Time Until Replacement [hrs]'] == approx(28947.55906846945,TOL)

    def test_lifetime_capacity_factor(self):
        assert self.H2_Res['Life: Capacity Factor'] == approx(0.3889240837784226,TOL)
        


class TestGridOnlySimulation(unittest.TestCase):
    """
    Test is based on land-based grid-only test case for TX
    - Power required for hydrogen demand is titled: 'GS_gridonly_power_signal.csv'
    - Has a hydrogen demand of 8366.311517 kg-H2/hr
    """
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestGridOnlySimulation, self).setUpClass()
        electrolyzer_size_mw = 480
        grid_connected = True
        hydrogen_dmd = 8366.311517 #kg-H2/hr
        
        greenheart_config_filename = os.path.join(input_library_path,"GS_greenheart_config.yaml")
        greenheart_config = load_yaml(greenheart_config_filename)

        greenheart_config['electrolyzer']['rating'] = electrolyzer_size_mw
        greenheart_config['project_parameters']['grid_connection'] = grid_connected
        greenheart_config["electrolyzer"]["grid_input_signal"] = 'hydrogen'
        greenheart_config['electrolyzer']['hydrogen_dmd'] = hydrogen_dmd

        electrolyzer_physics_results = he_elec.run_electrolyzer_physics(None, project_life_years, greenheart_config, wind_resource = None, design_scenario='off-grid', show_plots=False, save_plots=False, verbose=False)
        self.H2_Res = electrolyzer_physics_results["H2_Results"]["new_H2_Results"]
        self.power_profile = electrolyzer_physics_results["electrical_generation_timeseries"]
    
    def test_AEP_input_power(self):
        assert self.H2_Res['Sim: Total Input Power [kWh]'] == approx(4008524165.6783223,TOL)
    
    def test_AEP_input_power2(self):
        assert sum(self.power_profile) == approx(4008524165.6783223,TOL)

    def test_input_power_start_end(self):
        assert self.power_profile[-1] > self.power_profile[0]

    def test_electrolyzer_rated_BOL_H2_production_rate(self):
        assert self.H2_Res['Rated BOL: H2 Production [kg/hr]'] == approx(8789.649554728701,TOL)
        
    def test_electrolyzer_rated_BOL_power_consumption(self):
        assert self.H2_Res['Rated BOL: Power Consumed [kWh]'] == approx(480009.2007683235,TOL)

    def test_simulation_capacity_factor(self):
        assert self.H2_Res['Sim: Capacity Factor'] == approx(0.9518373167475502, TOL)

    def test_simulation_operation_time(self):
        assert self.H2_Res['Sim: Active Time / Sim Time'] == 1.0

    def test_simulation_off_cycles(self):
        assert self.H2_Res['Sim: Total Stack Off-Cycles'] == 0

    def test_simulation_warmup_losses(self):
        assert self.H2_Res['Sim: H2 Warm-Up Losses [kg]'] == 0

    def test_stack_life(self):
        assert self.H2_Res['Stack Life [hrs]'] == approx(78049.89571256597,TOL)
        
    def test_time_between_replacement(self):
        assert self.H2_Res['Time Until Replacement [hrs]'] == self.H2_Res['Stack Life [hrs]']

    def test_lifetime_capacity_factor(self):
        assert self.H2_Res['Life: Capacity Factor'] == approx(0.9518373167475502,TOL)
    
    def test_power_profile_start(self):
        power_profile_filename = 'GS_gridonly_power_signal.csv'
        grid_power_profile_filename = os.path.join(input_library_path,power_profile_filename)
        grid_power_profile = pd.read_csv(grid_power_profile_filename,index_col='Unnamed: 0')
        
        assert self.power_profile[0] == approx(grid_power_profile['combined_hybrid_power_production_hopp'].values[0],TOL)
    
    def test_power_profile_end(self):
        power_profile_filename = 'GS_gridonly_power_signal.csv'
        grid_power_profile_filename = os.path.join(input_library_path,power_profile_filename)
        grid_power_profile = pd.read_csv(grid_power_profile_filename,index_col='Unnamed: 0')
        assert self.power_profile[-1] == approx(grid_power_profile['combined_hybrid_power_production_hopp'].values[-1],TOL)

class TestElectrolysisTools(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestElectrolysisTools, self).setUpClass()
        greenheart_config_filename = os.path.join(input_library_path,"GS_greenheart_config.yaml")
        self.greenheart_config = load_yaml(greenheart_config_filename)
    
    def test_BOL_efficiency(self):
        bol_eff = he_elec.get_electrolyzer_BOL_efficiency(self.greenheart_config)
        assert bol_eff == 54.61
    
    def test_electrolyzer_sizing_for_hydrogen_demand(self):
        electrolyzer_capacity_BOL_MW = he_elec.size_electrolyzer_for_hydrogen_demand(self.greenheart_config)
        assert np.ceil(electrolyzer_capacity_BOL_MW) == 457

    def test_check_number_electrolyzer_clusters(self):
        gh_new = he_elec.check_capacity_based_on_clusters(self.greenheart_config)
        remainder = gh_new["electrolyzer"]["rating"] % gh_new["electrolyzer"]["cluster_rating_MW"]
        assert remainder == 0

    def test_grid_connected_electrolyzer_sizing(self):
        
        electrolyzer_capacity_BOL_MW = he_elec.size_electrolyzer_for_hydrogen_demand(self.greenheart_config)
        self.greenheart_config["electrolyzer"]["rating"] = electrolyzer_capacity_BOL_MW

        gh_new = he_elec.check_capacity_based_on_clusters(self.greenheart_config)
        assert gh_new["electrolyzer"]["rating"] == 480
        