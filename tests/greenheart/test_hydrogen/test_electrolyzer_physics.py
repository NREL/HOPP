import pytest
from pytest import approx
from pytest import fixture
import pandas as pd
import greenheart.tools.eco.electrolysis as he_elec
import greenheart.tools.plant_sizing_estimation as gh_sizing
from hopp.utilities import load_yaml
import os
import unittest
import numpy as np
from tests import TEST_ROOT_DIR

input_library_path = TEST_ROOT_DIR / "greenheart" / "test_hydrogen" / "input_files"
project_life_years = 30


TOL = 1e-3

default_config = {
    'project_parameters': 
            {
                'atb_year': 2022,
                'grid_connection': False,
                'ppa_price': 0.025,
                'hybrid_electricity_estimated_cf': 0.492,
                'project_lifetime':30,
            },
    # 'component_sizing': 
    #         {
    #             'hybrid_electricity_estimated_cf': 0.492,
    #             'electrolyzer': 
    #             {
    #                 'resize_for_enduse': False,
    #                 'size_for': 'BOL'
    #                 },
    #         },
    'electrolyzer': 
            {   'sizing': 
                    {
                    'resize_for_enduse': False,
                    'size_for': 'BOL',
                    'hydrogen_dmd': None,
                    },
                'rating': 180,
                'electrolyzer_capex': 700,
                'time_between_replacement': 62320,
                'replacement_cost_percent': 0.15,
                'cost_model': 'singlitico2021',
                'cluster_rating_MW': 40,
                'pem_control_type': 'basic',
                'include_degradation_penalty': True,
                'eol_eff_percent_loss': 13,
                'uptime_hours_until_eol': 77600,  #new
                'turndown_ratio': 0.1, #new
                # 'hydrogen_dmd': None,
            },
    'steel': 
            {   'capacity':{
                    'input_capacity_factor_estimate': 0.9,
                'annual_production_target': 1000000,
            }
            }
    # 'end_use': 
    #         {
    #             'annual_production_target': 1000000,
    #             'estimated_cf': 0.9,
    #             'product': 'steel'
    #         }
}
# NOTE: All test values came from the green steel project repository code.
# Run on Feb 21,2024 for Site 2"

# Test electrolyzer resizing for offgrid and grid
@fixture
def offgrid_resize_config():
    offgrid_resize_config = default_config.copy()
    offgrid_resize_config['project_parameters']['grid_connection'] = False
    offgrid_resize_config['project_parameters']['hybrid_electricity_estimated_cf'] = 0.492
    offgrid_resize_config["electrolyzer"]["sizing"]['resize_for_enduse'] = True
    return offgrid_resize_config

@fixture
def grid_resize_config():
    grid_resize_config = default_config.copy()
    grid_resize_config['project_parameters']['grid_connection'] = True
    grid_resize_config['project_parameters']['hybrid_electricity_estimated_cf'] = 1.0
    grid_resize_config["electrolyzer"]["sizing"]['resize_for_enduse'] = True

    return grid_resize_config

def test_resize_from_steel_grid(grid_resize_config):
    gh_test = gh_sizing.run_resizing_estimation(grid_resize_config)
    assert gh_test["electrolyzer"]["rating"] == 480
    assert gh_test["electrolyzer"]["sizing"]["hydrogen_dmd"] == approx(8366.311517,TOL)

def test_resize_from_steel_offgrid(offgrid_resize_config):
    gh_test = gh_sizing.run_resizing_estimation(offgrid_resize_config)
    assert gh_test["electrolyzer"]["rating"] == 960
    assert gh_test["electrolyzer"]["sizing"]["hydrogen_dmd"] == approx(17004.6982,TOL)

@fixture
def offgrid_physics():
    power_profile_filename = 'GS_offgrid_power_signal.csv'
    offgrid_power_profile_filename = os.path.join(input_library_path,power_profile_filename)
    offgrid_power_profile = pd.read_csv(offgrid_power_profile_filename)

    offgrid_config = default_config.copy()
    offgrid_config['project_parameters']['grid_connection'] = False
    offgrid_config['project_parameters']['hybrid_electricity_estimated_cf'] = 0.492
    offgrid_config["electrolyzer"]["sizing"]['resize_for_enduse'] = False
    offgrid_config['electrolyzer']['rating'] = 960
    
    # offgrid_config["electrolyzer"]["sizing"]["hydrogen_dmd"] = []

    electrolyzer_physics_results = he_elec.run_electrolyzer_physics(offgrid_power_profile, offgrid_config, wind_resource = None, design_scenario='off-grid', show_plots=False, save_plots=False, verbose=False)
    H2_Res_offgrid = electrolyzer_physics_results["H2_Results"]
    electrical_gen_ts_offgrid = electrolyzer_physics_results["power_to_electrolyzer_kw"]
    return [H2_Res_offgrid,electrical_gen_ts_offgrid]

def test_offgrid_electrolyzer_physics(offgrid_physics,subtests):
    H2_Res,power_profile = offgrid_physics
    
    with subtests.test("annual energy kwh input"):
        assert H2_Res['Sim: Total Input Power [kWh]'] == approx(3383382801.267635,TOL)
    
    with subtests.test("BOL rated hydrogen production rate"):
        assert H2_Res['Rated BOL: H2 Production [kg/hr]'] == approx(17579.2991094574,TOL)
    
    with subtests.test("BOL rated power consumption"):
        assert H2_Res['Rated BOL: Power Consumed [kWh]'] == approx(960018.4015366472,TOL)
    
    with subtests.test("BOL rated efficiency"):
        assert H2_Res['Rated BOL: Efficiency [kWh/kg]'] == approx(54.61073251891887, TOL)
    
    with subtests.test("simulation capacity factor"):
        assert H2_Res['Sim: Capacity Factor'] == approx(0.4104064586084918, TOL)
    
    with subtests.test("simulation stack off-cycles"):
        assert H2_Res['Sim: Total Stack Off-Cycles'] == 9483.0
    
    with subtests.test("simulation operation time fraction"):
        assert H2_Res['Sim: Active Time / Sim Time'] == approx(0.8112157534246575,TOL)
    
    with subtests.test("simulation hydrogen production "):
        assert H2_Res['Sim: Total H2 Produced [kg]'] == approx(63200403.136826776,TOL)
    
    with subtests.test("simulation hydrogen warm-up losses"):
        assert H2_Res['Sim: H2 Warm-Up Losses [kg]'] == approx(240892.61322440396,TOL)
    
    with subtests.test("stack life"):
        assert H2_Res['Stack Life [hrs]'] == approx(23633.651537254333,TOL)
    
    with subtests.test("time until replacement"):
        assert H2_Res['Time Until Replacement [hrs]'] == approx(28947.55906846945,TOL)
    
    with subtests.test("life average capacity factor"):
        assert H2_Res['Life: Capacity Factor'] == approx(0.3889240837784226,TOL)
    
    

    
@fixture
def grid_physics():
    grid_config = default_config.copy()
    grid_config['project_parameters']['grid_connection'] = True
    grid_config['project_parameters']['hybrid_electricity_estimated_cf'] = 1.0
    grid_config["electrolyzer"]["sizing"]['resize_for_enduse'] = False
    grid_config['electrolyzer']['rating'] = 480
    grid_config["electrolyzer"]["sizing"]["hydrogen_dmd"] = 8366.311517
    grid_power_profile = []
    electrolyzer_physics_results = he_elec.run_electrolyzer_physics(grid_power_profile, grid_config, wind_resource = None, design_scenario='grid-only', show_plots=False, save_plots=False, verbose=False)
    H2_Res_grid = electrolyzer_physics_results["H2_Results"]
    electrical_gen_ts_grid = electrolyzer_physics_results["power_to_electrolyzer_kw"]
    return [H2_Res_grid,electrical_gen_ts_grid]
    
@fixture
def grid_baseline_power_profile():
    power_profile_filename = 'GS_gridonly_power_signal.csv'
    grid_power_profile_filename = os.path.join(input_library_path,power_profile_filename)
    grid_power_profile = pd.read_csv(grid_power_profile_filename,index_col='Unnamed: 0')
    return grid_power_profile

def test_grid_electrolyzer_physics(grid_physics,grid_baseline_power_profile,subtests):
    H2_Res,power_profile = grid_physics

    with subtests.test("simulation AEP"):
        assert H2_Res['Sim: Total Input Power [kWh]'] == approx(4008524165.6783223,TOL)

    with subtests.test("power profile AEP"):
        assert sum(power_profile) == approx(4008524165.6783223,TOL)

    with subtests.test("power profile trend"):
        assert power_profile[-1] > power_profile[0]
    
    with subtests.test("grid simulation capacity factor"):
        assert H2_Res['Sim: Capacity Factor'] == approx(0.9518373167475502, TOL)

    with subtests.test("grid simulation operation time fraction"):
        assert H2_Res['Sim: Active Time / Sim Time'] == 1.0

    with subtests.test("grid simulation off-cycles"):
        assert H2_Res['Sim: Total Stack Off-Cycles'] == 0

    with subtests.test("grid simulation hydrogen warm-up losses"):
        assert H2_Res['Sim: H2 Warm-Up Losses [kg]'] == 0

    with subtests.test("grid stack life"):
        assert H2_Res['Stack Life [hrs]'] == approx(78049.89571256597,TOL)

    with subtests.test("grid time between replacement"):
        assert H2_Res['Time Until Replacement [hrs]'] == H2_Res['Stack Life [hrs]']

    with subtests.test("grid life average capacity factor"):
        assert H2_Res['Life: Capacity Factor'] == approx(0.9518373167475502,TOL)
    
    with subtests.test("grid power profile start"):
        assert power_profile[0] == approx(grid_baseline_power_profile['combined_hybrid_power_production_hopp'].values[0],TOL)
    
    with subtests.test("grid power profile end"):
        assert power_profile[-1] == approx(grid_baseline_power_profile['combined_hybrid_power_production_hopp'].values[-1],TOL)

def test_electrolyzer_tools(subtests):
    hydrogen_demand = 8366.311517
    cluster_cap_mw = 40
    bol_eff = he_elec.get_electrolyzer_BOL_efficiency()
    electrolyzer_capacity_BOL_MW = he_elec.size_electrolyzer_for_hydrogen_demand(hydrogen_demand)
    electrolyzer_size_mw = he_elec.check_capacity_based_on_clusters(electrolyzer_capacity_BOL_MW,cluster_cap_mw)
    
    with subtests.test("electrolyzer BOL efficiency"):
        assert bol_eff == 54.61

    with subtests.test("electrolyzer size for hydrogen demand"):
        assert np.ceil(electrolyzer_capacity_BOL_MW) == 457

    with subtests.test("electrolyzer size rounded to nearest cluster capacity"):
        assert electrolyzer_size_mw == 480
    