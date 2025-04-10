from pytest import approx, fixture

import numpy as np

from hopp.simulation import HoppInterface

from hopp import ROOT_DIR
from hopp.utilities import load_yaml

FLORIS_V4_TEMPLATE_PATH = ROOT_DIR.parent / "tests"/"hopp"/"inputs"/"floris_v4_empty_layout.yaml"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"

@fixture
def site_info():
    site_dict = {
        "data": {
        "lat": 35.2018863,
        "lon": -101.945027,
        "year": 2012,
        "site_details": {
            "site_shape": "square",
            "site_area_km2": 2.0,
        },
    },
        "solar_resource_file": DEFAULT_SOLAR_RESOURCE_FILE,
        "wind_resource_file": DEFAULT_WIND_RESOURCE_FILE,
        "solar": True,
        "wind": True,
        "hub_height": 90.0,
        # "follow_desired_schedule": True,
        "curtailment_value_type": "interconnect_kw",
        "desired_schedule": [18.0]*8760,
    }
    return site_dict

@fixture
def ghost_site():
    site_dict = {
        "data": {
        "lat": 35.2018863,
        "lon": -101.945027,
        "year": 2012,
        "site_details": {
            "site_shape": "square",
            "site_area_km2": 2.0,
        },
    },
        "solar": False,
        "wind": False,
        # "follow_desired_schedule": True,
        "curtailment_value_type": "interconnect_kw",
        "desired_schedule": [18.0]*8760,
        "n_timesteps": 8760,
    }
    return site_dict

@fixture
def hybrid_tech_config():
    """Loads the config YAML and updates site info to use resource files."""
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    technologies = {
        "pv": {
            "system_capacity_kw": 120600,
            "panel_tilt_angle": "lat-func",
            "dc_ac_ratio": 1.34,
            "inv_eff": 96.0,
            "losses": 14.0757,
        },
        "wind": {
            "num_turbines": 15,
            "turbine_name": "NREL_Reference_5MW_126",
            "model_name": "floris",
            "floris_config":floris_template,
            "layout_mode": "basicgrid",
            "layout_params": {
                "row_D_spacing": 5.0,
                "turbine_D_spacing": 5.0,
            }

        },
        "battery": {
            "system_capacity_kw": 25000,
            "system_capacity_kwh": 100000,
            "minimum_SOC": 10.0,
            "maximum_SOC": 100.0,
            "initial_SOC": 10.0,
        },
        "grid": {
            "interconnect_kw": 180000.0,
            "ppa_price": 0.01,
        }
    }
    return technologies

@fixture
def dispatch_options():
    dispatch_opt = {
        "battery_dispatch": "load_following_heuristic",
        "solver": "cbc",
        "n_look_ahead_periods": 48,
        "grid_charging": False,
        "pv_charging_only": False,
        "include_lifecycle_count": False,
    }
    return dispatch_opt


def test_ghost_hybrid_with_storage_dispatch(hybrid_tech_config,site_info,dispatch_options,ghost_site,subtests):
    """Test ghost plant functionality for a wind, pv, and battery system. 
    This uses GhostMultiSystem as the GhostPlant system_model.
    """

    hopp_config_renewables = {
        "site": site_info,
        "technologies": hybrid_tech_config,
        "config": {"dispatch_options":dispatch_options},
    }
    # simulate renewables
    hi = HoppInterface(hopp_config_renewables)
    hi.system.simulate(project_life = 1)
    hybrid_plant = hi.system

    pv_size_kwac = hybrid_plant.pv._system_model.SystemDesign.system_capacity/hybrid_plant.pv._system_model.SystemDesign.dc_ac_ratio
    wind_generation_profile = np.array(hybrid_plant.wind.generation_profile)
    pv_generation_profile = np.array(hybrid_plant.pv.generation_profile)
    wind_pv_generation = wind_generation_profile + pv_generation_profile

    hopp_config_ghost = {
        "site": ghost_site,
        "technologies": {
            "ghost": {
                "pv_system": {
                    "system_capacity_kw": hybrid_plant.pv._system_model.SystemDesign.system_capacity,
                    "system_capacity_kwac": pv_size_kwac,
                    "generation_profile_kw": np.array(hybrid_plant.pv.generation_profile).tolist(),
                },
                "wind_system": {
                    "system_capacity_kw": hybrid_plant.wind.system_capacity_kw,
                    "system_capacity_kwac": hybrid_plant.wind.system_capacity_kw,
                    "generation_profile_kw": np.array(hybrid_plant.wind.generation_profile).tolist(),
                },
            },
            "battery": hybrid_tech_config["battery"],
            "grid": hybrid_tech_config["grid"],
        },
        "config": {"dispatch_options":dispatch_options},
    }

    boo = HoppInterface(hopp_config_ghost)
    boo.system.simulate(project_life = 1)
    hybrid_ghost_plant = boo.system

    generation_hybrid = np.array(hybrid_plant.generation_profile.grid)
    generation_ghost = np.array(hybrid_ghost_plant.generation_profile.grid)
    
    # hybrid nominal capacity is set after simulate_grid_connection()
    # calculated in calc_nominal_capacity() - which is AC capacity
    with subtests.test("hybrid_nominal_capacity"):
        assert hybrid_ghost_plant.grid.hybrid_nominal_capacity == approx(hybrid_plant.grid.hybrid_nominal_capacity,1e-6)

    # hybrid_size_kw input to simulate_grid_connection()
    with subtests.test("hybrid_size_kw"):
        assert hybrid_ghost_plant.grid.system_capacity_kw == approx(hybrid_plant.grid.system_capacity_kw)
    
    # check that generation profile was set properly
    with subtests.test("Ghost Generation Profile"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.generation_profile.ghost, 
            wind_pv_generation,
            rtol = 1e-6
            )

    # check gen max feasible
    with subtests.test("total_gen_max_feasible_year1"):
        wind_gen_max_feasible = hybrid_plant.wind.calc_gen_max_feasible_kwh(hybrid_plant.interconnect_kw)
        pv_gen_max_feasible = hybrid_plant.pv.calc_gen_max_feasible_kwh(hybrid_plant.interconnect_kw)
        wind_pv_gen_max_feasible = np.array(wind_gen_max_feasible) + np.array(pv_gen_max_feasible)
        np.testing.assert_allclose(
            hybrid_ghost_plant.ghost.gen_max_feasible,
            wind_pv_gen_max_feasible,
            rtol = 1e-6
            )

    # based on total_gen_max_feasible_year1 input to simulate_grid_connection()
    with subtests.test("grid.gen_max_feasible"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid.gen_max_feasible,
            hybrid_plant.grid.gen_max_feasible,
            rtol = 1e-6
            )
    
    # total_gen_max_feasible_year1 input to simulate_grid_connection()
    with subtests.test("grid.total_gen_max_feasible_year1"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid.total_gen_max_feasible_year1,
            hybrid_plant.grid.total_gen_max_feasible_year1,
            rtol = 1e-6
            )

    with subtests.test("system_pre_interconnect_kwac"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid._system_model.Outputs.system_pre_interconnect_kwac, 
            hybrid_plant.grid._system_model.Outputs.system_pre_interconnect_kwac,
            rtol = 1e-6,
        )

    with subtests.test("Grid AEP"):
        assert np.sum(generation_ghost) == approx(np.sum(generation_hybrid),1e-6)
    
    # total_gen is input to simulate_grid_connection
    with subtests.test("generation_profile_wo_battery"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid.generation_profile_wo_battery,
            hybrid_plant.grid.generation_profile_wo_battery,
            rtol = 1e-6,
            )
    
    # hybrid_plant.grid.generation_profile
    with subtests.test("generation_profile.grid"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.generation_profile.grid, 
            hybrid_plant.generation_profile.grid,
            rtol = 1e-6,
            )

def test_ghost_wind_with_storage_dispatch(hybrid_tech_config,site_info,dispatch_options,ghost_site,subtests):
    """Test ghost plant functionality for a wind and battery system. 
    This uses GhostSystem as the GhostPlant system_model.
    """

    techs = ['wind','battery','grid']
    tech_config = {k:v for k,v in hybrid_tech_config.items() if k in techs}
    hopp_config_renewables = {
        "site": site_info,
        "technologies": tech_config,
        "config": {"dispatch_options":dispatch_options},
    }

    # simulate renewables
    hi = HoppInterface(hopp_config_renewables)
    hi.system.simulate_power(project_life = 1)
    hybrid_plant = hi.system
    
    wind_generation_profile = np.array(hybrid_plant.wind._system_model.gen)
   
    hopp_config_ghost = {
        "site": ghost_site,
        "technologies": {
            "ghost": {
                "system_capacity_kw": hybrid_plant.wind.system_capacity_kw,
                "generation_profile_kw": list(wind_generation_profile)
            },
            "battery": hybrid_tech_config["battery"],
            "grid": hybrid_tech_config["grid"],
        },
        "config": {"dispatch_options":dispatch_options},
    }

    boo = HoppInterface(hopp_config_ghost)
    boo.system.simulate_power(project_life = 1)
    hybrid_ghost_plant = boo.system

    generation_hybrid = np.array(hybrid_plant.generation_profile.grid)
    generation_ghost = np.array(hybrid_ghost_plant.generation_profile.grid)
    
    with subtests.test("grid.hybrid_nominal_capacity"):
        assert hybrid_ghost_plant.grid.hybrid_nominal_capacity == approx(hybrid_plant.grid.hybrid_nominal_capacity,1e-6)

    with subtests.test("hybrid_size_kw"):
        assert hybrid_ghost_plant.grid.system_capacity_kw == approx(hybrid_plant.grid.system_capacity_kw)

    with subtests.test("Grid AEP"):
        assert np.sum(generation_ghost) == approx(np.sum(generation_hybrid),1e-6)
    
    with subtests.test("generation_profile.grid"):
        np.testing.assert_allclose(hybrid_ghost_plant.generation_profile.grid, hybrid_plant.generation_profile.grid,rtol = 1e-6)


def test_ghost_pv_with_storage_dispatch(hybrid_tech_config,site_info,dispatch_options,ghost_site,subtests):
    """Test ghost plant functionality for a pv and battery system. 
    This uses GhostSystem as the GhostPlant system_model.
    """

    techs = ['pv','battery','grid']
    tech_config = {k:v for k,v in hybrid_tech_config.items() if k in techs}
    hopp_config_renewables = {
        "site": site_info,
        "technologies": tech_config,
        "config": {"dispatch_options":dispatch_options},
    }

    # simulate renewables
    hi = HoppInterface(hopp_config_renewables)
    hi.system.simulate_power(project_life = 1)
    hybrid_plant = hi.system
    pv_size_kwac = hybrid_plant.pv._system_model.SystemDesign.system_capacity/hybrid_plant.pv._system_model.SystemDesign.dc_ac_ratio
    hopp_config_ghost = {
        "site": ghost_site,
        "technologies": {
            "ghost": {
                "system_capacity_kw": hybrid_plant.pv._system_model.SystemDesign.system_capacity,
                "system_capacity_kwac": pv_size_kwac,
                "generation_profile_kw": list(np.array(hybrid_plant.pv._system_model.Outputs.gen))
            },
            "battery": hybrid_tech_config["battery"],
            "grid": hybrid_tech_config["grid"],
        },
        "config": {"dispatch_options":dispatch_options},
    }

    boo = HoppInterface(hopp_config_ghost)
    boo.system.simulate_power(project_life = 1)
    hybrid_ghost_plant = boo.system

    generation_hybrid = np.array(hybrid_plant.generation_profile.grid)
    generation_ghost = np.array(hybrid_ghost_plant.generation_profile.grid)
    
    with subtests.test("grid.hybrid_nominal_capacity"):
        assert hybrid_ghost_plant.grid.hybrid_nominal_capacity == approx(hybrid_plant.grid.hybrid_nominal_capacity,1e-6)

    with subtests.test("hybrid_size_kw"):
        assert hybrid_ghost_plant.grid.system_capacity_kw == approx(hybrid_plant.grid.system_capacity_kw)

    with subtests.test("Grid AEP"):
        assert np.sum(generation_ghost) == approx(np.sum(generation_hybrid),1e-6)
    
    with subtests.test("Grid generation profile"):
        np.testing.assert_allclose(generation_ghost, generation_hybrid,rtol = 1e-6)

def test_ghost_hybrid(hybrid_tech_config,site_info,dispatch_options,ghost_site,subtests):
    """Test ghost plant functionality for a wind and pv system. 
    This uses GhostMultiSystem as the GhostPlant system_model.
    """

    techs = ['pv','wind','grid']
    tech_config = {k:v for k,v in hybrid_tech_config.items() if k in techs}
    hopp_config_renewables = {
        "site": site_info,
        "technologies": tech_config,
        "config": {"dispatch_options":dispatch_options},
    }

    # simulate renewables
    hi = HoppInterface(hopp_config_renewables)
    hi.system.simulate_power(project_life = 1)
    hybrid_plant = hi.system

    pv_size_kwac = hybrid_plant.pv._system_model.SystemDesign.system_capacity/hybrid_plant.pv._system_model.SystemDesign.dc_ac_ratio
    wind_generation_profile = np.array(hybrid_plant.wind.generation_profile)
    pv_generation_profile = np.array(hybrid_plant.pv.generation_profile)
    wind_pv_generation = wind_generation_profile + pv_generation_profile

    hopp_config_ghost = {
        "site": ghost_site,
        "technologies": {
            "ghost": {
                "pv_system": {
                    "system_capacity_kw": hybrid_plant.pv._system_model.SystemDesign.system_capacity,
                    "system_capacity_kwac": pv_size_kwac,
                    "generation_profile_kw": np.array(hybrid_plant.pv.generation_profile).tolist(),
                },
                "wind_system": {
                    "system_capacity_kw": hybrid_plant.wind.system_capacity_kw,
                    "system_capacity_kwac": hybrid_plant.wind.system_capacity_kw,
                    "generation_profile_kw": np.array(hybrid_plant.wind.generation_profile).tolist(),
                },
            },
            "grid": hybrid_tech_config["grid"],
        },
        "config": {"dispatch_options":dispatch_options},
    }

    boo = HoppInterface(hopp_config_ghost)
    boo.system.simulate_power(project_life = 1)
    hybrid_ghost_plant = boo.system

    generation_hybrid = np.array(hybrid_plant.generation_profile.grid)
    generation_ghost = np.array(hybrid_ghost_plant.generation_profile.grid)
    
    # check hybrid nominal capacity
    with subtests.test("grid.hybrid_nominal_capacity"):
        assert hybrid_ghost_plant.grid.hybrid_nominal_capacity == approx(hybrid_plant.grid.hybrid_nominal_capacity,1e-6)

    # check gen max feasible
    with subtests.test("gen_max_feasible"):
        wind_gen_max_feasible = hybrid_plant.wind.calc_gen_max_feasible_kwh(hybrid_plant.interconnect_kw)
        pv_gen_max_feasible = hybrid_plant.pv.calc_gen_max_feasible_kwh(hybrid_plant.interconnect_kw)
        wind_pv_gen_max_feasible = np.array(wind_gen_max_feasible) + np.array(pv_gen_max_feasible)
        np.testing.assert_allclose(
            hybrid_ghost_plant.ghost.gen_max_feasible,
            wind_pv_gen_max_feasible,
            rtol = 1e-6
            )
    
    # check that generation profile was set properly
    with subtests.test("Ghost Generation Profile"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.generation_profile.ghost, 
            wind_pv_generation,
            rtol = 1e-6
            )
    
    # check total gen max feasible year 1
    with subtests.test("grid.total_gen_max_feasible_year1"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid.total_gen_max_feasible_year1,
            hybrid_plant.grid.total_gen_max_feasible_year1,
            rtol = 1e-6
            )

    with subtests.test("grid.system_pre_interconnect_kwac"):
        np.testing.assert_allclose(
            hybrid_ghost_plant.grid._system_model.Outputs.system_pre_interconnect_kwac, 
            hybrid_plant.grid._system_model.Outputs.system_pre_interconnect_kwac,
            rtol = 1e-6
        )

    with subtests.test("Grid AEP"):
        assert np.sum(generation_ghost) == approx(np.sum(generation_hybrid),1e-6)

    with subtests.test("generation_profile.grid"):
        np.testing.assert_allclose(generation_ghost, generation_hybrid,rtol = 1e-6)
