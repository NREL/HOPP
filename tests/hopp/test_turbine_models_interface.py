from pytest import fixture, approx
import pytest
import numpy as np
import json
from hopp.simulation.technologies.wind.floris import Floris
from hopp.utilities import load_yaml
from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.tools.design.wind.turbine_library_tools import check_turbine_library_for_turbine
from hopp.simulation.technologies.sites.site_info import SiteInfo
from hopp import ROOT_DIR
from hopp.simulation import HoppInterface

FLORIS_V4_TEMPLATE_PATH = ROOT_DIR.parent / "tests"/"hopp"/"inputs"/"floris_v4_empty_layout.yaml"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"

@fixture
def site_input():
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
        "hub_height": 80.0
    }
    return site_dict

def test_turbine_library_tools_for_valid_turbine_name(subtests):
    valid_turbine_name = "BergeyExcel15_15.6kW_9.6"
    is_valid = check_turbine_library_for_turbine(valid_turbine_name)
    
    with subtests.test("valid name (bool)"):
        assert is_valid is True


def test_turbine_library_tools_for_invalid_turbine_name(subtests):
    
    invalid_turbine_name_close_match = "BergeyExcel15"
    is_valid = check_turbine_library_for_turbine(invalid_turbine_name_close_match)
    
    with subtests.test("invalid name (bool)"): 
        assert is_valid is False
    
def test_floris_nrel_5mw(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    floris_library_turbine_name = "nrel_5MW"
    n_turbs = 4
    turbine_rating_kw = 5000.0
    layout_x = [0.0,1841.0,3682.0,5523.0]
    layout_y = [0.0]*n_turbs
    floris_template["farm"].update({"layout_x":layout_x,"layout_y":layout_y})
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": floris_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template,
        "layout_mode": "floris_layout"
    }
    project_life = 25
    site_input.update({"hub_height":90.0})
    site = SiteInfo.from_dict(site_input)
    wind_config = WindConfig.from_dict(wind_config_dict)
    wind_plant = WindPlant.from_dict({"site":site,"config":wind_config})
    wind_plant._system_model.execute(project_life)
    with subtests.test("capacity factor"):
        assert wind_plant.capacity_factor > 0.0
    with subtests.test("capacity factor"):
        assert wind_plant.capacity_factor < 100.0
    with subtests.test("aep"):
        wind_plant._system_model.annual_energy == approx(74149945, 1e-3)
    with subtests.test("wind capacity factor value"):
        assert wind_plant.capacity_factor == approx(42.0, abs = 1.0)

def test_floris_nrel_5mw_hopp(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    floris_library_turbine_name = "nrel_5MW"
    n_turbs = 4
    turbine_rating_kw = 5000.0
    layout_x = [0.0,1841.0,3682.0,5523.0]
    layout_y = [0.0]*n_turbs
    floris_template["farm"].update({"layout_x":layout_x,"layout_y":layout_y})
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": floris_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template,
        "layout_mode": "floris_layout"
    }
    site_input.update({"hub_height":90.0})
    system_capacity_kw = turbine_rating_kw*n_turbs
    technologies = {"wind":wind_config_dict,"grid":{"interconnect_kw":system_capacity_kw}}
    hybrid_config = {"site":site_input,"technologies":technologies}
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate(25)

    aeps = hybrid_plant.annual_energies
    with subtests.test("wind aep"):
        assert aeps.wind == approx(74149945, 1e-3)
    with subtests.test("wind capacity factor"):
        assert hybrid_plant.capacity_factors["wind"] == approx(42.0, abs = 1.0)

def test_floris_NREL_5MW_RWT_corrected_hopp(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    turbine_library_turbine_name = "NREL_Reference_5MW_126"
    n_turbs = 4
    turbine_rating_kw = 5000.0
    layout_x = [0.0,1841.0,3682.0,5523.0]
    layout_y = [0.0]*n_turbs
    floris_template["farm"].update({"layout_x":layout_x,"layout_y":layout_y})
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": turbine_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template,
        "layout_mode": "floris_layout"
    }
    site_input.update({"hub_height":90.0})
    system_capacity_kw = turbine_rating_kw*n_turbs
    technologies = {"wind":wind_config_dict,"grid":{"interconnect_kw":system_capacity_kw}}
    hybrid_config = {"site":site_input,"technologies":technologies}
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate(25)

    aeps = hybrid_plant.annual_energies
    with subtests.test("wind aep"):
        assert aeps.wind == approx(74149945, 1e-3)
    with subtests.test("wind capacity factor"):
        assert hybrid_plant.capacity_factors["wind"] == approx(42.0, abs = 1.0)

def test_floris_NREL_5MW_RWT_error_hopp(site_input, subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    turbine_library_turbine_name = "NREL_Reference_5MW_126"
    n_turbs = 4
    turbine_rating_kw = 6000.0
    layout_x = [0.0, 1841.0, 3682.0, 5523.0]
    layout_y = [0.0] * n_turbs
    floris_template["farm"].update({"layout_x": layout_x, "layout_y": layout_y})
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": turbine_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template,
        "layout_mode": "floris_layout"
    }
    site_input.update({"hub_height": 90.0})
    system_capacity_kw = turbine_rating_kw * n_turbs
    technologies = {"wind": wind_config_dict, "grid": {"interconnect_kw": system_capacity_kw}}
    hybrid_config = {"site": site_input, "technologies": technologies}
    
    with subtests.test("error on invalid turbine rating"):
        with pytest.raises(ValueError):
            hi = HoppInterface(hybrid_config)
            hi.simulate(25)

def test_floris_NREL_5MW_RWT_no_error_hopp(site_input, subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    turbine_library_turbine_name = "NREL_Reference_5MW_126"
    n_turbs = 4
    turbine_rating_kw = 4900.0
    layout_x = [0.0, 1841.0, 3682.0, 5523.0]
    layout_y = [0.0] * n_turbs
    floris_template["farm"].update({"layout_x": layout_x, "layout_y": layout_y})
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": turbine_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template,
        "layout_mode": "floris_layout"
    }
    site_input.update({"hub_height": 90.0})
    system_capacity_kw = turbine_rating_kw * n_turbs
    technologies = {"wind": wind_config_dict, "grid": {"interconnect_kw": system_capacity_kw}}
    hybrid_config = {"site": site_input, "technologies": technologies}
    
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate(25)

    aeps = hybrid_plant.annual_energies
    with subtests.test("wind aep"):
        assert aeps.wind == approx(74149945, 1e-3)
    with subtests.test("wind capacity factor"):
        assert hybrid_plant.capacity_factors["wind"] == approx(43.2, abs = 1.0)


def test_pysam_NREL_5MW_RWT_corrected_hopp(site_input,subtests):
    turbine_library_turbine_name = "NREL_Reference_5MW_126"
    n_turbs = 4
    turbine_rating_kw = 5000.0
    layout_x = [0.0,1841.0,3682.0,5523.0]
    layout_y = [0.0]*n_turbs
    layout_params = {"layout_x":layout_x,"layout_y":layout_y}
    wind_config_dict = {
        "num_turbines": n_turbs,
        "turbine_rating_kw": turbine_rating_kw,
        "turbine_name": turbine_library_turbine_name,
        "model_name": "pysam",
        "layout_mode": "custom",
        "layout_params": layout_params
    }
    site_input.update({"hub_height":90.0})
    system_capacity_kw = turbine_rating_kw*n_turbs
    technologies = {"wind":wind_config_dict,"grid":{"interconnect_kw":system_capacity_kw}}
    hybrid_config = {"site":site_input,"technologies":technologies}
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate(25)

    aeps = hybrid_plant.annual_energies
    with subtests.test("wind aep"):
        assert aeps.wind == approx(66040330, 1e-3)
    with subtests.test("wind capacity factor"):
        assert hybrid_plant.capacity_factors["wind"] == approx(37.7, abs = 1.0)