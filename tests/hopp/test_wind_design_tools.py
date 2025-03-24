from pytest import fixture
import pytest
from hopp.tools.design.wind.floris_helper_tools import (
    check_floris_library_for_turbine,
    load_turbine_from_floris_library,
    check_libraries_for_turbine_name_floris
)
from hopp.simulation.technologies.wind.wind_plant import WindConfig
from hopp.simulation.technologies.wind.floris import Floris
from hopp.utilities import load_yaml
from hopp.simulation.technologies.sites.site_info import SiteInfo
from hopp import ROOT_DIR
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
FLORIS_V4_TEMPLATE_PATH = ROOT_DIR.parent / "tests"/"hopp"/"inputs"/"floris_v4_empty_layout.yaml"

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


def test_floris_library_tools_for_valid_floris_turbine(subtests):
    floris_library_turbine_name = "nrel_5MW"
    is_floris_turbine = check_floris_library_for_turbine(floris_library_turbine_name)
    floris_turbine_specs = load_turbine_from_floris_library(floris_library_turbine_name)
    with subtests.test("valid floris turbine (bool)"):
        assert is_floris_turbine is True
    with subtests.test("valid floris turbine loader return type"):
        assert isinstance(floris_turbine_specs,dict)
    with subtests.test("valid floris turbine loader turbine specs"):
        assert floris_turbine_specs["turbine_type"] == floris_library_turbine_name

def test_floris_library_tools_for_invalid_floris_turbine(subtests):
    floris_library_invalid_turbine_name = "nrel_10MW"
    is_floris_turbine = check_floris_library_for_turbine(floris_library_invalid_turbine_name)
    
    with subtests.test("invalid floris turbine (bool)"):
        assert is_floris_turbine is False

    with pytest.raises(FileNotFoundError) as err:
        floris_turbine_specs = load_turbine_from_floris_library(floris_library_invalid_turbine_name)
    assert str(err.value) == f"Floris library file for turbine {floris_library_invalid_turbine_name} does not exist."

def test_floris_turbine_loader_valid_floris_turbine(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    floris_library_turbine_name = "nrel_5MW"
    wind_config_dict = {
        "num_turbines": 4,
        "layout_mode": "basicgrid",
        "turbine_rating_kw": 5000.0,
        "layout_params": {},
        "turbine_name": floris_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template
    }
    site_input.update({"hub_height":90.0})
    site = SiteInfo.from_dict(site_input)
    wind_config = WindConfig.from_dict(wind_config_dict)
    floris_model = Floris.from_dict({"site":site,"config":wind_config})
    
    floris_turb_res = check_libraries_for_turbine_name_floris(floris_library_turbine_name,floris_model)
    with subtests.test("return type"):
        assert isinstance(floris_turb_res,dict)
    with subtests.test("turbine type name"):
        assert floris_turb_res["turbine_type"] == floris_library_turbine_name
    with subtests.test("hub-height"):
        assert floris_turb_res["hub_height"] == 90.0
    with subtests.test("rated power"):
        assert max(floris_turb_res["power_thrust_table"]["power"]) == wind_config_dict["turbine_rating_kw"]

def test_floris_turbine_loader_multi_hub_height_turbine(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    turbine_library_turbine_name = "VestasV82_1.65MW_82" #has 80m hub-height option
    wind_config_dict = {
        "num_turbines": 4,
        "layout_mode": "basicgrid",
        "turbine_rating_kw": 1650.0,
        "layout_params": {},
        "turbine_name": turbine_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template
    }
    site_input.update({"hub_height":80.0})
    site = SiteInfo.from_dict(site_input)
    wind_config = WindConfig.from_dict(wind_config_dict)
    floris_model = Floris.from_dict({"site":site,"config":wind_config})
    
    floris_turb_res = check_libraries_for_turbine_name_floris(turbine_library_turbine_name,floris_model)

    with subtests.test("return type"):
        assert isinstance(floris_turb_res,dict)
    with subtests.test("turbine type name"):
        assert floris_turb_res["turbine_type"] == turbine_library_turbine_name
    with subtests.test("hub-height"):
        assert floris_turb_res["hub_height"] == 80.0
    with subtests.test("rated power"):
        assert max(floris_turb_res["power_thrust_table"]["power"]) == wind_config_dict["turbine_rating_kw"]


def test_floris_turbine_loader_single_hub_height_turbine(site_input,subtests):
    floris_template = load_yaml(str(FLORIS_V4_TEMPLATE_PATH))
    turbine_library_turbine_name = "DOE_GE_1.5MW_77" #only has 80m as valid hub-height
    wind_config_dict = {
        "num_turbines": 4,
        "layout_mode": "basicgrid",
        "turbine_rating_kw": 1500.0,
        "layout_params": {},
        "turbine_name": turbine_library_turbine_name,
        "model_name": "floris",
        "floris_config": floris_template
    }
    site_input.update({"hub_height":80.0})
    site = SiteInfo.from_dict(site_input)
    wind_config = WindConfig.from_dict(wind_config_dict)
    floris_model = Floris.from_dict({"site":site,"config":wind_config})
    floris_turb_res = check_libraries_for_turbine_name_floris(turbine_library_turbine_name,floris_model)

    with subtests.test("return type"):
        assert isinstance(floris_turb_res,dict)
    with subtests.test("turbine type name"):
        assert floris_turb_res["turbine_type"] == turbine_library_turbine_name
    with subtests.test("hub-height"):
        assert floris_turb_res["hub_height"] == 80.0
    with subtests.test("rated power"):
        assert max(floris_turb_res["power_thrust_table"]["power"]) == wind_config_dict["turbine_rating_kw"]