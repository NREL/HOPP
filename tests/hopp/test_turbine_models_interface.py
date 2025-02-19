import pytest
from pytest import approx
import numpy as np
import json
from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.tools.design.wind.turbine_library_interface_tools import get_floris_turbine_specs
from hopp.tools.design.wind.turbine_library_tools import check_turbine_name, check_turbine_library_for_turbine

def test_turbine_library_tools_for_valid_turbine_name(subtests):
    valid_turbine_name = "BergeyExcel15_15.6kW_9.6"
    is_valid = check_turbine_library_for_turbine(valid_turbine_name)
    res_valid_name = check_turbine_name(valid_turbine_name)
    
    with subtests.test("test valid name (bool)"):
        assert is_valid is True

    with subtests.test("test valid name (str)"):
        assert res_valid_name == valid_turbine_name

def test_turbine_library_tools_for_invalid_turbine_name(subtests):
    
    invalid_turbine_name_close_match = "BergeyExcel15"
    is_valid = check_turbine_library_for_turbine(invalid_turbine_name_close_match)
    res_invalid_name = check_turbine_name(invalid_turbine_name_close_match)

    with subtests.test("test invalid name (bool)"): 
        assert is_valid is False

    with subtests.test("test invalid name (str)"):     
        assert res_invalid_name != invalid_turbine_name_close_match
    
    with subtests.test("test invalid name best match"):
        assert res_invalid_name == "BergeyExcel15_15.6kW_9.6"