import pytest
import os
from pathlib import Path

import ORBIT as orbit
from greenheart.simulation.technologies.offshore.fixed_platform import install_platform, calc_platform_opex, calc_substructure_mass_and_cost

'''Sources:
    - [1]  M. Maness, B. Maples and A. Smith, "NREL Offshore Balance-of-System Model," National Renewable Energy Laboratory, 2017. https://www.nrel.gov/docs/fy17osti/66874.pdf
'''
@pytest.mark.skip(reason="no way of currently testing this")
@pytest.fixture
def config():
    offshore_path = Path(__file__).parents[3] / "greenheart" / "simulation" / "technologies" / "offshore"

    return orbit.load_config(os.path.join(offshore_path, "example_fixed_project.yaml"))
@pytest.mark.skip(reason="no way of currently testing this")
def test_install_platform(config):
    '''
    Test the code that calculates the platform installation cost
    [1]: equations (91),(113),(98)
    '''
    distance = 24
    mass = 2100
    area = 500
    
    cost = install_platform(mass, area, distance, install_duration=14)

    assert pytest.approx(cost) == 7200014

def test_calc_substructure_cost(config):
    '''
    Test the code that calculates the CapEx from fixed_platform.py
    [1]: equations (81),(83),(84)
    '''
    topmass = 200
    toparea = 1000
    depth = 45
    
    cost, _ = calc_substructure_mass_and_cost(topmass, toparea, depth)

    assert pytest.approx(cost) == 7640000

def test_calc_substructure_mass(config):
    '''
    Test the code that calculates the CapEx from fixed_platform.py
    [1]: equations (81),(83),(84)
    '''
    topmass = 200
    toparea = 1000
    depth = 45
    
    _,mass = calc_substructure_mass_and_cost(topmass, toparea, depth)

    assert pytest.approx(mass,.1) == 372.02

def test_calc_platform_opex():
    '''
    Test the code that calculates the OpEx from fixed_platform.py
    '''
    capex = 28e6
    opex_rate = 0.01
    cost = calc_platform_opex(capex, opex_rate)
    
    assert pytest.approx(cost) == 28e4
