import pytest
import os

from ORBIT import load_config
from hopp.offshore.fixed_platform import install_platform, calc_platform_opex, calc_substructure_mass_and_cost

@pytest.fixture
def config():
    offshore_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir,'hopp','offshore'))

    return load_config(os.path.join(offshore_path,"example_fixed_project.yaml"))

def test_install_platform(config):
    '''
    Test the code that calculates the platform installation cost
    '''
    distance = 24
    mass = 2100
    area = 500

    cost = install_platform(mass, area, distance, install_duration=14)

    assert pytest.approx(cost) == 7142871

def test_calc_substructure_mass_and_cost(config):
    '''
    Test the code that calculates the CapEx from fixed_platform.py
    '''
    topmass = 200
    toparea = 1000
    depth = 45

    cost, mass = calc_substructure_mass_and_cost(topmass, toparea, depth)

    assert pytest.approx(cost) == 8142358.32
    assert pytest.approx(mass, 1.) == 280.0 

def test_calc_platform_opex():
    '''
    Test the code that calculates the OpEx from fixed_platform_h2.py
    '''
    capex = 28e6
    opex_rate = 0.01
    cost = calc_platform_opex(capex, opex_rate)

    assert cost == 28e4