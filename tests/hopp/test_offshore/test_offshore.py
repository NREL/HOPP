import pytest
import os
from pathlib import Path

from ORBIT import load_config
from hopp.simulation.technologies.offshore.fixed_platform import install_platform, calc_platform_opex, calc_substructure_mass_and_cost

@pytest.fixture
def config():
    offshore_path = Path(__file__).parents[3] / "hopp" / "simulation" / "technologies" / "offshore"

    return load_config(os.path.join(offshore_path, "example_fixed_project.yaml"))

def test_install_platform(config):
    '''
    Test the code that calculates the platform installation cost
    '''
    distance = 24
    mass = 2100
    area = 500

    cost = install_platform(mass, area, distance, install_duration=14)

    assert pytest.approx(cost) == 7200014

def test_calc_substructure_mass_and_cost(config):
    '''
    Test the code that calculates the CapEx from fixed_platform.py
    '''
    topmass = 200
    toparea = 1000
    depth = 45

    cost, mass = calc_substructure_mass_and_cost(topmass, toparea, depth)

    assert pytest.approx(cost) == 7640000
    assert pytest.approx(mass, 1.) == 372.02 

def test_calc_platform_opex():
    '''
    Test the code that calculates the OpEx from fixed_platform_h2.py
    '''
    capex = 28e6
    opex_rate = 0.01
    cost = calc_platform_opex(capex, opex_rate)

    assert cost == 28e4