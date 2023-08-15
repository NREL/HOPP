import pytest
from hopp.simulation.technologies.steel.hdri_model import hdri_model


def test_mass_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)
    steel_out_desired = outputs[3]
    mass_iron_ore_input = outputs[4]
    mass_h2_input = outputs[5]
    mass_h2_output = outputs[6]
    mass_h2o_output = outputs[7]
    mass_pure_fe = outputs[8]
    mass_h2_h2o_output = outputs[9]
    mass_iron_ore_output = outputs[10]

    assert pytest.approx(steel_out_desired) == steel_output_desired
    assert pytest.approx(mass_iron_ore_input,.01) == 1601.07
    assert pytest.approx(mass_h2_input,.01) == 64.97
    assert pytest.approx(mass_h2_output,.01) == 10.83
    assert pytest.approx(mass_h2o_output,.01) == 483.89
    assert pytest.approx(mass_pure_fe,.01) == 1019.18
    assert pytest.approx(mass_h2_h2o_output,.01) == (mass_h2_output+mass_h2o_output)
    assert pytest.approx(mass_iron_ore_output,.01) == 63.83
 

def test_energy_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.energy_model(steel_output_desired)

    energy_balance = outputs[1]

    assert pytest.approx(energy_balance,.01) == -344.2

   

def test_heater_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.heater_mass_energy_model(steel_output_desired)

    energy_needed = outputs[1]

    assert pytest.approx(energy_needed,.01) == 286.97

def test_recuperator_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.recuperator_mass_energy_model(steel_output_desired)

    energy_exchange = outputs[1]

    assert pytest.approx(energy_exchange,.01) == 86.37

    

def test_financial_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.financial_model(steel_output_desired)
    capital_cost = outputs[1]
    operational_cost = outputs[2]
    maintenance_cost = outputs[3]
    depreciation_cost = outputs[4]
    iron_ore_total = outputs[5]
    labor_cost = outputs[6]

    assert pytest.approx(capital_cost) == .24
    assert pytest.approx(operational_cost) == .013
    assert pytest.approx(maintenance_cost) == .0036
    assert pytest.approx(depreciation_cost) == .006
    assert pytest.approx(iron_ore_total) == .09
    assert pytest.approx(labor_cost) == .02




