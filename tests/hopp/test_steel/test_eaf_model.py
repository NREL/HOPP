import pytest
from hopp.simulation.technologies.steel.eaf_model import eaf_model


def test_mass_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    steel_output_actual = outputs[1]
    carbon_total = outputs[2]
    lime_total = outputs[3]

    assert pytest.approx(steel_output_actual) == 1044.68
    assert pytest.approx(carbon_total) == 10
    assert pytest.approx(lime_total) == 50

def test_energy_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.energy_model(steel_output_desired)

    el_eaf = outputs[1]

    assert pytest.approx(el_eaf, .01) == 501.32

    

def test_emission_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.emission_model(steel_output_desired)

    indirect_emissions_total = outputs[1]
    direct_emissions = outputs[2]
    total_emissions = outputs[3]

    assert pytest.approx(indirect_emissions_total, 0.1) == 325.56
    assert pytest.approx(direct_emissions) == .233
    assert pytest.approx(total_emissions) == (indirect_emissions_total + direct_emissions)

  

def test_financial_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.financial_model(steel_output_desired)

    eaf_total_capital_cost = outputs[1]
    eaf_operational_cost_yr = outputs[2]
    eaf_maintenance_cost_yr = outputs[3]
    depreciation_cost = outputs[4]
    coal_total_cost_yr = outputs[5]
    total_labor_cost_yr = outputs[6]
    lime_cost_total = outputs[7]
    total_emission_cost = outputs[8]

    assert pytest.approx(eaf_total_capital_cost) == .42
    assert pytest.approx(eaf_operational_cost_yr) == .032
    assert pytest.approx(eaf_maintenance_cost_yr) == .0063
    assert pytest.approx(depreciation_cost) == .0105
    assert pytest.approx(coal_total_cost_yr) == .0012
    assert pytest.approx(total_labor_cost_yr) == .02
    assert pytest.approx(lime_cost_total) == .0056
    assert pytest.approx(total_emission_cost) == .00699
