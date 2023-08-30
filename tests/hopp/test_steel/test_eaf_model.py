import pytest
from hopp.simulation.technologies.steel.eaf_model import eaf_model
'''
Test files for eaf_model.py class and its functions.

All values were hand calculated and compared with the values in:
[1]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

'''

def test_mass_steel_output_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    steel_output_actual = outputs[1]

    assert pytest.approx(steel_output_actual) == 1044.68

def test_mass_carbon_total_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    carbon_total = outputs[2]

    assert pytest.approx(carbon_total) == 10
   

def test_mass_lime_total_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    lime_total = outputs[3]

    
    assert pytest.approx(lime_total) == 50


def test_energy_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.energy_model(steel_output_desired)

    el_eaf = outputs[1]

    assert pytest.approx(el_eaf, .01) == 501.32

    

def test_emission_indirect_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.emission_model(steel_output_desired)

    indirect_emissions_total = outputs[1]

    assert pytest.approx(indirect_emissions_total, 0.1) == 325.56

def test_emission_direct_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.emission_model(steel_output_desired)

    direct_emissions = outputs[2]

    assert pytest.approx(direct_emissions) == .233

def test_emission_total_model():
    model_instance = eaf_model()

    steel_output_desired = 1000

    outputs = model_instance.emission_model(steel_output_desired)

    indirect_emissions_total = outputs[1]
    direct_emissions = outputs[2]
    total_emissions = outputs[3]

    
    assert pytest.approx(total_emissions) == (indirect_emissions_total + direct_emissions)
  

def test_cap_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    eaf_total_capital_cost = outputs[1]

    assert pytest.approx(eaf_total_capital_cost) == .84
   

def test_op_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    eaf_operational_cost_yr = outputs[2]

    assert pytest.approx(eaf_operational_cost_yr) == .064
  

def test_maint_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    eaf_maintenance_cost_yr = outputs[3]

    assert pytest.approx(eaf_maintenance_cost_yr) == .0126
    

def test_dep_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    depreciation_cost = outputs[4]
    
    assert pytest.approx(depreciation_cost) == .021
    

def test_coal_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000000

    outputs = model_instance.financial_model(steel_output_desired)

    coal_total_cost_yr = outputs[5]

    assert pytest.approx(coal_total_cost_yr) == 4

def test_lab_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    total_labor_cost_yr = outputs[6]
    
    assert pytest.approx(total_labor_cost_yr) == .04

def test_lime_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000000

    outputs = model_instance.financial_model(steel_output_desired)

    lime_cost_total = outputs[7]

    assert pytest.approx(lime_cost_total) == 11.2
    

def test_emission_cost_model():
    model_instance = eaf_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    total_emission_cost = outputs[8]

    assert pytest.approx(total_emission_cost) == .01398





