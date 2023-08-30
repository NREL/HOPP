import pytest
from hopp.simulation.technologies.steel.hdri_model import hdri_model
'''
Test files for hdri_model.py class and its functions.

All values were hand calculated and compared with the values in:
[1]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

'''

def test_steel_out_desired_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    steel_out_desired = outputs[3]


    assert pytest.approx(steel_out_desired) == steel_output_desired
    

def test_iron_input_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_iron_ore_input = outputs[4]

    assert pytest.approx(mass_iron_ore_input,.01) == 1601.07


def test_h2_input_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_h2_input = outputs[5]

    assert pytest.approx(mass_h2_input,.01) == 64.97


def test_h2_output_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_h2_output = outputs[6]

    assert pytest.approx(mass_h2_output,.01) == 10.83


def test_water_out_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_h2o_output = outputs[7]

    assert pytest.approx(mass_h2o_output,.01) == 483.89


def test_iron_outputs_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_pure_fe = outputs[8]

    assert pytest.approx(mass_pure_fe,.01) == 1019.18


def test_gas_out_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_h2_output = outputs[6]
    mass_h2o_output = outputs[7]
 
    mass_h2_h2o_output = outputs[9]
  
    assert pytest.approx(mass_h2_h2o_output,.01) == (mass_h2_output+mass_h2o_output)


def test_iron_ore_output_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.mass_model(steel_output_desired)

    mass_iron_ore_output = outputs[10]

    assert pytest.approx(mass_iron_ore_output,.01) == 63.83

def test_energy_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.energy_model(steel_output_desired)

    energy_balance = outputs[1]

    assert pytest.approx(energy_balance,.01) == -332.2

   

def test_heater_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.heater_mass_energy_model(steel_output_desired)

    energy_needed = outputs[1]

    assert pytest.approx(energy_needed,.01) == 337.58

def test_recuperator_model():
    model_instance = hdri_model()

    steel_output_desired = 1000

    outputs = model_instance.recuperator_mass_energy_model(steel_output_desired)

    energy_exchange = outputs[1]

    assert pytest.approx(energy_exchange,.01) == 37.01

    

def test_cap_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    capital_cost = outputs[1]

    assert pytest.approx(capital_cost) == .48

def test_op_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    operational_cost = outputs[2]

    assert pytest.approx(operational_cost) == .026
 
def test_maint_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    maintenance_cost = outputs[3]

    assert pytest.approx(maintenance_cost) == .0072


def test_dep_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    depreciation_cost = outputs[4]

    assert pytest.approx(depreciation_cost) == .012


def test_iron_ore_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000000

    outputs = model_instance.financial_model(steel_output_desired)

    iron_ore_total = outputs[5]

    assert pytest.approx(iron_ore_total,.1) == 288.2


def test_lab_cost_model():
    model_instance = hdri_model()

    steel_output_desired = 2000

    outputs = model_instance.financial_model(steel_output_desired)

    labor_cost = outputs[6]

    assert pytest.approx(labor_cost) == .04


