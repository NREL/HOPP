import pytest

from hybrid.scenario import Scenario
from defaults.defaults_data import get_default
from hybrid.systems_behavior import *


e = 0.0001

solar_wind_input_info = {
    'Solar': {
        'Pvsamv1': {
            'SystemDesign': {
                'subarray1_gcr': [.1, .9],
                'subarray1_azimuth': [160, 200],
                'subarray1_tilt': [20, 34]
            }
        }
    },
    'Wind': {
        'Windpower': {
            'Farm': {
                'wind_farm_wake_model': (0, 1, 2),
            },
            'Losses': {
                'avail_bop_loss': [0, 5]
            }
        }
    }
}

geothermal_system = {

}


# outputs
solar_wind_output_info = {
    'Generic': {
        'GenericSystem': (
            'annual_energy',        #: 'Hybrid Annual Energy (MWh)',
            'gen',                  # 'Hybrid Power Generated (kWac)',
            'capacity_factor'       # 'Hybrid Capacity Factor (%)'
        ),
        'Singleowner': (
            'ppa_price',            # : 'Hybrid PPA price ($/MWh)',
            'analysis_period_irr'   #: 'Hybrid IRR (%)'
        )
    },
    'Solar': {
        'Pvsamv1': (
            'annual_energy',        # 'Solar Annual Energy (MWh)'
            'gen',                  # 'Solar Power Generated (kWac)'
            'capacity_factor'       # 'Solar Capacity Factor (%)'
        )
    },
    'Wind': {
        'Windpower': (
            'annual_energy',        #: 'Wind Annual Energy (MWh)',
            'gen',                  #: 'Wind Power Generated (kWac)',
            'capacity_factor'       #: 'Wind Capacity Factor (%)'
        )
    }
}


run_systems = get_system_behavior_fx(['Solar', 'Wind', 'Generic'])
# ordering of systems in defaults_data determines ordering for all other data (inputs, outputs)

@pytest.fixture
def scen():
    return Scenario(get_default(run_systems)[0], solar_wind_input_info, solar_wind_output_info, run_systems)

#
# Test system simulations are set up
#
def test_Scenario_setup(scen):
    systems = scen.systems
    assert('Singleowner' in systems[k] for k in ['Solar', 'Wind', 'Generic'])


#
# Test Scenario's map_parameters function
#
def test_Scenario_map_parameters(scen):
    assert(scen.parameter_names == ['subarray1_gcr', 'subarray1_azimuth', 'subarray1_tilt', 'wind_farm_wake_model',
                                    'avail_bop_loss'])
    assert(scen.coordinate_names == ['subarray1_gcr', 'subarray1_azimuth', 'subarray1_tilt', 'wind_farm_wake_model-0',
                                     'wind_farm_wake_model-1', 'wind_farm_wake_model-2', 'avail_bop_loss'])

    params_map = scen.parameter_map
    assert(params_map[0] == ('Solar', 'Pvsamv1', 'SystemDesign', 'subarray1_gcr'))
    assert(params_map[1] == ('Solar', 'Pvsamv1', 'SystemDesign', 'subarray1_azimuth'))
    assert(params_map[2] == ('Solar', 'Pvsamv1', 'SystemDesign', 'subarray1_tilt'))
    assert(params_map[3] == ('Wind', 'Windpower', 'Farm', 'wind_farm_wake_model'))
    assert(params_map[4] == ('Wind', 'Windpower', 'Losses', 'avail_bop_loss'))

    parameter_range = scen.parameter_range
    assert(parameter_range[0] == [.1, .9])
    assert(parameter_range[1] == [160, 200])
    assert(parameter_range[2] == [20, 34])
    assert(parameter_range[3] == (0, 1, 2))
    assert(parameter_range[4] == [0, 5])


#
# Test Scenario's map_outputs function
#
def test_Scenario_map_outputs(scen):
    outputs_map = scen.outputs_map
    assert(outputs_map[0] == ('Solar', "Pvsamv1", 'annual_energy'))
    assert(outputs_map[1] == ('Solar', "Pvsamv1", 'gen'))
    assert(outputs_map[2] == ('Solar', "Pvsamv1", 'capacity_factor'))
    assert(outputs_map[3] == ('Wind', "Windpower", 'annual_energy'))
    assert(outputs_map[4] == ('Wind', "Windpower", 'gen'))
    assert(outputs_map[5] == ('Wind', "Windpower", 'capacity_factor'))

#
# Test Scenario's sample_from_coordinate function
#
def test_Scenario_sample_from_coordinate(scen):
    coord = [0 for i in range(scen.vec_length)]
    coord[4] = True     # set wake model to 1

    sample = scen.values_from_coordinates(coord)
    assert(sample == [0.1, 160, 20, 1, 0])


#
# Test Scenario's single_scenario function
#
def test_Scenario_single_scenario(scen):
    coord = [0 for i in range(scen.vec_length)]
    coord[4] = True  # set wake model to 1

    scen.setup_single(coord)
    assert(abs(scen.systems['Solar']['Pvsamv1'].SystemDesign.subarray1_gcr - 0.1) < e)
    assert(abs(scen.systems['Solar']['Pvsamv1'].SystemDesign.subarray1_azimuth - 160) < e)
    assert(abs(scen.systems['Solar']['Pvsamv1'].SystemDesign.subarray1_tilt - 20) < e)
    assert(scen.systems['Wind']['Windpower'].Farm.wind_farm_wake_model == 1)
    assert(abs(scen.systems['Wind']['Windpower'].Losses.avail_bop_loss - 0) < e)


#
# Test Scenario's run_single function
#
def test_Scenario_run_single(scen):
    outputs = scen.run_single()
    assert(outputs[6] > 0)  # Generic: GenericSystem.Outputs.annual_energy


#
# Test conversion from coordinates to inputs
#
def test_Scenario_conversion(scen):
    coord = [0 for i in range(scen.vec_length)]
    coord[4] = True  # set wake model to 1

    inputs = scen.input_data_from_coordinates(coord)
    correct = {
        'Solar': {
            'Pvsamv1': {
                'SystemDesign': {
                    'subarray1_gcr': .1,
                    'subarray1_azimuth': 160,
                    'subarray1_tilt': 20
                }
            }
        },
        'Wind': {
            'Windpower': {
                'Farm': {
                    'wind_farm_wake_model': 1,
                },
                'Losses': {
                    'avail_bop_loss': 0,
                }
            }
        }
    }
    assert(inputs == correct)
