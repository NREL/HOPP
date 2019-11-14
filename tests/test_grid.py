from defaults.defaults_data import get_default
from hybrid.scenario import run_default_scenario
from hybrid.systems_behavior import get_system_behavior_fx
from parameters.parameter_data import get_input_output_data

def test_grid():
    technologies = ['Geothermal', 'Grid', 'Generic']
    systems = get_system_behavior_fx(technologies)  # defines which models get run in each system
    defaults, site = get_default(technologies)

    # test without interconnection limit
    input_data, output_data = get_input_output_data(systems)
    scenario, outputs = run_default_scenario(defaults=defaults,
                                             input_info=input_data,
                                             output_info=output_data,
                                             run_systems=systems)

    # geothermal plant runs at 30 MW for 8760
    assert outputs['Geothermal']['annual_energy'] == 262800000.0
    assert outputs['Generic']['annual_energy'] == 262800000


    # interconnection is limited to 20 MW, so curtail about 1/3 of energy
    defaults['Grid']['Grid']['Common']['enable_interconnection_limit'] = 1
    defaults['Grid']['Grid']['Common']['grid_interconnection_limit_kwac'] = 20000
    scenario, outputs = run_default_scenario(defaults=defaults,
                                             input_info=input_data,
                                             output_info=output_data,
                                             run_systems=systems)

    assert outputs['Geothermal']['annual_energy'] == 262800000
    assert outputs['Grid']['annual_energy'] == 175200000

    # test with hybrid techs
    technologies = ['Solar', 'Wind', 'Grid', 'Generic']
    systems = get_system_behavior_fx(technologies)  # defines which models get run in each system
    defaults, site = get_default(technologies)

    scenario, outputs = run_default_scenario(defaults=defaults,
                                             input_info=input_data,
                                             output_info=output_data,
                                             run_systems=systems)

    assert round(outputs['Generic']['annual_energy']) == 235352833
    assert round(outputs['Grid']['annual_energy']) == 129355369

