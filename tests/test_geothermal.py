from defaults.flatirons_site import get_default
from hybrid.scenario import run_default_scenario
from hybrid.systems_behavior import get_system_behavior_fx
from parameters.parameter_data import get_input_output_data

def test_geothermal():
    technologies = ['Geothermal', 'Generic']
    systems = get_system_behavior_fx(technologies)  # defines which models get run in each system
    defaults, site = get_default(technologies)

    input_data, output_data = get_input_output_data(systems)

    scenario, outputs = run_default_scenario(defaults=defaults,
                                             input_info=input_data,
                                             output_info=output_data,
                                             run_systems=systems)


    # geothermal plant runs at 30 MW for 8760
    assert outputs['Geothermal']['annual_energy'] == 262800000.0

