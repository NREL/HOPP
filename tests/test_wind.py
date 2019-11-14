from defaults.defaults_data import get_default
import defaults.wind_singleowner as wind_defaults
from hybrid.systems_behavior import get_system_behavior_fx
from hybrid.scenario import Scenario
from parameters.parameter_data import get_input_output_data

def test_wind_powercurve():

    technologies = ['Wind', 'Generic']
    defaults, site = get_default(technologies)
    systems = get_system_behavior_fx(technologies)
    input_data, output_data = get_input_output_data(systems)
    scen = Scenario(defaults=defaults, input_data=input_data, output_info=output_data, system_behavior=systems)
    systems = scen.systems

    model = systems['Wind']['Windpower']
    wind = defaults['Wind']['Windpower']

    # calculate system capacity.  To evaluate other turbines, update the defaults dictionary
    model.Turbine.calculate_powercurve(wind_defaults.wind_default_rated_output,
                                                  wind['Turbine']['wind_turbine_rotor_diameter'],
                                                  wind_defaults.wind_default_max_tip_speed,
                                                  wind_defaults.wind_default_max_tip_speed_ratio,
                                                  wind_defaults.wind_default_cut_in_speed,
                                                  wind_defaults.wind_default_cut_out_speed,
                                                  wind_defaults.wind_default_drive_train)

    windspeeds_truth = [round(x,2) for x in wind_defaults.powercurveWS]
    windspeeds_calc = [round(x,2) for x in model.Turbine.wind_turbine_powercurve_windspeeds]
    powercurve_truth = [round(x,0) for x in wind_defaults.powercurveKW]
    powercurve_calc = [round(x,0) for x in model.Turbine.wind_turbine_powercurve_powerout]


    assert all([a == b for a,b in zip(windspeeds_truth, windspeeds_calc )])
    assert all([a == b for a,b in zip(powercurve_truth, powercurve_calc )])


    print('Done')