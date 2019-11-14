from hybrid.optimizer_base import Optimizer
from defaults.defaults_data import get_default
from hybrid.systems_behavior import get_system_behavior_fx


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

# outputs
solar_wind_output_info = {
    'Generic': {
        'Singleowner': [
            'analysis_period_irr'   #: 'Hybrid IRR (%)'
        ]
    }
}

technologies = ['Solar', 'Wind', 'Generic']
run_systems = get_system_behavior_fx(technologies_to_run=technologies)

# test with short stopping condition
def test_Optimizer_stopcond():
    defaults, site = get_default(technologies)
    opt = Optimizer(defaults, solar_wind_input_info, solar_wind_output_info, run_systems)

    def max_iterations(n, max_output, max_output_prev):
        if n > 5:
            return True
        return False

    opt.setup(stopping_cond=max_iterations)

    optimal = opt.optimize()
    params_dict = opt.scenario.input_data_from_coordinates(optimal[1])

    assert(optimal[2] == 6)
    assert(optimal[0] > 4)
    for i in optimal[1]:
        assert(i > 0)

