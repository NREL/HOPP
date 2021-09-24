
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")

import pyDOE
import logging
from imp import reload
reload(logging)


def problem_setup():
    """

    """
    # Define Design Optimization Variables
    design_variables = dict(
        pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
                  'tilt':                {'bounds':(30,      60)},
                  },
        battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
                  'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
                  'system_voltage_volts':{'bounds':(400,     600)},
                  },
    )

    fixed_variables = dict(
        pv=      {'system_capacity_kw': 75*1e3
                 },
    )

    # Problem definition
    problem = HybridSizingProblem(design_variables, fixed_variables)

    return problem


if __name__ == '__main__':
    logging.basicConfig(filename='test_driver.log',
                        format='%(asctime)s - %(processName)s - %(threadName)s - %(module)s - %(funcName)s - %(message)s',
                        level=logging.DEBUG)

    logging.info("Main Startup")

    # Driver config
    driver_config = dict(eval_limit=100, obj_limit=-3e8, n_proc=6, time_limit=60)

    # Driver init
    driver = OptimizationDriver(problem_setup, **driver_config)

    # Optimizer callable init
    optimizers = humpday.OPTIMIZERS[:5]
    opt_config = dict(n_dim=5, n_trials=50, with_count=True)
    objective_keys = ['net_present_values', 'hybrid']

    # Call all optimizers in parallel
    # best_candidate, best_objective = driver.parallel_optimize(optimizers,
    #                                                           opt_config,
    #                                                           objective_keys)#, cache_file='driver_cache.pkl')
    # best_candidate, best_objective = driver.optimize(optimizers, opt_config, objective_keys)


    # Get experiment candidates, and evaluate objective in parallel
    candidates = pyDOE.lhs(5, criterion='center', samples=12)
    num_evals = driver.parallel_sample(candidates)
    # num_evals = driver.sample(candidates)

    logging.info("All Tasks Complete")
    driver.write_cache()

    # Check on the driver cache
    print(driver.cache_info)

    candidates = list(driver.cache.keys())
    results = list(driver.cache.values())

    # print(candidates[0])

    print(results[0])