
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver
import numpy as np

import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")

import pyDOE2 as pyDOE
import logging
from imp import reload
reload(logging)

from hybrid.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()


def problem_setup():
    """

    """
    # Define Design Optimization Variables
    # design_variables = dict(
    #     pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
    #               'tilt':                {'bounds':(30,      60)},
    #               },
    #     battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
    #               'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
    #               'system_voltage_volts':{'bounds':(400,     600)},
    #               },
    # )

    # fixed_variables = dict(
    #     pv=      {'system_capacity_kw': 75*1e3
    #              },
    # )

    design_variables = dict(
        tower=    {'cycle_capacity_kw':  {'bounds':(125*1e3, 125*1e3)},
                   'solar_multiple':     {'bounds':(1.5,     3.5)},
                   'tes_hours':          {'bounds':(6,       16)}
                  },
        pv=       {'system_capacity_kw': {'bounds':(25*1e3,  200*1e3)},
                   'tilt':               {'bounds':(15,      60)}
                  },
    )
    fixed_variables = dict()

    # Problem definition
    problem = HybridSizingProblem(design_variables, fixed_variables)

    return problem


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    logging.basicConfig(filename='test_driver.log',
                        format='%(asctime)s - %(processName)s - %(threadName)s - %(module)s - %(funcName)s - %(message)s',
                        level=logging.DEBUG)

    logging.info("Main Startup")

    # Driver config
    cache_file = 'fullfact_csp_pv.df.gz'
    driver_config = dict(n_proc=16, cache_file=cache_file, cache_interval=4)
    n_dim = 5

    # driver = None

    # Get experiment candidates, and evaluate objective in parallel
    # candidates = pyDOE.lhs(n_dim, criterion='center', samples=4)
    levels = np.array([1, 6, 6, 6, 5]) # 6, 6, 6, 5
    design = pyDOE.fullfact(levels)
    levels[0] = 2
    design_scaled = design / (levels - 1)

    driver = OptimizationDriver(problem_setup, **driver_config)

    chunk_size = 16
    for chunk in chunks(design_scaled, chunk_size):

        num_evals = driver.parallel_sample(chunk, design_name='16665FF', cache_file=cache_file)
        # num_evals = driver.sample(candidates, design_name='16665FF', cache_file=cache_file)

        # Check on the driver cache
        print(driver.cache_info)

    # candidates = list(driver.cache.keys())
    # results = list(driver.cache.values())
