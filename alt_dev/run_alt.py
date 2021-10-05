
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver
import numpy as np

import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")

import pyDOE2 as pyDOE
from hybrid.hybrid_simulation import HybridSimulation
from pathlib import Path
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations

from hybrid.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()


def init_simulation():
    """
    Create the simulation object needed to calculate the objective of the problem
    TODO: make this representative of the design variables, is there currently a tradeoff in objectives?

    :return: The HOPP simulation as defined for this problem
    """

    site = 'irregular'
    location = locations[1]

    if site == 'circular':
        site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
    elif site == 'irregular':
        site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
    else:
        raise Exception("Unknown site '" + site + "'")

    solar_file = Path(
        __file__).parent.parent / "resource_files" / "solar" / "Beni_Miha" / "659265_32.69_10.90_2019.csv"
    grid_file = Path(__file__).parent.parent / "resource_files" / "grid" / "tunisia_est_grid_prices.csv"

    site_info = SiteInfo(site_data, solar_resource_file=solar_file, grid_resource_file=grid_file)

    # set up hybrid simulation with all the required parameters
    solar_size_mw = 50
    # battery_capacity_mwh = 1
    interconnection_size_mw = 400

    technologies = {'tower': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 12.0,
                              'optimize_field_before_sim': True},
                    'pv': {'system_capacity_kw': solar_size_mw * 1000},
                    # 'battery': {'system_capacity_kwh': battery_capacity_mwh * 1000,
                    #             'system_capacity_kw': battery_capacity_mwh * 1000 / 10},
                    'grid': interconnection_size_mw * 1000}

    # Create model
    # TODO: turn these off to run full year simulation
    dispatch_options = {'is_test_start_year': True,
                        'is_test_end_year': False}

    # TODO: turn-on receiver and field optimization before... initial simulation
    hybrid_plant = HybridSimulation(technologies,
                                    site_info,
                                    interconnect_kw=interconnection_size_mw * 1000,
                                    dispatch_options=dispatch_options)

    # Customize the hybrid plant assumptions here...
    hybrid_plant.pv.value('inv_eff', 95.0)
    hybrid_plant.pv.value('array_type', 0)

    return hybrid_plant


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
    problem = HybridSizingProblem(init_simulation, design_variables, fixed_variables)

    return problem


if __name__ == '__main__':
    # logging.basicConfig(filename='test_driver.log',
    #                     format='%(asctime)s - %(processName)s - %(threadName)s - %(module)s - %(funcName)s - %(message)s',
    #                     level=logging.DEBUG)

    # logging.info("Main Startup")

    # Driver config
    cache_file = 'test_csp_pv.df.gz'
    driver_config = dict(n_proc=4, cache_file=cache_file, cache_interval=4, cache_dir='test')
    n_dim = 5

    # driver = None

    # Get experiment candidates, and evaluate objective in parallel
    design_scaled = pyDOE.lhs(n_dim, criterion='center', samples=4)
    
    # levels = np.array([1, 6, 6, 6, 5]) # 6, 6, 6, 5
    # design = pyDOE.fullfact(levels)
    # levels[0] = 2
    # design_scaled = design / (levels - 1)

    driver = OptimizationDriver(problem_setup, **driver_config)

    # chunk_size = 32
    # for chunk in chunks(design_scaled, chunk_size):

    num_evals = driver.parallel_sample(design_scaled, design_name='16665FF', cache_file=cache_file)
    # num_evals = driver.sample(candidates, design_name='16665FF', cache_file=cache_file)

    # Check on the driver cache
    print(driver.cache_info)

    # candidates = list(driver.cache.keys())
    # results = list(driver.cache.values())
