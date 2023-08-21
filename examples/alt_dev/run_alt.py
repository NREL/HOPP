from pathlib import Path
examples_dir = Path(__file__).parent.absolute()

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

    solar_file = examples_dir.parent / "resource_files" / "solar" / "Beni_Miha" / "659265_32.69_10.90_2019.csv"
    grid_file = examples_dir.parent / "resource_files" / "grid" / "tunisia_est_grid_prices.csv"

    site_info = SiteInfo(site_data, solar_resource_file=solar_file, grid_resource_file=grid_file)

    # set up hybrid simulation with all the required parameters
    solar_size_mw = 200
    tower_cycle_mw = 125
    # battery_capacity_mwh = 15
    interconnection_size_mw = 400

    technologies = {'tower': {'cycle_capacity_kw': tower_cycle_mw * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 12.0,
                              'optimize_field_before_sim': False}, # TODO: turn on
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


def init_problem():
    """

    """

    design_variables = dict(
        tower=    {'cycle_capacity_kw':  {'bounds':(125*1e3, 125*1e3)},
                   'solar_multiple':     {'bounds':(1.5,     3.5)},
                   'tes_hours':          {'bounds':(6,       16)}
                  },
        pv=       {'system_capacity_kw': {'bounds':(25*1e3,  200*1e3)},
                   'tilt':               {'bounds':(15,      60)}
                  },
    )

    # fixed_variables = dict(
    #     pv=      {'system_capacity_kw': 75*1e3
    #              },
    # )

    # Problem definition
    problem = HybridSizingProblem(init_simulation, design_variables) #, fixed_variables)

    return problem


def max_hybrid_energy(result):
    return -result['annual_energies']['hybrid']

def min_hybrid_lcoe(result):
    return result['lcoe_real']['hybrid']


if __name__ == '__main__':

    # Driver config
    cache_file = 'test_csp_pv.df.gz'
    driver_config = dict(n_proc=4, eval_limit=5, cache_file=cache_file, cache_dir='test')
    driver = OptimizationDriver(init_problem, **driver_config)
    n_dim = 5


    ### Sampling Example

    ## Parametric sweep
    levels = np.array([1, 1, 4, 1, 1])
    design = pyDOE.fullfact(levels)
    levels[levels == 1] = 2
    ff_scaled = design / (levels - 1)

    ## Latin Hypercube
    lhs_scaled = pyDOE.lhs(n_dim, criterion='center', samples=12)

    ## Execute Candidates
    num_evals = driver.sample(ff_scaled, design_name='test_s', cache_file=cache_file)
    num_evals = driver.parallel_sample(lhs_scaled, design_name='test_p', cache_file=cache_file)



    ### Optimization Example

    ## Show humpday optimizers
    # for i, f in humpday.OPTIMIZERS:
    #     print(i, f.__name__)

    ## Select optimization algorithms, common configuration
    optimizers = [humpday.OPTIMIZERS[0], humpday.OPTIMIZERS[53]]
    opt_config = dict(n_dim=n_dim, n_trials=50, with_count=True)

    ## Execute optimizer(s)
    best_energy, best_energy_candidate = driver.optimize(optimizers[:1], opt_config, max_hybrid_energy, cache_file=cache_file)
    best_lcoe, best_lcoe_candidate = driver.parallel_optimize(optimizers, opt_config, min_hybrid_lcoe, cache_file=cache_file)


    ## Print cache information
    print(driver.cache_info)