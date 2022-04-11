
# HOPP optimization problem and driver
from alt_dev.optimization_problem_alt import HybridSizingProblem, expand_financial_model 
from alt_dev.optimization_driver_alt import OptimizationDriver
from hybrid.hybrid_simulation import HybridSimulation
from PySAM import Singleowner


# Import humpday, optimization algorithms
# import warnings
# warnings.simplefilter("ignore")
# import humpday
# warnings.simplefilter("default")

# from bayes_opt import BayesianOptimization

# from skopt import gp_minimize, forest_minimize, gbrt_minimize

# from poap.controller import BasicWorkerThread, ThreadController
# from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube, TwoFactorial
# from pySOT.strategy import SRBFStrategy, EIStrategy, DYCORSStrategy,RandomStrategy, LCBStrategy
# from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant, GPRegressor
# from pySOT.optimization_problems.optimization_problem import OptimizationProblem

# import nevergrad as ng

# Import design of experiments and numpy
import pyDOE2 as pyDOE
# import idaes.surrogate.pysmo.sampling as sampling
import numpy as np

# NREL dev API key, for weather and data files and site information
from pathlib import Path
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hybrid.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()

from functools import partial
import os

WEATHER_FILE = "nchile_tmy.csv"
PRICE_FILE = "pricing-data-nchile_shift.csv"

def init_simulation_pv():
    """
    Create the simulation object needed to calculate the objective of the problem

    :return: The HOPP simulation as defined for this problem
    """

    # Create the site for the design evaluation
    site = 'irregular'
    location = locations[3]

    if site == 'circular':
        site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
    elif site == 'irregular':
        site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
    else:
        raise Exception("Unknown site '" + site + "'")

    # Load in weather and price data files
    solar_file = Path(
        __file__).parent.parent / "resource_files" / "solar" / WEATHER_FILE #"Beni_Miha" / "659265_32.69_10.90_2019.csv"
    grid_file = Path(__file__).parent.parent / "resource_files" / "grid" / PRICE_FILE #"tunisia_est_grid_prices.csv"

    # Combine the data into a site definition
    site_info = SiteInfo(site_data, solar_resource_file=solar_file, grid_resource_file=grid_file)

    # set up hybrid simulation with all the required parameters
    solar_size_mw = 200
    battery_capacity_mwh = 15
    battery_capacity_mw = 100
    interconnection_size_mw = 100

    technologies = {'pv': {'system_capacity_kw': solar_size_mw * 1000,
                           'array_type': 2,
                           'dc_ac_ratio': 1.1},
                    'battery': {'system_capacity_kwh': battery_capacity_mwh * 1000,
                                'system_capacity_kw': battery_capacity_mw * 1000},
                    'grid': interconnection_size_mw * 1000}

    # Create the hybrid plant simulation
    # TODO: turn these off to run full year simulation
    dispatch_options = {'is_test_start_year': False,
                        'is_test_end_year': False,
                        'solver': 'gurobi_ampl',
                        'grid_charging': False,  
                        'pv_charging_only': True}

    # TODO: turn-on receiver and field optimization before... initial simulation
    hybrid_plant = HybridSimulation(technologies,
                                    site_info,
                                    interconnect_kw=interconnection_size_mw * 1000,
                                    dispatch_options=dispatch_options)

    # Customize the hybrid plant assumptions here...
    hybrid_plant.pv.value('inv_eff', 95.0)
    hybrid_plant.pv.value('array_type', 0)
    hybrid_plant.pv.dc_degradation = [0] * 25

    return hybrid_plant


def init_problem_pv():
    """
    Create the optimization problem by defining design variables and bounds along with the function
    to initialize the simulation

    :return: The HOPP optimization problem
    """
    # Design variables and their bounds
    design_variables = dict(
        pv=       {'system_capacity_kw':  {'bounds':(100*1e3,  300*1e3)},
                   'dc_ac_ratio':         {'bounds':(1.0, 1.8)},
                  },
        battery=  {'system_capacity_kwh': {'bounds':(100*1e3, 600*1e3)},
                  },
    )

    options = dict(time_series_outputs=True, # add soc and curtailed series outputs
                   # dispatch_factors=False,    # add dispatch factors to objective output
                   generation_profile=True,  # add technology generation profile to output
                   financial_model=True,     # add financial model dictionary to output
                   shrink_output=True,       # keep only the first year of output
                   )

    # Problem definition
    problem = HybridSizingProblem(init_simulation_pv, design_variables, output_options=options) #, fixed_variables)

    return problem


def init_simulation_csp():
    """
    Create the simulation object needed to calculate the objective of the problem

    :return: The HOPP simulation as defined for this problem
    """

    # Create the site for the design evaluation
    site = 'irregular'
    location = locations[3]

    if site == 'circular':
        site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
    elif site == 'irregular':
        site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
    else:
        raise Exception("Unknown site '" + site + "'")

    # Load in weather and price data files
    solar_file = Path(
        __file__).parent.parent / "resource_files" / "solar" / WEATHER_FILE #"Beni_Miha" / "659265_32.69_10.90_2019.csv"
    grid_file = Path(__file__).parent.parent / "resource_files" / "grid" / PRICE_FILE #"tunisia_est_grid_prices.csv"

    # Combine the data into a site definition
    site_info = SiteInfo(site_data, solar_resource_file=solar_file, grid_resource_file=grid_file)

    # set up hybrid simulation with all the required parameters
    tower_cycle_mw = 100
    interconnection_size_mw = 100

    technologies = {'tower': {'cycle_capacity_kw': tower_cycle_mw * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 12.0,
                              'optimize_field_before_sim': True}, # TODO: turn on
                    'grid': interconnection_size_mw * 1000}

    # Create the hybrid plant simulation
    # TODO: turn these off to run full year simulation
    dispatch_options = {'is_test_start_year': False,
                        'is_test_end_year': False,
                        'solver': 'gurobi_ampl'}

    # TODO: turn-on receiver and field optimization before... initial simulation
    hybrid_plant = HybridSimulation(technologies,
                                    site_info,
                                    interconnect_kw=interconnection_size_mw * 1000,
                                    dispatch_options=dispatch_options)

    return hybrid_plant


def init_problem_csp():
    """
    Create the optimization problem by defining design variables and bounds along with the function
    to initialize the simulation

    :return: The HOPP optimization problem
    """
    # Design variables and their bounds
    design_variables = dict(
        tower=    {'solar_multiple': {'bounds':(0.8,  3.0)},
                   'tes_hours':      {'bounds':(4,    16)}
                  },
    )

    options = dict(time_series_outputs=True, # add soc and curtailed series outputs
                   # dispatch_factors=False,    # add dispatch factors to objective output
                   generation_profile=True,  # add technology generation profile to output
                   financial_model=True,     # add financial model dictionary to output
                   shrink_output=True,       # keep only the first year of output
                   )

    # Problem definition
    problem = HybridSizingProblem(init_simulation_csp, design_variables, output_options=options) #, fixed_variables)

    return problem
   
   
def init_simulation_hybrid():
    """
    Create the simulation object needed to calculate the objective of the problem

    :return: The HOPP simulation as defined for this problem
    """

    # Create the site for the design evaluation
    site = 'irregular'
    location = locations[3]

    if site == 'circular':
        site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
    elif site == 'irregular':
        site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
    else:
        raise Exception("Unknown site '" + site + "'")

    # Load in weather and price data files
    solar_file = Path(
        __file__).parent.parent / "resource_files" / "solar" / WEATHER_FILE #"Beni_Miha" / "659265_32.69_10.90_2019.csv"
    grid_file = Path(__file__).parent.parent / "resource_files" / "grid" / PRICE_FILE #"tunisia_est_grid_prices.csv"

    # Combine the data into a site definition
    site_info = SiteInfo(site_data, solar_resource_file=solar_file, grid_resource_file=grid_file)

    # set up hybrid simulation with all the required parameters
    tower_cycle_mw = 100
    solar_size_mw = 200
    battery_capacity_mwh = 15
    battery_capacity_mw = 100
    interconnection_size_mw = 100

    technologies = {'pv': {'system_capacity_kw': solar_size_mw * 1000,
                           'array_type': 2,
                           'dc_ac_ratio': 1.1},
                    'tower': {'cycle_capacity_kw': tower_cycle_mw * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 12.0,
                              'optimize_field_before_sim': True}, # TODO: turn on
                    'battery': {'system_capacity_kwh': battery_capacity_mwh * 1000,
                                'system_capacity_kw': battery_capacity_mw * 1000},
                    'grid': interconnection_size_mw * 1000}

    # Create the hybrid plant simulation
    # TODO: turn these off to run full year simulation
    dispatch_options = {'is_test_start_year': False,
                        'is_test_end_year': False,
                        'solver': 'gurobi_ampl',
                        'grid_charging': False,  
                        'pv_charging_only': True}

    # TODO: turn-on receiver and field optimization before... initial simulation
    hybrid_plant = HybridSimulation(technologies,
                                    site_info,
                                    interconnect_kw=interconnection_size_mw * 1000,
                                    dispatch_options=dispatch_options)

    # Customize the hybrid plant assumptions here...
    hybrid_plant.pv.value('inv_eff', 95.0)
    hybrid_plant.pv.value('array_type', 0)
    hybrid_plant.pv.dc_degradation = [0] * 25

    return hybrid_plant


def init_problem_hybrid():
    """
    Create the optimization problem by defining design variables and bounds along with the function
    to initialize the simulation

    :return: The HOPP optimization problem
    """
    # Design variables and their bounds
    design_variables = dict(
        tower=    {'solar_multiple': {'bounds':(0.8,      3.0)},
                   'tes_hours':      {'bounds':(4,        16)},
                   'cycle_capacity_kw': {'bounds':(50*1e3,90*1e3)},
                  },
        pv=       {'system_capacity_kw': {'bounds':(100*1e3,  300*1e3)},
                   'dc_ac_ratio':        {'bounds':(1.0, 1.8)},
                   },
        battery=  {'system_capacity_kwh': {'bounds':(100*1e3, 600*1e3)},
                  },
    )

    options = dict(time_series_outputs=True, # add soc and curtailed series outputs
                   # dispatch_factors=False,    # add dispatch factors to objective output
                   generation_profile=True,  # add technology generation profile to output
                   financial_model=True,     # add financial model dictionary to output
                   shrink_output=True,       # keep only the first year of output
                   )

    # Problem definition
    problem = HybridSizingProblem(init_simulation_hybrid, design_variables, output_options=options) #, fixed_variables)

    return problem
   

## Example optimization objectives, the objective recieves the nested result dictionary from the problem objective
def calc_capacity_credit_percent(model_dict, N=100):
    
    system_cap = model_dict['CapacityPayments']['cp_system_nameplate']
    price = model_dict['Revenue']['dispatch_factors_ts']
    gen = model_dict['SystemOutput ']['gen']
    
    
    df = pd.DataFrame({'price': price, 'gen': gen})
    
    selected = df.nlargest(N, 'price')
    
    capacity_value = selected['gen'].sum() / (system_cap * len(selected.index)) * 100
    capacity_value = min(100, capacity_value)
    
    return (capacity_value,)

def recalc_financials(result, assumptions):
    cols = [col for col in result.keys() if col.endswith('_financial_model')]
    
    models = []
    for col in cols:
        model_dict = expand_financial_model(result[col])
        model = Singleowner.new() #.default(defaults[tech_prefix[j]])
        model.assign(model_dict)
        
        for key, val in assumptions.items():
            if key == 'cp_capacity_credit_percent':
                val = calc_capacity_credit_percent(model_dict, N=val)
            model.value(key, val)
        model.execute()
                
        models.append(model)
        
    return models

def max_npv_itc(result):
    obj_col = 'project_return_aftertax_npv'
    assumptions = {'itc_fed_percent': 26}
    
    models = recalc_financials(result, assumptions)
    npvs = [model.value(obj_col) for model in models]
        
    # negate since max
    return -sum(npvs)
    
    
def minLCOE(result):
    return result['Hybrid Real Levelized Cost of Energy ($/kWh)']

def maxBCR(result):
    return -result['Hybrid Benefit cost Ratio (-)'] # negate for max
    

# pip install scikit-optimize
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dump


if __name__ == '__main__':

    N = 20
  
    ## Driver config
    driver_config = dict(n_proc=20, cache_dir='caiso20_pv', 
                                    dataframe_file='pv_results.df.gz', reconnect_cache=True)
    driver = OptimizationDriver(init_problem_pv, **driver_config)
    n_dim = 3
    
    ## Optimization
    bounds = [(a,b) for a,b in zip(driver.problem.lower_bounds, driver.problem.upper_bounds)]
    c = [k for k in driver.cache if isinstance(k, list)]

    # get 'x' values of designs in cache
    x0 = [[(x[1]- a) / (b-a) for x, (a,b) in zip(k, bounds)] for k in c]
    
    # get 'y' values of designs in cache (for each objective)
    BCR = [maxBCR(driver.cache[k]) for k in c]
    LCOE = [minLCOE(driver.cache[k]) for k in c]
    
    # base configuration for skopt minimizer
    base_config = dict(dimensions = [(0.,1.)]*n_dim,
                  n_calls=N,
                  x0=x0,
                  verbose=False,
                  n_initial_points=0,
                  acq_func='LCB',
                  kappa=0.1 # higher for more exploration, lower for less (default = 1.94)
                  )
    
    # list of minimizer functions
    optimizers = [gp_minimize, forest_minimize, gbrt_minimize] * 2
    opt_configs = [base_config.copy() for i in range(len(optimizers))]
    
    # edit config (for each objective)
    opt_configs[0]['noise'] = 1E-9 # tell Gaussian minimizer the objective is deterministic
    for i in [0, 1, 2]:
        opt_configs[i]['y0'] = BCR
        
    opt_configs[3]['noise'] = 1E-9 # tell Gaussian minimizer the objective is deterministic
    for i in [3, 4, 5]:
        opt_configs[i]['y0'] = LCOE
    
    # list of objective functions (accepting a problem result dictionary as input)
    objectives = [maxBCR]*3 + [minLCOE]*3
    
    # parallel execute optimizers on the objectives of interest
    out = driver.parallel_optimize(optimizers, opt_configs, objectives)

    # optionally save out the optimizer object for post analysis
    for key in out.keys():
         dump(out[key], f"alt_dev/p_{key}.pkl.gz", compress=9, store_objective=False)
    
    
    

    # ## Parametric sweep
    # # levels = np.array([2] * n_dim])
    # # design = pyDOE.fullfact(levels)
    # # levels[levels == 1] = 2
    # # ff_scaled = design / (levels - 1)
    
    # # lhs_scaled = pyDOE.lhs(n_dim, criterion='centermaximin', samples=N)
    
    # bounds = [[0]*n_dim, [1]*n_dim]
    # b = sampling.CVTSampling(bounds, N, tolerance = 1e-3, sampling_type="creation")
    # samples = b.sample_points()

    # ## Execute Candidates
    # num_evals = driver.parallel_sample(samples, design_name='cvt_750')
    
    
    
    
    
    # N = 500
    
    # ## Driver config
    # driver_config = dict(n_proc=20, cache_dir='nchile_csp_20', 
                                    # dataframe_file='csp_results.df.gz', reconnect_cache=False)
    # driver = OptimizationDriver(init_problem_csp, **driver_config)
    # n_dim = 2

    # ## Parametric sweep
    # # levels = np.array([2] * n_dim])
    # # design = pyDOE.fullfact(levels)
    # # levels[levels == 1] = 2
    # # ff_scaled = design / (levels - 1)
    
    # # lhs_scaled = pyDOE.lhs(n_dim, criterion='centermaximin', samples=N)
    
    # bounds = [[0]*n_dim, [1]*n_dim]
    # b = sampling.CVTSampling(bounds, N, tolerance = 1e-3, sampling_type="creation")
    # samples = b.sample_points()

    # ## Execute Candidates
    # num_evals = driver.parallel_sample(samples, design_name='cvt_500')
    
    
    
    
    
    # N = 1500    
    
    # # Driver config
    # driver_config = dict(n_proc=20, cache_dir='nchile_hybrid_20', 
                                    # dataframe_file='hybrid_results.df.gz', reconnect_cache=False)
    # driver = OptimizationDriver(init_problem_hybrid, **driver_config)
    # n_dim = 6
    
    # ## Parametric sweep
    # # levels = np.array([2] * n_dim])
    # # design = pyDOE.fullfact(levels)
    # # levels[levels == 1] = 2
    # # ff_scaled = design / (levels - 1)
    
    # lhs_scaled = pyDOE.lhs(n_dim, criterion='centermaximin', samples=N)
    
    # bounds = [[0]*n_dim, [1]*n_dim]
    # b = sampling.CVTSampling(bounds, N, tolerance = 1e-3, sampling_type="creation")
    # samples = b.sample_points()

    # ## Execute Candidates
    # num_evals = driver.parallel_sample(lhs_scaled, design_name='cvt_1500')
    
    
    
    
    os.system('echo \"Subject: solve complete on $(hostname) - $(date)\" | sendmail jlcox@mines.edu')
    


    # # bounds = [(a,b) for a,b in zip(driver.problem.lower_bounds, driver.problem.upper_bounds)]
    # # c = [k for k in driver.cache if isinstance(k, list)]

    # # x0 = [[(x[1]- a) / (b-a) for x, (a,b) in zip(k, bounds)] for k in c]
    # # y0 = [test(driver.cache[k]) for k in c]

    # from skopt import gp_minimize, forest_minimize, gbrt_minimize

    # # opt_config = dict(dimensions = [(0.,1.)]*n_dim,
                      # # n_calls=10,
                      # # x0=x0,
                      # # y0=y0,
                      # # verbose=True,
                      # # n_initial_points=0)
                      
    # optimizers = [gp_minimize, forest_minimize, gbrt_minimize]
    # # out = driver.optimize(optimizers[:1], opt_config, test)

    # from skopt import dump

    # # for key in out.keys():
         # # dump(out[key], f'alt_dev/s_{key}.pkl.gz', compress=9, store_objective=False)
         
    # bounds = [(a,b) for a,b in zip(driver.problem.lower_bounds, driver.problem.upper_bounds)]
    # c = [k for k in driver.cache if isinstance(k, list)]

    # x0 = [[(x[1]- a) / (b-a) for x, (a,b) in zip(k, bounds)] for k in c]
    # y0 = [test(driver.cache[k]) for k in c]

    # opt_config = dict(dimensions = [(0.,1.)]*n_dim,
                      # n_calls=10,
                      # x0=x0,
                      # y0=y0,
                      # verbose=True,
                      # n_initial_points=0,
                      # noise=1E-10,
                      # acq_func='LCB',
                      # kappa=0.01,
                      # )
         
    # out = driver.optimize(optimizers[:1], opt_config, test)

    # for key in out.keys():
         # dump(out[key], f'alt_dev/p_{key}.pkl.gz', compress=9, store_objective=False)

    # obj = partial(driver.wrapped_objective(), name='gp_test', objective=test)
    
    # res = gp_minimize(obj, 
                      # [(0,1)]*n_dim,
                      # acq_func="EI",
                      # n_calls=15,
                      # x0=x0,
                      # y0=y0)
                      
    
    
    ## Latin Hypercube
    # lhs_scaled = pyDOE.lhs(n_dim, criterion='center', samples=100)
    
    ## Parametric sweep
    # levels = np.array([10, 10])
    # design = pyDOE.fullfact(levels)
    # levels[levels == 1] = 2
    # ff_scaled = design / (levels - 1)

    ## Execute Candidates
    # num_evals = driver.sample(ff_scaled, design_name='test_s')
    # num_evals = driver.parallel_sample(ff_scaled, design_name='test_csp')
    
    
    # ## Select optimization algorithms, common configuration
    # optimizers = [humpday.OPTIMIZERS[0], humpday.OPTIMIZERS[31]]
    # opt_config = dict(n_dim=n_dim, n_trials=50, with_count=True)
    
    ## Execute optimizer(s)
    # best, best_candidate = driver.optimize(optimizers[:1], opt_config, max_npv_itc)
    # best, best_candidate = driver.parallel_optimize(optimizers, opt_config, max_npv_itc)

    # print(best, best_cankdidate)


    ### Sampling Example

    ## Parametric sweep
    # levels = np.array([1, 3, 3, 3, 3])
    # design = pyDOE.fullfact(levels)
    # levels[levels == 1] = 2
    # ff_scaled = design / (levels - 1)

    ## Latin Hypercube
    # lhs_scaled = pyDOE.lhs(n_dim, criterion='center', samples=2)

    ## Execute Candidates
    # num_evals = driver.sample(ff_scaled, design_name='test_s')
    # num_evals = driver.parallel_sample(lhs_scaled, design_name='test_p')



    ### Optimization Example

    ## Show humpday optimizers
    # for i, f in humpday.OPTIMIZERS:
    #     print(i, f.__name__)

    ## Select optimization algorithms, common configuration
    # optimizers = [humpday.OPTIMIZERS[0], humpday.OPTIMIZERS[53]]
    # opt_config = dict(n_dim=n_dim, n_trials=50, with_count=True)

    ## Execute optimizer(s)
    # best_energy, best_energy_candidate = driver.optimize(optimizers[:1], opt_config, max_hybrid_energy)
    # best_lcoe, best_lcoe_candidate = driver.parallel_optimize(optimizers, opt_config, min_hybrid_lcoe)


    ## Print cache information
    print(driver.cache_info)