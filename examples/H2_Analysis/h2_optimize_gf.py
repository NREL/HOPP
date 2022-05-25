import os
from pathlib import Path
from examples.H2_Analysis.h2_setup_optimize import calculate_h_lcoe_continuous
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.gradient_free import GeneticAlgorithm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def objective_function(x):
    """
    This is the objective function to be used in the gradient-free optimization,
    specifically with the genetic algorithm from gradient_free.py.
    Right now it is setup with global variables, which should be fixed soon.
    
    INPUTS
    x: array or list, the design variables considered in the optimization

    OUTPUTS:
    h_lcoe: float, the levelized cost of hydrogen that we want to minimize
    """
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid

    electrolyzer_size_mw = x[0]
    solar_capacity_mw = x[1]
    battery_storage_mwh = x[2]
    n_turbines = int(x[3])

    wind_capacity_mw = n_turbines * scenario['Turbine Rating']

    h_lcoe, _, _, _, _, _ = calculate_h_lcoe_continuous(bat_model,electrolyzer_size_mw,wind_capacity_mw,solar_capacity_mw,
                                                        battery_storage_mwh,battery_storage_mwh,battery_storage_mwh,
                                                        scenario,buy_from_grid=False,sell_to_grid=False)
    
    return h_lcoe


def optimize_gf(workdir=os.getcwd(), show_plot=False):
    """
    Run the plant optimization to minimize LCOH using gradient-free optimization,
    specifically with the genetic algorithm from gradient_free.py.
    Right now it is setup with global variables, which should be fixed soon.
    This function gives a template on how to set up and run the genetic algorithm.
    """

    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid

    bat_model = SimpleDispatch()
    buy_from_grid = False
    sell_to_grid = False

    h2_examples_path = Path(__file__).absolute().parent

    scenarios_df = pd.read_csv(h2_examples_path / 'single_scenario.csv') 
    for i, s in scenarios_df.iterrows():
        scenario = s

    scenario["Powercurve File"] = h2_examples_path / scenario["Powercurve File"]

    ga = GeneticAlgorithm()
    ga.objective_function = objective_function
    ga.bits = np.array([8,8,8,6])
    ga.bounds = np.array([(1E-6,450),(0,450),(1E-6,450),(0,64)])
    ga.variable_type = np.array(["float","float","float","int"])
    
    ga.max_generation = 3
    ga.population_size = 2
    ga.convergence_iters = 10
    ga.tol = 1E-6
    ga.crossover_rate = 0.1
    ga.mutation_rate = 0.01

    ga.optimize_ga(print_progress=False)

    solution_history = ga.solution_history
    opt_lcoh = ga.optimized_function_value
    opt_vars = ga.optimized_design_variables

    opt_electrolyzer_size_mw = opt_vars[0]
    opt_solar_capacity_mw = opt_vars[1]
    opt_battery_storage_mwh = opt_vars[2]
    opt_n_turbines = int(opt_vars[3])

    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(solution_history)
        plt.show()

    return opt_lcoh, opt_electrolyzer_size_mw, opt_solar_capacity_mw, opt_battery_storage_mwh, opt_n_turbines

if __name__=="__main__":

    import time

    start = time.time()
    opt_lcoh, opt_electrolyzer_size_mw, opt_solar_capacity_mw, opt_battery_storage_mwh, opt_n_turbines = optimize_gf()

    print("time to run: ", time.time()-start)
    print("opt_lcoh: ", opt_lcoh)
    print("opt_electrolyzer: ", opt_electrolyzer_size_mw)
    print("opt_solar: ", opt_solar_capacity_mw)
    print("opt_battery: ", opt_battery_storage_mwh)
    print("opt_n_turbs: ", opt_n_turbines)
