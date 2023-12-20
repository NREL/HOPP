from greenheart.to_organize.H2_Analysis.h2_setup_optimize import calculate_h_lcoe
from greenheart.to_organize.H2_Analysis.simple_dispatch import SimpleDispatch
from greenheart.to_organize.gradient_free import GeneticAlgorithm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# h_lcoe, np.sum(combined_pv_wind_power_production_hopp), H2_Results['hydrogen_annual_output'], total_system_installed_cost, total_annual_operating_costs

def objective_function(x):
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global best_solution

    electrolyzer_size_mw = x[0]
    solar_capacity_mw = x[1]
    battery_storage_mwh = x[2]
    n_turbines = int(x[3])

    h_lcoe, _, _, _, _ = calculate_h_lcoe(bat_model,electrolyzer_size_mw,n_turbines,solar_capacity_mw,battery_storage_mwh,
                                scenario,buy_from_grid=buy_from_grid,sell_to_grid=sell_to_grid)


    if h_lcoe < best_solution:
        best_solution = h_lcoe
        print("_____________________________")
        print("best_solution: ", h_lcoe)
        print("electrolyzer_size_mw: ", electrolyzer_size_mw)
        print("solar_capacity_mw: ", solar_capacity_mw)
        print("battery_storage_mwh: ", battery_storage_mwh)
        print("n_turbines: ", n_turbines)

    return h_lcoe

def optimize_gf():
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global best_solution

    bat_model = SimpleDispatch()
    scenario = pd.read_csv('single_scenario.csv') 
    buy_from_grid = False
    sell_to_grid = False
    best_solution = 1E16

    ga = GeneticAlgorithm()
    ga.objective_function = objective_function
    ga.bits = np.array([8,8,8,8])
    ga.bounds = np.array([(1E-6,200),(0,200),(0,200),(0,100)])
    ga.variable_type = np.array(["float","float","float","int"])
    
    ga.max_generation = 30
    ga.population_size = 15
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
    return opt_electrolyzer_size_mw, opt_solar_capacity_mw, opt_battery_storage_mwh, opt_n_turbines

if __name__=="__main__":
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global best_solution

    bat_model = SimpleDispatch()
    scenario = pd.read_csv('single_scenario.csv')
    buy_from_grid = False
    sell_to_grid = False
    best_solution = 1E16

    ga = GeneticAlgorithm()
    ga.objective_function = objective_function
    ga.bits = np.array([8,8,8,8])
    ga.bounds = np.array([(1E-6,200),(0,200),(0,200),(0,100)])
    ga.variable_type = np.array(["float","float","float","int"])

    ga.max_generation = 30
    ga.population_size = 15
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

    print("opt_lcoh: ", opt_lcoh)
    print("opt_electrolyzer: ", opt_electrolyzer_size_mw)
    print("opt_solar: ", opt_solar_capacity_mw)
    print("opt_battery: ", opt_battery_storage_mwh)
    print("opt_n_turbs: ", opt_n_turbines)

    import matplotlib.pyplot as plt
    plt.plot(solution_history)
    plt.show()
