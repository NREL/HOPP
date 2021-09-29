from h2_setup_optimize import calculate_h_lcoe
from simple_dispatch import SimpleDispatch
import pyoptsparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# h_lcoe, np.sum(combined_pv_wind_power_production_hopp), H2_Results['hydrogen_annual_output'], total_system_installed_cost, total_annual_operating_costs

def objective_function(x):
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global n_turbines
    global best_solution

    electrolyzer_size_mw = x["electrolyzer_size_mw"]
    solar_capacity_mw = x["solar_capacity_mw"]
    battery_storage_mwh = x["battery_storage_mwh"]

    h_lcoe, _, _, _, _ = calculate_h_lcoe(bat_model,electrolyzer_size_mw,n_turbines,solar_capacity_mw,battery_storage_mwh,
                                scenario,buy_from_grid=buy_from_grid,sell_to_grid=sell_to_grid)
    

    if h_lcoe < best_solution:
        best_solution = h_lcoe
        print("_____________________________")
        print("best_solution: ", h_lcoe)
        print("electrolyzer_size_mw: ", electrolyzer_size_mw)
        print("solar_capacity_mw: ", solar_capacity_mw)
        print("battery_storage_mwh: ", battery_storage_mwh)

    funcs = {}
    fail = False
    funcs["h_lcoe"] = h_lcoe
    
    return funcs, fail


if __name__=="__main__":
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global n_turbines
    global best_solution

    bat_model = SimpleDispatch()
    scenario = pd.read_csv('single_scenario.csv') 
    buy_from_grid = False
    sell_to_grid = False
    n_turbines = 0
    best_solution = 1E16

    start_electrolyzer = 76.8627457137254
    start_solar = 134.901960784313
    start_battery = 185.882352941176

    x = {}
    x["electrolyzer_size_mw"] = start_electrolyzer
    x["solar_capacity_mw"] = start_solar
    x["battery_storage_mwh"] = start_battery

    funcs,_ = objective_function(x)
    start_lcoh = funcs["h_lcoe"]
    print("start_lcoh: ", start_lcoh)

    optProb = pyoptsparse.Optimization("optimize_sizing",objective_function)
    optProb.addVar("electrolyzer_size_mw",type="c",lower=1E-6,upper=300,value=start_electrolyzer)
    optProb.addVar("solar_capacity_mw",type="c",lower=0,upper=300,value=start_solar)
    optProb.addVar("battery_storage_mwh",type="c",lower=0,upper=300,value=start_battery)

    optProb.addObj("h_lcoe")
    optimize = pyoptsparse.SLSQP()
    optimize.setOption("MAXIT",value=50)
    optimize.setOption("ACC",value=1E-6)

    print("start GB optimization")
    solution = optimize(optProb,sens="FD")
    print("******************************************")
    print("finished optimization")

    opt_DVs = solution.getDVs()
    opt_electrolyzer = opt_DVs["electrolyzer_size_mw"]
    opt_solar = opt_DVs["solar_capacity_mw"]
    opt_battery = opt_DVs["battery_storage_mwh"]

    funcs,fail = objective_function(opt_DVs)
    opt_lcoh = funcs["h_lcoe"]

    print("opt_lcoh: ", opt_lcoh)
    print("opt_electrolyzer: ", opt_electrolyzer)
    print("opt_solar: ", opt_solar)
    print("opt_battery: ", opt_battery)
