"""
This file shows an example of how to set up a gradinet-based optiization with
pyoptsparse. Currently, the LCOH with respect to wind capacity is discontinuous,
because HOPP rounds wind capacity to integer values of wind turbines. 
h2_setup_optimize currently fixes the wind energy production to be continuous, but
not the costs. So, this file will have trouble running for certain values, but
can still be used as a reference for setting up a gradient-based optimization.
"""

from numpy.lib.npyio import save
from h2_setup_optimize import calculate_h_lcoe_continuous
from simple_dispatch import SimpleDispatch
import pyoptsparse
import pandas as pd
import warnings
import os
import numpy as np
import csv
warnings.filterwarnings("ignore")

def objective_function(x):
    """
    This is the onjective function to be used in the gradient-based optimization.
    Right now it is setup with global variables, which should be fixed soon.
    This objective function is setup to run with pyoptsparse, meaning that it takes
    a dictionary (x in this case) as the input, and returns funcs and fail. funcs is a
    dictionary with the objective, and constraints if there are any, and fail is the flag
    indicating if the function failed.
    """
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global best_solution
    global obj_scale

    electrolyzer_size_mw = x["electrolyzer_size_mw"]
    wind_capacity_mw = x["wind_capacity_mw"]
    solar_capacity_mw = x["solar_capacity_mw"]
    battery_storage_mwh = x["battery_storage_mwh"]
    battery_storage_mw = x["battery_storage_mw"]    

    h_lcoe, _, _, _, _, _ = calculate_h_lcoe_continuous(bat_model,electrolyzer_size_mw,wind_capacity_mw,solar_capacity_mw,
                                                        battery_storage_mwh,battery_storage_mw,battery_storage_mw,
                                                        scenario,buy_from_grid=False,sell_to_grid=False)
    

    if h_lcoe < best_solution:
        best_solution = h_lcoe
        print("_____________________________")
        print("best_solution: ", h_lcoe)
        print("electrolyzer_size_mw: ", electrolyzer_size_mw)
        print("wind_capacity_mw: ", wind_capacity_mw)
        print("solar_capacity_mw: ", solar_capacity_mw)
        print("battery_storage_mwh: ", battery_storage_mwh)
        print("battery_storage_mw: ", battery_storage_mw)

    funcs = {}
    fail = False
    funcs["h_lcoe"] = h_lcoe/obj_scale
    
    return funcs, fail


if __name__=="__main__":
    global bat_model
    global scenario
    global buy_from_grid
    global sell_to_grid
    global best_solution
    global obj_scale

    save_filename = "test_scaling2.csv"

    scenarios_df = pd.read_csv('single_scenario.csv') 
    for i, s in scenarios_df.iterrows():
        scenario = s

    scales = np.array([1E-4])
    buy_from_grid = False
    sell_to_grid = False

    import time

    for k in range(len(scales)):
        start_time = time.time()

        obj_scale = scales[k]

        bat_model = SimpleDispatch()

        best_solution = 1E16

        start_electrolyzer = 250.0
        start_wind = 500.0
        start_solar = 1.0
        start_battery_mwh = 100.0
        start_battery_mw = 100.0

        x = {}
        x["electrolyzer_size_mw"] = start_electrolyzer
        x["wind_capacity_mw"] = start_wind
        x["solar_capacity_mw"] = start_solar
        x["battery_storage_mwh"] = start_battery_mwh
        x["battery_storage_mw"] = start_battery_mw

        funcs,_ = objective_function(x)
        start_lcoh = funcs["h_lcoe"]*obj_scale
        print("start_lcoh: ", start_lcoh)

        optProb = pyoptsparse.Optimization("optimize_sizing",objective_function)
        optProb.addVar("electrolyzer_size_mw",type="c",lower=1,upper=250,value=start_electrolyzer)
        optProb.addVar("wind_capacity_mw",type="c",lower=1,upper=1000,value=start_wind)
        optProb.addVar("solar_capacity_mw",type="c",lower=1,upper=500,value=start_solar)
        optProb.addVar("battery_storage_mwh",type="c",lower=0,upper=1000,value=start_battery_mwh)
        optProb.addVar("battery_storage_mw",type="c",lower=0,upper=1000,value=start_battery_mw)

        optProb.addObj("h_lcoe")
        optimize = pyoptsparse.SLSQP()
        optimize.setOption("MAXIT",value=5)
        # optimize.setOption("ACC",value=1E-6)
        # optimize = pyoptsparse.SNOPT()
        

        print("start GB optimization")
        solution = optimize(optProb,sens="FD")
        print("******************************************")
        print("finished optimization")

        opt_DVs = solution.getDVs()
        opt_electrolyzer = opt_DVs["electrolyzer_size_mw"]
        opt_wind = opt_DVs["wind_capacity_mw"]
        opt_solar = opt_DVs["solar_capacity_mw"]
        opt_battery_mwh = opt_DVs["battery_storage_mwh"]
        opt_battery_mw = opt_DVs["battery_storage_mw"]

        funcs,fail = objective_function(opt_DVs)
        opt_lcoh = funcs["h_lcoe"]
        time_to_run = time.time()-start_time

        print("opt_lcoh: ", opt_lcoh*obj_scale)
        print("opt_electrolyzer: ", opt_electrolyzer)
        print("opt_wind: ", opt_wind)
        print("opt_solar: ", opt_solar)
        print("opt_battery_mwh: ", opt_battery_mwh)
        print("opt_battery_mw: ", opt_battery_mw)
        print("time_to_run: ", time_to_run)


        # save the results
        results_array = [scenario['Lat'],scenario['Long'],opt_lcoh*obj_scale,time_to_run,obj_scale,opt_electrolyzer,opt_wind,opt_solar,opt_battery_mwh,opt_battery_mw]
        if os.path.exists(save_filename) == False:
            header_array = ["latitude","longitude","lcoh ($/kg)","time (s)","scale","electrolyzer (MW)","wind (MW)","solar (MW)","battery (MWh)","battery (MW)"]
            f = open(save_filename, 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(header_array)
            f.close()

        f = open(save_filename, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(results_array)
        f.close()