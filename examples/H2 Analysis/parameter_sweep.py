from h2_setup_optimize import calculate_h_lcoe_continuous
from simple_dispatch import SimpleDispatch
import pandas as pd
import warnings
import os
import numpy as np
import csv
warnings.filterwarnings("ignore")
import time

if __name__=="__main__":

    save_filename = "test_georgia.csv"
    scenarios_df = pd.read_csv('Georgia_scenarios.csv')
    buy_from_grid = False
    sell_to_grid = False

    start = time.time()

    electrolyzer_sizes = np.array([100,250,500,1000])

    for i, s in scenarios_df.iterrows():
        for k in range(len(electrolyzer_sizes)):
            
                scenario = s

                bat_model = SimpleDispatch()
                electrolyzer_size_mw = electrolyzer_sizes[k]
                wind_capacity_mw = scenario["Wind Size (MW)"]
                solar_capacity_mw = scenario["Solar Size (MW)"]
                battery_storage_mwh = scenario["Storage Size (MWh)"]
                battery_storage_mw = scenario["Storage Size (MW)"]


                h_lcoe,pv_wind,hydrogen_annual_output,total_installed_cost,total_operating_cost,capacity_factor = \
                                calculate_h_lcoe_continuous(bat_model,electrolyzer_size_mw,wind_capacity_mw,
                                solar_capacity_mw,battery_storage_mwh,battery_storage_mw,battery_storage_mw,
                                scenario,buy_from_grid=buy_from_grid,sell_to_grid=sell_to_grid)

                # save the results
                results_array = [scenario["Scenario Number"],scenario["Scenario Name"],scenario["Site Name"],scenario['Latitude'],scenario['Longitude'],electrolyzer_size_mw,wind_capacity_mw,
                                solar_capacity_mw,battery_storage_mwh,battery_storage_mw,h_lcoe,pv_wind,hydrogen_annual_output,
                                total_installed_cost,total_operating_cost,capacity_factor,
                                scenario["Force Plant Size"],
                                scenario["Force Electrolyzer Cost"],
                                scenario["Year"],scenario["Power Curve File"],scenario["PTC"],scenario["ITC"],scenario["Debt Equity"],
                                scenario["Turbine Rating"],scenario["Tower Height"],scenario["Rotor Diameter"],scenario["Wind Cost (kW)"],scenario["Solar Cost (kW)"],
                                scenario["Storage Cost (kW)"],scenario["Storage Cost (kWh)"],scenario["Electrolyzer Cost (kW)"]
                                ]

                                                                                                                            
                if os.path.exists(save_filename) == False:
                    header_array = ["Scenario Number","Scenario Name","Site Name","Latitude","Longitude",
                                    "Electrolyzer Size (MW)","Wind Size (MW)","Solar Size (MW)","Storage Size (MWh)",
                                    "Storage Size (MW)","LCOH ($/kg)","Total PV+Wind Generation","Annual H2 Output",
                                    "Installed Cost","Operating Cost","Capacity Factor","Force Plant Size",
                                    "Force Electrolyzer Cost","Year","Power Curve File","PTC","ITC","Debt Equity",
                                    "Turbine Rating","Tower Height","Rotor Diameter","Wind Cost (kW)","Solar Cost (kW)",
                                    "Storage Cost (kW)","Storage Cost (kWh)","Electrolyzer Cost (kW)"]
                    f = open(save_filename, 'w')
                    writer = csv.writer(f)
                    writer.writerow(header_array)
                    f.close()
                
                f = open(save_filename, 'a')
                writer = csv.writer(f)
                writer.writerow(results_array)
                f.close()

    print("run time: ", time.time()-start)