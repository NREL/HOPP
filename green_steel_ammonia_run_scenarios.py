import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hybrid.sites import SiteInfo
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import time
warnings.filterwarnings("ignore")

import hopp_tools
import hopp_tools_steel
import inputs_py
import copy 
import plot_results
import run_RODeO
import run_pyfast_for_hydrogen
import run_pyfast_for_steel

def batch_generator_kernel(arg_list):

    from hybrid.sites import flatirons_site as sample_site # For some reason we have to pull this inside the definition
    

    
    # Read in arguments
    [policy, i, atb_year, site_location, turbine_name, turbine_model,run_RODeO_selector,grid_connected_rodeo,parent_path,results_dir,rodeo_output_dir,floris_dir,orbit_path] = arg_list
    
    # i = 'option 1'
    # policy = {'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0}}
    # atb_year = 2022
    # site_location = 'Site 1'
    # turbine_model = '18MW'
    # run_RODeO_selector = True
    # grid_connected_rodeo = False
    # # Set paths for results, floris and orbit
    # parent_path = os.path.abspath('')
    # results_dir = parent_path + '/examples/H2_Analysis/results/'
    # floris_dir = parent_path + '/floris_input_files/'
    # orbit_path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
    # rodeo_output_dir = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\'

    """
    Perform a LCOH analysis for an offshore wind + Hydrogen PEM system
    
    Missing Functionality:
    1. Figure out H2A Costs or ammortize cost/kw electrolyzer figure and add opex
    
    ~1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~
    
    2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
    3. Cost Scaling Based on Plant Size (Shields et. Al report)
    4. Integration Required:
    * Pressure Vessel Model~
    * HVDC Model 
    * Pipeline Model
    
    5. Model Development Required:
    - Floating Electrolyzer Platform
    """
    
    #Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env
    
    #Step 1: User Inputs for scenario
    resource_year = 2013
    
    sample_site['year'] = resource_year
    useful_life = 30
    critical_load_factor = 1
    run_reopt_flag = False
    custom_powercurve = True    #A flag that is applicable when using PySam WindPower (not FLORIS)
    storage_used = False
    battery_can_grid_charge = False
    grid_connected_hopp = False
    # grid_connected_rodeo = False
    # run_RODeO_selector = False
    # Technology sizing
    interconnection_size_mw = 1000
    electrolyzer_size_mw = 1000
    wind_size_mw = 1000
    solar_size_mw = 0
    storage_size_mw = 0
    storage_size_mwh = 0
    
    scenario_choice = 'Offshore Wind-H2 Analysis'
    
    scenario = dict()
    kw_continuous = electrolyzer_size_mw * 1000
    load = [kw_continuous for x in
            range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    
    #Site lat and lon will be set by data loaded from Orbit runs
    
    # Financial inputs
    discount_rate = 0.07
    debt_equity_split = 60
    
    # Wind costs input from ORBIT analysis
    h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
    # h2_model = 'H2A'
    
    # These inputs are not used in this analysis (no solar or storage)
    solar_cost_kw = 9999
    storage_cost_kw = 250
    storage_cost_kwh = 240
    
    # Flags (TODO: remove them and update documentation)
    forced_sizes = True
    force_electrolyzer_cost = False
    
    
    # Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
    # sell_price = 0.01
    # buy_price = 0.01
    sell_price = False
    buy_price = False
        
    print('Parent path = ', parent_path)
    
    # ORBIT financial information
    #orbit_path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
    xl = pd.ExcelFile(orbit_path)
    
    save_outputs_dict = inputs_py.establish_save_output_dict()
    save_all_runs = list()
    
    rodeo_scenarios = list()
    h2a_scenarios = list()
    
    # which plots to show
    plot_power_production = True
    plot_battery = True
    plot_grid = True
    plot_h2 = True
    plot_desal = True
    plot_wind = True
    plot_hvdcpipe = True
    plot_hvdcpipe_lcoh = True
    
    # Read in gams exe and license location
    # Create a .txt file in notepad with the locations of the gams .exe file, the .gms RODeO
    # version that you want to use, and the location of the gams license file. The text
    # should look something like this: 
    # "C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt
    # Do not push this file to the remote repository because it will be different for every user
    # and for every machine, depending on what version of gams they are using and where it is installed
    with open('gams_exe_license_locations.txt') as f:
        gams_locations_rodeo_version = f.readlines()
    f.close()
    
    # Setup policy scenario
    scenario, policy_option = hopp_tools.set_policy_values(scenario, policy, i)
    print(scenario['Wind PTC'])
    

    site_number = site_location.split(' ')[1]
    
    # set turbine values
    scenario, nTurbs, floris_config = hopp_tools.set_turbine_model(turbine_model, scenario, parent_path,floris_dir)

    scenario['Useful Life'] = useful_life

    # financials
    scenario = hopp_tools.set_financial_info(scenario, debt_equity_split, discount_rate)

    # set electrolyzer information
    electrolyzer_capex_kw, time_between_replacement =  hopp_tools.set_electrolyzer_info(atb_year)

    # Extract Scenario Information from ORBIT Runs
    # Load Excel file of scenarios
    # OSW sites and cost file including turbines 8/16/2022 

    # site info
    path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
    xl = pd.ExcelFile(path)
    site_df, sample_site = hopp_tools.set_site_info(xl, turbine_model, site_location, sample_site)
    site_name = site_df['Representative region']
    fixed_or_floating_wind = site_df['Substructure technology']
    site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])

    #Assign Orbit results to scenario cost details
    total_capex = site_df['Total CapEx']
    wind_cost_kw = copy.deepcopy(total_capex)
    wind_om_cost_kw = site_df['OpEx, $/kW-yr']
    wind_net_cf = site_df['Assumed NCF']

    # set export financials
    wind_cost_kw, wind_om_cost_kw, total_export_system_cost, total_export_om_cost = hopp_tools.set_export_financials(wind_size_mw, 
                                                                                                                    wind_cost_kw,
                                                                                                                    wind_om_cost_kw,
                                                                                                                    useful_life,
                                                                                                                    site_df)
    # set wind financials
    new_wind_cost_kw, new_wind_om_cost_kw, new_wind_net_cf = hopp_tools.set_turbine_financials(turbine_model, 
                                                                                                fixed_or_floating_wind,
                                                                                                atb_year,
                                                                                                wind_cost_kw,
                                                                                                wind_om_cost_kw,
                                                                                                wind_net_cf,
                                                                                                parent_path)
    #Plot Wind Data to ensure offshore data is sound
    wind_data = site.wind_resource._data['data']
    wind_speed = [W[2] for W in wind_data]
    plot_results.plot_wind_results(wind_data, site_name, site_df['Representative coordinates'], results_dir, plot_wind)

    #Plot Wind Cost Contributions
    # Plot a nested pie chart of results
    # TODO: Remove export system from pieplot
    plot_results.plot_pie(site_df, site_name, turbine_model, results_dir)
    
    # Run HOPP
    floris = False
    combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe = \
        hopp_tools.run_HOPP(scenario,
                            site,
                            sample_site,
                            forced_sizes,
                            solar_size_mw,
                            wind_size_mw,
                            storage_size_mw,
                            storage_size_mwh,
                            new_wind_cost_kw, 
                            solar_cost_kw, 
                            storage_cost_kw, 
                            storage_cost_kwh,
                            kw_continuous, 
                            load,
                            electrolyzer_size_mw,
                            wind_om_cost_kw,
                            nTurbs,
                            floris_config,
                            floris)

    #Step 4: Plot HOPP Results
    plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
                            energy_shortfall_hopp,
                            combined_pv_wind_curtailment_hopp,
                            load,
                            results_dir,
                            site_name,
                            atb_year,
                            turbine_model,
                            hybrid_plant,
                            plot_power_production)

    #Step 5: Run Simple Dispatch Model
    combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used, excess_energy = \
        hopp_tools.run_battery(energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, combined_pv_wind_power_production_hopp)
    
    plot_results.plot_battery_results(combined_pv_wind_curtailment_hopp, 
             energy_shortfall_hopp,
             combined_pv_wind_storage_power_production_hopp,
             combined_pv_wind_power_production_hopp,
             battery_SOC,
             battery_used,
             results_dir,
             site_name,atb_year,turbine_model,
             load,
             plot_battery)

    

    # grid information
    cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer = hopp_tools.grid(combined_pv_wind_storage_power_production_hopp,
                                                                         sell_price,
                                                                         excess_energy,
                                                                         buy_price,
                                                                         kw_continuous,
                                                                         plot_grid)
    
    
    # Step 6: Run RODeO/H2A
    
    if run_RODeO_selector == True:
        rodeo_scenario,lcoh,electrolyzer_capacity_factor,storage_duration_hr,hydrogen_annual_production,water_consumption_hourly,RODeO_summary_results_dict,hydrogen_hourly_results_RODeO,electrical_generation_timeseries\
            = run_RODeO.run_RODeO(atb_year,site_location,turbine_model,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
                      energy_to_electrolyzer,hybrid_plant,electrolyzer_capex_kw,useful_life,time_between_replacement,\
                      grid_connected_rodeo,gams_locations_rodeo_version,rodeo_output_dir)
            
    else:
    # If not running RODeO, run H2A via PyFAST
        # Currently only works for offgrid
        grid_string = 'offgrid'    
        scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','-') +'_'+turbine_model+'_'+grid_string
    
        # Specify a few things PyFAST will need. These could also be specified elsewhere or more generally
        electrolyzer_energy_kWh_per_kg = 55.5
        # Convert electrolysis energy consumption into LHV efficiency
        hydrogen_LHV = 120000 #kJ/kg
        eta_LHV = hydrogen_LHV/3600/electrolyzer_energy_kWh_per_kg
        # Initially just assign CF. Eventually we should actually run the H2OPP PEM electrolyzer module
        electrolyzer_capacity_factor = sum(energy_to_electrolyzer)/1000/(electrolyzer_size_mw*8760)
        # Assign storage capacity; eventually, need to develop a module to calculate this based on renewable profile
        storage_capacity_MWh = 223.0440086*electrolyzer_size_mw
        storage_duration_hr = storage_capacity_MWh/eta_LHV/electrolyzer_size_mw
        water_consumption_galperkg = 3.78
        water_cost = 0.01
        
        electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
        electrical_generation_timeseries[:] = energy_to_electrolyzer[:]
        # Put electrolyzer input into MW
        electrical_generation_timeseries = electrical_generation_timeseries/1000
        
        h2a_solution,h2a_summary = run_pyfast_for_hydrogen.run_pyfast_for_hydrogen(site_location,electrolyzer_size_mw,electrolyzer_energy_kWh_per_kg,electrolyzer_capacity_factor,\
                                    electrolyzer_capex_kw,storage_capacity_MWh,useful_life,water_consumption_galperkg,water_cost,lcoe/100)
        
        lcoh = h2a_solution.get('price')
        h2a_scenarios.append([scenario_name,lcoh])
        
    # Step 7: Calculate break-even cost of steel and ammonia production
    lime_unitcost = 0.01812
    carbon_unitcost = 0.0538
    iron_ore_pellet_unitcost = 1.62927
    steel_economics_from_pyfast, steel_economics_summary, steel_breakeven_price, steel_annual_production_mtpy = hopp_tools_steel.steel_LCOS(lcoh,hydrogen_annual_production,lime_unitcost,
    carbon_unitcost,
    iron_ore_pellet_unitcost)
                
    # Step 7: Write outputs to file
    
    if run_RODeO_selector == True:             
        policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
        scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
        discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
        wind_itc_total, total_itc_hvdc = hopp_tools.write_outputs_RODeO(electrical_generation_timeseries,\
                             hybrid_plant,
                             total_export_system_cost,
                             total_export_om_cost,
                             cost_to_buy_from_grid,
                             electrolyzer_capex_kw, 
                             time_between_replacement,
                             profit_from_selling_to_grid,
                             useful_life,
                             atb_year,
                             policy_option,
                             scenario,
                             wind_cost_kw,
                             solar_cost_kw,
                             discount_rate,
                             solar_size_mw,
                             results_dir,
                             site_name,
                             turbine_model,
                             scenario_choice,
                             lcoe,
                             run_RODeO_selector,
                             lcoh,
                             electrolyzer_capacity_factor,
                             storage_duration_hr,
                             hydrogen_annual_production,
                             water_consumption_hourly,
                             RODeO_summary_results_dict,
                             steel_breakeven_price) 
        

                
 
                    

                


                

        #         #Step 6: Run the H2_PEM model
        #         h2_model = 'Simple'
        #         H2_Results, H2A_Results, electrical_generation_timeseries = hopp_tools.run_H2_PEM_sim(hybrid_plant,
        #                                                                                                 energy_to_electrolyzer,
        #                                                                                                 scenario,
        #                                                                                                 wind_size_mw,
        #                                                                                                 solar_size_mw,
        #                                                                                                 electrolyzer_size_mw,
        #                                                                                                 kw_continuous,
        #                                                                                                 electrolyzer_capex_kw,
        #                                                                                                 lcoe)

        #         plot_results.plot_h2_results(H2_Results, 
        #                                     electrical_generation_timeseries,
        #                                     results_dir,
        #                                     site_name,atb_year,turbine_model,
        #                                     load,
        #                                     plot_h2)

        #         #Step 6b: Run desal model
        #         desal_capex, desal_opex, desal_annuals = hopp_tools.desal_model(H2_Results, 
        #                                                         electrolyzer_size_mw, 
        #                                                         electrical_generation_timeseries, 
        #                                                         useful_life)

        #         # compressor model
        #         compressor, compressor_results = hopp_tools.compressor_model()

        #         #Pressure Vessel Model Example
        #         storage_input, storage_output = hopp_tools.pressure_vessel()

        #         # pipeline model
        #         total_h2export_system_cost, opex_pipeline, dist_to_port_value = hopp_tools.pipeline(site_df, 
        #                                                                         H2_Results, 
        #                                                                         useful_life, 
        #                                                                         storage_input)
                
                
        #         # plot HVDC vs pipe 
        #         plot_results.plot_hvdcpipe(total_export_system_cost,
        #                                     total_h2export_system_cost,
        #                                     site_name,
        #                                     atb_year,
        #                                     dist_to_port_value,
        #                                     results_dir)

                

        #         # Step 6.5: Intermediate financial calculation

        #         LCOH_cf_method_wind, LCOH_cf_method_pipeline, LCOH_cf_method_hvdc, LCOH_cf_method_solar,\
        # LCOH_cf_method_h2_costs, LCOH_cf_method_desal_costs, LCOH_cf_method_total_hvdc, LCOH_cf_method_total_pipeline, \
        # total_elec_production, lifetime_h2_production, gut_check_h2_cost_kg_pipeline, gut_check_h2_cost_kg_hvdc, \
        # wind_itc_total, total_itc_pipeline, total_itc_hvdc, total_annual_operating_costs_hvdc, total_annual_operating_costs_pipeline, \
        # h_lcoe_hvdc, h_lcoe_pipeline, tlcc_wind_costs, tlcc_solar_costs, tlcc_h2_costs, tlcc_desal_costs, tlcc_pipeline_costs,\
        # tlcc_hvdc_costs, tlcc_total_costs, tlcc_total_costs_pipeline, \
        #     electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc = \
        #             hopp_tools.calculate_financials(electrical_generation_timeseries,
        #                  hybrid_plant,
        #                  H2A_Results,
        #                  H2_Results,
        #                  desal_opex,
        #                  desal_annuals,
        #                  total_h2export_system_cost,
        #                  opex_pipeline,
        #                  total_export_system_cost,
        #                  total_export_om_cost,
        #                  cost_to_buy_from_grid,
        #                  profit_from_selling_to_grid,
        #                  useful_life,
        #                  atb_year,
        #                  policy_option,
        #                  scenario,
        #                  h2_model,
        #                  desal_capex,
        #                  wind_cost_kw,
        #                  solar_cost_kw,
        #                  discount_rate,
        #                  solar_size_mw,
        #                  electrolyzer_size_mw,
        #                  results_dir,
        #                  site_name,
        #                  turbine_model,
        #                  scenario_choice)

        #         # Step 7: Plot Results
                
        #         # plot bars in stack manner
        #         if plot_hvdcpipe_lcoh:
        #             plt.figure(figsize=(9,6))
        #             barx = ['HVDC', 'Pipeline']
        #             plt.bar(barx, [LCOH_cf_method_wind,LCOH_cf_method_wind], color='blue')
        #             plt.bar(barx, [LCOH_cf_method_h2_costs,LCOH_cf_method_h2_costs], bottom=[LCOH_cf_method_wind,LCOH_cf_method_wind], color='orange')
        #             plt.bar(barx, [LCOH_cf_method_desal_costs,LCOH_cf_method_desal_costs], bottom =[(LCOH_cf_method_wind + LCOH_cf_method_h2_costs), (LCOH_cf_method_wind + LCOH_cf_method_h2_costs)], color='g')
        #             plt.bar(barx, [LCOH_cf_method_hvdc,LCOH_cf_method_pipeline], bottom =[(LCOH_cf_method_wind + LCOH_cf_method_h2_costs + LCOH_cf_method_desal_costs), (LCOH_cf_method_wind + LCOH_cf_method_h2_costs+LCOH_cf_method_desal_costs)], color='black')
                    
        #             plt.ylabel("LCOH")
        #             plt.legend(["Wind", "Electrolyzer", "Desalination","Export System"])
        #             plt.title("Levelized Cost of hydrogen - Cost Contributors\n {}\n {}\n {} \n{}".format(site_name,atb_year,turbine_model,policy_option))
        #             plt.savefig(os.path.join(results_dir,'LCOH Barchart_{}_{}_{}_{}.jpg'.format(site_name,atb_year,turbine_model,policy_option)),bbox_inches='tight')
        #             # plt.show()

        #         print_results = False
        #         print_h2_results = True
        #         # save_outputs_dict = inputs_py.save_the_things()
        #         # save_all_runs.append(save_outputs_dict)
        #         # save_outputs_dict = inputs_py.establish_save_output_dict()

        #         if print_results:
        #             # ------------------------- #
        #             #TODO: Tidy up these print statements
        #             print("Future Scenario: {}".format(scenario['Scenario Name']))
        #             print("Wind Cost per KW: {}".format(scenario['Wind Cost KW']))
        #             print("PV Cost per KW: {}".format(scenario['Solar Cost KW']))
        #             print("Storage Cost per KW: {}".format(scenario['Storage Cost kW']))
        #             print("Storage Cost per KWh: {}".format(scenario['Storage Cost kWh']))
        #             print("Wind Size built: {}".format(wind_size_mw))
        #             print("PV Size built: {}".format(solar_size_mw))
        #             print("Storage Size built: {}".format(storage_size_mw))
        #             print("Storage Size built: {}".format(storage_size_mwh))
        #             print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
        #             print("Total Yearly Electrical Output: {}".format(total_elec_production))
        #             print("Total Yearly Hydrogen Production: {}".format(H2_Results['hydrogen_annual_output']))
        #             #print("Levelized Cost H2/kg (new method - no operational costs)".format(h_lcoe_no_op_cost))
        #             print("Capacity Factor of Electrolyzer: {}".format(H2_Results['cap_factor']))

        #         if print_h2_results:
        #             print('Total Lifetime H2(kg) produced: {}'.format(lifetime_h2_production))
        #             print("Gut-check H2 cost/kg: {}".format(gut_check_h2_cost_kg_pipeline))
        #         #     print("h_lcoe: ", h_lcoe)
        #            # print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
        #             #     H2_Results['feedstock_cost_h2_levelized_hopp']))
        #             # print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(H2A_Results['Total Hydrogen Cost ($/kgH2)']))
        #             # print("Total unit cost of H2 ($/kg) : {}".format(H2_Results['total_unit_cost_of_hydrogen']))
        #             # print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(
        #             #     H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp']))

        #             #Step 9: Summarize Results
        #             print('For a {}MW Offshore Wind Plant of turbine size {} with {}MW onshore electrolyzer \n located at {} \n (average wind speed {}m/s) in {} \n with a Wind CAPEX cost of {}$/kW,  and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} $/kg '.
        #                         format(wind_size_mw,turbine_model,electrolyzer_size_mw,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_capex_kw,LCOH_cf_method_total_hvdc))
        #             print('For a {}MW Offshore Wind Plant of turbine size {} with {}MW offshore electrolyzer \n located at {} \n (average wind speed {}m/s) in {} \n with a Wind CAPEX cost of {}$/kW,  and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} $/kg '.
        #                         format(wind_size_mw,turbine_model,electrolyzer_size_mw,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_capex_kw,LCOH_cf_method_total_pipeline))

# save_outputs = True
# if save_outputs:
#     #save_outputs_dict_df = pd.DataFrame(save_all_runs)
#     save_all_runs_df = pd.DataFrame(save_all_runs)
#     save_all_runs_df.to_csv(os.path.join(results_dir, "H2_Analysis_OSW_All.csv"))


# print('Done')

