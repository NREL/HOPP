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

    # Read in arguments
    [policy, i, atb_year, site_location, turbine_name, turbine_model,electrolysis_scale,run_RODeO_selector,grid_connected_rodeo,parent_path,results_dir,rodeo_output_dir,floris_dir,orbit_path] = arg_list
    
    
    from hybrid.sites import flatirons_site as sample_site # For some reason we have to pull this inside the definition
    i = 'option 1'
    policy = {'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0}}
    atb_year = 2022
    site_location = 'Site 1'
    turbine_model = '18MW'
    electrolysis_scale = 'Centralized'
    run_RODeO_selector = True
    grid_connected_rodeo = False
    # Set paths for results, floris and orbit
    parent_path = os.path.abspath('')
    results_dir = parent_path + '/examples/H2_Analysis/results/'
    floris_dir = parent_path + '/floris_input_files/'
    orbit_path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
    rodeo_output_dir = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\'

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
    electrolyzer_capex_kw, time_between_replacement =  hopp_tools.set_electrolyzer_info(atb_year,electrolysis_scale)

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
    
    
    # Step 6: Run RODeO or Pyfast for hydrogen
    
    if run_RODeO_selector == True:
        rodeo_scenario,lcoh,electrolyzer_capacity_factor,hydrogen_storage_duration_hr,hydrogen_annual_production,water_consumption_hourly,RODeO_summary_results_dict,hydrogen_hourly_results_RODeO,electrical_generation_timeseries\
            = run_RODeO.run_RODeO(atb_year,site_location,turbine_model,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
                      energy_to_electrolyzer,hybrid_plant,electrolyzer_capex_kw,useful_life,time_between_replacement,\
                      grid_connected_rodeo,gams_locations_rodeo_version,rodeo_output_dir)
            
    else:
    # If not running RODeO, run H2A via PyFAST
        # Currently only works for offgrid
        grid_string = 'offgrid'    
        scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','-') +'_'+turbine_model+'_'+grid_string
        
        #Run the H2_PEM model to get hourly hydrogen output, capacity factor, water consumption, etc.
        h2_model = 'Simple'
        H2_Results, H2A_Results, electrical_generation_timeseries = hopp_tools.run_H2_PEM_sim(hybrid_plant,
                                                                                                energy_to_electrolyzer,
                                                                                                scenario,
                                                                                                wind_size_mw,
                                                                                                solar_size_mw,
                                                                                                electrolyzer_size_mw,
                                                                                                kw_continuous,
                                                                                                electrolyzer_capex_kw,
                                                                                                lcoe)
        
        #Step 6b: Run desal model
        desal_capex, desal_opex, desal_annuals = hopp_tools.desal_model(H2_Results, 
                                                        electrolyzer_size_mw, 
                                                        electrical_generation_timeseries, 
                                                        useful_life)
        
        hydrogen_annual_production = H2_Results['hydrogen_annual_output']
        
        # Calculate required storage capacity to meet a flat demand profile. In the future, we could customize this to
        # work with any demand profile
        storage_type = 'Salt cavern'
        hydrogen_production_storage_system_output_kgprhr,hydrogen_storage_capacity_kg,hydrogen_storage_capacity_MWh_HHV,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg,storage_status_message\
            = hopp_tools.hydrogen_storage_capacity_cost_calcs(H2_Results,electrolyzer_size_mw,storage_type)   
        print(storage_status_message)
        
        # Run PyFAST to get LCOH
        water_cost = 0.01
    
        h2a_solution,h2a_summary,lcoh_breakdown = run_pyfast_for_hydrogen. run_pyfast_for_hydrogen(site_location,electrolyzer_size_mw,H2_Results,\
                                        electrolyzer_capex_kw,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                                        desal_capex,desal_opex,useful_life,water_cost,lcoe)
        
        lcoh = h2a_solution['price']

        

        
        
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
                             electrolysis_scale,
                             scenario_choice,
                             lcoe,
                             run_RODeO_selector,
                             lcoh,
                             electrolyzer_capacity_factor,
                             hydrogen_storage_duration_hr,
                             hydrogen_annual_production,
                             water_consumption_hourly,
                             RODeO_summary_results_dict,
                             steel_breakeven_price) 
    else:
        policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
        scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
        discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
        wind_itc_total, total_itc_hvdc = hopp_tools.write_outputs_PyFAST(electrical_generation_timeseries,\
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
                             electrolysis_scale,
                             scenario_choice,
                             lcoe,
                             run_RODeO_selector,
                             lcoh,
                             H2_Results,
                             hydrogen_storage_duration_hr,
                             lcoh_breakdown,
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

                



