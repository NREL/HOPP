# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:50:54 2022

@author: ereznic2
"""
import os
import pandas as pd
import numpy as np
import time
import subprocess

def run_RODeO(atb_year,site_name,turbine_model,electrolysis_scale,policy_option,policy,i,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
              energy_to_electrolyzer,electrolyzer_energy_kWh_per_kg,hybrid_plant,revised_renewable_cost,electrolyzer_capex_kw,capex_ratio_dist,wind_om_cost_kw,\
              useful_life,time_between_replacement,\
              grid_connection_scenario,grid_price_scenario,gams_locations_rodeo_version,rodeo_output_dir):

     # Renewable generation profile
     system_rating_mw = wind_size_mw + solar_size_mw
     # Renewable output profile needs to be same length as number of time periods in RODeO.
     # Ideally it would be 8760 but if for some reason a couple hours less, this is a simple fix
     while len(energy_to_electrolyzer)<8760:
         energy_to_electrolyzer.append(energy_to_electrolyzer[-1])
         
     electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
     electrical_generation_timeseries[:] = energy_to_electrolyzer[:]
     # Put electrolyzer input into MW
     electrical_generation_timeseries = electrical_generation_timeseries/1000
     # Normalize renewable profile to system rating (RODeO then scales it back up)
     electrical_generation_timeseries = electrical_generation_timeseries/system_rating_mw
     # Get renewable generation profile into a format that works for RODeO
     electrical_generation_timeseries_df = pd.DataFrame(electrical_generation_timeseries).reset_index().rename(columns = {'index':'Interval',0:1})
     electrical_generation_timeseries_df['Interval'] = electrical_generation_timeseries_df['Interval']+1
     electrical_generation_timeseries_df = electrical_generation_timeseries_df.set_index('Interval')
     
     # Fill in renewable profile for RODeO with zeros for years 2-20 (because for some reason it neesd this)
     extra_zeroes = np.zeros_like(energy_to_electrolyzer)
     for j in range(19):
         #j=0
         extra_zeroes_df = pd.DataFrame(extra_zeroes,columns = [j+2]).reset_index().rename(columns = {'index':'Interval',0:j+2})
         extra_zeroes_df['Interval'] = extra_zeroes_df['Interval']+1
         extra_zeroes_df = extra_zeroes_df.set_index('Interval')
         electrical_generation_timeseries_df = electrical_generation_timeseries_df.join(extra_zeroes_df)
         # normalized_demand_df = normalized_demand_df.join(extra_zeroes_df)

     # Write the renewable generation profile to a .csv file in the RODeO repository, assuming RODeO is installed in the same folder as HOPP
     ren_profile_name = 'ren_profile_'+str(atb_year) + '_'+site_name + '_'+ turbine_model
     electrical_generation_timeseries_df.to_csv("examples/H2_Analysis/RODeO_files/Data_files/TXT_files/Ren_profile/" + ren_profile_name + '.csv',sep = ',')
     
     equation_year_CEPCI = 603.1
     model_year_CEPCI = 607.5
     
     # Storage costs as a function of location. Eventuall change this to specify type of storage to RODeO
     if site_name == 'IA':
         h2_storage_cost_USDperkg = 540
         #balancing_area = 'p65'
     elif site_name == 'TX':
         h2_storage_cost_USDperkg = model_year_CEPCI/equation_year_CEPCI*12.30
         #balancing_area ='p124'
     elif site_name == 'IN':
         h2_storage_cost_USDperkg = 540
         #balancing_area = 'p128'
     elif site_name == 'MS':
         h2_storage_cost_USDperkg = model_year_CEPCI/equation_year_CEPCI*12.30
         #balancing_area = 'p9'
         
     # Municipal water rates and wastewater treatment rates combined ($/gal)
     if site_name == 'IA':
         water_cost = 0.00612
         #balancing_area = 'p65'
     elif site_name == 'TX':
         water_cost = 0.00811
         #balancing_area ='p124'
     elif site_name == 'IN':
         water_cost = 0.00634
         #balancing_area = 'p128'
     elif site_name == 'MS':
         water_cost = 0.00844  
         #balancing_area = 'p9'

     
     # Format renewable system cost for RODeO
     #hybrid_installed_cost = hybrid_plant.grid.total_installed_cost
     hybrid_installed_cost = revised_renewable_cost
     hybrid_installed_cost_perMW = hybrid_installed_cost/system_rating_mw  
     
     
     #if electrolysis_scale == 'Centralized':
         # Installed capital cost
     electrolyzer_installation_factor = 12/100  #[%] for stack cost 
    
    # Indirect capital cost as a percentage of installed capital cost
     site_prep = 2/100   #[%]
     engineering_design = 10/100 #[%]
     permitting = 15/100     #[%]         
     project_contingency = 15/100 #[%]
    #[%]
     land_cost = 250000   #[$]
     # elif electrolysis_scale == 'Distributed':
     #     electrolyzer_installation_factor = 12/100/capex_ratio_dist  #[%] for stack cost 
         
     #     # Indirect capital cost as a percentage of installed capital cost
     #     site_prep = 0/100   #[%]
     #     engineering_design = 0/100 #[%]
     #     permitting = 15/100/capex_ratio_dist     #[%]         
     #     project_contingency = 15/100/capex_ratio_dist #[%]
     #     land_cost = 0   #[$]
     

     stack_replacement_cost = 15/100  #[% of installed capital cost]
     fixed_OM = 0.24     #[$/kg H2]    
     
     # Calculate electrolyzer installation cost
     total_direct_electrolyzer_cost_kw = (electrolyzer_capex_kw * (1+electrolyzer_installation_factor)) \
     
     electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000
     
     electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)
     
     #electrolyzer_installation_cost = electrolyzer_system_capex_kw*stack_installation_factor*electrolyzer_size_mw\
     #                               + electrolyzer_indirect_cost                             
     
     compressor_capex_USDprkWe_of_electrolysis = 39
     
     # Calculate capital costs
     electrolyzer_total_capital_cost = electrolyzer_total_installed_capex + electrolyzer_indirect_cost
         
     electrolyzer_system_capex_kw = electrolyzer_total_capital_cost/electrolyzer_size_mw/1000
     
     # O&M costs
     # https://www.sciencedirect.com/science/article/pii/S2542435121003068
     fixed_OM = 12.8 #[$/kWh-y]
     property_tax_insurance = 1.5/100    #[% of Cap/y]
     variable_OM = 1.30  #[$/MWh]
     
          
     # Tell RODeO if grid connected or not
     if grid_connection_scenario == 'off-grid':
         grid_connected_rodeo = False
         grid_string = grid_connection_scenario
         grid_imports = 0
         # Electrolyzer CF estimation for ammortized variable O&M cost calculation
         elec_cf = sum(energy_to_electrolyzer)/(electrolyzer_size_mw*1000*8760)
     elif grid_connection_scenario == 'grid-only':
         grid_connected_rodeo = True
         grid_string = grid_connection_scenario+'-'+grid_price_scenario
         grid_imports = 1
         # Electrolyzer CF estimation for ammortized variable O&M cost calculation
         elec_cf = 0.97
     elif grid_connection_scenario == 'hybrid-grid':
         grid_connected_rodeo = True
         grid_string = grid_connection_scenario+'-'+grid_price_scenario
         grid_imports = 1
         # Electrolyzer CF estimation for ammortized variable O&M cost calculation
         elec_cf = 0.97

     # Amortized refurbishment expense [$/MWh]
     amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacement_cost)\
             *max(((useful_life*8760*elec_cf)/time_between_replacement-1),0)/useful_life/8760/elec_cf*1000

     total_variable_OM = variable_OM+amortized_refurbish_cost
     
     # Define electrolyzer capex, fixed opex, and energy consumption (if not pulling from external data)
     electrolyzer_capex_USD_per_MW = electrolyzer_system_capex_kw*1000
     electrolyzer_fixed_opex_USD_per_MW_year = fixed_OM*1000
     #electrolyzer_energy_kWh_per_kg = 55.5 # Eventually get from input loop
     
     # Define dealination conversion factors
     desal_energy_conversion_factor_kWh_per_m3_water = 4 # kWh per m3-H2O
     m3_water_per_kg_h2 = 0.01 # m3-H2O per kg-H2
     
     # Calculate desalination energy requirement per kg of produced hydrogen
     desal_energy_kWh_per_kg_H2 = m3_water_per_kg_h2*desal_energy_conversion_factor_kWh_per_m3_water
     
     # Desalination system size
     desal_sys_size_m3_hr = electrolyzer_size_mw*1000/electrolyzer_energy_kWh_per_kg*m3_water_per_kg_h2
     desal_sys_size_mw = desal_sys_size_m3_hr*desal_energy_conversion_factor_kWh_per_m3_water/1000
     
     # Total desal capex and opex
     desal_capex_total = 32894 * (997 * desal_sys_size_m3_hr / 3600) # Output in USD
     desal_opex_total = 4841 * (997 * desal_sys_size_m3_hr / 3600) # Output in USD/yr
     
     # Desal capex and opex in $/MW
     desal_capex_per_mw = desal_capex_total/desal_sys_size_mw
     desal_opex_per_mw = desal_opex_total/desal_sys_size_mw
     
     # Calculate desal capex and opex per MW of electrolysis power
     # desal_capex_USD_per_MW_of_electrolysis = 32894*(997/3600*1000/electrolyzer_energy_kWh_per_kg*m3_water_per_kg_h2)
     # desal_opex_USD_per_MW_of_EC_per_year = 4841*(997/3600*1000/electrolyzer_energy_kWh_per_kg*m3_water_per_kg_h2)
     
     # Incorporate desal cost and efficiency into electrolyzer capex, opex, and energy consumption
     #electrolysis_desal_total_capex_per_MW = electrolyzer_capex_USD_per_MW + desal_capex_USD_per_MW_of_electrolysis
     #electrolysis_desal_total_opex_per_MW_per_year = electrolyzer_fixed_opex_USD_per_MW_year + desal_opex_USD_per_MW_of_EC_per_year
     electrolysis_desal_total_energy_consumption = electrolyzer_energy_kWh_per_kg + desal_energy_kWh_per_kg_H2
     
     # Convert electrolysis energy consumption into LHV efficiency
     hydrogen_LHV = 120000 #kJ/kg
     eta_LHV = hydrogen_LHV/3600/electrolysis_desal_total_energy_consumption

     
         
     # Financial parameters
     inflation_rate = 0.0/100
     equity_percentage = 42/100
     bonus_depreciation = 0/100
     
     # Set hydrogen break even price guess value
     # Could in the future replace with H2OPP or H2A estimates 
     lcoh_guessvalue =50
     # Placeholder for if not doing optimization; may want to move this elsewhere or higher level
     h2_storage_duration = 10
     optimize_storage_duration = 1
     
     # Specify grid cost year for ATB year
     if atb_year == 2020:
         grid_year = 2025
     elif atb_year == 2025:
         grid_year = 2030
     elif atb_year == 2030:
         grid_year = 2035
     elif atb_year == 2035:
         grid_year = 2040
         
     # Policy impacts
     policy_scenario = policy[i]
        
     # Set up batch file
     dir0 = "..\\RODeO\\"
     dir1 = 'examples\\H2_Analysis\\RODeO_files\\Data_files\\TXT_files\\'
     dirout = rodeo_output_dir
     
    # txt1 = '"C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt'
     txt1 = gams_locations_rodeo_version[0]
     
     # # Putting this as a conditional just so it will run with existing data, but we should change this to the second one eventually
     # if electrolysis_scale == 'Centralized':
     #     scenario_name = str(atb_year)+'_'+ site_name +'_'+turbine_model+'_'+grid_string
     # elif electrolysis_scale == 'Distributed':
     #     scenario_name = str(atb_year)+'_'+ site_name +'_'+turbine_model+'_'+policy_option.replace(' ','-') + '_'+electrolysis_scale+ '_' + grid_string
         
     scenario_name = str(atb_year)+'_'+ site_name +'_'+turbine_model+'_'+policy_option.replace(' ','-') + '_'+electrolysis_scale+ '_' + grid_string
     
     scenario_inst = ' --file_name_instance='+scenario_name
     #scenario_name = ' --file_name_instance='+Scenario1
     # demand_prof = ' --product_consumed_inst=' + dem_profile_name
     demand_prof = ' --product_consumed_inst=Product_consumption_flat_hourly_ones'
     load_prof = ' --load_prof_instance=Additional_load_none_hourly'
     ren_prof = ' --ren_prof_instance=Ren_profile\\'+ren_profile_name

     
     #energy_price = ' --energy_purchase_price_inst=Elec_prices\\Elec_purch_price_wholesale_MWh_hourly'
     energy_price = ' --energy_purchase_price_inst=Elec_prices\\Elec_purch_price_MWh_MC95by35_'+grid_price_scenario+'_'+site_name+'_'+str(grid_year)

     #energy_price = ' --energy_purchase_price_inst=Elec_prices\\Elec_purch_price_WS_MWh_MC95by35_'+str(balancing_area)+'_'+str(atb_year)
     #energy_price = ' --energy_purchase_price_inst=Netload_'+str(i1)+' --energy_sale_price_inst=Netload_'+str(i1)
     #max_input_entry = ' --Max_input_prof_inst=Max_input_cap_'+str(i1)
     #capacity_values = ' --input_cap_instance='+str(electrolyzer_size_mw)#+str(storage_power_increment)#+' --output_cap_instance='+str(storage_power_increment)
     efficiency = ' --input_efficiency_inst='+str(round(eta_LHV,4))#'0.611'#+str(round(math.sqrt(RTE[i1-1]),6))#+' --output_efficiency_inst='+str(round(math.sqrt(RTE[i1-1]),6))

     wacc_instance = ' --wacc_instance=0.055'                    
     equity_perc_inst = ' --perc_equity_instance=' + str(round(equity_percentage,4))
     ror_inst = ' --ror_instance=0.0489'
     roe_inst = ' --roe_instance=0.104'
     debt_interest_inst = ' --debt_interest_instance=0.0489'
     cftr_inst = ' --cftr_instance=0.27'
     inflation_inst = ' --inflation_inst=' + str(round(inflation_rate,3))
     bonus_dep_frac_inst = ' --bonus_deprec_instance=' + str(round(bonus_depreciation,1))
     
     storage_init_inst = ' --storage_init_instance=0.5'
     storage_final_inst = ' --storage_final_instance=0.5'
     max_storage_dur_inst= ' --max_stor_disch_inst=10000'
     
     storage_cap = ' --storage_cap_instance='+str(h2_storage_duration)#'1000'#+str(stor_dur[i1-1])
     storage_opt = ' --opt_storage_cap ='+str(optimize_storage_duration)
     out_dir = ' --outdir='+dirout
     in_dir = ' --indir='+dir1
     #out_dir = ' --outdir=C:\\Users\\ereznic2\\Documents\\Projects\\SCS_CRADA\\RODeO\\Projects\\SCS\\Output_GSA_test'
     #in_dir = ' --indir=C:\\Users\\ereznic2\\Documents\\Projects\\SCS_CRADA\\RODeO\\Projects\\SCS\\Data_files\\TXT_files'
     product_price_inst = ' --Product_price_instance='+str(lcoh_guessvalue)
     device_ren_inst = ' --devices_ren_instance=1'
     input_cap_inst = ' --input_cap_instance='+str(electrolyzer_size_mw)#1'
     allow_import_inst = ' --allow_import_instance='+str(grid_imports)
     input_LSL_inst = ' --input_LSL_instance=0'
     
     # If grid only, set renewable costs and capacity to zero
     if grid_connection_scenario == 'grid-only':
         ren_cap = ' --Renewable_MW_instance=0'
         ren_capcost = ' --renew_cap_cost_inst=0'
         ren_fom = ' --renew_FOM_cost_inst=0'
     else:
         ren_cap = ' --Renewable_MW_instance='+str(system_rating_mw)#'1'
         ren_capcost = ' --renew_cap_cost_inst='+str(round(hybrid_installed_cost_perMW))#'1230000'
         ren_fom = ' --renew_FOM_cost_inst='+str(1000*round(wind_om_cost_kw))
         
     ren_vom = ' --renew_VOM_cost_inst=0'
     
     input_capcost= ' --input_cap_cost_inst='+str(round(electrolyzer_capex_USD_per_MW))#'1542000'
     prodstor_capcost = ' --ProdStor_cap_cost_inst='+str(round(h2_storage_cost_USDperkg))#'26'
     input_fom = ' --input_FOM_cost_inst='+str(round(electrolyzer_fixed_opex_USD_per_MW_year))#'34926.3'
     input_vom = ' --input_VOM_cost_inst='+str(round(total_variable_OM,2))
     
     water_charge_inst = ' --water_charge_inst='+str(round(water_cost,5))
     desal_cap_cost_inst = ' --desal_cap_cost_inst='+str(round(desal_capex_per_mw))
     desal_FOM_inst = ' --desal_FOM_inst='+str(round(desal_opex_per_mw))
     desal_sys_size_inst = ' --desal_sys_size_inst='+str(round(desal_sys_size_mw,4))
     
     
     ren_itc = ' --itc_ren_inst='+str(round(policy_scenario['Wind ITC'],5))
     stor_itc = ' --itc_stor_inst='+str(round(policy_scenario['Storage ITC'],5))
     
     if atb_year == 2035:
         ren_itc = ' --itc_ren_inst=0'
         stor_itc = ' --itc_stor_inst=0'
     
     # Create batch file
     batch_string = txt1+scenario_inst+demand_prof+ren_prof+load_prof+energy_price+efficiency+storage_cap+storage_opt+ren_cap+out_dir+in_dir\
                  + product_price_inst+device_ren_inst+input_cap_inst+allow_import_inst+input_LSL_inst+ren_capcost+input_capcost+prodstor_capcost\
                  +ren_fom+input_fom+ren_vom+input_vom+water_charge_inst+desal_sys_size_inst+desal_cap_cost_inst+desal_FOM_inst\
                  + wacc_instance+equity_perc_inst+ror_inst+roe_inst+debt_interest_inst+cftr_inst+inflation_inst+bonus_dep_frac_inst\
                  + storage_init_inst+storage_final_inst  +max_storage_dur_inst + ren_itc + stor_itc                           
     
     #   # # For troubleshooting only
     # with open(os.path.join(dir0, 'Output_batch.bat'), 'w') as OPATH:
     #     OPATH.writelines([batch_string,'\n','pause']) # Remove '\n' and 'pause' if not trouble shooting   
     # os.startfile(r'..\\RODeO\\Output_batch.bat')  
       
     #temp = subprocess.run(batch_string,capture_output = True)
     #print(temp)  
     
     #--------------------------- Post processing ---------------------------------
     
     # Get RODeO results summary (high level outputs such as LCOH, capacity factor, cost breakdown, etc.)
     RODeO_results_summary = pd.read_csv(dirout+'Storage_dispatch_summary_'+scenario_name + '.csv',header = 1,sep=',')
     RODeO_results_summary = RODeO_results_summary.rename(columns = {'Elapsed Time (minutes):':'Parameter',RODeO_results_summary.columns[1]:'Value'}).set_index('Parameter')
     # Put results into a dictionary
     RODeO_results_summary_T = RODeO_results_summary.T
     RODeO_results_summary_dict = RODeO_results_summary_T.iloc[0].to_dict()
    
     # Examples for reading out RODeO summary results of interest
     lcoh = RODeO_results_summary_dict['Product NPV cost (US$/kg)']
     electrolyzer_capacity_factor = RODeO_results_summary_dict['input capacity factor']
     electrolyzer_renewable_curtailment_MWh = RODeO_results_summary_dict['Curtailment (MWh)']
     electyrolyzer_renewable_curtailment_percent = 100*RODeO_results_summary_dict['Curtailment (MWh)']/(RODeO_results_summary_dict['Renewable Electricity Input (MWh)']+0.00000001)
     storage_duration_hr = RODeO_results_summary_dict['storage capacity (MWh)']/RODeO_results_summary_dict['input efficiency (%)']/system_rating_mw
     storage_capacity_kg = RODeO_results_summary_dict['storage capacity (MWh)']/electrolysis_desal_total_energy_consumption*1000
    
     # Get RODeO operational results (e.g., electrolyzer and storage hourly operation)
     hydrogen_hourly_inputs_RODeO = pd.read_csv(dirout+'Storage_dispatch_inputs_'+scenario_name + '.csv',index_col = None,header = 29)
     hydrogen_hourly_results_RODeO = pd.read_csv(dirout+'Storage_dispatch_results_'+scenario_name + '.csv',index_col = None,header = 26)
     hydrogen_hourly_results_RODeO['Storage Level (%)'] = 100*hydrogen_hourly_results_RODeO['Storage Level (MW-h)']/(RODeO_results_summary_dict['storage capacity (MWh)'])
     hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]'] = hydrogen_hourly_results_RODeO['Input Power (MW)']*1000/54.55
     hydrogen_hourly_results_RODeO['Water consumption [kg/hr]'] = hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]']*10 #15.5 might be a better number for centralized electrolysis
    
     hydrogen_annual_production = sum(hydrogen_hourly_results_RODeO['Product Sold (units of product)'])
     water_consumption_hourly_array = hydrogen_hourly_results_RODeO['Water consumption [kg/hr]'].to_numpy()
     
     return(scenario_name,lcoh,electrolyzer_capacity_factor,storage_duration_hr,storage_capacity_kg,hydrogen_annual_production,water_consumption_hourly_array,RODeO_results_summary_dict,hydrogen_hourly_results_RODeO,electrical_generation_timeseries,electrolyzer_system_capex_kw,h2_storage_cost_USDperkg)