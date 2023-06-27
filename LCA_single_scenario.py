# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:09:20 2022

@author: ereznic2
"""
import pandas as pd

dircambium = 'examples/H2_Analysis/Cambium_data/StdScen21_MidCase95by2035_hourly_' 

# grid_connection_scenario = 'hybrid-grid'
# atb_year = 2020
# site_name = 'TX'
# turbine_model = '6MW'
# electrolysis_scale = 'Centralized'
# policy_option = 'no policy'
# grid_price_scenario = 'retail-flat'
# electrolyzer_energy_kWh_per_kg = 55

def hydrogen_LCA_singlescenario(grid_connection_scenario,atb_year,site_name,turbine_model,electrolysis_scale,policy_option,grid_price_scenario,electrolyzer_energy_kWh_per_kg,hydrogen_hourly_results_RODeO):

    #==============================================================================
    # DATA
    #==============================================================================        
    # Conversions
    g_to_kg_conv  = 0.001  # Conversion from grams to kilograms
    kg_to_MT_conv = 0.001 # Converion from kg to metric tonnes
    MT_to_kg_conv = 1000 # Conversion from metric tonne to kilogram
    kWh_to_MWh_conv = 0.001 # Conversion from kWh to MWh
    
    #------------------------------------------------------------------------------
    # Renewable infrastructure embedded emission intensities
    #------------------------------------------------------------------------------
    system_life        = 30
    ely_stack_capex_EI = 0.019 # PEM electrolyzer CAPEX emissions (kg CO2e/kg H2)
    wind_capex_EI      = 10    # Electricity generation from wind, nominal value taken (g CO2e/kWh)
    solar_pv_capex_EI  = 37    # Electricity generation from solar pv, nominal value taken (g CO2e/kWh)

    
    #------------------------------------------------------------------------------
    # Hydrogen production via water electrolysis
    #------------------------------------------------------------------------------
    
    grid_trans_losses   = 0.05 # Grid losses of 5% are assumed (-)
    fuel_to_grid_curr   = 48   # Fuel mix emission intensity for current power grid (g CO2e/kWh)
    fuel_to_grid_futu   = 14   # Fuel mix emission intensity for future power grid (g CO2e/kWh)
    
    if atb_year == 2020:
        cambium_year = 2025
    elif atb_year == 2025:
        cambium_year = 2030
    elif atb_year == 2030:
        cambium_year =2035
    elif atb_year == 2035:
        cambium_year = 2040
        
    # Read in Cambium data  
    cambiumdata_filepath = dircambium + site_name + '_'+str(cambium_year) + '.csv'
    cambium_data = pd.read_csv(cambiumdata_filepath,index_col = None,header = 4,usecols = ['lrmer_co2_c','lrmer_ch4_c','lrmer_n2o_c','lrmer_co2_p','lrmer_ch4_p','lrmer_n2o_p','lrmer_co2e_c','lrmer_co2e_p','lrmer_co2e'])
    
    cambium_data = cambium_data.reset_index().rename(columns = {'index':'Interval','lrmer_co2_c':'LRMER CO2 combustion (kg-CO2/MWh)','lrmer_ch4_c':'LRMER CH4 combustion (g-CH4/MWh)','lrmer_n2o_c':'LRMER N2O combustion (g-N2O/MWh)',\
                                                  'lrmer_co2_p':'LRMER CO2 production (kg-CO2/MWh)','lrmer_ch4_p':'LRMER CH4 production (g-CH4/MWh)','lrmer_n2o_p':'LRMER N2O production (g-N2O/MWh)','lrmer_co2e_c':'LRMER CO2 equiv. combustion (kg-CO2e/MWh)',\
                                                  'lrmer_co2e_p':'LRMER CO2 equiv. production (kg-CO2e/MWh)','lrmer_co2e':'LRMER CO2 equiv. total (kg-CO2e/MWh)'})
    
    cambium_data['Interval']=cambium_data['Interval']+1
    cambium_data = cambium_data.set_index('Interval')   
    
    # Read in rodeo data
    rodeo_data = hydrogen_hourly_results_RODeO[['Interval','Input Power (MW)','Non-Ren Import (MW)','Renewable Input (MW)','Curtailment (MW)','Product Sold (units of product)']]
    rodeo_data = rodeo_data.rename(columns = {'Input Power (MW)':'Electrolyzer Power (MW)','Non-Ren Import (MW)':'Grid Import (MW)','Renewable Input (MW)':'Renewable Input (MW)', 'Curtailment (MW)':'Curtailment (MW)','Product Sold (units of product)':'Hydrogen production (kg-H2)'})
    # Combine RODeO and Cambium data into one dataframe
    combined_data = rodeo_data.merge(cambium_data, on = 'Interval',how = 'outer')   
    
    # Calculate hourly grid emissions factors of interest. If we want to use different GWPs, we can do that here. The Grid Import is an hourly data i.e., in MWh
    combined_data['Total grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. total (kg-CO2e/MWh)']
    combined_data['Scope 2 (combustion) grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)']
    combined_data['Scope 3 (production) grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)']
    
    # Sum total emissions
    scope2_grid_emissions_sum = combined_data['Scope 2 (combustion) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    scope3_grid_emissions_sum = combined_data['Scope 3 (production) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    h2prod_sum = combined_data['Hydrogen production (kg-H2)'].sum()*system_life*kg_to_MT_conv
    h2prod_grid_frac = combined_data['Grid Import (MW)'].sum() / combined_data['Electrolyzer Power (MW)'].sum()
           
    if grid_connection_scenario == 'hybrid-grid' :
        # Calculate grid-connected electrolysis emissions/ future cases should reflect targeted electrolyzer electricity usage
        electrolysis_Scope3_EI =  h2prod_grid_frac*scope3_grid_emissions_sum/h2prod_sum + wind_capex_EI * electrolyzer_energy_kWh_per_kg * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI =  h2prod_grid_frac*scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI  = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI 
        electrolysis_total_EI_policy_grid = electrolysis_total_EI - wind_capex_EI * electrolyzer_energy_kWh_per_kg * g_to_kg_conv - ely_stack_capex_EI
        electrolysis_total_EI_policy_offgrid = 0
    elif grid_connection_scenario == 'grid-only':
        # Calculate grid-connected electrolysis emissions
        electrolysis_Scope3_EI = scope3_grid_emissions_sum/h2prod_sum  + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        electrolysis_total_EI_policy_grid = electrolysis_total_EI - ely_stack_capex_EI
        electrolysis_total_EI_policy_offgrid = 0
    elif grid_connection_scenario == 'off-grid':    
        # Calculate renewable only electrolysis emissions        
        electrolysis_Scope3_EI = wind_capex_EI * electrolyzer_energy_kWh_per_kg * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = 0
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        electrolysis_total_EI_policy_offgrid = electrolysis_total_EI - wind_capex_EI * electrolyzer_energy_kWh_per_kg * g_to_kg_conv - ely_stack_capex_EI
        electrolysis_total_EI_policy_grid = 0
    
    return(electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid,h2prod_grid_frac)    


