# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:44:27 2022

@author: ereznic2
"""

import fnmatch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sqlite3

import glob
import csv
import sys
import heapq


# Directory from which to pull outputs from
parent_path = os.path.abspath('')
dir0 = 'Examples/H2_Analysis/RODeO_files/Output_test/' 
dirfinancial = 'Examples/H2_Analysis/financial_summary_results/'
dircambium = 'Examples/H2_Analysis/Cambium_data/StdScen21_MidCase95by2035_hourly_' 
dir_plot = 'Examples/H2_Analysis/RODeO_files/Plots/LCA_Plots/'

# Specify grid price scenario if interest for down-selection in case multiple grid scenarios
# exist within the output folder
# grid_connection_cases ['off-grid','grid-only','hybrid-grid']
# Grid price scenario ['wholesale','retail-peaks','retail-flat']
grid_price_scenario = 'retail-peaks'

c0 = [0,0,0]

files2load_results={}
files2load_results_title={}
files2load_results_categories={}


for files2load in os.listdir(dir0):
    if fnmatch.fnmatch(files2load, 'Storage_dispatch_results*'):
        c0[0]=c0[0]+1
        files2load_results[c0[0]] = files2load
        int1 = files2load.split("_")
        int1 = int1[3:]
        int1[-1] = int1[-1].replace('.csv', '')
        files2load_results_title[c0[0]] = int1
    files2load_title_header = ['Year','Site','Turbine Size','Policy Option','Electrolysis case','Grid Case']
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
# Steam methane reforming (SMR) - Incumbent H2 production process
#------------------------------------------------------------------------------

smr_NG_combust = 56.2 # Natural gas combustion (g CO2e/MJ)
smr_NG_consume = 167  # Natural gas consumption (MJ/kg H2)
smr_PO_consume = 0.13    # Power consumption in SMR plant (kWh/kg H2)
smr_ccs_PO_consume = 1.5 # Power consumption in SMR CCS plant (kWh/kg H2)
smr_steam_prod = 17.4 # Steam production on SMR site (MJ/kg H2)
smr_HEX_eff    = 0.9  # Heat exchanger efficiency (-)
smr_NG_supply  = 9    # Natural gas extraction and supply to SMR plant assuming 2% CH4 leakage rate (g CO2e/MJ)
ccs_PO_consume = 0   # Power consumption for CCS (kWh/kg CO2)
ccs_perc_capture = 0.95 # Carbon capture rate (-)

#------------------------------------------------------------------------------
# Hydrogen production via water electrolysis
#------------------------------------------------------------------------------

grid_trans_losses   = 0.05 # Grid losses of 5% are assumed (-)
fuel_to_grid_curr   = 48   # Fuel mix emission intensity for current power grid (g CO2e/kWh)
fuel_to_grid_futu   = 14   # Fuel mix emission intensity for future power grid (g CO2e/kWh)

#------------------------------------------------------------------------------
# Ammonia
#------------------------------------------------------------------------------

NH3_PO_consume = 0.53      # Electricity usage (kWh/kg NH3)
NH3_H2_consume = 0.2       # Hydrogen consumption (kg H2/kg NH3)
NH3_boiler_EI  = 0.5       # Boiler combustion of methane (kg CO2e/kg NH3)

#------------------------------------------------------------------------------
# Steel
#------------------------------------------------------------------------------

steel_H2_consume = 0.06596 # metric tonnes of H2 per tonne of steel
steel_NG_consume = 0.71657 # GJ-LHV per tonne of steel
steel_lime_consume = 0.01812 # metric tonne of lime per tonne of steel
steel_iron_ore_consume = 1.629 # metric tonnes of iron ore per metric tonne of steel
steel_PO_consume = 0.5502 # MWh per metric tonne of steel
steel_H2O_consume = 0.8037 # metric tonnes of H2O per tonne of steel
steel_CH4_prod = 39.29	# kg of CO2e emission/metric tonne of annual steel slab production 
steel_CO2_prod = 174.66	# kg of CO2 emission/metric tonne of annual steel slab production 

steel_NG_supply_EI  = 9    # Natural gas extraction and supply to plant assuming 2% CH4 leakage rate (g CO2e/MJ)
steel_lime_EI = 1.28   # kg CO2e/kg lime
steel_iron_ore_EI = 0.048 # kg CO2e/kg iron ore
steel_H2O_EI = 0.00013 # kg CO2e/gal H2O
gal_to_ton_conv = 0.001336 # for water conversions

smr_Scope3_EI = 'NA'
smr_Scope2_EI = 'NA'
smr_Scope1_EI = 'NA'
smr_total_EI  = 'NA'
smr_ccs_Scope3_EI = 'NA'
smr_ccs_Scope2_EI = 'NA'
smr_ccs_Scope1_EI = 'NA'
smr_ccs_total_EI  = 'NA'
NH3_smr_Scope3_EI = 'NA'
NH3_smr_Scope2_EI = 'NA'
NH3_smr_Scope1_EI = 'NA'
NH3_smr_total_EI  = 'NA'
NH3_smr_ccs_Scope3_EI = 'NA'
NH3_smr_ccs_Scope2_EI = 'NA'
NH3_smr_ccs_Scope1_EI = 'NA'
NH3_smr_ccs_total_EI  = 'NA'
steel_smr_Scope3_EI = 'NA'
steel_smr_Scope2_EI = 'NA'
steel_smr_Scope1_EI = 'NA'
steel_smr_total_EI  = 'NA'
steel_smr_ccs_Scope3_EI = 'NA'
steel_smr_ccs_Scope2_EI = 'NA'
steel_smr_ccs_Scope1_EI = 'NA'
steel_smr_ccs_total_EI  = 'NA'
electrolysis_Scope3_EI = 'NA'
electrolysis_Scope2_EI = 'NA'
electrolysis_Scope1_EI = 'NA'
electrolysis_total_EI  = 'NA'
NH3_electrolysis_Scope3_EI = 'NA'
NH3_electrolysis_Scope2_EI = 'NA'
NH3_electrolysis_Scope1_EI = 'NA'
NH3_electrolysis_total_EI  = 'NA'
steel_electrolysis_Scope3_EI = 'NA'
steel_electrolysis_Scope2_EI = 'NA'
steel_electrolysis_Scope1_EI = 'NA'
steel_electrolysis_total_EI  = 'NA'

#==============================================================================
    
# Loop through all scenarios in output folder
for i0 in range(len(files2load_results)):
    # Read in applicable Cambium file
    filecase = files2load_results_title[i0+1]
    # Extract year and site location to identify which cambium file to import
    year = int(filecase[0])
    site = filecase[1]
    grid_case = filecase[5]
    # The arguments below are just starting points
    
    if year == 2020:
        cambium_year = 2025
        ely_PO_consume = 55  # Electrolyzer current total electrical usage (kWh/kg H2)
    elif year == 2025:
        cambium_year = 2030
        ely_PO_consume = 55  # Electrolyzer current total electrical usage (kWh/kg H2)
    elif year == 2030:
        cambium_year =2035
        ely_PO_consume = 46 # Electrolyzer future total electrical usage (kWh/kg H2)
    elif year == 2035:
        cambium_year = 2040
        ely_PO_consume = 46 # Electrolyzer future total electrical usage (kWh/kg H2)
    
    # Read in the cambium 
    cambiumdata_filepath = dircambium + site + '_'+str(cambium_year) + '.csv'
    cambium_data = pd.read_csv(cambiumdata_filepath,index_col = None,header = 4,usecols = ['lrmer_co2_c','lrmer_ch4_c','lrmer_n2o_c','lrmer_co2_p','lrmer_ch4_p','lrmer_n2o_p','lrmer_co2e_c','lrmer_co2e_p','lrmer_co2e'])
    
    cambium_data = cambium_data.reset_index().rename(columns = {'index':'Interval','lrmer_co2_c':'LRMER CO2 combustion (kg-CO2/MWh)','lrmer_ch4_c':'LRMER CH4 combustion (g-CH4/MWh)','lrmer_n2o_c':'LRMER N2O combustion (g-N2O/MWh)',\
                                                  'lrmer_co2_p':'LRMER CO2 production (kg-CO2/MWh)','lrmer_ch4_p':'LRMER CH4 production (g-CH4/MWh)','lrmer_n2o_p':'LRMER N2O production (g-N2O/MWh)','lrmer_co2e_c':'LRMER CO2 equiv. combustion (kg-CO2e/MWh)',\
                                                  'lrmer_co2e_p':'LRMER CO2 equiv. production (kg-CO2e/MWh)','lrmer_co2e':'LRMER CO2 equiv. total (kg-CO2e/MWh)'})
    
    cambium_data['Interval']=cambium_data['Interval']+1
    cambium_data = cambium_data.set_index('Interval')        
        
    # Read in rodeo data
    rodeo_filepath = dir0+files2load_results[i0+1]
    rodeo_data = pd.read_csv(rodeo_filepath,index_col = None, header = 26,usecols = ['Interval','Input Power (MW)','Non-Ren Import (MW)','Renewable Input (MW)','Curtailment (MW)','Product Sold (units of product)'])
    # What was the rationale for calling the nominal values normalized [(-) and (kg/MW)]? Was the idea to divide by the electrolyzer capacity (not currently in combined_data output)? Shout if I am wrong but I don't think it is necessary because we need the data in its original units.
    # rodeo_data = rodeo_data.rename(columns = {'Input Power (MW)':'Normalized Electrolyzer Power (-)','Non-Ren Import (MW)':'Normalized Grid Import (-)','Renewable Input (MW)':'Normalized Renewable Input (-)', 'Curtailment (MW)':'Normalized Curtailment (-)','Product Sold (units of product)':'Hydrogen production (kg/MW)'})
    rodeo_data = rodeo_data.rename(columns = {'Input Power (MW)':'Electrolyzer Power (MW)','Non-Ren Import (MW)':'Grid Import (MW)','Renewable Input (MW)':'Renewable Input (MW)', 'Curtailment (MW)':'Curtailment (MW)','Product Sold (units of product)':'Hydrogen production (kg-H2)'})
    # Combine RODeO and Cambium data into one dataframe
    combined_data = rodeo_data.merge(cambium_data, on = 'Interval',how = 'outer')
    
    # Calculate hourly grid emissions factors of interest. If we want to use different GWPs, we can do that here. The Grid Import is an hourly data i.e., in MWh
    combined_data['Total grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. total (kg-CO2e/MWh)']
    combined_data['Scope 2 (combustion) grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)']
    combined_data['Scope 3 (production) grid emissions (kg-CO2e)'] = combined_data['Grid Import (MW)']*combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)']
    
    # Sum total emissions
    total_grid_emissions_sum = combined_data['Total grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    scope2_grid_emissions_sum = combined_data['Scope 2 (combustion) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    scope3_grid_emissions_sum = combined_data['Scope 3 (production) grid emissions (kg-CO2e)'].sum()*system_life*kg_to_MT_conv
    h2prod_sum = combined_data['Hydrogen production (kg-H2)'].sum()*system_life*kg_to_MT_conv
    grid_emission_intensity_annual_average = combined_data['LRMER CO2 equiv. total (kg-CO2e/MWh)'].mean()
       
    if 'hybrid-grid' in grid_case:
        # Calculate grid-connected electrolysis emissions/ future cases should reflect targeted electrolyzer electricity usage
        electrolysis_Scope3_EI = scope3_grid_emissions_sum/h2prod_sum + wind_capex_EI * ely_PO_consume * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI  = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI 
        # Calculate ammonia emissions via hybrid grid electrolysis
        NH3_electrolysis_Scope3_EI = NH3_H2_consume * electrolysis_total_EI + NH3_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope2_EI = NH3_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope1_EI = NH3_boiler_EI
        NH3_electrolysis_total_EI  = NH3_electrolysis_Scope1_EI + NH3_electrolysis_Scope2_EI + NH3_electrolysis_Scope3_EI
        # Calculate steel emissions via hybrid grid electrolysis
        steel_electrolysis_Scope3_EI = (steel_H2_consume * electrolysis_total_EI * MT_to_kg_conv + steel_lime_EI * steel_lime_consume * MT_to_kg_conv + steel_iron_ore_EI  * steel_iron_ore_consume  * MT_to_kg_conv + steel_NG_supply_EI * steel_NG_consume  + combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * steel_PO_consume + steel_H2O_EI * steel_H2O_consume * gal_to_ton_conv)  # kg CO2e/metric tonne steel
        steel_electrolysis_Scope2_EI = steel_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean()  
        steel_electrolysis_Scope1_EI = steel_CH4_prod + steel_CO2_prod
        steel_electrolysis_total_EI  = steel_electrolysis_Scope1_EI + steel_electrolysis_Scope2_EI + steel_electrolysis_Scope3_EI
    if 'grid-only' in grid_case:
        # Calculate SMR emissions. SMR and SMR + CCS are always grid-connected
        smr_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv + smr_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
        smr_Scope2_EI = smr_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
        smr_Scope1_EI = smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
        smr_total_EI  = smr_Scope1_EI + smr_Scope2_EI + smr_Scope3_EI
        
        # Calculate ammonia emissions via SMR process
        NH3_smr_Scope3_EI = NH3_H2_consume * smr_total_EI + NH3_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_smr_Scope2_EI = NH3_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_smr_Scope1_EI = NH3_boiler_EI
        NH3_smr_total_EI = NH3_smr_Scope1_EI + NH3_smr_Scope2_EI + NH3_smr_Scope3_EI   
        
        # Calculate steel emissions via SMR process
        steel_smr_Scope3_EI = (smr_total_EI * steel_H2_consume * MT_to_kg_conv + steel_lime_EI * steel_lime_consume * MT_to_kg_conv + steel_iron_ore_EI  * steel_iron_ore_consume  * MT_to_kg_conv + steel_NG_supply_EI * steel_NG_consume  + combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * steel_PO_consume + steel_H2O_EI * steel_H2O_consume * gal_to_ton_conv)  # kg CO2e/metric tonne steel
        steel_smr_Scope2_EI = combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * steel_PO_consume 
        steel_smr_Scope1_EI = steel_CH4_prod + steel_CO2_prod
        steel_smr_total_EI  = steel_smr_Scope1_EI + steel_smr_Scope2_EI + steel_smr_Scope3_EI
        
        # Calculate SMR + CCS emissions
        smr_ccs_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv + (smr_ccs_PO_consume +  ccs_PO_consume) * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
        smr_ccs_Scope2_EI = (smr_ccs_PO_consume +  ccs_PO_consume) * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
        smr_ccs_Scope1_EI = (1-ccs_perc_capture)* smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
        smr_ccs_total_EI  = smr_ccs_Scope1_EI + smr_ccs_Scope2_EI + smr_ccs_Scope3_EI    
        
        # Calculate ammonia emissions via SMR with CCS process
        NH3_smr_ccs_Scope3_EI = NH3_H2_consume * smr_ccs_total_EI + NH3_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_smr_ccs_Scope2_EI = NH3_smr_Scope2_EI
        NH3_smr_ccs_Scope1_EI = NH3_smr_Scope1_EI
        NH3_smr_ccs_total_EI = NH3_smr_ccs_Scope1_EI + NH3_smr_ccs_Scope2_EI + NH3_smr_ccs_Scope3_EI   
        
        # Calculate steel emissions via SMR with CCS process
        steel_smr_ccs_Scope3_EI = (smr_ccs_total_EI * steel_H2_consume * MT_to_kg_conv + steel_lime_EI * steel_lime_consume * MT_to_kg_conv + steel_iron_ore_EI  * steel_iron_ore_consume  * MT_to_kg_conv + steel_NG_supply_EI * steel_NG_consume  + combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * steel_PO_consume + steel_H2O_EI * steel_H2O_consume * gal_to_ton_conv)  # kg CO2e/metric tonne steel
        steel_smr_ccs_Scope2_EI = steel_smr_Scope2_EI 
        steel_smr_ccs_Scope1_EI = steel_smr_Scope1_EI 
        steel_smr_ccs_total_EI  = steel_smr_Scope1_EI + steel_smr_Scope2_EI + steel_smr_ccs_Scope3_EI        
        # Calculate grid-connected electrolysis emissions
        electrolysis_Scope3_EI = scope3_grid_emissions_sum/h2prod_sum  + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = scope2_grid_emissions_sum/h2prod_sum 
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        # Calculate ammonia emissions via grid only electrolysis
        NH3_electrolysis_Scope3_EI = NH3_H2_consume * electrolysis_total_EI + NH3_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope2_EI = NH3_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope1_EI = NH3_boiler_EI
        NH3_electrolysis_total_EI  = NH3_electrolysis_Scope1_EI + NH3_electrolysis_Scope2_EI + NH3_electrolysis_Scope3_EI
        # Calculate steel emissions via grid only electrolysis
        steel_electrolysis_Scope3_EI = (steel_H2_consume * electrolysis_total_EI * MT_to_kg_conv + steel_lime_EI * steel_lime_consume * MT_to_kg_conv + steel_iron_ore_EI  * steel_iron_ore_consume * MT_to_kg_conv + steel_NG_supply_EI * steel_NG_consume  + combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * steel_PO_consume + steel_H2O_EI * steel_H2O_consume * gal_to_ton_conv)  # kg CO2e/metric tonne steel
        steel_electrolysis_Scope2_EI = steel_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean()  
        steel_electrolysis_Scope1_EI = steel_CH4_prod + steel_CO2_prod
        steel_electrolysis_total_EI  = steel_electrolysis_Scope1_EI + steel_electrolysis_Scope2_EI + steel_electrolysis_Scope3_EI
    if 'off-grid' in grid_case:
        # Calculate renewable only electrolysis emissions        
        electrolysis_Scope3_EI = wind_capex_EI * ely_PO_consume * g_to_kg_conv + ely_stack_capex_EI # kg CO2e/kg H2
        electrolysis_Scope2_EI = 0
        electrolysis_Scope1_EI = 0
        electrolysis_total_EI = electrolysis_Scope1_EI + electrolysis_Scope2_EI + electrolysis_Scope3_EI
        # Calculate ammonia emissions via renewable electrolysis
        NH3_electrolysis_Scope3_EI = NH3_H2_consume * electrolysis_total_EI + NH3_PO_consume * combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope2_EI = NH3_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv
        NH3_electrolysis_Scope1_EI = NH3_boiler_EI
        NH3_electrolysis_total_EI = NH3_electrolysis_Scope1_EI + NH3_electrolysis_Scope2_EI + NH3_electrolysis_Scope3_EI
        # Calculate steel emissions via renewable electrolysis
        steel_electrolysis_Scope3_EI = (steel_H2_consume * electrolysis_total_EI * MT_to_kg_conv + steel_lime_EI * steel_lime_consume * MT_to_kg_conv + steel_iron_ore_EI  * steel_iron_ore_consume * MT_to_kg_conv + steel_NG_supply_EI * steel_NG_consume  + combined_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * steel_PO_consume + steel_H2O_EI * steel_H2O_consume * gal_to_ton_conv)  # kg CO2e/metric tonne steel
        steel_electrolysis_Scope2_EI = steel_PO_consume * combined_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() 
        steel_electrolysis_Scope1_EI = steel_CH4_prod + steel_CO2_prod
        steel_electrolysis_total_EI  = steel_electrolysis_Scope1_EI + steel_electrolysis_Scope2_EI + steel_electrolysis_Scope3_EI
        
    '''
    For reference:
        Steel production with DRI using 100% natural gas has GHG emissions of ~1.8 kg CO2e/kg steel according to GREET 2022.
        Steel production with DRI using 100% clean hydrogen has GHG emissions of ~0.8 kg CO2e/kg hydrogen according to GREET 2022.
        For virgin steel production, we expect GHG around ~2,750 kg CO2e/ton steel (we are not using that case)
        Using 100% H2 (clean H2) in EAF, GHG should be ~260 kg CO2e/ton steel
    '''
    
    # Put all cumulative metrics into a dictionary, and then a dataframe
    d = {'Total Life Cycle H2 Production (tonnes-H2/MW)': [h2prod_sum],'Total Scope 2 (Combustion) GHG Emissions (tonnes-CO2e/MW)': [scope2_grid_emissions_sum],
          'Total Scope 3 (Production) GHG Emissions (tonnes-CO2e/MW)': [scope3_grid_emissions_sum],'Total Life Cycle Emissions (tonnes-CO2e/MW)' : [total_grid_emissions_sum],
          'Annaul Average Grid Emission Intensity (kg-CO2/MWh)': [grid_emission_intensity_annual_average],
          'SMR Scope 3 GHG Emissions (kg-CO2e/kg-H2)': [smr_Scope3_EI],'SMR Scope 2 GHG Emissions (kg-CO2e/kg-H2)': [smr_Scope2_EI],
          'SMR Scope 1 GHG Emissions (kg-CO2e/kg-H2)': [smr_Scope1_EI],'SMR Total GHG Emissions (kg-CO2e/kg-H2)': [smr_total_EI],     
          'Ammonia SMR Scope 3 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_Scope3_EI],
          'Ammonia SMR Scope 2 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_Scope2_EI], 
          'Ammonia SMR Scope 1 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_Scope1_EI],
          'Ammonia SMR Total GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_total_EI],    
          'Steel SMR Scope 3 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_Scope3_EI],
          'Steel SMR Scope 2 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_Scope2_EI],
          'Steel SMR Scope 1 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_Scope1_EI],
          'Steel SMR Total GHG Emissions (kg-CO2e/MT steel)': [steel_smr_total_EI],          
          'SMR with CCS Scope 3 GHG Emissions (kg-CO2e/kg-H2)': [smr_ccs_Scope3_EI],'SMR with CCS Scope 2 GHG Emissions (kg-CO2e/kg-H2)': [smr_ccs_Scope2_EI],
          'SMR with CCS Scope 1 GHG Emissions (kg-CO2e/kg-H2)': [smr_ccs_Scope1_EI],'SMR with CCS Total GHG Emissions (kg-CO2e/kg-H2)': [smr_ccs_total_EI],     
          'Ammonia SMR with CCS Scope 3 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_ccs_Scope3_EI],
          'Ammonia SMR with CCS Scope 2 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_ccs_Scope2_EI], 
          'Ammonia SMR with CCS Scope 1 GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_ccs_Scope1_EI],
          'Ammonia SMR with CCS Total GHG Emissions (kg-CO2e/kg-NH3)': [NH3_smr_ccs_total_EI],    
          'Steel SMR with CCS Scope 3 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_ccs_Scope3_EI],
          'Steel SMR with CCS Scope 2 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_ccs_Scope2_EI],
          'Steel SMR with CCS Scope 1 GHG Emissions (kg-CO2e/MT steel)': [steel_smr_ccs_Scope1_EI],
          'Steel SMR with CCS Total GHG Emissions (kg-CO2e/MT steel)': [steel_smr_ccs_total_EI],                  
          'Electrolysis Scope 3 GHG Emissions (kg-CO2e/kg-H2)':[electrolysis_Scope3_EI],
          'Electrolysis Scope 2 GHG Emissions (kg-CO2e/kg-H2)':[electrolysis_Scope2_EI],
          'Electrolysis Scope 1 GHG Emissions (kg-CO2e/kg-H2)':[electrolysis_Scope1_EI],   
          'Electrolysis Total GHG Emissions (kg-CO2e/kg-H2)':[electrolysis_total_EI],                 
          'Ammonia Electrolysis Scope 3 GHG Emissions (kg-CO2e/kg-NH3)':[NH3_electrolysis_Scope3_EI],
          'Ammonia Electrolysis Scope 2 GHG Emissions (kg-CO2e/kg-NH3)':[NH3_electrolysis_Scope2_EI],
          'Ammonia Electrolysis Scope 1 GHG Emissions (kg-CO2e/kg-NH3)':[NH3_electrolysis_Scope1_EI],   
          'Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)':[NH3_electrolysis_total_EI],                                   
          'Steel Electrolysis Scope 3 GHG Emissions (kg-CO2e/MT steel)':[steel_electrolysis_Scope3_EI],
          'Steel Electrolysis Scope 2 GHG Emissions (kg-CO2e/MT steel)':[steel_electrolysis_Scope2_EI],
          'Steel Electrolysis Scope 1 GHG Emissions (kg-CO2e/MT steel)':[steel_electrolysis_Scope1_EI],   
          'Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)':[steel_electrolysis_total_EI]}
    emissionsandh2 = pd.DataFrame(data = d)
    #trial = pd.concat(emissionsandh2,ignore_index = True)
    for i1 in range(len(files2load_title_header)):
        emissionsandh2[files2load_title_header[i1]] = files2load_results_title[i0+1][i1]
    if i0 == 0:
        emissionsandh2_output = emissionsandh2
    else:
        emissionsandh2_output = pd.concat([emissionsandh2_output,emissionsandh2],ignore_index = True)
       # emissionsandh2_output = emissionsandh2_output.append(emissionsandh2,ignore_index = True)
emissionsandh2_output.to_csv(parent_path+'/examples/H2_Analysis/LCA_results/LCA_results.csv')
# Downselect to grid cases of interest
# emissionsandh2_output = emissionsandh2_output.loc[emissionsandh2_output['Grid Case'].isin(['grid-only-'+grid_price_scenario,'hybrid-grid-'+grid_price_scenario,'off-grid'])]

steel_scope_1 = {}
steel_scope_2 = {}
steel_scope_3 = {}

ammonia_scope_1 = {}
ammonia_scope_2 = {}
ammonia_scope_3 = {}

electrolysis_cases = [
                    'Centralized',
                    #'Distributed'
                    ]

locations = [
        'IN',
        'TX',
        'IA',
        'MS'
        ]

use_cases = [
          'SMR',
          'SMR + CCS', 
          'Grid Only',
          #'Grid + Renewables',
          #'Off Grid, Centralized EC',
          #'Off Grid, Distributed EC'
          ]

retail_string = 'retail-flat'

if retail_string == 'retail-flat':
    emissionsandh2_output  = emissionsandh2_output.loc[(emissionsandh2_output['Grid Case']!='grid-only-wholesale') & (emissionsandh2_output['Grid Case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    emissionsandh2_output = emissionsandh2_output.loc[(emissionsandh2_output['Grid Case']!='grid-only-retail-flat') & (emissionsandh2_output['Grid Case']!='hybrid-grid-retail-flat')]
    
grid_cases = [
    'grid-only-'+retail_string,
    #'hybrid-grid-'+retail_string,
    #'off-grid'
    ]

policy_options = [
                'max',
                #'no-policy'
                ]

font = 'Arial'
title_size = 10
axis_label_size = 10
legend_size = 6
tick_size = 10
resolution = 150

years = [2020, 2025, 2030, 2035]
years = pd.unique(years).astype(int).astype(str).tolist()  

for electrolysis_case in electrolysis_cases:
    for policy_option in policy_options:  
        for grid_case in grid_cases:
            emissionsandh2_plots = emissionsandh2_output.loc[(emissionsandh2_output['Electrolysis case']==electrolysis_case) & (emissionsandh2_output['Policy Option']==policy_option) & (emissionsandh2_output['Grid Case']==grid_case)]
            for site in locations:   
                for use_case in use_cases:
                        if use_case == 'SMR':
                            steel_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR Scope 1 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR Scope 2 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR Scope 3 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                        elif use_case == 'SMR + CCS':
                            steel_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR with CCS Scope 1 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR with CCS Scope 2 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel SMR with CCS Scope 3 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                        else:
                            #if 'grid-only' in grid_case:
                            steel_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel Electrolysis Scope 1 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel Electrolysis Scope 2 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                            steel_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Steel Electrolysis Scope 3 GHG Emissions (kg-CO2e/MT steel)')].values.tolist())
                

                        width = 0.5
                        #fig, ax = plt.subplots()
                        fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
                        ax.bar(years,steel_scope_1[site],width,label='GHG Scope 1 Emissions',edgecolor='steelblue',color='deepskyblue')
                        barbottom=steel_scope_1[site]
                        ax.bar(years,steel_scope_2[site],width,bottom=barbottom,label = 'GHG Scope 2 Emissions',edgecolor='dimgray',color='dimgrey')
                        barbottom=barbottom+steel_scope_2[site]
                        ax.bar(years,steel_scope_3[site],width,bottom=barbottom,label='GHG Scope 3 Emissions',edgecolor='black',color='navy')
                        barbottom=barbottom+steel_scope_3[site]
                        ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1)
            
                        # Decorations
                        scenario_title = site + ', ' + use_case 
                        ax.set_title(scenario_title, fontsize=title_size)
                        
                        ax.set_ylabel('GHG (kg CO2e/MT steel)', fontname = font, fontsize = axis_label_size)
                        #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
                        ax.legend(fontsize = legend_size, ncol = 1, prop = {'family':'Arial','size':7})
                        max_y = np.max(barbottom)
                        ax.set_ylim([0,2000])
                        ax.tick_params(axis = 'y',labelsize = 7,direction = 'in')
                        ax.tick_params(axis = 'x',labelsize = 7,direction = 'in',rotation=45)
                        #ax2 = ax.twinx()
                        #ax2.set_ylim([0,10])
                        #plt.xlim(x[0], x[-1])
                        plt.tight_layout()
                        
                       
                        if use_case == 'SMR':
                            ammonia_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR Scope 1 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR Scope 2 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR Scope 3 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                        elif use_case == 'SMR + CCS':
                            ammonia_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR with CCS Scope 1 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR with CCS Scope 2 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia SMR with CCS Scope 3 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                        else:
                            #if 'grid-only' in grid_case:
                            ammonia_scope_1[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia Electrolysis Scope 1 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_2[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia Electrolysis Scope 2 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                            ammonia_scope_3[site] = np.array(emissionsandh2_plots.loc[(emissionsandh2_plots['Site']==site,'Ammonia Electrolysis Scope 3 GHG Emissions (kg-CO2e/kg-NH3)')].values.tolist())
                

                        width = 0.5
                        #fig, ax = plt.subplots()
                        fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
                        ax.bar(years,ammonia_scope_1[site],width,label='GHG Scope 1 Emissions',edgecolor='teal',color='lightseagreen')
                        barbottom=ammonia_scope_1[site]
                        ax.bar(years,ammonia_scope_2[site],width,bottom=barbottom,label = 'GHG Scope 2 Emissions',edgecolor='dimgray',color='grey')
                        barbottom=barbottom+ammonia_scope_2[site]
                        ax.bar(years,ammonia_scope_3[site],width,bottom=barbottom,label='GHG Scope 3 Emissions',edgecolor='chocolate',color='darkorange')
                        barbottom=barbottom+ammonia_scope_3[site]
                        ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1)
            
                        # Decorations
                        scenario_title = site + ', ' + use_case #+ ',' +retail_string
                        ax.set_title(scenario_title, fontsize=title_size)
                        
                        ax.set_ylabel('GHG (kg CO2e/kg NH3)', fontname = font, fontsize = axis_label_size)
                        #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
                        ax.legend(fontsize = legend_size, ncol = 1, prop = {'family':'Arial','size':7})
                        max_y = np.max(barbottom)
                        ax.set_ylim([0,5])
                        ax.tick_params(axis = 'y',labelsize = 7,direction = 'in')
                        ax.tick_params(axis = 'x',labelsize = 7,direction = 'in',rotation=45)
                        #ax2 = ax.twinx()
                        #ax2.set_ylim([0,10])
                        #plt.xlim(x[0], x[-1])
                        plt.tight_layout()
#plt.savefig(parent_path + '/examples/H2_Analysis/LCA_results/best_GHG_steel.png')

### ATTENTION!!! Plotting below doesn't really work. I think we need to change the way we are doing plots
# since we have more distinction between locations

# years = pd.unique(emissionsandh2_output['Year']).tolist()

# for year in years:
#     year = 2030
#     gridonly_emissions = emissionsandh2_output.loc[(emissionsandh2_output['Year'] == year) & (emissionsandh2_output['Grid Case'] == 'grid-only-'+grid_price_scenario)]
#     offgrid_emissions = emissionsandh2_output.loc[(emissionsandh2_output['Year'] == year) & (emissionsandh2_output['Grid Case'] == 'off-grid') ]
#     hybridgrid_emissions = emissionsandh2_output.loc[(emissionsandh2_output['Year'] == year) & (emissionsandh2_output['Grid Case'] == 'hybrid-grid-'+grid_price_scenario) ]

#     smr_emissions = offgrid_emissions.drop(labels = ['Scope 1 Emissions (kg-CO2/kg-H2)','Scope 2 Emissions (kg-CO2/kg-H2)','Scope 3 Emissions (kg-CO2/kg-H2)'],axis=1)
#     # just use IA since all are the same right now
#     smr_emissions = smr_emissions.loc[smr_emissions['Site']=='IA'].drop(labels = ['Site'],axis=1)
#     smr_emissions['Site'] = 'SMR - \n all sites'
#     smr_emissions = smr_emissions.rename(columns = {'SMR Scope 3 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 3 Emissions (kg-CO2/kg-H2)','SMR Scope 2 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 2 Emissions (kg-CO2/kg-H2)',
#                                                     'SMR Scope 1 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 1 Emissions (kg-CO2/kg-H2)'})
    
#     # The current plotting method will not work for all grid cases; we will need to change how we do it
#     # This at least makes it possible to compare grid-only emissions with SMR emissions
#     aggregate_emissions = pd.concat([gridonly_emissions,smr_emissions])

#     smr_total_emissions = aggregate_emissions.loc[aggregate_emissions['Site'] == 'SMR - \n all sites','Scope 3 Emissions (kg-CO2/kg-H2)'] + aggregate_emissions.loc[aggregate_emissions['Site'] == 'SMR - \n all sites','Scope 2 Emissions (kg-CO2/kg-H2)'] \
#                         + aggregate_emissions.loc[aggregate_emissions['Site'] == 'SMR - \n all sites','Scope 1 Emissions (kg-CO2/kg-H2)'] 
#     smr_total_emissions = smr_total_emissions.tolist()
#     smr_total_emissions = smr_total_emissions[0]
    
#     labels = pd.unique(aggregate_emissions['Site']).tolist()
    
#     scope3 = aggregate_emissions['Scope 3 Emissions (kg-CO2/kg-H2)']
#     scope2 = aggregate_emissions['Scope 2 Emissions (kg-CO2/kg-H2)']
#     scope1 = aggregate_emissions['Scope 1 Emissions (kg-CO2/kg-H2)']
#     width = 0.3
#     fig, ax = plt.subplots()
#     #ax.set_ylim([0, 18])
#     ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
#     ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
#     ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
#     #valuelabel(scope1, scope2, scope3, labels)
#     ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
#     ax.set_title('GHG Emission Intensities - All Sites ' + str(year))
#     plt.axhline(y = smr_total_emissions, color='red', linestyle ='dashed', label = 'GHG emissions baseline')
#     ax.legend(loc='upper right', 
#                       #bbox_to_anchor=(0.5, 1),
#              ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
#             #fig.tight_layout() 
#     plt.savefig(dir_plot+'GHG Emission Intensities_all_sites_'+str(year)+'.png', dpi = 1000)
    
#Pull in TEA data
# Read in the summary data from the database
# conn = sqlite3.connect(dirfinancial+'Default_summary.db')
# TEA_data = pd.read_sql_query("SELECT * From Summary",conn)

# conn.commit()
# conn.close()


# TEA_data = TEA_data[['Hydrogen model','Site','Year','Turbine Size','Electrolysis case','Policy Option','Grid Case','Hydrogen annual production (kg)',\
#                      'Steel annual production (tonne/year)','Ammonia annual production (kg/year)','LCOH ($/kg)','Steel price: Total ($/tonne)','Ammonia price: Total ($/kg)']]
# TEA_data = TEA_data.loc[(TEA_data['Hydrogen model']=='RODeO') & (TEA_data['Grid Case'].isin(['grid-only-'+grid_price_scenario,'hybrid-grid-'+grid_price_scenario,'off-grid']))]
# TEA_data['Year'] = TEA_data['Year'].astype(np.int32)
# TEA_data = TEA_data.drop(labels = ['Hydrogen model'],axis =1)
# TEA_data['Policy Option'] = TEA_data['Policy Option'].replace(' ','-')

# # Combine data into one dataframe
# combined_TEA_LCA_data = TEA_data.merge(emissionsandh2_output,how = 'outer', left_index = False,right_index = False)

# # Example of calculating carbon abatement cost. 
# # This section is mostly just to give a sense for how things like carbon abatement cost could be calculated for the above 
# # structure
# smr_cost_no_ccs = 1 # USD/kg-H2; just an approximation for now

# combined_TEA_LCA_data['Total SMR Emissions (kg-CO2e/kg-H2)'] = combined_TEA_LCA_data['SMR Scope 3 GHG Emissions (kg-CO2e/kg-H2)'] +combined_TEA_LCA_data['SMR Scope 2 GHG Emissions (kg-CO2e/kg-H2)'] + combined_TEA_LCA_data['SMR Scope 1 GHG Emissions (kg-CO2e/kg-H2)']

# combined_TEA_LCA_data['CO2 abatement cost ($/MT-CO2)'] = (combined_TEA_LCA_data['LCOH ($/kg)'] - smr_cost_no_ccs)/(combined_TEA_LCA_data['Total SMR Emissions (kg-CO2e/kg-H2)']-combined_TEA_LCA_data['Total Life Cycle Emissions (kg-CO2e/kg-H2)'])*1000

# # Segregate data by grid scenario
# TEALCA_data_offgrid = combined_TEA_LCA_data.loc[combined_TEA_LCA_data['Grid Case'].isin(['off-grid'])] 
# TEALCA_data_gridonly = combined_TEA_LCA_data.loc[combined_TEA_LCA_data['Grid Case'].isin(['grid-only-'+grid_price_scenario])]
# TEALCA_data_hybridgrid = combined_TEA_LCA_data.loc[combined_TEA_LCA_data['Grid Case'].isin(['hybrid-grid-'+grid_price_scenario])]

# # Pivot tables for Emissions plots vs year
# hydrogen_abatementcost_offgrid = TEALCA_data_offgrid.pivot_table(index = 'Year',columns = ['Site','Grid Case'], values = 'CO2 abatement cost ($/MT-CO2)')
# hydrogen_abatementcost_gridonly = TEALCA_data_gridonly.pivot_table(index = 'Year',columns = ['Site','Grid Case'], values = 'CO2 abatement cost ($/MT-CO2)')
# hydrogen_abatementcost_hybridgrid = TEALCA_data_hybridgrid.pivot_table(index = 'Year',columns = ['Site','Grid Case'], values = 'CO2 abatement cost ($/MT-CO2)')

# # Create lists of scenario names for plot legends
# names_gridonly = hydrogen_abatementcost_gridonly.columns.values.tolist()
# names_gridonly_joined = []
# for j in range(len(hydrogen_abatementcost_gridonly.columns)):
#     names_gridonly_joined.append(', '.join(names_gridonly[j]))
    
# names_hybridgrid = hydrogen_abatementcost_hybridgrid.columns.values.tolist()
# names_hybridgrid_joined = []
# for j in range(len(hydrogen_abatementcost_hybridgrid.columns)):
#     names_hybridgrid_joined.append(', '.join(names_hybridgrid[j]))
    
# names_offgrid = hydrogen_abatementcost_offgrid.columns.values.tolist()
# names_offgrid_joined = []
# for j in range(len(hydrogen_abatementcost_offgrid.columns)):
#     names_offgrid_joined.append(', '.join(names_offgrid[j]))

# # Abatement cost vs year
# fig5, ax5 = plt.subplots(3,1,sharex = 'all',figsize = (4,8),dpi = 150)
# ax5[0].plot(hydrogen_abatementcost_gridonly,marker = '.')
# ax5[1].plot(hydrogen_abatementcost_hybridgrid,marker = '.')
# ax5[2].plot(hydrogen_abatementcost_offgrid ,marker = '.')
# for ax in ax5.flat:
#     ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
#     ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
# ax5[0].set_ylabel('Grid-Only CO2 Abatement Cost \n($/t-CO2)',fontsize = 10, fontname = 'Arial')
# ax5[1].set_ylabel('Hybrid-Grid CO2 Abatement Cost \n($/t-CO2)',fontsize = 10, fontname='Arial')
# ax5[2].set_ylabel('Off-Grid CO2 Abatement Cost \n($/t-CO2)',fontsize = 10, fontname = 'Arial')
# ax5[2].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
# ax5[0].legend(names_gridonly_joined,prop = {'family':'Arial','size':6})
# ax5[1].legend(names_hybridgrid_joined,prop = {'family':'Arial','size':6})
# ax5[2].legend(names_offgrid_joined ,prop = {'family':'Arial','size':6})
# plt.tight_layout()
# plt.savefig(dir_plot+'hydrogen_abatement_cost.png',pad_inches = 0.1)
# plt.close(fig = None)
