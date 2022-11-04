# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:45:28 2022

@author: mkoleva
"""

#==============================================================================
# Abbreviations
# EI  = emission intensity
# ELY = electrolyzer
# SMR = steam methane reforming
# MT = metric ton
# LCA = Life cycle assessment
#==============================================================================
# Import packages
#==============================================================================
import os
import glob
import csv
import sys
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tabulate import tabulate  # currently not working but would like to fix

#==============================================================================
# Set directories
#==============================================================================

# Pull data from \HOPP\examples\H2_Analysis which is where the .py code will be located
path = os.getcwd()
dirin_cambium  = '\\RODeO_files\\Cambium\\'
dirin_rodeo = '\\RODeO_files\\Output\\'

#==============================================================================
# Input assumptions
#==============================================================================

g_to_kg_conv = 0.001

# Global Warming Potential Values for 100 years and 20 years
#------------------------------------------------------------------------------

gwp = {'Greenhouse Gas': ['CO2', 'CH4', 'N2O'], 
       'GWP_100': [1, 30, 265],
       'GWP_20': [1, 85, 264]
       }

#==============================================================================
# WIP
#==============================================================================
# gwp2 = {'Greenhouse Gas': ['CO2', 'CH4', 'N2O', 'C2F6', 'CF4'], 
#        'GWP_100': [1, 30, 265, 11100, 6630],
#        'GWP_20': [1, 85, 264, 8210, 4880]
#        }
#==============================================================================
#print(tabulate(GWP, headers = 'keys'))

gwp = pd.DataFrame(data = gwp)

print('\n The Global Warming Potential data are:\n', gwp)

#==============================================================================
# Scenarios
# 1. Steam methane reforming
# 2. Steam methane reforming with CCS
# 3. Grid-renewable electrolysis
# 4. Renewable electrolysis
#==============================================================================

# Steam methane reforming (SMR) - Incumbent H2 production process
#------------------------------------------------------------------------------

smr_NG_combust = 56.2 # Natural gas combustion (g CO2e/MJ)
smr_NG_consume = 167  # Natural gas consumption (MJ/kg H2)
smr_PO_consume = 0    # Power consumption in SMR plant (kWh/kg H2)
smr_steam_prod = 17.4 # Steam production on SMR site (MJ/kg H2)
smr_HEX_eff    = 0.9  # Heat exchanger efficiency (-)
smr_NG_supply  = 9    # Natural gas extraction and supply to SMR plant assuming 2% CH4 leakage rate (g CO2e/MJ)

# Hydrogen production via water electrolysis
#------------------------------------------------------------------------------

ely_PO_consume_curr = 55   # Electrolyzer current total electrical usage (kWh/kg H2)
ely_PO_consume_futu = 46   # Electrolyzer future total electrical usage (kWh/kg H2)
grid_trans_losses   = 0.05 # Grid losses of 5% are assumed (-)
fuel_to_grid_curr   = 48   # Fuel mix emission intensity for current power grid (g CO2e/kWh)
fuel_to_grid_futu   = 14   # Fuel mix emission intensity for future power grid (g CO2e/kWh)

#==============================================================================

# Sites: 
# 1: Gulf of Mexico: 40% renewable for 2022
# 2: Central Atlantic: 44% renewable for 2022
# 3: New York: 47% renewable for 2022
# 4: California: 45% renewable for 2022

cambium_files = glob.glob(os.path.join(path + dirin_cambium, 'StdScen21_MidCase95_hourly_p*.csv'))
years = ['2022', '2025', '2030', '2035']



smr_Scope3_EI_all = {'year': [], 'value': []}
ely_Scope3_grid_EI_all = {'year': [], 'value': []}
ely_Scope3_ren_EI_all = {'year': [], 'value': []}
smr_Scope2_EI_all = {'year': [], 'value': []}
ely_Scope2_grid_EI_all = {'year': [], 'value': []}
ely_Scope2_ren_EI_all = {'year': [], 'value': []}
smr_Scope1_EI_all = {'year': [], 'value': []}
ely_Scope1_grid_EI_all = {'year': [], 'value': []}
ely_Scope1_ren_EI_all = {'year': [], 'value': []}
years_all = []

for year in years:
    for cambium_file in cambium_files:
        if year in cambium_file:
#    os.rename(cambium_file, cambium_file.replace('by2035', ''))  
#==============================================================================
# Scope 3 emissions: Resources extraction, processing and transportation (indirect emissions, 
# currently excluding CAPEX but good to add in future, when information becomes available)
#==============================================================================
            grid_Scope3_EI = pd.read_csv(cambium_file, skiprows=4, 
                                         usecols = ['lrmer_co2_p', 'lrmer_ch4_p', 'lrmer_n2o_p']
                                         ) # CO2 comes in kg/MWh while CH4 and N2O come in g/MWh. Data will be converted to be in g X/kWh
            grid_Scope3_EI['lrmer_ch4_p'] = grid_Scope3_EI['lrmer_ch4_p'].div(1000)
            grid_Scope3_EI['lrmer_n2o_p'] = grid_Scope3_EI['lrmer_n2o_p'].div(1000)
            grid_Scope3_CO2e_EI = grid_Scope3_EI * np.array([gwp['GWP_100']])
            po_Scope3_EI = grid_Scope3_CO2e_EI.sum(axis=1) # g CO2e/kWh
            smr_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
            smr_Scope3_EI_all['year'].append(year)
            smr_Scope3_EI_all['value'].append(smr_Scope3_EI)
            ely_Scope3_grid_EI = po_Scope3_EI.mean() * ely_PO_consume_curr * g_to_kg_conv # kg CO2e/kg H2
            ely_Scope3_grid_EI_all['year'].append(year)
            ely_Scope3_grid_EI_all['value'].append(ely_Scope3_grid_EI)
            ely_Scope3_ren_EI = 0
            ely_Scope3_ren_EI_all['year'].append(year)
            ely_Scope3_ren_EI_all['value'].append(ely_Scope3_ren_EI)

#==============================================================================
# Scope 2 emissions: Electricity generation and transmission (indirect emissions)
#==============================================================================
    
            grid_Scope2_EI = pd.read_csv(cambium_file, skiprows=4, 
                                          usecols = ['lrmer_co2_c','lrmer_ch4_c', 'lrmer_n2o_c']
                                          )
            grid_Scope2_EI['lrmer_ch4_c'] = grid_Scope2_EI['lrmer_ch4_c'].div(1000)
            grid_Scope2_EI['lrmer_n2o_c'] = grid_Scope2_EI['lrmer_n2o_c'].div(1000)
            grid_Scope2_CO2e_EI = grid_Scope2_EI * np.array([gwp['GWP_100']])
            po_Scope2_EI = grid_Scope2_CO2e_EI.sum(axis=1) # g CO2e/kWh
            smr_Scope2_EI = smr_PO_consume * po_Scope2_EI.mean() * g_to_kg_conv # kg CO2e/kg H2
            smr_Scope2_EI_all['year'].append(year)
            smr_Scope2_EI_all['value'].append(smr_Scope2_EI)
            ely_Scope2_grid_EI = ely_PO_consume_curr * po_Scope2_EI.mean() * g_to_kg_conv  # kg CO2e/kg H2
            ely_Scope2_grid_EI_all['year'].append(year)
            ely_Scope2_grid_EI_all['value'].append(ely_Scope2_grid_EI)
            ely_Scope2_ren_EI = 0
            ely_Scope2_ren_EI_all['year'].append(year)
            ely_Scope2_ren_EI_all['value'].append(ely_Scope2_ren_EI)
        
#==============================================================================
# Scope 1 emissions: Fuel (H2) production process emissions (direct emissions)
#==============================================================================

            smr_Scope1_EI = smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
            smr_Scope1_EI_all['year'].append(year)
            smr_Scope1_EI_all['value'].append(smr_Scope1_EI)
            ely_Scope1_grid_EI = 0 # kg CO2e/ kg H2
            ely_Scope1_grid_EI_all['year'].append(year)
            ely_Scope1_grid_EI_all['value'].append(ely_Scope1_grid_EI)
            ely_Scope1_ren_EI = 0 # kg CO2e/ kg H2
            ely_Scope1_ren_EI_all['year'].append(year)
            ely_Scope1_ren_EI_all['value'].append(ely_Scope1_ren_EI)
            
            years_all.append(year)

smr_Scope3_EI_all = pd.DataFrame(smr_Scope3_EI_all)   
smr_Scope2_EI_all = pd.DataFrame(smr_Scope2_EI_all)  
smr_Scope1_EI_all = pd.DataFrame(smr_Scope1_EI_all) 

ely_Scope3_grid_EI_all = pd.DataFrame(ely_Scope3_grid_EI_all) 
ely_Scope2_grid_EI_all = pd.DataFrame(ely_Scope2_grid_EI_all) 
ely_Scope1_grid_EI_all = pd.DataFrame(ely_Scope1_grid_EI_all) 

ely_Scope3_ren_EI_all = pd.DataFrame(ely_Scope3_ren_EI_all)
ely_Scope2_ren_EI_all = pd.DataFrame(ely_Scope2_ren_EI_all)
ely_Scope1_ren_EI_all = pd.DataFrame(ely_Scope1_ren_EI_all)        
#==============================================================================
# Total emission intensities
#==============================================================================

locations = ['California', 'Gulf of Mexico', 'Central Atlantic', 'New York',
         'California', 'Gulf of Mexico', 'Central Atlantic', 'New York',
         'California', 'Gulf of Mexico', 'Central Atlantic', 'New York',
         'California', 'Gulf of Mexico', 'Central Atlantic', 'New York']

headers = ['Location', 'Year', 'GHG Emissions kg CO2e/kg H2']

locations = pd.DataFrame(locations)
years_all = pd.DataFrame(years_all)

smr_total_EI = smr_Scope3_EI_all['value'] + smr_Scope2_EI_all['value'] + smr_Scope1_EI_all['value'] # kg CO2e/kg H2
ely_total_grid_EI = ely_Scope3_grid_EI_all['value'] + ely_Scope2_grid_EI_all['value'] + ely_Scope1_grid_EI_all['value']
ely_total_ren_EI = ely_Scope3_ren_EI_all['value'] + ely_Scope2_ren_EI_all['value'] + ely_Scope1_ren_EI_all['value'] # kg CO2e/kg H2

smr_total_EI = pd.DataFrame(smr_total_EI)
smr_total_EI = pd.concat([locations, years_all, smr_total_EI], axis = 1)
smr_total_EI.columns = headers

ely_total_grid_EI = pd.DataFrame(ely_total_grid_EI)
ely_total_grid_EI = pd.concat([locations, years_all, ely_total_grid_EI], axis = 1)
ely_total_grid_EI.columns = headers

ely_total_ren_EI = pd.DataFrame(ely_total_ren_EI)
ely_total_ren_EI = pd.concat([locations, years_all, ely_total_ren_EI], axis = 1)
ely_total_ren_EI.columns = headers

print('\n The Well-to-gate GHG emission intensity of the SMR system is: \n', smr_total_EI, 'kg CO2e/kg H2')
print('\n The Well-to-gate GHG emission intensity of the grid-connected electrolyzer system is: \n', ely_total_grid_EI, 'kg CO2e/kg H2')
print('\n The Well-to-gate GHG emission intensity of the off-grid electrolyzer system is: \n', ely_total_ren_EI, 'kg CO2e/kg H2')

smr_Scope3_EI_all = pd.concat([locations, smr_Scope3_EI_all], axis = 1)  
smr_Scope3_EI_all.columns = headers 
smr_Scope2_EI_all = pd.concat([locations, smr_Scope2_EI_all], axis = 1) 
smr_Scope2_EI_all.columns = headers  
smr_Scope1_EI_all = pd.concat([locations, smr_Scope1_EI_all], axis = 1) 
smr_Scope1_EI_all.columns = headers 

ely_Scope3_grid_EI_all = pd.concat([locations, ely_Scope3_grid_EI_all], axis = 1) 
ely_Scope3_grid_EI_all.columns = headers 
ely_Scope2_grid_EI_all = pd.concat([locations, ely_Scope2_grid_EI_all], axis = 1) 
ely_Scope2_grid_EI_all.columns = headers 
ely_Scope1_grid_EI_all = pd.concat([locations, ely_Scope1_grid_EI_all], axis = 1) 
ely_Scope1_grid_EI_all.columns = headers 

ely_Scope3_ren_EI_all = pd.concat([locations, ely_Scope3_ren_EI_all], axis = 1)
ely_Scope3_ren_EI_all.columns = headers 
ely_Scope2_ren_EI_all = pd.concat([locations, ely_Scope2_ren_EI_all], axis = 1)
ely_Scope2_ren_EI_all.columns = headers 
ely_Scope1_ren_EI_all = pd.concat([locations, ely_Scope1_ren_EI_all], axis = 1) 
ely_Scope1_ren_EI_all.columns = headers 

def valuelabel(scope1, scope2, scope3, labels):
    for i in range(len(labels)):
        plt.text(i,round(scope1[i],1),round(scope1[i],1), ha = 'center')
        plt.text(i,round(scope2[i],1),round(scope2[i],1), ha = 'center')
        plt.text(i,round(scope3[i],1),round(scope3[i],1), ha = 'center')

# Sites: 
# 1: Gulf of Mexico: 40% renewable for 2022
# 2: Central Atlantic: 44% renewable for 2022
# 3: New York: 47% renewable for 2022
# 4: California: 45% renewable for 2022    

# Sites: 
# 1: Gulf of Mexico: 40% renewable for 2025
# 2: Central Atlantic: 47% renewable for 2025
# 3: New York: 47% renewable for 2025
# 4: California: 45% renewable for 2025   

# Sites: 
# 1: Gulf of Mexico: 39% renewable for 2030
# 2: Central Atlantic: 44% renewable for 2030
# 3: New York: 47% renewable for 2030
# 4: California: 45% renewable for 2030  

# Sites: 
# 1: Gulf of Mexico: 41% renewable for 2035
# 2: Central Atlantic: 44% renewable for 2035
# 3: New York: 47% renewable for 2035
# 4: California: 45% renewable for 2035    

# Numbers should be calculated from RODeO outputs

non_renewable_portion = [0.55, 0.60, 0.56, 0.53,
                         0.55, 0.60, 0.53, 0.53,
                         0.55, 0.61, 0.56, 0.53,
                         0.55, 0.59, 0.56, 0.53]
                         
non_renewable_portion = pd.DataFrame(non_renewable_portion)
non_renewable_portion = pd.concat([locations, years_all, non_renewable_portion], axis = 1)
headers_non_ren = ['Location', 'Year', 'Non-ren portion']
non_renewable_portion.columns = headers_non_ren

# Not sure I can cycle through the years

#==============================================================================
# Year 2022
#==============================================================================

labels = ['RE - \n all sites', 'GE - \n California', 'GE - \n Gulf of \n Mexico', 
                  'GE - \n Central \n Atlantic', 'GE - \n New York', 'SMR']
scope3 = [ely_Scope3_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[0]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[0], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[0]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[0]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[0]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  smr_Scope3_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope2 = [ely_Scope2_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[0]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[0], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[0]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[0]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[0]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  smr_Scope2_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope1 = [ely_Scope1_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[0], 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[0],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[0], 
                  smr_Scope1_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
width = 0.3
fig, ax = plt.subplots()
ax.set_ylim([0, 16])
ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
valuelabel(scope1, scope2, scope3, labels)
ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
ax.set_title('GHG Emission Intensities - All Sites 2022')
plt.axhline(y = smr_total_EI['GHG Emissions kg CO2e/kg H2'].mean(), color='red', linestyle ='dashed', label = 'GHG emissions baseline')
ax.legend(loc='upper right', 
                  #bbox_to_anchor=(0.5, 1),
         ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
        #fig.tight_layout() 
plt.savefig('GHG Emission Intensities_all_sites_2022.png', dpi = 1000)
        #plt.show()


#==============================================================================
# Year 2025
#==============================================================================

labels = ['RE - \n all sites', 'GE - \n California', 'GE - \n Gulf of \n Mexico', 
                  'GE - \n Central \n Atlantic', 'GE - \n New York', 'SMR']
scope3 = [ely_Scope3_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[1]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[1], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[1]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[1]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[1]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  smr_Scope3_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope2 = [ely_Scope2_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[1]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[1], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[1]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[1]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[1]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  smr_Scope2_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope1 = [ely_Scope1_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[1], 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[1],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[1], 
                  smr_Scope1_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
width = 0.3
fig, ax = plt.subplots()
ax.set_ylim([0, 16])
ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
valuelabel(scope1, scope2, scope3, labels)
ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
ax.set_title('GHG Emission Intensities - All Sites 2025')
plt.axhline(y = smr_total_EI['GHG Emissions kg CO2e/kg H2'].mean(), color='red', linestyle ='dashed', label = 'GHG emissions baseline')
ax.legend(loc='upper right', 
                  #bbox_to_anchor=(0.5, 1),
         ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
        #fig.tight_layout() 
plt.savefig('GHG Emission Intensities_all_sites_2025.png', dpi = 1000)
        #plt.show()
        
#==============================================================================
# Year 2030
#==============================================================================

labels = ['RE - \n all sites', 'GE - \n California', 'GE - \n Gulf of \n Mexico', 
                  'GE - \n Central \n Atlantic', 'GE - \n New York', 'SMR']
scope3 = [ely_Scope3_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[2]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[2], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[2]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[2]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[2]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  smr_Scope3_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope2 = [ely_Scope2_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[2]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[2], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[2]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[2]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[2]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  smr_Scope2_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope1 = [ely_Scope1_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[2], 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[2],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[2], 
                  smr_Scope1_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
width = 0.3
fig, ax = plt.subplots()
ax.set_ylim([0, 16])
ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
valuelabel(scope1, scope2, scope3, labels)
ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
ax.set_title('GHG Emission Intensities - All Sites 2030')
plt.axhline(y = smr_total_EI['GHG Emissions kg CO2e/kg H2'].mean(), color='red', linestyle ='dashed', label = 'GHG emissions baseline')
ax.legend(loc='upper right', 
                  #bbox_to_anchor=(0.5, 1),
         ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
        #fig.tight_layout() 
plt.savefig('GHG Emission Intensities_all_sites_2030.png', dpi = 1000)
        #plt.show()
        
#==============================================================================
# Year 2035
#==============================================================================

labels = ['RE - \n all sites', 'GE - \n California', 'GE - \n Gulf of \n Mexico', 
                  'GE - \n Central \n Atlantic', 'GE - \n New York', 'SMR']
scope3 = [ely_Scope3_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[3]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[3], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[3]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[3]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[3]*ely_Scope3_grid_EI_all.loc[ely_Scope3_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  smr_Scope3_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope2 = [ely_Scope2_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'California', 'Non-ren portion'].to_numpy()[3]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[3], 
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Gulf of Mexico', 'Non-ren portion'].to_numpy()[3]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'Central Atlantic', 'Non-ren portion'].to_numpy()[3]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  non_renewable_portion.loc[non_renewable_portion['Location'] == 'New York', 'Non-ren portion'].to_numpy()[3]*ely_Scope2_grid_EI_all.loc[ely_Scope2_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  smr_Scope2_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
scope1 = [ely_Scope1_ren_EI_all['GHG Emissions kg CO2e/kg H2'].mean(), 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'California','GHG Emissions kg CO2e/kg H2'].to_numpy()[3], 
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Gulf of Mexico','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'Central Atlantic','GHG Emissions kg CO2e/kg H2'].to_numpy()[3],
                  ely_Scope1_grid_EI_all.loc[ely_Scope1_grid_EI_all['Location'] == 'New York','GHG Emissions kg CO2e/kg H2'].to_numpy()[3], 
                  smr_Scope1_EI_all['GHG Emissions kg CO2e/kg H2'].mean()]
width = 0.3
fig, ax = plt.subplots()
ax.set_ylim([0, 16])
ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
valuelabel(scope1, scope2, scope3, labels)
ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
ax.set_title('GHG Emission Intensities - All Sites 2035')
plt.axhline(y = smr_total_EI['GHG Emissions kg CO2e/kg H2'].mean(), color='red', linestyle ='dashed', label = 'GHG emissions baseline')
ax.legend(loc='upper right', 
                  #bbox_to_anchor=(0.5, 1),
         ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
        #fig.tight_layout() 
plt.savefig('GHG Emission Intensities_all_sites_2035.png', dpi = 1000)
        #plt.show()

# #==============================================================================
# # Loading hourly hydrogen production
# #==============================================================================

csv_files = glob.glob(os.path.join(path + dirin_rodeo, "*_results_*_Site_*50_hrstor_gridconnected.csv"))
  
# loop over the list of csv files

h2_sold_annually_grid_all = []
annual_emissions_Scope1_all = []
annual_emissions_Scope2_all = []
annual_emissions_Scope3_all = []
total_annual_emissions_grid = []
emissions_reduction_grid = []

# Could the outputs from RODeO be specificed in terms of location name i.e., 'California' or the same way the Cambium files are saved i.e., 'p128' etc.?
# Results not quite correct as I have taken only the 50-hr storage duration files
for filename in csv_files:      
    RODeO_results_data = pd.read_csv(filename, on_bad_lines='skip', skiprows = (29))
    RODeO_results_data = pd.DataFrame(RODeO_results_data)
    h2_sold_annually = RODeO_results_data['Product Sold (units of product)'].sum() / 1000 # metric ton H2
    h2_sold_annually_grid_all.append(h2_sold_annually)
h2_sold_annually_grid_all = pd.DataFrame(h2_sold_annually_grid_all)    
locations_RODeO = ['Gulf of Mexico', 'Central Atlantic', 'New York', 'California',
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California'
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California'
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California']
locations_RODeO = pd.DataFrame(locations_RODeO)
h2_sold_annually_grid_all = pd.concat([locations, years_all, h2_sold_annually_grid_all], axis = 1)
h2_sold_annually_grid_all.columns = ['Location', 'Year', 'Sold Hydrogen (MT H2)']

ely_Scope1_grid_EI_all.set_index('Location', inplace=True)
h2_sold_annually_grid_all.set_index('Location', inplace=True)    
h2_sold_annually_grid_all = h2_sold_annually_grid_all.reindex(ely_Scope1_grid_EI_all.index)
annual_emissions_Scope1 = ely_Scope1_grid_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_grid_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

ely_Scope2_grid_EI_all.set_index('Location', inplace=True)   
annual_emissions_Scope2 = ely_Scope2_grid_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_grid_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

ely_Scope3_grid_EI_all.set_index('Location', inplace=True)   
annual_emissions_Scope3 = ely_Scope3_grid_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_grid_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

total_annual_emissions_grid = annual_emissions_Scope1 + annual_emissions_Scope2 + annual_emissions_Scope3  # tons CO2e per year

smr_total_EI.set_index('Location', inplace=True)   
emissions_reduction_grid = smr_total_EI['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_grid_all['Sold Hydrogen (MT H2)'] - total_annual_emissions_grid  # tons CO2e per year
print(emissions_reduction_grid)

csv_files = glob.glob(os.path.join(path + dirin_rodeo, "*_results_*_Site_*500_hrstor_offgrid.csv"))
  
# loop over the list of csv files

h2_sold_annually_ren_all = []
annual_emissions_Scope1_all = []
annual_emissions_Scope2_all = []
annual_emissions_Scope3_all = []
total_annual_emissions_ren = []
emissions_reduction_ren = []

# Could the outputs from RODeO be specificed in terms of location name i.e., 'California' or the same way the Cambium files are saved i.e., 'p128' etc.?
# Results not quite correct as I have taken only the 500-hr storage duration files
for filename in csv_files:      
    RODeO_results_data = pd.read_csv(filename, on_bad_lines='skip', skiprows = (29))
    RODeO_results_data = pd.DataFrame(RODeO_results_data)
    h2_sold_annually = RODeO_results_data['Product Sold (units of product)'].sum() / 1000 # metric ton H2
    h2_sold_annually_ren_all.append(h2_sold_annually)
h2_sold_annually_ren_all = pd.DataFrame(h2_sold_annually_ren_all)    
locations_RODeO = ['Gulf of Mexico', 'Central Atlantic', 'New York', 'California',
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California'
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California'
                   'Gulf of Mexico', 'Central Atlantic', 'New York', 'California']
locations_RODeO = pd.DataFrame(locations_RODeO)
h2_sold_annually_ren_all = pd.concat([locations, years_all, h2_sold_annually_ren_all], axis = 1)
h2_sold_annually_ren_all.columns = ['Location', 'Year', 'Sold Hydrogen (MT H2)']

ely_Scope1_ren_EI_all.set_index('Location', inplace=True)
h2_sold_annually_ren_all.set_index('Location', inplace=True)    
h2_sold_annually_ren_all = h2_sold_annually_ren_all.reindex(ely_Scope1_ren_EI_all.index)
annual_emissions_Scope1 = ely_Scope1_ren_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_ren_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

ely_Scope2_ren_EI_all.set_index('Location', inplace=True)   
annual_emissions_Scope2 = ely_Scope2_ren_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_ren_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

ely_Scope3_ren_EI_all.set_index('Location', inplace=True)   
annual_emissions_Scope3 = ely_Scope3_ren_EI_all['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_ren_all['Sold Hydrogen (MT H2)'] # tons CO2e per year

total_annual_emissions_ren = annual_emissions_Scope1 + annual_emissions_Scope2 + annual_emissions_Scope3  # tons CO2e per year
  
emissions_reduction_ren = smr_total_EI['GHG Emissions kg CO2e/kg H2'] * h2_sold_annually_ren_all['Sold Hydrogen (MT H2)'] - total_annual_emissions_ren  # tons CO2e per year
print(emissions_reduction_ren)
# #==============================================================================
# # Carbon abatement costs
# #==============================================================================
# lcoh_smr = 1 # $/kg H2

# # csv_summary_files = glob.glob(os.path.join(path + dirout, "*_summary_*_Site_1_*_50_hrstor_gridconnected.csv"))

# # lcoh_ely_all = []
# # lcca_all = []

# # for file_summary in csv.reader(csv_summary_files):
# #     lcoh_ely = pd.read_csv(file_summary, on_bad_lines='skip',dtype = object)
# #     lcoh_ely = lcoh_ely['Product NPV cost (US$/kg)'].T # $/kg H2
# #     lcoh_ely_all.append(lcoh_ely)
# #     lcca = (lcoh_ely - lcoh_smr)/emissions_reduction # $/tCO2e
# #     lcca_all.append(lcca)
# # print('\n The levelized cost of hydrogen is: \n', lcoh_ely_all, '$/kg H2')
# # print('\n The levelized cost of carbon abatement is: \n', lcoh_ely_all, '$/tCO2')

# # creating LCOH for 2022, 2025, 2030 and 2035; grid-connected scenarios
# lcohGM_grid = [4.8, 3.9, 3.5, 3.7]
# lcohCenA_grid = [5.9, 5.1, 4.3, 4.2]
# lcohNY_grid = [5.0, 4.1, 4.1, 3.7]
# lcohCA_grid = [5.3, 4.6, 4.4, 4.8]

# # creating LCOH for 2022, 2025, 2030 and 2035; off-grid scenarios

# lcohGM = [15.7, 12.4, 11.3, 10.7]
# lcohCenA = [27.3, 22.5, 21.0, 20.0]
# lcohNY = [11.7, 9.3, 8.5, 8.0]
# lcohCA = [14.0, 11.4, 10.4, 9.9]

# lcohGM = np.array(lcohGM)

# ely_totalGM_EI_years = np.array(ely_totalGM_EI_years)
# ely_totalCenA_EI_years = np.array(ely_totalCenA_EI_years)
# ely_totalNY_EI_years = np.array(ely_totalNY_EI_years)
# ely_totalCA_EI_years = np.array(ely_totalCA_EI_years)

# lccoh_offgrid = (lcohGM - lcoh_smr) / (smr_Total_EI - ely_Total_EI)
# lccohGM_grid = np.divide((lcohGM - lcoh_smr), (smr_Total_EI - ely_totalGM_EI_years.T)) # $/tCO2e
# lccohCenA_grid = np.divide((lcohGM - lcoh_smr), (smr_Total_EI - ely_totalCenA_EI_years.T)) # $/tCO2e
# lccohCA_grid = np.divide((lcohGM - lcoh_smr), (smr_Total_EI - ely_totalCA_EI_years.T)) # $/tCO2e
# lccohNY_grid = np.divide((lcohGM - lcoh_smr), (smr_Total_EI - ely_totalNY_EI_years.T)) # $/tCO2e

# lccohGM = np.concatenate((lccoh_offgrid, lccohGM_grid ), axis=None)
# emissions_reductionGM = pd.DataFrame(emissions_reductionGM)
# lccohGM = pd.DataFrame(lccohGM)



# # # Data generation
# # x = np.linspace(0, 5, num=150)
# # y = np.sin(x) + (np.random.normal(size=len(x)))/10

# # # Model fitting
# # lowess_model = lowess.Lowess()
# # lowess_model.fit(x, y)

# # # Model prediction
# # x_pred = np.linspace(0, 5, 26)
# # y_pred = lowess_model.predict(x_pred)

# # Plotting (example, not the real data but a placeholder for what I would like us to do):

# cases = ['']
# # lccoh GM
# lccoh_GM_list = [1.53, 1.18, 1.07, 1.01, -1.33, -1.62, -13.03, 1.31]

# # emission reductions

# emission_reduction_list = [1142, 1134, 1139, 1048, 1142, 1134, 1139, 1048]
# cumulative = [1142, 2276, 3415, 4463, 5605, 6739, 7878, 8926 ]


# def Cumulative(emission_reduction_list):
#     cu_list = []
#     length = len(emission_reduction_list)
#     cu_list = [sum(emission_reduction_list[0:x:1]) for x in range(0, length+1)]
#     return cu_list[1:]

# cumulative_emission_reduction = Cumulative(emission_reduction_list)

# heapq.heapify(lccoh_GM_list)
# lccoh_GM_list_sort = []
# while lccoh_GM_list:
#     lccoh_GM_list_sort.append(heapq.heappop(lccoh_GM_list))
    
# colors = ["yellow","limegreen","skyblue","pink","limegreen","black","orange","grey"]
# xpos = cumulative
# y = lccoh_GM_list_sort
#     #width of each bar
# w = emission_reduction_list


# fig = plt.bar(xpos, 
#             height = y,
#             width = w,
#             fill = True,
#             color = colors)

# emission_reduction_list = pd.DataFrame(emission_reduction_list)
# lccoh_GM_list_sort = pd.DataFrame(lccoh_GM_list_sort)
# plt.xlim(0, emission_reduction_list.sum())
# plt.ylim(0, lccoh_GM_list_sort.max() + 20)


# plt.legend(fig.patches, cases,
#            loc = "best",
#            ncol = 3)

#     # plt.text(x = demand - df.loc[cut_off_power_plant, "Available capacity (MW)"]/2,
#     #         y = df.loc[cut_off_power_plant, "Marginal Costs"] + 10,
#     #         s = f"Electricity price: \n    {df.loc[cut_off_power_plant, 'Marginal Costs']} $/MWh")

# plt.xlabel("Power plant capacity (MW)")
# plt.ylabel("Marginal Cost ($/MWh)")
# plt.figure()
# plt.show()

#==============================================================================
# End-use applications
#==============================================================================
# Today, in the U.S., 26.4% of the steel plants use recycled steel while 73.6% 
# use virgin steel
#==============================================================================

# recycled_steel_plants = 0.264 # fraction
# virgin_steel_plants = 0.736 # fraction
# recycled_steel_EI = 0.7014 # kg CO2e/kg steel
# virgin_steel_EI =  3.0323 # kg CO2e/kg steel
# average_steel_EI_US = (recycled_steel_plants * recycled_steel_EI + virgin_steel_plants * virgin_steel_EI) * 1000 #kg CO2e/ton steel
# print(average_steel_EI_US, 'kg CO2e/kg steel')

#coke_GHG_intensity = 13.3457 # g CO2e/MJ coke
#iron_ore_GHG_intensity = 45.1148 # g CO2e/kg iron ore
#lime_GHG_intensity = 1.2826 # kg CO2e/kg lime
#diesel_for_steel = 15.4712 g CO2e/ MJ diesel
#NG_for_steel = 12.4013  g CO2e/MJ NG

#==============================================================================
# Qualification for credits 
#==============================================================================

