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
import pandas as pd
import matplotlib.pyplot as plt
# from tabulate import tabulate  # currently not working but would like to fix

#==============================================================================
# Set directories
#==============================================================================

# Pull data from \HOPP\examples\H2_Analysis which is where the .py code will be located
path = os.getcwd()
dirin  = '\\RODeO_files\\Cambium\\'
dirout = '\\RODeO_files\\Output\\'
c0 = [0,0,0]

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

#print(tabulate(GWP, headers = 'keys'))

gwp = pd.DataFrame(data = gwp)

print('\n The Global Warming Potential data are:\n', gwp)

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
# Scope 3 emissions: Resources extraction, processing and transportation (indirect emissions, 
# currently excluding CAPEX but good to add in future, when information becomes available)
#==============================================================================

grid_Scope3_EI = pd.read_csv(path + dirin + 'StdScen21_MidCase95by2035_hourly_AL_2022.csv', skiprows=4, 
                      usecols = ['lrmer_co2_p', 'lrmer_ch4_p', 'lrmer_n2o_p']
                      ) # CO2 comes in kg/MWh while CH4 and N2O come in g/MWh. Data will be converted to be in g X/kWh
grid_Scope3_EI['lrmer_ch4_p'] = grid_Scope3_EI['lrmer_ch4_p'].div(1000)
grid_Scope3_EI['lrmer_n2o_p'] = grid_Scope3_EI['lrmer_n2o_p'].div(1000)

grid_Scope3_CO2e_EI = grid_Scope3_EI * gwp['GWP_100'] 

smr_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
ely_Scope3_EI = 0 # kg CO2e/kg H2


#==============================================================================
# Scope 2 emissions: Electricity generation and transmission (indirect emissions)
#==============================================================================

ren_gen_EI = 0 # kg CO2e/kg H2
grid_Scope2_EI = pd.read_csv(path + dirin + 'StdScen21_MidCase95by2035_hourly_AL_2022.csv', skiprows=4, 
                      usecols = ['lrmer_co2_c','lrmer_ch4_c', 'lrmer_n2o_c']
                      )
grid_Scope2_EI['lrmer_ch4_c'] = grid_Scope2_EI['lrmer_ch4_c'].div(1000)
grid_Scope2_EI['lrmer_n2o_c'] = grid_Scope2_EI['lrmer_n2o_c'].div(1000)

grid_Scope2_CO2e_EI = grid_Scope2_EI * gwp['GWP_100'] 
po_EI = grid_Scope2_CO2e_EI.sum(axis=1) # g CO2e/kWh

smr_Scope2_EI = smr_PO_consume * po_EI.mean() * g_to_kg_conv # kg CO2e/kg H2
ely_Scope2_EI = ely_PO_consume_curr * ren_gen_EI * g_to_kg_conv # kg CO2e/kg H2
#==============================================================================
# Scope 1 emissions: Fuel (H2) production process emissions (direct emissions)
#==============================================================================

smr_Scope1_EI = smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
ely_Scope1_EI = 0 # kg CO2e/ kg H2

#==============================================================================
# Total emission intensities
#==============================================================================

smr_Total_EI = smr_Scope3_EI + smr_Scope2_EI + smr_Scope1_EI # kg CO2e/kg H2
ely_Total_EI = ely_Scope3_EI + ely_Scope2_EI + ely_Scope1_EI # kg CO2e/kg H2

print('\n The Well-to-gate GHG emission intensity of the SMR system is: \n', round(smr_Total_EI,2), 'kg CO2e/kg H2')
print('\n The Well-to-gate GHG emission intensity of the electrolyzer system is: \n', ely_Total_EI, 'kg CO2e/kg H2')

#==============================================================================
# Loading hourly hydrogen production
#==============================================================================

csv_files = glob.glob(os.path.join(path + dirout, "*_results_*.csv"))
  
# loop over the list of csv files
for filename in csv_files:
      
    RODeO_results_data = pd.read_csv(filename, on_bad_lines='skip', skiprows = 29)
    h2_sold_annually = RODeO_results_data['Product Sold (units of product)'].sum() / 1000 # metric ton H2
    total_annual_emissions = ely_Total_EI * h2_sold_annually  # tons CO2e per year
    emissions_reduction = (smr_Total_EI - ely_Total_EI) * h2_sold_annually  # tons CO2e per year
    print('\n Scenario', filename, 'has the following results: \n'
          'The annual hydrogen sales are:\n', round(h2_sold_annually, 2), 'MT H2 \n'
          'The annual emissions of the electrolyzer system are:\n', round(total_annual_emissions, 2), 'MT CO2e \n'
          'The annual emissions reduction, compared to the incumbent H2 production process, are:\n', round(emissions_reduction, 2), 'MT CO2e'
          ) 
    lca_data = {'Total Annual H2 Production (metric tons H2/yr)': [h2_sold_annually],
                'Total Annual Emissions (tCO2e/yr)' : [total_annual_emissions], 
                'Total Annual Emissions Reduction (tCO2e/yr)' : [emissions_reduction]
                }
    lca_data = pd.DataFrame(data = lca_data)
    lca_data.to_csv(filename + '_LCA.csv', index = False)
    
#==============================================================================
# Plots
#==============================================================================

labels = ['Renewable electrolysis', 'Steam Methane Reforming']
scope3 = [ely_Scope3_EI, smr_Scope3_EI]
scope2 = [ely_Scope2_EI, smr_Scope2_EI]
scope1 = [ely_Scope1_EI, smr_Scope1_EI]
width = 0.3
fig, ax = plt.subplots()
ax.set_ylim([0, 10])
ax.bar(labels, scope3, width, label = 'Scope 3 emissions')
ax.bar(labels, scope2, width, label = 'Scope 2 emissions')
ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emissions')
ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
ax.set_title('GHG Emission Intensities by LCA Scope')
ax.legend()
plt.savefig('GHG Emission Intensities.jpg')
#plt.show()

#==============================================================================
# Qualification for credits and carbon abatement costs
#==============================================================================
