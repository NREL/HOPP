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
import sqlite3

import glob
import csv
import sys
import heapq


# Directory from which to pull outputs from
#year = '2022'
dir0 = 'Examples/H2_Analysis/RODeO_files/Output/' 
dircambium = 'Examples/H2_Analysis/RODeO_files/Cambium/StdScen21_MidCase95by2035_hourly_' 
dir_plot = 'Examples/H2_Analysis/RODeO_files/Plots/LCA_Plots/'

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
        int1[8] = int1[8].replace('.csv', '')
        files2load_results_title[c0[0]] = int1
    files2load_title_header = ['Steel String','Year','Site String','Site Number','Turbine Year','Turbine Size','Storage Duration','Storage String','Grid Case']
    
    
# SMR emissions
g_to_kg_conv = 0.001
smr_NG_combust = 56.2 # Natural gas combustion (g CO2e/MJ)
smr_NG_consume = 167  # Natural gas consumption (MJ/kg H2)
smr_PO_consume = 0    # Power consumption in SMR plant (kWh/kg H2)
smr_steam_prod = 17.4 # Steam production on SMR site (MJ/kg H2)
smr_HEX_eff    = 0.9  # Heat exchanger efficiency (-)
smr_NG_supply  = 9    # Natural gas extraction and supply to SMR plant assuming 2% CH4 leakage rate (g CO2e/MJ)


    
# Loop through all scenarios in output folder
for i0 in range(len(files2load_results)):
    #i0 = 0
    # Read in applicable Cambium file
    filecase = files2load_results_title[i0+1]
    # Extract year and site location to identify which cambium file to import
    year = int(filecase[1])
    site = filecase[3]
    
    # Specify balancing area based on site
    if site == '1':
        balancing_area = 'p65'
    elif site == '2':
        balancing_area ='p124'
    elif site == '3':
        balancing_area = 'p128'
    elif site == '4':
        balancing_area = 'p9'
    
    # Read in the cambium 
    cambiumdata_filepath = dircambium + balancing_area + '_'+str(year) + '.csv'
    cambium_data = pd.read_csv(cambiumdata_filepath,index_col = None,header = 4,usecols = ['Interval','lrmer_co2_c','lrmer_ch4_c','lrmer_n2o_c','lrmer_co2_p','lrmer_ch4_p','lrmer_n2o_p','lrmer_co2e_c','lrmer_co2e_p','lrmer_co2e'])
    cambium_data = cambium_data.rename(columns = {'lrmer_co2_c':'LRMER CO2 combustion (kg/MWh)','lrmer_ch4_c':'LRMER CH4 combustion (g/MWh)','lrmer_n2o_c':'LRMER N2O combustion (g/MWh)',\
                                                  'lrmer_co2_p':'LRMER CO2 production (kg/MWh)','lrmer_ch4_p':'LRMER CH4 production (g/MWh)','lrmer_n2o_p':'LRMER N2O production (g/MWh)','lrmer_co2e_c':'LRMER CO2 equiv. combustion (kg/MWh)',\
                                                  'lrmer_co2e_p':'LRMER CO2 equiv. production (kg/MWh)','lrmer_co2e':'LRMER CO2 equiv. total (kg/MWh)'})
    # Read in rodeo data
    rodeo_filepath = dir0+files2load_results[i0+1]
    rodeo_data = pd.read_csv(rodeo_filepath,index_col = None, header = 26,usecols = ['Interval','Input Power (MW)','Non-Ren Import (MW)','Renewable Input (MW)','Curtailment (MW)','Product Sold (units of product)'])
    rodeo_data = rodeo_data.rename(columns = {'Input Power (MW)':'Normalized Electrolyzer Power (-)','Non-Ren Import (MW)':'Normalized Grid Import (-)','Renewable Input (MW)':'Normalized Renewable Input (-)', 'Curtailment (MW)':'Normalized Curtailment (-)','Product Sold (units of product)':'Hydrogen production (kg/MW)'})

    # Combine RODeO and Cambium data into one dataframe
    combined_data = rodeo_data.merge(cambium_data, on = 'Interval',how = 'outer')
    
    # Calculate hourly emissions factors of interest. If we want to use different GWPs, we can do that here.
    combined_data['Total grid emissions (kg-CO2/MW)'] = combined_data['Normalized Grid Import (-)']*combined_data['LRMER CO2 equiv. total (kg/MWh)']
    combined_data['Scope 2 (combustion) grid emissions (kg-CO2/MW)'] = combined_data['Normalized Grid Import (-)']*combined_data['LRMER CO2 equiv. combustion (kg/MWh)']
    combined_data['Scope 3 (production) grid emissions (kg-CO2/MW)'] = combined_data['Normalized Grid Import (-)']*combined_data['LRMER CO2 equiv. production (kg/MWh)']
    
    # Sum total emissions. Note that the 30 indicates system life; the 1000 converts kg to metric tonnes
    total_emissions_sum = combined_data['Total grid emissions (kg-CO2/MW)'].sum()*30/1000
    scope2_combustion_emissions_sum = combined_data['Scope 2 (combustion) grid emissions (kg-CO2/MW)'].sum()*30/1000
    scope3_production_emissions_sum = combined_data['Scope 3 (production) grid emissions (kg-CO2/MW)'].sum()*30/1000
    h2prod_sum = combined_data['Hydrogen production (kg/MW)'].sum()*30/1000
    grid_emission_intensity_annual_average = combined_data['LRMER CO2 equiv. total (kg/MWh)'].mean()
    
    # Calculate SMR emissions
    smr_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
    smr_Scope2_EI = smr_PO_consume * grid_emission_intensity_annual_average * g_to_kg_conv # kg CO2e/kg H2
    smr_Scope1_EI = smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
    
    # Put all cumulative metrics into a dictionary, and then a dataframe
    d = {'Total Life Cycle H2 Production (tonnes-H2/MW)': [h2prod_sum],'Total Scope 2 (Combustion) Life Cycle Emissions (tonnes-CO2/MW)':[scope2_combustion_emissions_sum],\
         'Total Scope 3 (Production) Life Cycle Emissions (tonnes-CO2/MW)': [scope3_production_emissions_sum],'Total Life Cycle Emissions (tonnes-CO2/MW)' : [total_emissions_sum],\
         'Annaul Average Grid Emission Intensity (kg-CO2/MWh)':grid_emission_intensity_annual_average,'SMR Scope 3 Life Cycle Emissions (kg-CO2/kg-H2)':smr_Scope3_EI,\
         'SMR Scope 2 Life Cycle Emissions (kg-CO2/kg-H2)':smr_Scope2_EI, 'SMR Scope 1 Life Cycle Emissions (kg-CO2/kg-H2)':smr_Scope1_EI}
    emissionsandh2 = pd.DataFrame(data = d)
    for i1 in range(len(files2load_title_header)):
        emissionsandh2[files2load_title_header[i1]] = files2load_results_title[i0+1][i1]
    if i0 == 0:
        emissionsandh2_output = emissionsandh2
    else:
        emissionsandh2_output = pd.concat([emissionsandh2_output,emissionsandh2],ignore_index = True)
        #emissionsandh2_output = emissionsandh2_output.append(emissionsandh2,ignore_index = True)

# In case you want to plot against storage duration or something like that
emissionsandh2_output['Storage Duration'] = emissionsandh2_output['Storage Duration'].astype(int)
# Calculate life cycle emissions on a kg-CO2/kg-H2 basis
emissionsandh2_output['Year'] = emissionsandh2_output['Year'].astype(int)
emissionsandh2_output['Total Life Cycle Emissions (kg-CO2/kg-H2)'] = emissionsandh2_output['Total Life Cycle Emissions (tonnes-CO2/MW)']/emissionsandh2_output['Total Life Cycle H2 Production (tonnes-H2/MW)']
emissionsandh2_output['Scope 2 Emissions (kg-CO2/kg-H2)'] = emissionsandh2_output['Total Scope 2 (Combustion) Life Cycle Emissions (tonnes-CO2/MW)']/emissionsandh2_output['Total Life Cycle H2 Production (tonnes-H2/MW)']
emissionsandh2_output['Scope 3 Emissions (kg-CO2/kg-H2)'] = emissionsandh2_output['Total Scope 3 (Production) Life Cycle Emissions (tonnes-CO2/MW)']/emissionsandh2_output['Total Life Cycle H2 Production (tonnes-H2/MW)']

# Set up Scope 1 emissions as zeros since we currently don't have anything for Scope 1 emissions
scope1_emissions = [0]*emissionsandh2_output.shape[0]
scope1_emissions = pd.DataFrame(scope1_emissions,columns = ['Scope 1 Emissions (kg-CO2/kg-H2)'])
emissionsandh2_output = emissionsandh2_output.join(scope1_emissions)

# Reformat grid case names
emissionsandh2_output.loc[emissionsandh2_output['Grid Case'] =='gridconnected', 'Grid Case']= 'Grid-Connected'
emissionsandh2_output.loc[emissionsandh2_output['Grid Case'] =='offgrid', 'Grid Case']= 'Off-Grid'

# Format storage duration output and calculate hydrogen sold
emissionsandh2_output['Storage Duration'] = emissionsandh2_output['Storage Duration'].astype(np.int64)
emissionsandh2_output['Storage Duration'] = emissionsandh2_output['Storage Duration'].astype(str)
emissionsandh2_output['Storage Duration'] = emissionsandh2_output['Storage Duration'] + ' hr'

# Reformat location names
# emissionsandh2_output['Site Number'] = emissionsandh2_output['Site Number'].astype(np.int64)
# emissionsandh2_output['Site Number'] = emissionsandh2_output['Site Number'].astype(str)
# emissionsandh2_output['Site Number'] = 'Site ' + emissionsandh2_output['Site Number']
emissionsandh2_output = emissionsandh2_output.rename(columns ={'Site Number':'Site Name'})
emissionsandh2_output.loc[emissionsandh2_output['Site Name']=='Site 1','Site Name'] = 'Gulf of Mexico'
emissionsandh2_output.loc[emissionsandh2_output['Site Name']=='Site 2','Site Name'] = 'Central Atlantic'
emissionsandh2_output.loc[emissionsandh2_output['Site Name']=='Site 3','Site Name'] = 'New York'
emissionsandh2_output.loc[emissionsandh2_output['Site Name']=='Site 4','Site Name'] = 'Northern California'


# Downselect to optimal storage durations. This section will be unnecessary once we optimize storae duration within RODeO
emissionsandh2_output_gulf_offgrid = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Gulf of Mexico') & (emissionsandh2_output['Grid Case'] == 'Off-Grid') & (emissionsandh2_output['Storage Duration'] =='500 hr')]
emissionsandh2_output_gulf_gridconnected = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Gulf of Mexico') & (emissionsandh2_output['Grid Case'] == 'Grid-Connected') & (emissionsandh2_output['Storage Duration'] =='50 hr')]
emissionsandh2_output_atlantic_offgrid = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Central Atlantic') & (emissionsandh2_output['Grid Case'] == 'Off-Grid') & (emissionsandh2_output['Storage Duration'] =='100 hr')]
emissionsandh2_output_atlantic_gridconnected = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Central Atlantic') & (emissionsandh2_output['Grid Case'] == 'Grid-Connected') & (emissionsandh2_output['Storage Duration'] =='10 hr')]

emissionsandh2_output_newyork_offgrid = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='New York') & (emissionsandh2_output['Grid Case'] == 'Off-Grid') & (emissionsandh2_output['Storage Duration'] =='500 hr')]
emissionsandh2_output_newyork_gridconnected = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='New York') & (emissionsandh2_output['Grid Case'] == 'Grid-Connected') & (emissionsandh2_output['Storage Duration'] =='50 hr')]

emissionsandh2_output_cal_offgrid = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Northern California') & (emissionsandh2_output['Grid Case'] == 'Off-Grid') & (emissionsandh2_output['Storage Duration'] =='500 hr')]
emissionsandh2_output_cal_gridconnected = emissionsandh2_output.loc[(emissionsandh2_output['Site Name']=='Northern California') & (emissionsandh2_output['Grid Case'] == 'Grid-Connected') & (emissionsandh2_output['Storage Duration'] =='50 hr')]


emissionsandh2_output_optstorage = pd.concat([emissionsandh2_output_gulf_offgrid,emissionsandh2_output_gulf_gridconnected,emissionsandh2_output_atlantic_offgrid,emissionsandh2_output_atlantic_gridconnected,\
                                              emissionsandh2_output_newyork_offgrid,emissionsandh2_output_newyork_gridconnected,emissionsandh2_output_cal_offgrid,emissionsandh2_output_cal_gridconnected])

# Plot life cycle emissions bar charts for each year

years = pd.unique(emissionsandh2_output_optstorage['Year']).tolist()

for year in years:
    year = 2022
    gridconnected_emissions = emissionsandh2_output_optstorage.loc[(emissionsandh2_output_optstorage['Year'] == year) & (emissionsandh2_output_optstorage['Grid Case'] == 'Grid-Connected')]
    # Just use GoM for all offgrid sites since for now they are all the same
    offgrid_emissions = emissionsandh2_output_optstorage.loc[(emissionsandh2_output_optstorage['Year'] == year) & (emissionsandh2_output_optstorage['Grid Case'] == 'Off-Grid') & (emissionsandh2_output_optstorage['Site Name'] == 'Gulf of Mexico')]
    offgrid_emissions = offgrid_emissions.drop(labels = ['Site Name'],axis = 1)
    offgrid_emissions['Site Name'] = 'Off-grid - \n all sites'
    smr_emissions = offgrid_emissions.drop(labels = ['Scope 1 Emissions (kg-CO2/kg-H2)','Scope 2 Emissions (kg-CO2/kg-H2)','Scope 3 Emissions (kg-CO2/kg-H2)','Site Name'],axis=1)
    smr_emissions['Site Name'] = 'SMR - \n all sites'
    smr_emissions = smr_emissions.rename(columns = {'SMR Scope 3 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 3 Emissions (kg-CO2/kg-H2)','SMR Scope 2 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 2 Emissions (kg-CO2/kg-H2)',
                                                    'SMR Scope 1 Life Cycle Emissions (kg-CO2/kg-H2)':'Scope 1 Emissions (kg-CO2/kg-H2)'})
    
    aggregate_emissions = pd.concat([offgrid_emissions,gridconnected_emissions,smr_emissions])
    aggregate_emissions.loc[aggregate_emissions['Site Name'] =='Gulf of Mexico','Site Name'] = 'GE - \n Gulf of \n Mexico'
    aggregate_emissions.loc[aggregate_emissions['Site Name'] =='Central Atlantic','Site Name'] = 'GE - \n Central \n Atlantic'
    aggregate_emissions.loc[aggregate_emissions['Site Name'] =='New York','Site Name'] = 'GE - \n New York'
    aggregate_emissions.loc[aggregate_emissions['Site Name'] =='Northern California','Site Name'] = 'GE - \n California'

    smr_total_emissions = aggregate_emissions.loc[aggregate_emissions['Site Name'] == 'SMR - \n all sites','Scope 3 Emissions (kg-CO2/kg-H2)'] + aggregate_emissions.loc[aggregate_emissions['Site Name'] == 'SMR - \n all sites','Scope 2 Emissions (kg-CO2/kg-H2)'] \
                        + aggregate_emissions.loc[aggregate_emissions['Site Name'] == 'SMR - \n all sites','Scope 1 Emissions (kg-CO2/kg-H2)'] 
    smr_total_emissions = smr_total_emissions.tolist()
    smr_total_emissions = smr_total_emissions[0]
    
    labels = pd.unique(aggregate_emissions['Site Name']).tolist()
    
    scope3 = aggregate_emissions['Scope 3 Emissions (kg-CO2/kg-H2)']
    scope2 = aggregate_emissions['Scope 2 Emissions (kg-CO2/kg-H2)']
    scope1 = aggregate_emissions['Scope 1 Emissions (kg-CO2/kg-H2)']
    width = 0.3
    fig, ax = plt.subplots()
    ax.set_ylim([0, 18])
    ax.bar(labels, scope3, width, label = 'Scope 3 emission intensities', color = 'darkcyan')
    ax.bar(labels, scope2, width, bottom = scope3, label = 'Scope 2 emission intensities', color = 'darkorange')
    ax.bar(labels, scope1, width, bottom = scope3, label = 'Scope 1 emission intensities', color = 'goldenrod')
    #valuelabel(scope1, scope2, scope3, labels)
    ax.set_ylabel('GHG Emission Intensities (kg CO2e/kg H2)')
    ax.set_title('GHG Emission Intensities - All Sites ' + str(year))
    plt.axhline(y = smr_total_emissions, color='red', linestyle ='dashed', label = 'GHG emissions baseline')
    ax.legend(loc='upper right', 
                      #bbox_to_anchor=(0.5, 1),
             ncol=1, fancybox=True, shadow=False, borderaxespad=0, framealpha=0.2)
            #fig.tight_layout() 
    plt.savefig(dir_plot+'GHG Emission Intensities_all_sites_'+str(year)+'.png', dpi = 1000)
    
#Pull in TEA data
# Read in the summary data from the database
conn = sqlite3.connect(dir0+'Default_summary.db')
TEA_data = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

TEA_data['Storage Duration'] = TEA_data['Storage Duration'].astype(int)
TEA_data['Year'] = TEA_data['Year'].astype(int)
TEA_data = TEA_data[['Steel String','Year','Site String','Site Number','Turbine Year','Turbine Size','Storage Duration','Storage String','Grid Case','Product NPV cost (US$/kg)']]

# Reformat grid case names
TEA_data.loc[TEA_data['Grid Case'] =='gridconnected', 'Grid Case']= 'Grid-Connected'
TEA_data.loc[TEA_data['Grid Case'] =='offgrid', 'Grid Case']= 'Off-Grid'

# Reformat location names
TEA_data['Site Number'] = TEA_data['Site Number'].astype(np.int64)
TEA_data['Site Number'] = TEA_data['Site Number'].astype(str)
TEA_data['Site Number'] = 'Site ' + TEA_data['Site Number']
TEA_data = TEA_data.rename(columns ={'Site Number':'Site Name'})
TEA_data.loc[TEA_data['Site Name']=='Site 1','Site Name'] = 'Gulf of Mexico'
TEA_data.loc[TEA_data['Site Name']=='Site 2','Site Name'] = 'Central Atlantic'
TEA_data.loc[TEA_data['Site Name']=='Site 3','Site Name'] = 'New York'
TEA_data.loc[TEA_data['Site Name']=='Site 4','Site Name'] = 'Northern California'

# Combine data into one dataframe
combined_TEA_LCA_data = TEA_data.merge(emissionsandh2_output,how = 'outer', left_index = False,right_index = False)

# Example of calculating carbon abatement cost. NOte that this should probably be cleaned up by optimal storage duration,
# though hopefully by the time we are focused on these plots we will be optimizing storage duration.
# This section is mostly just to give a sense for how things like carbon abatement cost could be calculated for the above 
# structure
smr_cost_no_ccs = 1 # USD/kg-H2; just an approximation for now

combined_TEA_LCA_data['Total SMR Emissions (kg-CO2/kg-H2)'] = combined_TEA_LCA_data['SMR Scope 3 Life Cycle Emissions (kg-CO2/kg-H2)'] +combined_TEA_LCA_data['SMR Scope 2 Life Cycle Emissions (kg-CO2/kg-H2)'] + combined_TEA_LCA_data['SMR Scope 1 Life Cycle Emissions (kg-CO2/kg-H2)']

combined_TEA_LCA_data['CO2 abatement cost ($/MT-CO2)'] = (combined_TEA_LCA_data['Product NPV cost (US$/kg)'] - smr_cost_no_ccs)/(combined_TEA_LCA_data['Total SMR Emissions (kg-CO2/kg-H2)']-combined_TEA_LCA_data['Total Life Cycle Emissions (kg-CO2/kg-H2)'])*1000

# Format storage duration output as a string for plot legends
combined_TEA_LCA_data['Storage Duration'] = combined_TEA_LCA_data['Storage Duration'].astype(np.int64)
combined_TEA_LCA_data['Storage Duration'] = combined_TEA_LCA_data['Storage Duration'].astype(str)
combined_TEA_LCA_data['Storage Duration'] = combined_TEA_LCA_data['Storage Duration'] + ' hr'

# Segregate data by grid scenario
TEALCA_data_offgrid = combined_TEA_LCA_data[combined_TEA_LCA_data['Grid Case'].isin(['Off-Grid'])] 
TEALCA_data_gridconnected = combined_TEA_LCA_data[combined_TEA_LCA_data['Grid Case'].isin(['Grid-Connected'])]

# Pivot tables for Emissions plots vs year
hydrogen_abatementcost_offgrid = TEALCA_data_offgrid.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'CO2 abatement cost ($/MT-CO2)')
hydrogen_abatementcost_gridconnected = TEALCA_data_gridconnected.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'CO2 abatement cost ($/MT-CO2)')

# Create lists of scenario names for plot legends
names_gridconnected = hydrogen_abatementcost_gridconnected.columns.values.tolist()
names_gridconnected_joined = []
for j in range(len(hydrogen_abatementcost_gridconnected.columns)):
    names_gridconnected_joined.append(', '.join(names_gridconnected[j]))
    
names_offgrid = hydrogen_abatementcost_offgrid.columns.values.tolist()
names_offgrid_joined = []
for j in range(len(hydrogen_abatementcost_offgrid.columns)):
    names_offgrid_joined.append(', '.join(names_offgrid[j]))

# Abatement cost vs year
fig5, ax5 = plt.subplots(2,1,sharex = 'all',figsize = (4,8),dpi = 150)
ax5[0].plot(hydrogen_abatementcost_gridconnected,marker = '.')
ax5[1].plot(hydrogen_abatementcost_gridconnected ,marker = '.')
for ax in ax5.flat:
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
ax5[0].set_ylabel('Grid-Connected CO2 Abatement Cost \n($/t-CO2)',fontsize = 10, fontname = 'Arial')
ax5[1].set_ylabel('Off-Grid CO2 Abatement Cost \n($/t-CO2)',fontsize = 10, fontname = 'Arial')
ax5[1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0].legend(names_gridconnected_joined,prop = {'family':'Arial','size':6})
ax5[1].legend(names_offgrid_joined ,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(dir_plot+'hydrogen_abatement_cost.png',pad_inches = 0.1)
plt.close(fig = None)
