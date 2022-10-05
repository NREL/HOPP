# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:27:41 2021

@author: ktopolsk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# Initialization and Global Settings
#Specify directory name
output_directory = 'examples/H2_Analysis/RODeO_files/Output'
plot_directory = 'examples/H2_Analysis/RODeO_files/Plots'
plot_subdirectory = 'Stacked_Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
RODeO_summary_outputs  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Eventually remove this when you fix the naming scheme for cases
RODeO_summary_outputs = RODeO_summary_outputs.drop(['Steel String','Site String','Storage String'],axis=1)

# Format storage duration output and calculate hydrogen sold
RODeO_summary_outputs['Storage Duration'] = RODeO_summary_outputs['Storage Duration'].astype(np.int64)
RODeO_summary_outputs['Storage Duration'] = RODeO_summary_outputs['Storage Duration'].astype(str)
RODeO_summary_outputs['Storage Duration'] = RODeO_summary_outputs['Storage Duration'] + ' hr'
# Reformat grid case names
RODeO_summary_outputs.loc[RODeO_summary_outputs['Grid Case'] =='gridconnected', 'Grid Case']= 'Grid-Connected'
RODeO_summary_outputs.loc[RODeO_summary_outputs['Grid Case'] =='offgrid', 'Grid Case']= 'Off-Grid'

# Reformat location names
RODeO_summary_outputs['Site Number'] = RODeO_summary_outputs['Site Number'].astype(np.int64)
RODeO_summary_outputs['Site Number'] = RODeO_summary_outputs['Site Number'].astype(str)
RODeO_summary_outputs['Site Number'] = 'Site ' + RODeO_summary_outputs['Site Number']
RODeO_summary_outputs = RODeO_summary_outputs.rename(columns ={'Site Number':'Site Name'})
RODeO_summary_outputs.loc[RODeO_summary_outputs['Site Name']=='Site 1','Site Name'] = 'Gulf of Mexico'
RODeO_summary_outputs.loc[RODeO_summary_outputs['Site Name']=='Site 2','Site Name'] = 'Central Atlantic'
RODeO_summary_outputs.loc[RODeO_summary_outputs['Site Name']=='Site 3','Site Name'] = 'New York'
RODeO_summary_outputs.loc[RODeO_summary_outputs['Site Name']=='Site 4','Site Name'] = 'Northern California'

# Downselect to optimal storage durations
RODeO_summary_outputs_gulf_offgrid = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Gulf of Mexico') & (RODeO_summary_outputs['Grid Case'] == 'Off-Grid') & (RODeO_summary_outputs['Storage Duration'] =='500 hr')]
RODeO_summary_outputs_gulf_gridconnected = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Gulf of Mexico') & (RODeO_summary_outputs['Grid Case'] == 'Grid-Connected') & (RODeO_summary_outputs['Storage Duration'] =='50 hr')]
RODeO_summary_outputs_atlantic_offgrid = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Central Atlantic') & (RODeO_summary_outputs['Grid Case'] == 'Off-Grid') & (RODeO_summary_outputs['Storage Duration'] =='100 hr')]
RODeO_summary_outputs_atlantic_gridconnected = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Central Atlantic') & (RODeO_summary_outputs['Grid Case'] == 'Grid-Connected') & (RODeO_summary_outputs['Storage Duration'] =='10 hr')]

RODeO_summary_outputs_newyork_offgrid = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='New York') & (RODeO_summary_outputs['Grid Case'] == 'Off-Grid') & (RODeO_summary_outputs['Storage Duration'] =='500 hr')]
RODeO_summary_outputs_newyork_gridconnected = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='New York') & (RODeO_summary_outputs['Grid Case'] == 'Grid-Connected') & (RODeO_summary_outputs['Storage Duration'] =='50 hr')]

RODeO_summary_outputs_cal_offgrid = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Northern California') & (RODeO_summary_outputs['Grid Case'] == 'Off-Grid') & (RODeO_summary_outputs['Storage Duration'] =='500 hr')]
RODeO_summary_outputs_cal_gridconnected = RODeO_summary_outputs.loc[(RODeO_summary_outputs['Site Name']=='Northern California') & (RODeO_summary_outputs['Grid Case'] == 'Grid-Connected') & (RODeO_summary_outputs['Storage Duration'] =='50 hr')]


RODeO_summary_outputs_optstorage = pd.concat([RODeO_summary_outputs_gulf_offgrid,RODeO_summary_outputs_gulf_gridconnected,RODeO_summary_outputs_atlantic_offgrid,RODeO_summary_outputs_atlantic_gridconnected,\
                                              RODeO_summary_outputs_newyork_offgrid,RODeO_summary_outputs_newyork_gridconnected,RODeO_summary_outputs_cal_offgrid,RODeO_summary_outputs_cal_gridconnected])


RODeO_summary_outputs_optstorage = RODeO_summary_outputs_optstorage.reset_index().drop(columns = 'index')

RODeO_summary_outputs_optstorage = RODeO_summary_outputs_optstorage.rename(columns = {'Energy charge (US$/kg)':'Grid energy (US$/kg)','Input CAPEX (US$/kg)':'Electrolyzer CAPEX (US$/kg)','Input FOM (US$/kg)':'Electrolyzer FOM (US$/kg)','Input VOM (US$/kg)':'Electrolyzer VOM (US$/kg)','Renewable capital cost (US$/kg)':'Renewable CAPEX (US$/kg)'})



# Global Plot Settings
font = 'Arial'
title_size = 10
axis_label_size = 10
legend_size = 6
tick_size = 10
resolution = 150

# Loop iteration though scenarios

for i in RODeO_summary_outputs_optstorage.index:
    
    # Scenario of Interest    
    grid_case = RODeO_summary_outputs_optstorage.iloc[i, 127]
    storage_case = RODeO_summary_outputs_optstorage.iloc[i, 126]
    location_case = RODeO_summary_outputs_optstorage.iloc[i, 123]
 
    file_name = location_case + '_'  + storage_case + '_'+grid_case 
    scenario_title = location_case + ', '  + storage_case + ', '+grid_case
    # Database Discretization

    RODeO_scenario = RODeO_summary_outputs_optstorage[['Year', 'Site Name','Storage Duration', 'Grid Case', 'Grid energy (US$/kg)', 'Storage & compression cost (US$/kg)', 'Electrolyzer CAPEX (US$/kg)', 'Electrolyzer FOM (US$/kg)', 'Electrolyzer VOM (US$/kg)','Renewable CAPEX (US$/kg)', 'Renewable FOM (US$/kg)', 'Taxes (US$/kg)', 'Product NPV cost (US$/kg)']].copy()
    #
    RODeO_scenario = RODeO_scenario[RODeO_scenario['Site Name'].isin([location_case])]
    RODeO_scenario = RODeO_scenario[RODeO_scenario['Storage Duration'].isin([storage_case])]
    RODeO_scenario = RODeO_scenario[RODeO_scenario['Grid Case'].isin([grid_case])]


    # Draw Plot and Annotate
    fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)

    columns = RODeO_scenario.columns[4:]
    lab = columns.values.tolist()
    count = 0
    for i in lab:
        lab[count] = i.replace(' (US$/kg)', '')
        count = count + 1

    # Manipulation data
    x  = RODeO_scenario['Year'].values.tolist()

    Energy_cost = RODeO_scenario['Grid energy (US$/kg)'].values.tolist()
    Stor_Comp = RODeO_scenario['Storage & compression cost (US$/kg)'].values.tolist()
    Elec_cap_cost = RODeO_scenario['Electrolyzer CAPEX (US$/kg)'].values.tolist()
    Elec_FOM = RODeO_scenario['Electrolyzer FOM (US$/kg)'].values.tolist()
    Elec_VOM = RODeO_scenario['Electrolyzer VOM (US$/kg)'].values.tolist()
    Renew_cap_cost = RODeO_scenario['Renewable CAPEX (US$/kg)'].values.tolist()
    Renew_FOM = RODeO_scenario['Renewable FOM (US$/kg)'].values.tolist()
    Taxes = RODeO_scenario['Taxes (US$/kg)'].values.tolist()
    
    y = np.vstack([Energy_cost, Stor_Comp, Elec_cap_cost, Elec_FOM, Elec_VOM, Renew_cap_cost, Renew_FOM, Taxes])
    # 
    # Plot for each column
    labels = columns.values.tolist()
    ax = plt.gca()
    ax.stackplot(x, y, labels=lab)
    
    # Decorations
    ax.set_title(scenario_title, fontsize=title_size)
    ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
    ax.set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size)
    ax.set_xlabel('Year', fontname = font, fontsize = axis_label_size)
    
    if location_case == 'Central Atlantic':
        ax.set_ylim([0,16])
    else:
        ax.set_ylim([0,10])
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
    #ax2 = ax.twinx()
    #ax2.set_ylim([0,10])
    plt.xlim(x[0], x[-1])
    plt.tight_layout()
    plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + file_name + '.png',pad_inches = 0.1)
    plt.close(fig = None)
    


''' Left over code to repurpose for other figures'''

'''
Energy_cost = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Energy charge (US$/kg)')
Stor_Comp = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Storage & compression cost (US$/kg)')
Elec_cap_cost = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Input CAPEX (US$/kg)')
Elec_FOM = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Input FOM (US$/kg)')
Renew_cap_cost = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Renewable capital cost (US$/kg)')
Renew_FOM = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Renewable FOM (US$/kg)')
Taxes = SCS_data.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Taxes (US$/kg)')
'''

# Scenario of Interest
'''
grid_case = 'Grid-Only'
EC_case = 'EC-Low'
PV_case = 'PV-None'
storage_case = 'Geo-Storage'
storage_dur = '1000'
demand_case = 'Statewide'
'''