# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:27:41 2021

@author: erezni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.colors as mcolors
import plotly.express as px

# Initialization and Global Settings
#Specify directory name
output_directory = 'examples/H2_Analysis/Phase1B/Fin_sum'
plot_directory = 'examples/H2_Analysis/Phase1B/Plots'
#plot_subdirectory = 'Stacked_Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Retail price of interest ['retail-flat','wholesale']
retail_string = 'retail-flat'
plot_subdirectory = 'Stacked_Plots_' + retail_string

# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary  = financial_summary.loc[(financial_summary['Grid case']!='grid-only-wholesale') & (financial_summary['Grid case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    financial_summary = financial_summary.loc[(financial_summary['Grid case']!='grid-only-retail-flat') & (financial_summary['Grid case']!='hybrid-grid-retail-flat')]

# Loop iteration though scenarios

# Note that if you set this to 'Distributed', you must only run 'off-grid' for grid-cases
electrolysis_cases = [
                    'Centralized',
                    #'Distributed'
                    ]

grid_cases = [
    'grid-only-'+retail_string,
    'hybrid-grid-'+retail_string,
    'off-grid'
    ]

    # Select hybrids case 'Wind' or 'Wind+PV+bat'
hybrids_cases = [
            'Wind',
            'Wind+PV+bat',
                ]

locations = [
        'IN',
        'TX',
        'IA',
        'MS',
        #'WY'
        ]

location_strings = [
        'Indiana',
        'Texas',
        'Iowa',
        'Mississippi',
        #'Wyoming'
]

# True if you want to plot quad of four locations. Must select four locations above
plot_quad = True

for electrolysis_case in electrolysis_cases:
    for grid_case in grid_cases:
        for hybrids_case in hybrids_cases:

            if grid_case =='grid-only-'+retail_string:
                renewables_case = 'No-ren'
            else:
                renewables_case = hybrids_case
            
            fin_sum_usecase = financial_summary.loc[(financial_summary['Electrolysis case']==electrolysis_case) & (financial_summary['Grid case']==grid_case)&(financial_summary['Renewables case']==renewables_case)]
            
            #Calculate policy savings
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH: Base policy savings ($/kg)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='base','LCOH ($/kg)'].values
            
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH: Max policy savings ($/kg)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','LCOH ($/kg)'].values
                
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Base policy savings ($/tonne)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='base','Steel price: Total ($/tonne)'].values
            
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Max policy savings ($/tonne)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Steel price: Total ($/tonne)'].values
            
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Base policy savings ($/kg)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='base','Ammonia price: Total ($/kg)'].values
            
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Max policy savings ($/kg)'] = \
                fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Ammonia price: Total ($/kg)'].values
            
            fin_sum_usecase = fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy']
            
            labels  = pd.unique(fin_sum_usecase['ATB Year']).astype(int).astype(str).tolist()    
            
    #----------------------------- Preallocate dictionaries with LCOH categories and plot each one--------------------------------------------------------------------------------------------
            storage_compression_cost = {}
            elec_cap_cost = {}
            desal_cap_cost = {}
            elec_FOM = {}
            elec_capex_FOM={}
            desal_FOM = {}
            elec_VOM= {}
            wind_cap_cost = {}
            wind_FOM ={}
            solar_cap_cost = {}
            solar_FOM ={}
            bat_cap_cost = {}
            bat_FOM ={}
            wind_capex_FOM = {}
            solar_capex_FOM={}
            bat_capex_FOM={}
            grid_electricity = {}
            water_consumption = {}
            desal_and_water = {}
            taxes_and_financial = {}
            bulk_transmission = {}
            lcoh_nopolicy={}
            lcoh_base_policy_savings={}
            lcoh_max_policy_savings = {}
            
            for site in locations:
                storage_compression_cost[site]=np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Compression & storage ($/kg)'].values.tolist())
                elec_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Electrolyzer CAPEX ($/kg)'].values.tolist())
                desal_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Desalination CAPEX ($/kg)'].values.tolist())
                elec_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site, 'LCOH: Electrolyzer FOM ($/kg)'].values.tolist())
                desal_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH:Desalination FOM ($/kg)'].values.tolist())
                elec_VOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site, 'LCOH: Electrolyzer VOM ($/kg)'].values.tolist())
                wind_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Wind Plant CAPEX ($/kg)'].values.tolist())
                wind_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Wind Plant FOM ($/kg)'].values.tolist())
                solar_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Solar Plant CAPEX ($/kg)'].values.tolist())
                solar_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Solar Plant FOM ($/kg)'].values.tolist())
                bat_cap_cost[site]=np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Battery Storage CAPEX ($/kg)'].values.tolist())
                bat_FOM[site]=np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Battery Storage FOM ($/kg)'].values.tolist())
                grid_electricity[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Grid electricity ($/kg)'].values.tolist())
                water_consumption[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Water consumption ($/kg)'].values.tolist())
                desal_and_water[site] = desal_cap_cost[site]+desal_FOM[site]+water_consumption[site]
                taxes_and_financial[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Taxes and Finances ($/kg)'].values.tolist())
                bulk_transmission[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Bulk H2 Transmission ($/kg)'].values.tolist())
                elec_capex_FOM[site]=elec_cap_cost[site]+elec_FOM[site]
                wind_capex_FOM[site]=wind_cap_cost[site]+wind_FOM[site]
                solar_capex_FOM[site]=solar_cap_cost[site]+solar_FOM[site]
                bat_capex_FOM[site]=bat_cap_cost[site]+bat_FOM[site]
                
                # Plot individual LCOH Plots
                resolution = 150
                width = 0.5
                fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
                ax.bar(labels,storage_compression_cost[site], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
                barbottom=storage_compression_cost[site]
                ax.bar(labels,elec_capex_FOM[site],width,bottom=barbottom,label = 'Electrolyzer CAPEX & FOM',edgecolor='teal',color='teal')
                barbottom=barbottom+elec_capex_FOM[site]
                ax.bar(labels,elec_VOM[site],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
                barbottom=barbottom+elec_VOM[site]
                ax.bar(labels,desal_and_water[site],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen', color='forestgreen')
                barbottom=barbottom+desal_and_water[site]
                ax.bar(labels,bulk_transmission[site],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
                barbottom = barbottom+bulk_transmission[site]
    
                if grid_case == 'off-grid' or 'hybrid-grid-'+retail_string:
                    ax.bar(labels,wind_capex_FOM[site],width,bottom=barbottom,label = 'Wind CAPEX & FOM',edgecolor='deepskyblue',color='deepskyblue')
                    barbottom=barbottom+wind_capex_FOM[site]
                    if hybrids_case =='Wind+PV+bat':
                        ax.bar(labels,solar_capex_FOM[site],width,bottom=barbottom,label = 'Solar CAPEX & FOM',edgecolor = 'gold', color='gold')
                        barbottom=barbottom+solar_capex_FOM[site]
                        ax.bar(labels,bat_capex_FOM[site],width,bottom=barbottom,label = 'Battery CAPEX & FOM',edgecolor = 'palevioletred', color='palevioletred')
                        barbottom=barbottom+bat_capex_FOM[site]

                if grid_case == 'grid-only-'+retail_string  or grid_case == 'hybrid-grid-'+retail_string:
                    ax.bar(labels,grid_electricity[site],width,bottom=barbottom,label = 'Grid Electricity',edgecolor = 'darkorange',color='darkorange')
                    barbottom = barbottom+grid_electricity[site]

                ax.bar(labels,taxes_and_financial[site],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor = 'goldenrod', color='goldenrod')
                barbottom=barbottom+taxes_and_financial[site]

                # Plot policy savings
                lcoh_nopolicy[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH ($/kg)'].values.tolist())
                lcoh_base_policy_savings[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Base policy savings ($/kg)'].values.tolist())
                lcoh_max_policy_savings[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Max policy savings ($/kg)'].values.tolist())

                ax.plot([0,1,2,3], lcoh_nopolicy[site]-lcoh_base_policy_savings[site], color='black', marker='o', linestyle='none', markersize=4,label='Base Policy')
                ax.plot([0,1,2,3], lcoh_nopolicy[site]-lcoh_max_policy_savings[site], color='dimgray', marker='s', linestyle='none', markersize=4,label='Max Policy')
                arrow_top = np.zeros(len(labels))
                ax.errorbar(labels,lcoh_nopolicy[site],yerr=[arrow_top,arrow_top], fmt='none',elinewidth=1,ecolor='black',capsize=10,markeredgewidth=1.25) 
                for j in range(len(labels)): 
                    ax.arrow(j,lcoh_nopolicy[site][j],0,-1*lcoh_base_policy_savings[site][j],head_width=0.1,head_length=0.4,length_includes_head=True,color='black')
                    ax.arrow(j,lcoh_nopolicy[site][j]-lcoh_base_policy_savings[site][j],0,-1*(lcoh_max_policy_savings[site][j]-lcoh_base_policy_savings[site][j]),head_width=0.1,head_length=0.25,length_includes_head=True,color='dimgray')

                ax.axhline(y=0, color='k', linestyle='-',linewidth=1.5)

                scenario_title = site + ', ' + electrolysis_case + ', ' + grid_case
                file_name = site + '_' + electrolysis_case + '_' + grid_case
                
                # Global Plot Settings
                font = 'Arial'
                title_size = 10
                axis_label_size = 10
                legend_size = 8
                tick_size = 10

                
                # Decorations
                ax.set_title(scenario_title, fontsize=title_size)

                ax.set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size)
                ax.set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size)
                ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7},loc='upper right')
                max_y = np.max(barbottom)
                ax.set_ylim([-2,16])
                ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
                ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
                plt.tight_layout()
                plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'single_lcoh_barchart_'+file_name +'_'+ retail_string+'_'+renewables_case+'.png',pad_inches = 0.1)
                plt.close(fig = None)
                
    #-------------------------- Plot LCOH quad-plot-----------------------------------------------------------------------------------------------------------------------
            if plot_quad:

                title_size_quad = 16
                axis_label_size_quad = 14
                legend_size_quad = 9
                tick_size = 10
                resolution = 150
                width = 0.5
                fig, ax = plt.subplots(2,2,figsize=(12,10), dpi= resolution)
                # Loop through the four subplots
                for axi,site in enumerate(locations):
                    if axi <= 1:
                        axi1 = 0
                    elif axi > 1:
                        axi1=1
                    axi2 = axi % 2

                    ax[axi1,axi2].bar(labels,storage_compression_cost[site], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
                    barbottom=storage_compression_cost[site]
                    ax[axi1,axi2].bar(labels,elec_capex_FOM[site],width,bottom=barbottom,label = 'Electrolyzer CAPEX & FOM',edgecolor='teal',color='teal')
                    barbottom=barbottom+elec_capex_FOM[site]
                    ax[axi1,axi2].bar(labels,elec_VOM[site],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
                    barbottom=barbottom+elec_VOM[site]
                    ax[axi1,axi2].bar(labels,desal_and_water[site],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen', color='forestgreen')
                    barbottom=barbottom+desal_and_water[site]
                    ax[axi1,axi2].bar(labels,bulk_transmission[site],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
                    barbottom = barbottom+bulk_transmission[site]

                    if grid_case == 'off-grid' or 'hybrid-grid-'+retail_string:
                        ax[axi1,axi2].bar(labels,wind_capex_FOM[site],width,bottom=barbottom,label = 'Wind CAPEX & FOM',edgecolor='deepskyblue',color='deepskyblue')
                        barbottom=barbottom+wind_capex_FOM[site]
                        if hybrids_case =='Wind+PV+bat':
                            ax[axi1,axi2].bar(labels,solar_capex_FOM[site],width,bottom=barbottom,label = 'Solar CAPEX & FOM',edgecolor = 'gold', color='gold')
                            barbottom=barbottom+solar_capex_FOM[site]
                            ax[axi1,axi2].bar(labels,bat_capex_FOM[site],width,bottom=barbottom,label = 'Battery CAPEX & FOM',edgecolor = 'palevioletred', color='palevioletred')
                            barbottom=barbottom+bat_capex_FOM[site]

                    if grid_case == 'grid-only-'+retail_string  or grid_case == 'hybrid-grid-'+retail_string:
                        ax[axi1,axi2].bar(labels,grid_electricity[site],width,bottom=barbottom,label = 'Grid Electricity',edgecolor = 'darkorange',color='darkorange')
                        barbottom = barbottom+grid_electricity[site]

                    ax[axi1,axi2].bar(labels,taxes_and_financial[site],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor = 'goldenrod', color='goldenrod')
                    barbottom=barbottom+taxes_and_financial[site]

                    # Plot policy savings
                    ax[axi1,axi2].plot([0,1,2,3], lcoh_nopolicy[site]-lcoh_base_policy_savings[site], color='black', marker='o', linestyle='none', markersize=4,label='Base Policy')
                    ax[axi1,axi2].plot([0,1,2,3], lcoh_nopolicy[site]-lcoh_max_policy_savings[site], color='dimgray', marker='s', linestyle='none', markersize=4,label='Max Policy')
                    arrow_top = np.zeros(len(labels))
                    ax[axi1,axi2].errorbar(labels,lcoh_nopolicy[site],yerr=[arrow_top,arrow_top], fmt='none',elinewidth=1,ecolor='black',capsize=10,markeredgewidth=1.25) 
                    for j in range(len(labels)): 
                        ax[axi1,axi2].arrow(j,lcoh_nopolicy[site][j],0,-1*lcoh_base_policy_savings[site][j],head_width=0.1,head_length=0.4,length_includes_head=True,color='black')
                        ax[axi1,axi2].arrow(j,lcoh_nopolicy[site][j]-lcoh_base_policy_savings[site][j],0,-1*(lcoh_max_policy_savings[site][j]-lcoh_base_policy_savings[site][j]),head_width=0.1,head_length=0.25,length_includes_head=True,color='dimgray')

                    ax[axi1,axi2].axhline(y=0, color='k', linestyle='-',linewidth=1.5)
                    ax[axi1,axi2].set_title(location_strings[axi], fontsize=title_size_quad)
                    ax[axi1,axi2].set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size_quad)
                    if axi > 1:
                        ax[axi1,axi2].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
                    if axi ==1:
                        ax[axi1,axi2].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
                    max_y = np.max(barbottom)
                    ax[axi1,axi2].set_ylim([-2,16])
                    ax[axi1,axi2].tick_params(axis = 'y',labelsize = 12,direction = 'in')
                    ax[axi1,axi2].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
                plt.tight_layout()
                file_name = electrolysis_case + '_' + grid_case
                plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'quad_lcoh_barchart_'+file_name + '_'+ retail_string+'_'+renewables_case+'.png',pad_inches = 0.1)
                plt.close(fig = None)
