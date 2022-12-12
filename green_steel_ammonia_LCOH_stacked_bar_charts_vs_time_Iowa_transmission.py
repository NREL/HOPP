# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:27:41 2021

@author: ktopolsk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.colors as mcolors
import plotly.express as px

# Initialization and Global Settings
#Specify directory name
output_directory = 'examples/H2_Analysis/RODeO_financial_summary_Iowa_transmission_LRC'
plot_directory = 'examples/H2_Analysis/Plots/'
#plot_subdirectory = 'Stacked_Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Retail price of interest ['retail-flat','wholesale']
retail_string = 'retail-flat'
plot_subdirectory = 'Stacked_Plots_Iowa_trans_LRC'


# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary  = financial_summary.loc[(financial_summary['Grid Case']!='grid-only-wholesale') & (financial_summary['Grid Case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    financial_summary = financial_summary.loc[(financial_summary['Grid Case']!='grid-only-retail-flat') & (financial_summary['Grid Case']!='hybrid-grid-retail-flat')]

# Loop iteration though scenarios

# Note that if you set this to 'Distributed', you must only run 'off-grid' for grid-cases
electrolysis_cases = [
                    #'Centralized',
                    'Distributed'
                    ]

grid_cases = [
    #'grid-only-'+retail_string,
    #'hybrid-grid-'+retail_string,
    'off-grid'
    ]

locations = [
        #'IN',
        #'TX',
        'IA',
        #'MS'
        ]

for electrolysis_case in electrolysis_cases:
    for grid_case in grid_cases:
        #electrolysis_case = 'Centralized'
        #grid_case = 'off-grid'
        
        fin_sum_usecase = financial_summary.loc[(financial_summary['Electrolysis case']==electrolysis_case) & (financial_summary['Grid Case']==grid_case)]
        
        #Calculate policy savings
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH: Policy savings ($/kg)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','LCOH ($/kg)'].values
            
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Policy savings ($/tonne)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Steel price: Total ($/tonne)'].values
        
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Policy savings ($/kg)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Ammonia price: Total ($/kg)'].values
        
        fin_sum_usecase = fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy']
        
        labels  = pd.unique(fin_sum_usecase['ATB Year']).astype(int).astype(str).tolist()    
        
#----------------------------- Preallocate dictionaries with LCOH categories and plot each one--------------------------------------------------------------------------------------------
        storage_compression_cost = {}
        elec_cap_cost = {}
        desal_cap_cost = {}
        elec_FOM = {}
        desal_FOM = {}
        elec_VOM= {}
        renew_cap_cost = {}
        renew_FOM ={}
        grid_electricity = {}
        water_consumption = {}
        desal_and_water = {}
        taxes_and_financial = {}
        bulk_transmission = {}
        policy_savings_h2 = {}
        
        for site in locations:
            #site = 'IN'
            storage_compression_cost[site]=np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Compression & storage ($/kg)'].values.tolist())
            elec_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Electrolyzer CAPEX ($/kg)'].values.tolist())
            desal_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Desalination CAPEX ($/kg)'].values.tolist())
            elec_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site, 'LCOH: Electrolyzer FOM ($/kg)'].values.tolist())
            desal_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH:Desalination FOM ($/kg)'].values.tolist())
            elec_VOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site, 'LCOH: Electrolyzer VOM ($/kg)'].values.tolist())
            renew_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Renewable CAPEX ($/kg)'].values.tolist())
            renew_FOM[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Renewable FOM ($/kg)'].values.tolist())
            grid_electricity[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Grid Electricity ($/kg)'].values.tolist())
            water_consumption[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Water consumption ($/kg)'].values.tolist())
            desal_and_water[site] = desal_cap_cost[site]+desal_FOM[site]+water_consumption[site]
            taxes_and_financial[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Taxes and Finances ($/kg)'].values.tolist())
            bulk_transmission[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Bulk H2 Transmission ($/kg)'].values.tolist())
            policy_savings_h2[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'LCOH: Policy savings ($/kg)'])
            
            # Plot individual LCOH Plots
            
            resolution = 150
            
            width = 0.5
            #fig, ax = plt.subplots()
            fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
            ax.bar(labels,storage_compression_cost[site], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
            barbottom=storage_compression_cost[site]
            #ax.bar(labels,compression_cost,width,bottom=storage_cost,label = 'Compression')
            ax.bar(labels,elec_cap_cost[site],width,bottom=barbottom,label = 'Electrolyzer CAPEX',edgecolor='teal',color='teal')
            barbottom=barbottom+elec_cap_cost[site]
            ax.bar(labels,elec_FOM[site],width,bottom = barbottom,label = 'Electrolyzer FOM',edgecolor='cadetblue',color='cadetblue')
            barbottom=barbottom+elec_FOM[site]
            ax.bar(labels,elec_VOM[site],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
            barbottom=barbottom+elec_VOM[site]
            ax.bar(labels,desal_and_water[site],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen', color='forestgreen')
            barbottom=barbottom+desal_and_water[site]
            ax.bar(labels,bulk_transmission[site],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
            barbottom = barbottom+bulk_transmission[site]
            ax.bar(labels,taxes_and_financial[site],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor = 'gold', color='gold')
            barbottom=barbottom+taxes_and_financial[site]
 
            if grid_case == 'off-grid' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
                ax.bar(labels,renew_cap_cost[site],width,bottom=barbottom,label = 'Renewable CAPEX',edgecolor='goldenrod',color='goldenrod')
                barbottom=barbottom+renew_cap_cost[site]
                ax.bar(labels,renew_FOM[site],width,bottom=barbottom,label = 'Renewable FOM',edgecolor = 'darkgoldenrod', color='darkgoldenrod')
                barbottom=barbottom+renew_FOM[site]

            if grid_case == 'grid-only-retail-flat' or grid_case =='grid-only-retail-peaks' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
                ax.bar(labels,grid_electricity[site],width,bottom=barbottom,label = 'Grid Electricity',edgecolor = 'darkorange',color='darkorange')
                barbottom = barbottom+grid_electricity[site]

            barbottom_policy = barbottom  - policy_savings_h2[site]
            ax.bar(labels,policy_savings_h2[site],width,bottom=barbottom_policy,label='Policy savings',color = 'white',edgecolor = 'goldenrod',alpha = 0.35,hatch='...')    

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
            ax.set_ylim([0,10])
            ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
            ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
            #ax2 = ax.twinx()
            #ax2.set_ylim([0,10])
            #plt.xlim(x[0], x[-1])
            plt.tight_layout()
            plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'single_lcoh_barchart_'+file_name +'_'+ retail_string+'_IA_trans_LRC.png',pad_inches = 0.1)
            plt.close(fig = None)
            
#-------------------------- Plot LCOH quad-plot-----------------------------------------------------------------------------------------------------------------------
        # width = 0.5
        # title_size_quad = 16
        # axis_label_size_quad = 14
        # legend_size_quad = 9
        # tick_size = 10
        # resolution = 150
        # fig, ax = plt.subplots(2,2,figsize=(12,10), dpi= resolution)
        # # Top left
        # ax[0,0].bar(labels,storage_compression_cost['IN'], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
        # barbottom=storage_compression_cost['IN']
        # ax[0,0].bar(labels,elec_cap_cost['IN'],width,bottom=barbottom,label = 'Electrolyzer CAPEX',edgecolor='teal',color='teal')
        # barbottom=barbottom+elec_cap_cost['IN']
        # ax[0,0].bar(labels,elec_FOM['IN'],width,bottom = barbottom,label = 'Electrolyzer FOM',edgecolor='cadetblue',color='cadetblue')
        # barbottom=barbottom+elec_FOM['IN']
        # ax[0,0].bar(labels,elec_VOM['IN'],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
        # barbottom=barbottom+elec_VOM['IN']
        # ax[0,0].bar(labels,desal_and_water['IN'],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen',color='forestgreen')
        # barbottom=barbottom+desal_and_water['IN']
        # ax[0,0].bar(labels,bulk_transmission['IN'],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
        # barbottom = barbottom+bulk_transmission['IN']
        # ax[0,0].bar(labels,taxes_and_financial['IN'],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor='gold',color='gold')
        # barbottom=barbottom+taxes_and_financial['IN']
        # if grid_case == 'off-grid' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[0,0].bar(labels,renew_cap_cost['IN'],width,bottom=barbottom,label = 'Renewable CAPEX',edgecolor='goldenrod',color='goldenrod')
        #     barbottom=barbottom+renew_cap_cost['IN']
        #     ax[0,0].bar(labels,renew_FOM['IN'],width,bottom=barbottom,label = 'Renewable FOM',edgecolor='darkgoldenrod',color='darkgoldenrod')
        #     barbottom=barbottom+renew_FOM['IN']
        # if grid_case == 'grid-only-retail-flat' or grid_case =='grid-only-retail-peaks' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[0,0].bar(labels,grid_electricity['IN'],width,bottom=barbottom,label = 'Grid Electricity',edgecolor='darkorange',color='darkorange')
        #     barbottom = barbottom+grid_electricity['IN']
        # barbottom_policy = barbottom  - policy_savings_h2['IN']
        # ax[0,0].bar(labels,policy_savings_h2['IN'],width,bottom=barbottom_policy,label='Policy savings',color = 'white',edgecolor = 'goldenrod',alpha = 0.4,hatch='..')
        # ax[0,0].set_title('Indiana', fontsize=title_size_quad)
        # ax[0,0].set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        # #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        # ax[0,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        # max_y = np.max(barbottom)
        # ax[0,0].set_ylim([0,10])
        # #ax[0,0].set_ylim([0,1.4*max_y])
        # ax[0,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        # ax[0,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        
        # # Top right
        # ax[0,1].bar(labels,storage_compression_cost['IA'], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
        # barbottom=storage_compression_cost['IA']
        # ax[0,1].bar(labels,elec_cap_cost['IA'],width,bottom=barbottom,label = 'Electrolyzer CAPEX',edgecolor='teal',color='teal')
        # barbottom=barbottom+elec_cap_cost['IA']
        # ax[0,1].bar(labels,elec_FOM['IA'],width,bottom = barbottom,label = 'Electrolyzer FOM',edgecolor='cadetblue',color='cadetblue')
        # barbottom=barbottom+elec_FOM['IA']
        # ax[0,1].bar(labels,elec_VOM['IA'],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
        # barbottom=barbottom+elec_VOM['IA']
        # ax[0,1].bar(labels,desal_and_water['IA'],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen',color='forestgreen')
        # barbottom=barbottom+desal_and_water['IA']
        # ax[0,1].bar(labels,bulk_transmission['IA'],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
        # barbottom = barbottom+bulk_transmission['IA']
        # ax[0,1].bar(labels,taxes_and_financial['IA'],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor='gold',color='gold')
        # barbottom=barbottom+taxes_and_financial['IA']
        # if grid_case == 'off-grid' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[0,1].bar(labels,renew_cap_cost['IA'],width,bottom=barbottom,label = 'Renewable CAPEX',edgecolor='goldenrod',color='goldenrod')
        #     barbottom=barbottom+renew_cap_cost['IA']
        #     ax[0,1].bar(labels,renew_FOM['IA'],width,bottom=barbottom,label = 'Renewable FOM',edgecolor='darkgoldenrod',color='darkgoldenrod')
        #     barbottom=barbottom+renew_FOM['IA']
        # if grid_case == 'grid-only-retail-flat' or grid_case =='grid-only-retail-peaks' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[0,1].bar(labels,grid_electricity['IA'],width,bottom=barbottom,label = 'Grid Electricity',edgecolor='darkorange',color='darkorange')
        #     barbottom = barbottom+grid_electricity['IA']
        # barbottom_policy = barbottom  - policy_savings_h2['IA']
        # ax[0,1].bar(labels,policy_savings_h2['IA'],width,bottom=barbottom_policy,label='Policy savings',color = 'white',edgecolor = 'goldenrod',alpha = 0.4,hatch='..')
        # ax[0,1].set_title('Iowa', fontsize=title_size_quad)
        # ax[0,1].set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        # #ax[0,1].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        # ax[0,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        # max_y = np.max(barbottom)
        # ax[0,1].set_ylim([0,10])
        # #ax[0,0].set_ylim([0,1.4*max_y])
        # ax[0,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        # ax[0,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)   
        
        # # Bottom left
        # ax[1,0].bar(labels,storage_compression_cost['TX'], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
        # barbottom=storage_compression_cost['TX']
        # ax[1,0].bar(labels,elec_cap_cost['TX'],width,bottom=barbottom,label = 'Electrolyzer CAPEX',edgecolor='teal',color='teal')
        # barbottom=barbottom+elec_cap_cost['TX']
        # ax[1,0].bar(labels,elec_FOM['TX'],width,bottom = barbottom,label = 'Electrolyzer FOM',edgecolor='cadetblue',color='cadetblue')
        # barbottom=barbottom+elec_FOM['TX']
        # ax[1,0].bar(labels,elec_VOM['TX'],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
        # barbottom=barbottom+elec_VOM['TX']
        # ax[1,0].bar(labels,desal_and_water['TX'],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen',color='forestgreen')
        # barbottom=barbottom+desal_and_water['TX']
        # ax[1,0].bar(labels,bulk_transmission['TX'],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
        # barbottom = barbottom+bulk_transmission['TX']
        # ax[1,0].bar(labels,taxes_and_financial['TX'],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor='gold',color='gold')
        # barbottom=barbottom+taxes_and_financial['TX']
        # if grid_case == 'off-grid' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[1,0].bar(labels,renew_cap_cost['TX'],width,bottom=barbottom,label = 'Renewable CAPEX',edgecolor='goldenrod',color='goldenrod')
        #     barbottom=barbottom+renew_cap_cost['TX']
        #     ax[1,0].bar(labels,renew_FOM['TX'],width,bottom=barbottom,label = 'Renewable FOM',edgecolor='darkgoldenrod',color='darkgoldenrod')
        #     barbottom=barbottom+renew_FOM['TX']
        # if grid_case == 'grid-only-retail-flat' or grid_case =='grid-only-retail-peaks' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[1,0].bar(labels,grid_electricity['TX'],width,bottom=barbottom,label = 'Grid Electricity',edgecolor='darkorange',color='darkorange')
        #     barbottom = barbottom+grid_electricity['TX']
        # barbottom_policy = barbottom  - policy_savings_h2['TX']
        # ax[1,0].bar(labels,policy_savings_h2['TX'],width,bottom=barbottom_policy,label='Policy savings',color = 'white',edgecolor = 'goldenrod',alpha = 0.4,hatch='..')
        # ax[1,0].set_title('Texas', fontsize=title_size_quad)
        # ax[1,0].set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        # ax[1,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        # ax[1,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        # max_y = np.max(barbottom)
        # ax[1,0].set_ylim([0,10])
        # #ax[0,0].set_ylim([0,1.4*max_y])
        # ax[1,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        # ax[1,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
            
        # # Bottom right
        # ax[1,1].bar(labels,storage_compression_cost['MS'], width, label='Storage & compression',edgecolor='darkslategray',color='darkslategray')
        # barbottom=storage_compression_cost['MS']
        # ax[1,1].bar(labels,elec_cap_cost['MS'],width,bottom=barbottom,label = 'Electrolyzer CAPEX',edgecolor='teal',color='teal')
        # barbottom=barbottom+elec_cap_cost['MS']
        # ax[1,1].bar(labels,elec_FOM['MS'],width,bottom = barbottom,label = 'Electrolyzer FOM',edgecolor='cadetblue',color='cadetblue')
        # barbottom=barbottom+elec_FOM['MS']
        # ax[1,1].bar(labels,elec_VOM['MS'],width,bottom=barbottom,label = 'Electrolyzer VOM',edgecolor='darkturquoise',color='darkturquoise')
        # barbottom=barbottom+elec_VOM['MS']
        # ax[1,1].bar(labels,desal_and_water['MS'],width,bottom=barbottom,label='Desalination and water',edgecolor='forestgreen',color='forestgreen')
        # barbottom=barbottom+desal_and_water['MS']
        # ax[1,1].bar(labels,bulk_transmission['MS'],width,bottom=barbottom,label='Bulk H2 Transmission',edgecolor='crimson',color='crimson')
        # barbottom = barbottom+bulk_transmission['MS']
        # ax[1,1].bar(labels,taxes_and_financial['MS'],width,bottom=barbottom,label = 'Taxes and Finances',edgecolor='gold',color='gold')
        # barbottom=barbottom+taxes_and_financial['MS']
        # if grid_case == 'off-grid' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[1,1].bar(labels,renew_cap_cost['MS'],width,bottom=barbottom,label = 'Renewable CAPEX',edgecolor='goldenrod',color='goldenrod')
        #     barbottom=barbottom+renew_cap_cost['MS']
        #     ax[1,1].bar(labels,renew_FOM['MS'],width,bottom=barbottom,label = 'Renewable FOM',edgecolor='darkgoldenrod',color='darkgoldenrod')
        #     barbottom=barbottom+renew_FOM['MS']
        # if grid_case == 'grid-only-retail-flat' or grid_case =='grid-only-retail-peaks' or grid_case == 'hybrid-grid-retail-flat' or grid_case =='hybrid-grid-retail-peaks':
        #     ax[1,1].bar(labels,grid_electricity['MS'],width,bottom=barbottom,label = 'Grid Electricity',edgecolor='darkorange',color='darkorange')
        #     barbottom = barbottom+grid_electricity['MS']
        # barbottom_policy = barbottom  - policy_savings_h2['MS']
        # ax[1,1].bar(labels,policy_savings_h2['MS'],width,bottom=barbottom_policy,label='Policy savings',color = 'white',edgecolor = 'goldenrod',alpha = 0.4,hatch='..')
        # ax[1,1].set_title('Mississippi', fontsize=title_size_quad)
        # ax[1,1].set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        # ax[1,1].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        # ax[1,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        # max_y = np.max(barbottom)
        # ax[1,1].set_ylim([0,10])
        # #ax[0,0].set_ylim([0,1.4*max_y])
        # ax[1,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        # ax[1,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        # plt.tight_layout()
        # file_name = electrolysis_case + '_' + grid_case
        # plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'quad_lcoh_barchart_'+file_name + '_'+ retail_string+'_IA_trans_LRC.png',pad_inches = 0.1)
        # plt.close(fig = None)