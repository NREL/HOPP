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
output_directory = 'examples/H2_Analysis/RODeO_financial_summary_results'
plot_directory = 'examples/H2_Analysis/Plots/'
#plot_subdirectory = 'Stacked_Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Retail price of interest ['retail-flat','wholesale']
retail_string = 'wholesale'
plot_subdirectory = 'Stacked_Plots_' + retail_string


# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary  = financial_summary.loc[(financial_summary['Grid Case']!='grid-only-wholesale') & (financial_summary['Grid Case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    financial_summary = financial_summary.loc[(financial_summary['Grid Case']!='grid-only-retail-flat') & (financial_summary['Grid Case']!='hybrid-grid-retail-flat')]

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

locations = [
        'IN',
        'TX',
        'IA',
        'MS'
        ]

for electrolysis_case in electrolysis_cases:
    for grid_case in grid_cases:
        #electrolysis_case = 'Centralized'
        #grid_case = 'hybrid-grid-'+retail_string
        
        fin_sum_usecase= financial_summary.loc[(financial_summary['Electrolysis case']==electrolysis_case) & (financial_summary['Grid Case']==grid_case)]
        
        labels  = pd.unique(fin_sum_usecase['ATB Year']).astype(int).astype(str).tolist()    
        
#----------------------------- Preallocate dictionaries with LCOH categories and plot each one--------------------------------------------------------------------------------------------
            
        cap_factor_no_policy = {}
        storage_duration_no_policy = {}

        cap_factor_with_policy = {}
        storage_duration_with_policy = {}
        
        for site in locations:
            #site = 'TX'
            cap_factor_no_policy[site]=np.array(fin_sum_usecase.loc[(fin_sum_usecase['Site']==site) & (fin_sum_usecase['Policy Option']=='no-policy'),'Electrolyzer CF (-)'].values.tolist())
            cap_factor_with_policy[site]=np.array(fin_sum_usecase.loc[(fin_sum_usecase['Site']==site) & (fin_sum_usecase['Policy Option']=='max'),'Electrolyzer CF (-)'].values.tolist())
            storage_duration_no_policy[site]=np.array(fin_sum_usecase.loc[(fin_sum_usecase['Site']==site) & (fin_sum_usecase['Policy Option']=='no-policy'),'Hydrogen storage duration (hr)'].values.tolist())
            storage_duration_with_policy[site]=np.array(fin_sum_usecase.loc[(fin_sum_usecase['Site']==site) & (fin_sum_usecase['Policy Option']=='max'),'Hydrogen storage duration (hr)'].values.tolist())
            
            # Plot individual LCOH Plots
            scenario_title = site + ', ' + electrolysis_case + ', ' + grid_case
            file_name = site + '_' + electrolysis_case + '_' + grid_case
            
            resolution = 150
            width = 0.5
            
            fig, ax = plt.subplots(1,1,sharex = 'all',figsize=(4.8,3.6), dpi= resolution)
            ax.plot(labels,cap_factor_no_policy[site],label = 'Capacity factor (No policy)',marker = 'd',markersize = 6,linestyle = 'none',color='darkblue')
            ax.plot(labels,cap_factor_with_policy[site],label = 'Capacity factor (With policy)',marker = '.',markersize = 6,linestyle = 'none', color='orange')
            
            ax2 = ax.twinx()
            ax2.plot(labels,storage_duration_no_policy[site],label = 'Storage duration (No policy)',marker = 's',markersize = 6,linestyle = 'none',color='darkslategrey')
            ax2.plot(labels,storage_duration_with_policy[site],label = 'With policy',marker = '.',markersize = 6,linestyle = 'none',color='goldenrod')
            
            # Decorations
            font = 'Arial'
            title_size = 10
            axis_label_size = 10
            tickfontsize = 10
            legend_size = 6
            tick_size = 10
            ax.set_title(scenario_title, fontsize=title_size)

            ax.set_ylabel('Capacity factor (-)', fontname = font, fontsize = axis_label_size)
            ax2.set_ylabel('Storage duration (hr)',fontname = font, fontsize = axis_label_size)
            ax.set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size)
            ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc=[0,0.05])#)
            ax2.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper right')

            ax.set_ylim([0,1.2])
            ax2.set_ylim([0,600])
            ax.tick_params(axis = 'y',labelsize = 12,direction = 'in')
            ax.tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
            ax2.tick_params(axis = 'y',labelsize = 12,direction = 'in')
            #ax[1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
            plt.tight_layout()
            plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'single_CF_storageduration_'+file_name +'_'+ retail_string+'.png',pad_inches = 0.1)
            plt.close(fig = None)
            
            
                    
#-------------------------- Plot LCOH quad-plot-----------------------------------------------------------------------------------------------------------------------
        width = 0.5
        font = 'Arial'
        title_size_quad = 16
        axis_label_size_quad = 14
        legend_size_quad = 9
        tick_size = 10
        resolution = 150
        marker_size = 10
        cf_max = 1.25
        storage_max = 1250
        fig, ax = plt.subplots(2,2,figsize=(12,10), dpi= resolution)
        
        # Top left
        site = 'IN'
        ax[0,0].plot(labels,cap_factor_no_policy[site],label = 'Capacity factor (No policy)',marker = 'd',markersize = marker_size,linestyle = 'none',color='darkblue')
        ax[0,0].plot(labels,cap_factor_with_policy[site],label = 'Capacity factor (With policy)',marker = '.',markersize = marker_size,linestyle = 'none', color='orange')
        ax2 = ax[0,0].twinx()
        ax2.plot(labels,storage_duration_no_policy[site],label = 'Storage duration (No policy)',marker = 's',markersize = marker_size,linestyle = 'none',color='darkslategrey')
        ax2.plot(labels,storage_duration_with_policy[site],label = 'With policy',marker = '.',markersize = marker_size,linestyle = 'none',color='goldenrod')
        ax[0,0].set_title('Indiana', fontsize=title_size_quad)
        ax[0,0].set_ylabel('Capacity factor (-)', fontname = font, fontsize = axis_label_size_quad)
        ax2.set_ylabel('Storage duration (hr)',fontname = font, fontsize = axis_label_size_quad)
        ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc='upper left')#[0.02,0.85])#)
        ax2.legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc=[0.01,0.85])#'upper right')
        ax[0,0].set_ylim([0,cf_max])
        ax2.set_ylim([0,storage_max])
        ax[0,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
        ax2.tick_params(axis = 'y',labelsize = 12,direction = 'in')

        # Top right
        site = 'IA'
        ax[0,1].plot(labels,cap_factor_no_policy[site],label = 'Capacity factor (No policy)',marker = 'd',markersize = marker_size,linestyle = 'none',color='darkblue')
        ax[0,1].plot(labels,cap_factor_with_policy[site],label = 'Capacity factor (With policy)',marker = '.',markersize = marker_size,linestyle = 'none', color='orange')
        ax2 = ax[0,1].twinx()
        ax2.plot(labels,storage_duration_no_policy[site],label = 'Storage duration (No policy)',marker = 's',markersize = marker_size,linestyle = 'none',color='darkslategrey')
        ax2.plot(labels,storage_duration_with_policy[site],label = 'With policy',marker = '.',markersize = marker_size,linestyle = 'none',color='goldenrod')
        ax[0,1].set_title('Iowa', fontsize=title_size_quad)
        ax[0,1].set_ylabel('Capacity factor (-)', fontname = font, fontsize = axis_label_size_quad)
        ax2.set_ylabel('Storage duration (hr)',fontname = font, fontsize = axis_label_size_quad)
        ax[0,1].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc='upper left')##)
        ax2.legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc=[0.01,0.85])
        ax[0,1].set_ylim([0,cf_max])
        ax2.set_ylim([0,storage_max])
        ax[0,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
        ax2.tick_params(axis = 'y',labelsize = 12,direction = 'in')
        
        # Bottom left
        site = 'TX'
        ax[1,0].plot(labels,cap_factor_no_policy[site],label = 'Capacity factor (No policy)',marker = 'd',markersize = marker_size,linestyle = 'none',color='darkblue')
        ax[1,0].plot(labels,cap_factor_with_policy[site],label = 'Capacity factor (With policy)',marker = '.',markersize = marker_size,linestyle = 'none', color='orange')
        ax2 = ax[1,0].twinx()
        ax2.plot(labels,storage_duration_no_policy[site],label = 'Storage duration (No policy)',marker = 's',markersize = marker_size,linestyle = 'none',color='darkslategrey')
        ax2.plot(labels,storage_duration_with_policy[site],label = 'With policy',marker = '.',markersize = marker_size,linestyle = 'none',color='goldenrod')
        ax[1,0].set_title('Texas', fontsize=title_size_quad)
        ax[1,0].set_ylabel('Capacity factor (-)', fontname = font, fontsize = axis_label_size_quad)
        ax2.set_ylabel('Storage duration (hr)',fontname = font, fontsize = axis_label_size_quad)
        ax[1,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc='upper left')#)
        ax2.legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc=[0.01,0.85])
        ax[1,0].set_ylim([0,cf_max])
        ax2.set_ylim([0,storage_max])
        ax[1,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
        ax2.tick_params(axis = 'y',labelsize = 12,direction = 'in')
        
        # Bottom right
        site = 'MS'
        ax[1,1].plot(labels,cap_factor_no_policy[site],label = 'Capacity factor (No policy)',marker = 'd',markersize = marker_size,linestyle = 'none',color='darkblue')
        ax[1,1].plot(labels,cap_factor_with_policy[site],label = 'Capacity factor (With policy)',marker = '.',markersize = marker_size,linestyle = 'none', color='orange')
        ax2 = ax[1,1].twinx()
        ax2.plot(labels,storage_duration_no_policy[site],label = 'Storage duration (No policy)',marker = 's',markersize = marker_size,linestyle = 'none',color='darkslategrey')
        ax2.plot(labels,storage_duration_with_policy[site],label = 'With policy',marker = '.',markersize = marker_size,linestyle = 'none',color='goldenrod')
        ax[1,1].set_title('Mississippi', fontsize=title_size_quad)
        ax[1,1].set_ylabel('Capacity factor (-)', fontname = font, fontsize = axis_label_size_quad)
        ax2.set_ylabel('Storage duration (hr)',fontname = font, fontsize = axis_label_size_quad)
        ax[1,1].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc='upper left')#)
        ax2.legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad},loc=[0.01,0.85])
        ax[1,1].set_ylim([0,cf_max])
        ax2.set_ylim([0,storage_max])
        ax[1,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45)
        ax2.tick_params(axis = 'y',labelsize = 12,direction = 'in')
  
        plt.tight_layout(pad=2)
        
        file_name = electrolysis_case + '_' + grid_case
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'quad_cf_storageduration_'+file_name + '_'+ retail_string+'.png',pad_inches = 0.1)
        plt.close(fig = None)