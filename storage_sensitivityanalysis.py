# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:03 2021

@author: ereznic2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#Specify directory name
output_directory = 'examples/H2_Analysis/RODeO_files/Output'
plot_directory = 'examples/H2_Analysis/RODeO_files/Plots/Storage_duration'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
RODeO_summary_outputs = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

system_rating_mw = 1000

# Eventually remove this when you fix the naming scheme for cases
RODeO_summary_outputs = RODeO_summary_outputs.drop(['Steel String','Site String','Storage String'],axis=1)

# Convert year to string
RODeO_summary_outputs['Year'] = RODeO_summary_outputs['Year'].astype(np.int64)
RODeO_summary_outputs['Year'] = RODeO_summary_outputs['Year'].astype(str)

# Convert storage duration to int
RODeO_summary_outputs['Storage Duration'] = RODeO_summary_outputs['Storage Duration'].astype(np.int64)

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

RODeO_summary_outputs['Curtailment (%)'] = 100*RODeO_summary_outputs['Curtailment (MWh)']/(RODeO_summary_outputs['Renewable Electricity Input (MWh)']+RODeO_summary_outputs['Curtailment (MWh)'])

# Segregate databse by grid scenario
RODeO_outputs_offgrid = RODeO_summary_outputs[RODeO_summary_outputs['Grid Case'].isin(['Off-Grid'])]
RODeO_outputs_gridconnected = RODeO_summary_outputs[RODeO_summary_outputs['Grid Case'].isin(['Grid-Connected'])]

project_locations = pd.unique(RODeO_summary_outputs['Site Name']).tolist()
project_years = pd.unique(RODeO_summary_outputs['Year']).tolist()

for location in project_locations:
    
    #location = 'Gulf of Mexico'
    
    RODeO_outputs_offgrid_location = RODeO_outputs_offgrid[RODeO_outputs_offgrid['Site Name']==location]
    RODeO_outputs_gridconnected_location = RODeO_outputs_gridconnected[RODeO_outputs_gridconnected['Site Name']==location]


    # Create dataframes for each year
    breakeven_storageduration_offgrid = RODeO_outputs_offgrid_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'Product NPV cost (US$/kg)')
    inputcapfac_storageduration_offgrid = RODeO_outputs_offgrid_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'input capacity factor')
    renewable_curtailment_offgrid = RODeO_outputs_offgrid_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'Curtailment (%)')

    # Create dataframes for each year
    breakeven_storageduration_gridconnected = RODeO_outputs_gridconnected_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'Product NPV cost (US$/kg)')
    inputcapfac_storageduration_gridconnected = RODeO_outputs_gridconnected_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'input capacity factor')
    renewable_curtailment_gridconnected = RODeO_outputs_gridconnected_location.pivot_table(index = 'Storage Duration',columns = ['Year','Grid Case'], values = 'Curtailment (%)')

    # Create lists of scenario names for plot legends
    names_gridconnected = breakeven_storageduration_gridconnected.columns.values.tolist()
    names_gridconnected_joined = []
    for j in range(len(breakeven_storageduration_gridconnected.columns)):
        names_gridconnected_joined.append(', '.join(names_gridconnected[j]))
        
    names_offgrid = breakeven_storageduration_offgrid.columns.values.tolist()
    names_offgrid_joined = []
    for j in range(len(breakeven_storageduration_offgrid.columns)):
        names_offgrid_joined.append(', '.join(names_offgrid[j]))

    #Plot all cases on log scale
    fig4, ax4 = plt.subplots(2,2,sharex = 'all',figsize = (8,6),dpi = 150)
    ax4[0,0].plot(breakeven_storageduration_offgrid,marker = '.')
    ax4[1,0].plot(inputcapfac_storageduration_offgrid,marker = '.')
    ax4[0,1].plot(breakeven_storageduration_gridconnected,marker = '.')
    ax4[1,1].plot(inputcapfac_storageduration_gridconnected,marker = '.')
    ax4[0,0].set_title(location + ', Off-Grid',fontname = 'Arial',fontsize = 10)
    ax4[0,1].set_title(location + ', Grid-Connected',fontname = 'Arial',fontsize = 10)
    for ax in ax4.flat:
        ax.set_xscale('log')
        ax.set_xlabel('Storage duration (hr)',fontsize = 10,fontname = 'Arial')
        ax.label_outer()
        ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
        ax.tick_params(which = 'both',axis = 'x',labelsize = 10,direction = 'in')
    ax4[0,0].set_ylabel('LCOH ($/kg)',fontsize = 10,fontname = 'Arial')
    ax4[1,0].set_ylabel('Capacity factor (-)',fontsize = 10,fontname = 'Arial')
    ax4[1,0].set_ylim([0,1])
    ax4[1,1].set_ylim([0,1])
    plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
    plt.yticks(fontname = 'Arial',fontsize = 10)
    plt.tick_params(direction = 'in',width = 1)
    ax4[0,0].legend(names_offgrid_joined,prop = {'family':'Arial','size':10})
    ax4[0,1].legend(names_gridconnected_joined,prop = {'family':'Arial','size':10})
    plt.tight_layout()
    #plt.savefig(plot_directory+'/storageduration_parametric_'+location+'_logscale.png',pad_inches = 0.1)
    plt.close(fig = None)
    
    #Plot rnewable curtailment on log scale
    fig4, ax4 = plt.subplots(1,2,sharey = 'all',figsize = (8,4),dpi = 150)
    ax4[0].plot(renewable_curtailment_offgrid,marker = '.')
    #ax4[1,0].plot(inputcapfac_storageduration_offgrid,marker = '.')
    ax4[1].plot(renewable_curtailment_gridconnected,marker = '.')
    #ax4[1,1].plot(inputcapfac_storageduration_gridconnected,marker = '.')
    ax4[0].set_title(location + ', Off-Grid',fontname = 'Arial',fontsize = 10)
    ax4[1].set_title(location + ', Grid-Connected',fontname = 'Arial',fontsize = 10)
    for ax in ax4.flat:
        ax.set_xscale('log')
        ax.set_xlabel('Storage duration (hr)',fontsize = 10,fontname = 'Arial')
        ax.label_outer()
        ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
        ax.tick_params(which = 'both',axis = 'x',labelsize = 10,direction = 'in')
    ax4[0].set_ylabel('Renewable Curtailment (%)',fontsize = 10,fontname = 'Arial')
    #ax4[1,0].set_ylabel('Capacity factor (-)',fontsize = 10,fontname = 'Arial')
    #ax4[1,0].set_ylim([0,1])
    #ax4[1,1].set_ylim([0,1])
    plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
    plt.yticks(fontname = 'Arial',fontsize = 10)
    plt.tick_params(direction = 'in',width = 1)
    ax4[0].legend(names_offgrid_joined,prop = {'family':'Arial','size':10})
    ax4[1].legend(names_gridconnected_joined,prop = {'family':'Arial','size':10})
    plt.tight_layout()
    #plt.savefig(plot_directory+'/curtailment_parametric_'+location+'_logscale.png',pad_inches = 0.1)
    plt.close(fig = None)
    
    #Plot all cases on log scale
    fig4, ax4 = plt.subplots(3,2,sharex = 'col',sharey='row',figsize = (8,8),dpi = 150)
    ax4[0,0].plot(breakeven_storageduration_offgrid,marker = '.')
    ax4[1,0].plot(inputcapfac_storageduration_offgrid,marker = '.')
    ax4[2,0].plot(renewable_curtailment_offgrid,marker = '.')
    ax4[0,1].plot(breakeven_storageduration_gridconnected,marker = '.')
    ax4[1,1].plot(inputcapfac_storageduration_gridconnected,marker = '.')
    ax4[2,1].plot(renewable_curtailment_gridconnected,marker = '.')
    ax4[0,0].set_title(location + ', Off-Grid',fontname = 'Arial',fontsize = 12)
    ax4[0,1].set_title(location + ', Grid-Connected',fontname = 'Arial',fontsize = 12)
    for ax in ax4.flat:
        ax.set_xscale('log')
        ax.set_xlabel('Storage duration (hr)',fontsize = 12,fontname = 'Arial')
        ax.label_outer()
        ax.tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax.tick_params(which = 'both',axis = 'x',labelsize = 12,direction = 'in')
    ax4[0,0].set_ylabel('LCOH ($/kg)',fontsize = 12,fontname = 'Arial')
    ax4[1,0].set_ylabel('Capacity factor (-)',fontsize = 12,fontname = 'Arial')
    ax4[2,0].set_ylabel('Renewable Curtailment (%)',fontsize = 12, fontname = 'Arial')
    ax4[1,0].set_ylim([0,1])
    ax4[1,1].set_ylim([0,1])
    plt.xticks(fontname = 'Arial',fontsize = 12,rotation = 45)
    plt.yticks(fontname = 'Arial',fontsize = 12)
    plt.tick_params(direction = 'in',width = 1)
    ax4[0,0].legend(names_offgrid_joined,prop = {'family':'Arial','size':12})
    ax4[0,1].legend(names_gridconnected_joined,prop = {'family':'Arial','size':12})
    plt.tight_layout()
    plt.savefig(plot_directory+'/storageduration_curtailment_parametric_'+location+'_logscale.png',pad_inches = 0.1)
    plt.close(fig = None)
    
    



