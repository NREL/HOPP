# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:43:30 2021

@author: ereznic2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#Specify directory name
output_directory = 'examples/H2_Analysis/RODeO_files/Output'
plot_directory = 'examples/H2_Analysis/RODeO_files/Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
RODeO_summary_outputs = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

system_rating_mw = 1000

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

# Calculate annual hydrogen sold and curtailment
RODeO_summary_outputs['Hydrogen sold (tonne/yr)'] = RODeO_summary_outputs['hydrogen revenue ($)']/RODeO_summary_outputs['Product NPV cost (US$/kg)']/1000*system_rating_mw
RODeO_summary_outputs['Curtailment (%)'] = 100*RODeO_summary_outputs['Curtailment (MWh)']/(RODeO_summary_outputs['Renewable Electricity Input (MWh)']+RODeO_summary_outputs['Curtailment (MWh)'])

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


# Create dataframes for each metric of interest across all data
breakeven_price = RODeO_summary_outputs_optstorage.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Product NPV cost (US$/kg)')
input_capfac = RODeO_summary_outputs_optstorage.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'input capacity factor')
hydrogen_production = RODeO_summary_outputs_optstorage.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Hydrogen sold (tonne/yr)')

# Segregate databse by grid scenario
RODeO_outputs_offgrid = RODeO_summary_outputs_optstorage[RODeO_summary_outputs['Grid Case'].isin(['Off-Grid'])]
RODeO_outputs_gridconnected = RODeO_summary_outputs_optstorage[RODeO_summary_outputs['Grid Case'].isin(['Grid-Connected'])]

# Define dataframes for each metric of interest for the Off Grid Grid scenario
breakevenprice_offgrid = RODeO_outputs_offgrid.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Product NPV cost (US$/kg)')
inputcapfac_offgrid = RODeO_outputs_offgrid.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'input capacity factor')
hydrogenproduction_offgrid = RODeO_outputs_offgrid.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Hydrogen sold (tonne/yr)')

# Define dataframes for each metric of interest for the Grid-Only Grid scenario
breakevenprice_gridconnected = RODeO_outputs_gridconnected.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Product NPV cost (US$/kg)')
inputcapfac_gridconnected = RODeO_outputs_gridconnected.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'input capacity factor')
hydrogenproduction_gridconnected = RODeO_outputs_gridconnected.pivot_table(index = 'Year',columns = ['Site Name','Storage Duration','Grid Case'], values = 'Hydrogen sold (tonne/yr)')

#Renewable penetration and curtailment
#renewable_penetration = SCSdata_PVGrid.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Renewable Penetration for Input (%)')
renewable_curtailment = SCSdata_PVGrid.pivot_table(index = 'Year',columns = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case'], values = 'Curtailment (MWh)')/1617.55*100

# Create lists of scenario names for plot legends
names_gridconnected = breakevenprice_gridconnected.columns.values.tolist()
names_gridconnected_joined = []
for j in range(len(breakevenprice_gridconnected.columns)):
    names_gridconnected_joined.append(', '.join(names_gridconnected[j]))
    
names_offgrid = breakevenprice_offgrid.columns.values.tolist()
names_offgrid_joined = []
for j in range(len(breakevenprice_offgrid.columns)):
    names_offgrid_joined.append(', '.join(names_offgrid[j]))

# Plot breakeven price and CF - portrait
fig5, ax5 = plt.subplots(2,2,sharex = 'all',figsize = (8,6),dpi = 150)
ax5[0,0].plot(breakevenprice_offgrid,marker = '.')
ax5[0,1].plot(inputcapfac_offgrid,marker = '.')
ax5[1,0].plot(breakevenprice_gridconnected,marker = '.')
ax5[1,1].plot(inputcapfac_gridconnected,marker = '.')
for ax in ax5.flat:
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
#ax5[0,0].set_ylim([0,30])
#ax5[1,0].set_ylim([0,30])
ax5[0,1].set_ylim([0,1])
ax5[1,1].set_ylim([0,1])
#ax5[0,0].set_xticks([2022,2026,2030,2034,2038,2042,2046,2050])
ax5[0,0].set_ylabel('Off-Grid LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax5[0,1].set_ylabel('Off-Grid Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax5[1,0].set_ylabel('Grid-Connected LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax5[1,1].set_ylabel('Grid-Connected Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax5[1,0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[1,1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0,0].legend(names_offgrid_joined,prop = {'family':'Arial','size':6})
ax5[1,0].legend(names_gridconnected_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/LCOHandCF_alllocations_paramoptstorage_portrait.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot renewable curtailment
fig0, ax0 = plt.subplots(1,sharex = 'all',figsize = (4,3),dpi = 150)
ax0.plot(renewable_curtailment,marker = '.')
#for ax in ax0.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
ax0.set_ylabel('PV Curtailment (%)',fontsize = 10,fontname = 'Arial')
ax0.set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax0.label_outer()
ax0.tick_params(axis = 'y',labelsize = 10)
plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 10)
plt.tick_params(direction = 'in',width = 1)
ax0.legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/renewablecurtailment.png',pad_inches = 0.1)
plt.close(fig = None)

















# Plot LCOH
fig0, ax0 = plt.subplots(2,sharex = 'all',figsize = (6,9),dpi = 150)
ax0[0].plot(breakevenprice_offgrid)
ax0[1].plot(breakevenprice_gridconnected)
for ax in ax0.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
    ax.set_ylabel('LCOH ($/kg)',fontsize = 16,fontname = 'Arial')
    ax.set_xlabel('Year',fontsize = 16,fontname = 'Arial')
    ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 16)
plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 16)
plt.tick_params(direction = 'in',width = 1)
ax0[0].legend(names_offgrid_joined,prop = {'family':'Arial','size':7})
ax0[1].legend(names_PVGrid_joined,prop = {'family':'Arial','size':7})
ax0[2].legend(names_PVonly_joined,prop = {'family':'Arial','size':7})
plt.tight_layout()
plt.savefig(plot_directory+'/breakevenprice5.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot LCOH of PV Only
fig0, ax0 = plt.subplots(figsize = (4,3),dpi = 150)
ax0.plot(breakevenprice_PVonly,marker = '.')
ax0.set_ylabel('LCOH ($/kg)',fontsize = 10,fontname = 'Arial')
ax0.set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax0.label_outer()
ax0.tick_params(axis = 'y',labelsize = 10)
plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 10)
plt.tick_params(direction = 'in',width = 1)
ax0.legend(names_PVonly_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/breakevenprice_PVonly.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot LCOH and CFof PV Only
fig0, ax0 = plt.subplots(1,2,figsize = (8,3.3),dpi = 150)
ax0[0].plot(breakevenprice_PVonly,marker = '.')
ax0[1].plot(inputcapfac_PVonly,marker = '.')
for ax in ax0.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
#    ax.set_xlabel('Year',fontsize = 8,fontname = 'Arial')
#   ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
#ax0[0].set_ylim([0,10])
ax0[1].set_ylim([0,1])
ax0[0].set_ylabel('PV-Only LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax0[1].set_ylabel('PV-Only Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax0[0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax0[1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')    
ax0[0].legend(names_PVonly_joined,prop = {'family':'Arial','size':6})   
plt.tight_layout()
plt.savefig(plot_directory+'/LCOHandCF_PVonly.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot capacity factor
fig1, ax1 = plt.subplots(3,sharex = 'all',figsize = (6,9),dpi = 150)
ax1[0].plot(inputcapfac_Gridonly)
ax1[1].plot(inputcapfac_PVGrid)
ax1[2].plot(inputcapfac_PVonly)
for ax in ax1.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
    ax.set_ylabel('Capacity factor (-)',fontsize = 16,fontname = 'Arial')
    ax.set_xlabel('Year',fontsize = 16,fontname = 'Arial')
    ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 16)
plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 16)
plt.tick_params(direction = 'in',width = 1)
ax1[0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':7})
ax1[1].legend(names_PVGrid_joined,prop = {'family':'Arial','size':7})
ax1[2].legend(names_PVonly_joined,prop = {'family':'Arial','size':7})
plt.tight_layout()
plt.savefig(plot_directory+'/capfac.png',pad_inches = 0.1)
plt.close(fig = None)

#Plot Grid Only case
fig2, ax2 = plt.subplots(3,sharex = 'all',figsize = (6,9),dpi = 150)
ax2[0].plot(breakevenprice_Gridonly)
ax2[1].plot(inputcapfac_Gridonly)
ax2[2].plot(hydrogenproduction_Gridonly)
ax2[0].set_title('Grid Only',fontname = 'Arial',fontsize = 20)
for ax in ax2.flat:
    ax.set_xlabel('Year',fontsize = 16,fontname = 'Arial')
    ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 16,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 16,direction = 'in')
ax2[0].set_ylabel('LCOH ($/kg)',fontsize = 16,fontname = 'Arial')
ax2[1].set_ylabel('Capacity factor (-)',fontsize = 16,fontname = 'Arial')
ax2[2].set_ylabel('Hydrogen Production',fontsize = 16,fontname = 'Arial')
plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 16)
plt.tick_params(direction = 'in',width = 1)
#ax2[0].legend(breakevenprice_Gridonly.columns.values.tolist(),prop = {'family':'Arial','size':7})
ax2[0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':7})
plt.tight_layout()
plt.savefig(plot_directory+'/gridonly.png',pad_inches = 0.1)
plt.close(fig = None)

#Plot PV+Grid case
fig3, ax3 = plt.subplots(3,sharex = 'all',figsize = (6,9),dpi = 150)
ax3[0].plot(breakevenprice_PVGrid)
ax3[1].plot(inputcapfac_PVGrid)
ax3[2].plot(hydrogenproduction_PVGrid)
ax3[0].set_title('PV + Grid',fontname = 'Arial',fontsize = 20)
for ax in ax3.flat:
    ax.set_xlabel('Year',fontsize = 16,fontname = 'Arial')
    ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 16,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 16,direction = 'in')
ax3[0].set_ylabel('LCOH ($/kg)',fontsize = 16,fontname = 'Arial')
ax3[1].set_ylabel('Capacity factor (-)',fontsize = 16,fontname = 'Arial')
ax3[2].set_ylabel('H2 Production (tonne/yr)',fontsize = 16,fontname = 'Arial')
plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 16)
plt.tick_params(direction = 'in',width = 1)
ax3[0].legend(names_PVGrid_joined,prop = {'family':'Arial','size':7})
plt.tight_layout()
plt.savefig(plot_directory+'/PVGrid.png',pad_inches = 0.1)
plt.close(fig = None)

#Plot PV Only
fig4, ax4 = plt.subplots(3,sharex = 'all',figsize = (4,4),dpi = 150)
ax4[0].plot(breakevenprice_PVonly)
ax4[1].plot(inputcapfac_PVonly)
ax4[2].plot(hydrogenproduction_PVonly)
ax4[0].set_title('PV Only',fontname = 'Arial',fontsize = 20)
for ax in ax4.flat:
    ax.set_xlabel('Year',fontsize = 16,fontname = 'Arial')
    ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 16,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 16,direction = 'in')
ax4[0].set_ylabel('LCOH ($/kg)',fontsize = 16,fontname = 'Arial')
ax4[1].set_ylabel('Capacity factor (-)',fontsize = 16,fontname = 'Arial')
ax4[2].set_ylabel('H2 Production (tonne/yr)',fontsize = 16,fontname = 'Arial')
plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 16)
plt.tick_params(direction = 'in',width = 1)
ax4[0].legend(names_PVonly_joined,prop = {'family':'Arial','size':7})
plt.tight_layout()
plt.savefig(plot_directory+'/PVonly.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot everything
fig5, ax5 = plt.subplots(3,3,sharex = 'all',figsize = (12,9),dpi = 150)
ax5[0,0].plot(breakevenprice_Gridonly)
ax5[1,0].plot(inputcapfac_Gridonly)
ax5[2,0].plot(hydrogenproduction_Gridonly)
ax5[0,0].set_title('Grid Only',fontname = 'Arial',fontsize = 16)
ax5[0,1].plot(breakevenprice_PVGrid)
ax5[1,1].plot(inputcapfac_PVGrid)
ax5[2,1].plot(hydrogenproduction_PVGrid)
ax5[0,1].set_title('PV+Grid',fontname = 'Arial',fontsize = 16)
ax5[0,2].plot(breakevenprice_PVonly)
ax5[1,2].plot(inputcapfac_PVonly)
ax5[2,2].plot(hydrogenproduction_PVonly)
ax5[0,2].set_title('PV Only',fontname = 'Arial',fontsize = 16)
for ax in ax5.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
#    ax.set_xlabel('Year',fontsize = 8,fontname = 'Arial')
#   ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
ax5[0,0].set_ylabel('LCOH ($/kg)',fontsize = 10,fontname = 'Arial')
ax5[1,0].set_ylabel('Capacity factor (-)',fontsize = 10,fontname = 'Arial')
ax5[2,0].set_ylabel('H2 Production (tonne/yr)',fontsize = 10,fontname = 'Arial')
ax5[2,0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[2,1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[2,2].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0,0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':6})
ax5[0,1].legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
ax5[0,2].legend(names_PVonly_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/allcases.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot breakeven price and CF
fig5, ax5 = plt.subplots(2,3,sharex = 'all',figsize = (12,6),dpi = 150)
ax5[0,0].plot(breakevenprice_Gridonly)
ax5[1,0].plot(inputcapfac_Gridonly)
ax5[0,0].set_title('Grid Only',fontname = 'Arial',fontsize = 16)
ax5[0,1].plot(breakevenprice_PVGrid)
ax5[1,1].plot(inputcapfac_PVGrid)
ax5[0,1].set_title('PV+Grid',fontname = 'Arial',fontsize = 16)
ax5[0,2].plot(breakevenprice_PVonly)
ax5[1,2].plot(inputcapfac_PVonly)
ax5[0,2].set_title('PV Only',fontname = 'Arial',fontsize = 16)
for ax in ax5.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
#    ax.set_xlabel('Year',fontsize = 8,fontname = 'Arial')
#   ax.label_outer()
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
ax5[0,0].set_ylim([0,10])
ax5[0,1].set_ylim([0,10])
ax5[0,2].set_ylim([0,10])
ax5[1,0].set_ylim([0,0.8])
ax5[1,1].set_ylim([0,0.8])
ax5[1,2].set_ylim([0,0.8])
ax5[0,0].set_ylabel('LCOH ($/kg)',fontsize = 10,fontname = 'Arial')
ax5[1,0].set_ylabel('Capacity factor (-)',fontsize = 10,fontname = 'Arial')
ax5[1,0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[1,1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0,0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':6})
ax5[0,1].legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
ax5[0,2].legend(names_PVonly_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/allcases_LCOHandCF.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot breakeven price and CF
fig5, ax5 = plt.subplots(1,2,figsize = (8,4),dpi = 150)
ax5[0].plot(breakevenprice_Gridonly)
ax5[1].plot(inputcapfac_Gridonly)
for ax in ax5.flat:
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
ax5[0].set_ylim([0,10])
ax5[0].set_ylabel('LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax5[1].set_ylabel('Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax5[0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/GridOnly_LCOHandCF.png',pad_inches = 0.1)
plt.close(fig = None)



# Plot breakeven price and CF for grid cases - portrait
fig5, ax5 = plt.subplots(2,2,sharex = 'all',figsize = (8,6),dpi = 150)
ax5[0,0].plot(breakevenprice_Gridonly,marker = '.')
ax5[0,1].plot(inputcapfac_Gridonly,marker = '.')
ax5[1,0].plot(breakevenprice_PVGrid,marker = '.')
ax5[1,1].plot(inputcapfac_PVGrid,marker = '.')
#ax5[2,0].plot(breakevenprice_PVonly,marker = '.')
#ax5[2,1].plot(inputcapfac_PVonly,marker = '.')
for ax in ax5.flat:
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
ax5[0,0].set_ylim([0,10])
ax5[1,0].set_ylim([0,10])
#ax5[2,0].set_ylim([0,10])
ax5[0,1].set_ylim([0,1])
ax5[1,1].set_ylim([0,1])
#ax5[2,1].set_ylim([0,1])
#ax5[0,0].set_xticks([2022,2026,2030,2034,2038,2042,2046,2050])
ax5[0,0].set_ylabel('Grid-Only LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax5[0,1].set_ylabel('Grid-Only Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax5[1,0].set_ylabel('PV+Grid LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
ax5[1,1].set_ylabel('PV+Grid Capacity factor (-)',fontsize = 10, fontname = 'Arial')
#ax5[2,0].set_ylabel('PV-Only LCOH ($/kg)',fontsize = 10, fontname = 'Arial')
#ax5[2,1].set_ylabel('PV-Only Capacity factor (-)',fontsize = 10, fontname = 'Arial')
ax5[1,0].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[1,1].set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax5[0,0].legend(names_Gridonly_joined,prop = {'family':'Arial','size':6})
ax5[1,0].legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
#ax5[2,0].legend(names_PVonly_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/gridcases_LCOHandCF_portrait.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot renewable penetration
fig0, ax0 = plt.subplots(1,sharex = 'all',figsize = (4,3),dpi = 150)
ax0.plot(renewable_penetration*100,marker = '.')
#for ax in ax0.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
ax0.set_ylabel('PV Penetration (%)',fontsize = 10,fontname = 'Arial')
ax0.set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax0.label_outer()
ax0.tick_params(axis = 'y',labelsize = 10)
plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 10)
plt.tick_params(direction = 'in',width = 1)
ax0.legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/renewablepenetration.png',pad_inches = 0.1)
plt.close(fig = None)

# Plot renewable curtailment
fig0, ax0 = plt.subplots(1,sharex = 'all',figsize = (4,3),dpi = 150)
ax0.plot(renewable_curtailment,marker = '.')
#for ax in ax0.flat:
#    ax.set(xlabel = 'Year',ylabel = 'LCOH ($/kg)')
ax0.set_ylabel('PV Curtailment (%)',fontsize = 10,fontname = 'Arial')
ax0.set_xlabel('Year',fontsize = 10,fontname = 'Arial')
ax0.label_outer()
ax0.tick_params(axis = 'y',labelsize = 10)
plt.xticks(fontname = 'Arial',fontsize = 10,rotation = 45)
plt.yticks(fontname = 'Arial',fontsize = 10)
plt.tick_params(direction = 'in',width = 1)
ax0.legend(names_PVGrid_joined,prop = {'family':'Arial','size':6})
plt.tight_layout()
plt.savefig(plot_directory+'/renewablecurtailment.png',pad_inches = 0.1)
plt.close(fig = None)