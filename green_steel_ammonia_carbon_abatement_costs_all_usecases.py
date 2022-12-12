# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.axes as axes
import sqlite3

# Initialization and Global Settings
#Specify directory name
electrolysis_directory = 'examples/H2_Analysis/RODeO_financial_summary_results'
#electrolysis_directory = 'examples/H2_Analysis/Financial_summary_TX_offgrid_80k'
sensitivity_directory = 'examples/H2_Analysis/Financial_summary_distributed_sensitivity'
smr_directory = 'examples/H2_Analysis/SMR_results'
lca_directory = 'examples/H2_Analysis/LCA_results'
plot_directory = 'examples/H2_Analysis/Plots/'

# Retail price of interest ['retail-flat','wholesale']
retail_string = 'retail-flat'

plot_subdirectory = 'Stacked_Plots_CAC_all_technologies_'+retail_string

# Read in the summary data from the electrolysis case database
conn = sqlite3.connect(electrolysis_directory+'/Default_summary.db')
financial_summary_electrolysis  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Read in the summary data from the smr case database
conn = sqlite3.connect(smr_directory+'/Default_summary.db')
financial_summary_smr  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Open distributed case sensitivity
# Read in the summary data from the electrolysis case database
conn = sqlite3.connect(sensitivity_directory+'/Default_summary.db')
financial_summary_electrolysis_distributed_sensitivity  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Read in LCA results
lca_data = pd.read_csv(lca_directory + '/LCA_results.csv',index_col = None,header = 0,usecols = ['SMR Total GHG Emissions (kg-CO2e/kg-H2)','Ammonia SMR Total GHG Emissions (kg-CO2e/kg-NH3)','Steel SMR Total GHG Emissions (kg-CO2e/MT steel)',\
                                                                                                 'SMR with CCS Total GHG Emissions (kg-CO2e/kg-H2)','Ammonia SMR with CCS Total GHG Emissions (kg-CO2e/kg-NH3)','Steel SMR with CCS Total GHG Emissions (kg-CO2e/MT steel)',\
                                                                                                 'Electrolysis Total GHG Emissions (kg-CO2e/kg-H2)','Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)','Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)',\
                                                                                                 'Year','Site','Turbine Size','Policy Option','Electrolysis case','Grid Case'])

# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']!='grid-only-wholesale') & (financial_summary_electrolysis['Grid Case']!='hybrid-grid-wholesale')]
    lca_data = lca_data.loc[(lca_data['Grid Case']!='grid-only-wholesale') & (lca_data['Grid Case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    financial_summary_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']!='grid-only-retail-flat') & (financial_summary_electrolysis['Grid Case']!='hybrid-grid-retail-flat')]
    lca_data = lca_data.loc[(lca_data['Grid Case']!='grid-only-retail-flat') & (lca_data['Grid Case']!='hybrid-grid-retail-flat')]
    
lca_data['Year'] = lca_data['Year'].astype(str)
# Add labels for plotting
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='grid-only-'+retail_string,'Label']='Grid Only'
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='grid-only-'+retail_string,'Order']= 1
lca_data.loc[lca_data['Grid Case']=='grid-only-'+retail_string,'Label']='Grid Only'
lca_data.loc[lca_data['Grid Case']=='grid-only-'+retail_string,'Order']=1


financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='hybrid-grid-'+retail_string,'Label']='Grid + \n Renewables'
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='hybrid-grid-'+retail_string,'Order']=2
lca_data.loc[lca_data['Grid Case']=='hybrid-grid-'+retail_string,'Label']='Grid + \n Renewables'
lca_data.loc[lca_data['Grid Case']=='hybrid-grid-'+retail_string,'Order']=2

financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Centralized'),'Label']='Off Grid, \n Centralized EC'
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Centralized'),'Order']=3
lca_data.loc[(lca_data['Grid Case']=='off-grid') & (lca_data['Electrolysis case']=='Centralized'),'Label']='Off Grid, \n Centralized EC'
lca_data.loc[(lca_data['Grid Case']=='off-grid') & (lca_data['Electrolysis case']=='Centralized'),'Order']=3

financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Distributed'),'Label']='Off Grid, \n Distributed EC'
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Distributed'),'Order']=4
lca_data.loc[(lca_data['Grid Case']=='off-grid') & (lca_data['Electrolysis case']=='Distributed'),'Label']='Off Grid, \n Distributed EC'
lca_data.loc[(lca_data['Grid Case']=='off-grid') & (lca_data['Electrolysis case']=='Distributed'),'Order']=4

financial_summary_smr.loc[financial_summary_smr['CCS Case']=='woCCS','Label']= 'SMR'
#financial_summary_smr.loc[financial_summary_smr['CCS Case']=='woCCS','Order']= 0
financial_summary_smr.loc[financial_summary_smr['CCS Case']=='wCCS','Label']= 'SMR + CCS'
financial_summary_smr.loc[financial_summary_smr['CCS Case']=='wCCS','Order']= 0

# Rename things as necessary
financial_summary_electrolysis = financial_summary_electrolysis.rename(columns={'(-) Steel price: BOS savings ($/tonne)':'Steel price: Labor savings ($/tonne)'})
financial_summary_smr = financial_summary_smr.rename(columns={'(-) Steel price: BOS savings ($/tonne)':'Steel price: Labor savings ($/tonne)'})
financial_summary_smr.loc[financial_summary_smr['Policy Option']=='no policy','Policy Option']='no-policy'

# Global Plot Settings
font = 'Arial'
title_size = 20
axis_label_size = 16
legend_size = 14
tick_size = 10
tickfontsize = 16
resolution = 150

locations = [
            'IN',
            'TX',
            'IA',
            'MS'
             ]
years = [
    '2020',
    '2025',
    '2030',
    '2035'
    ]

for site in locations:
    for atb_year in years:
        #site = 'IN'
        #atb_year = '2020'
        
        scenario_title = site + ', ' + atb_year
        file_name = site+'_' + atb_year
        
        # Limit to cases for specific site and year
        site_year_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Site']==site) & (financial_summary_electrolysis['Year']==atb_year)]
        site_year_electrolysis['CCS Case'] = 'NA'
        site_year_smr = financial_summary_smr.loc[(financial_summary_smr['Site']==site) & (financial_summary_smr['Year']==atb_year)]
        site_year_smr['Electrolysis case']=  'NA'
        site_year_smr['Grid Case'] = 'NA'
        site_year_lca = lca_data.loc[(lca_data['Site']==site) & (lca_data['Year']==atb_year)]
        site_year_lca['CCS Case']='NA'
        
        # Calculate o2/thermal integration savings
        site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','Steel price: Total with Integration savings ($/tonne)']= site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','Steel Price with Integration ($/tonne)'] - site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','Steel price: Labor savings ($/tonne)']
        site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Steel price: Total with Integration savings ($/tonne)']=site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Steel price: Total ($/tonne)']
        
        site_year_smr['Steel price: Total with Integration savings ($/tonne)'] = site_year_smr['Steel price: Total ($/tonne)']
        site_year_combined = pd.concat([site_year_smr,site_year_electrolysis],join='inner',ignore_index=True) 
        
        # Extract SMR LCOH, LCOS, and LCOA
        lcoh_smr = site_year_combined.loc[(site_year_combined['Label']=='SMR') & (site_year_combined['Policy Option']=='no-policy'),'LCOH ($/kg)'].values[0]
        lcos_smr = site_year_combined.loc[(site_year_combined['Label']=='SMR') & (site_year_combined['Policy Option']=='no-policy'),'Steel price: Total ($/tonne)'].values[0]
        lcoa_smr = site_year_combined.loc[(site_year_combined['Label']=='SMR') & (site_year_combined['Policy Option']=='no-policy'),'Ammonia price: Total ($/kg)'].values[0]
        
        # Drop SMR case
        site_year_combined = site_year_combined.loc[site_year_combined['Label']!='SMR']
        
        site_year_combined = site_year_combined.sort_values(by='Order',ignore_index=True)
        
        # Add a CCS case to the lca results
        site_year_lca_CCS = site_year_lca.loc[site_year_lca['Label']=='Grid Only']
        site_year_lca_CCS['Label']='SMR + CCS'
        site_year_lca_CCS['Order']=0
        site_year_lca_CCS['Electrolysis case'] = 'NA'
        site_year_lca_CCS['Grid Case']='NA'
        site_year_lca_CCS['CCS Case'] = 'wCCS'
        
        site_year_lca['Hydrogen Total GHG Emissions (kg-CO2e/kg-H2)'] = site_year_lca['Electrolysis Total GHG Emissions (kg-CO2e/kg-H2)']#,'Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)','Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)']
        site_year_lca['Ammonia Total GHG Emissions (kg-CO2e/kg-NH3)'] = site_year_lca['Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)']#,'Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)','Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)']
        site_year_lca['Steel Total GHG Emissions (kg-CO2e/MT steel)'] = site_year_lca['Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)']#,'Ammonia Electrolysis Total GHG Emissions (kg-CO2e/kg-NH3)','Steel Electrolysis Total GHG Emissions (kg-CO2e/MT steel)']
        
        site_year_lca_CCS['Hydrogen Total GHG Emissions (kg-CO2e/kg-H2)'] = site_year_lca_CCS['SMR with CCS Total GHG Emissions (kg-CO2e/kg-H2)']
        site_year_lca_CCS['Ammonia Total GHG Emissions (kg-CO2e/kg-NH3)'] = site_year_lca_CCS['Ammonia SMR with CCS Total GHG Emissions (kg-CO2e/kg-NH3)']
        site_year_lca_CCS['Steel Total GHG Emissions (kg-CO2e/MT steel)'] = site_year_lca_CCS['Steel SMR with CCS Total GHG Emissions (kg-CO2e/MT steel)']
        
        site_year_lca_combined= pd.concat([site_year_lca,site_year_lca_CCS],join='inner',ignore_index=True) 
        site_year_lca_combined = site_year_lca_combined.sort_values(by='Order',ignore_index=True)
        
        site_year_combined_TEA_lca = site_year_combined.merge(site_year_lca_combined,how = 'outer', left_index = False,right_index = False)
        
        site_year_combined_TEA_lca['SMR LCOH ($/kg)'] = lcoh_smr
        site_year_combined_TEA_lca['SMR Steel price ($/tonne)'] = lcos_smr
        site_year_combined_TEA_lca['SMR Ammonia price ($/kg)'] = lcoa_smr
        
        # Calculate Carbon abatement cost
        site_year_combined_TEA_lca['hydrogen GHG emission difference (kg-CO2e/kg-H2)'] = site_year_combined_TEA_lca['SMR Total GHG Emissions (kg-CO2e/kg-H2)'] - site_year_combined_TEA_lca['Hydrogen Total GHG Emissions (kg-CO2e/kg-H2)']
        
        # Zero out cases with negative emissions reduction (there is no logical CAC for these cases)
        site_year_combined_TEA_lca.loc[(site_year_combined_TEA_lca['hydrogen GHG emission difference (kg-CO2e/kg-H2)'] > 0),'Multiplier']=1
        site_year_combined_TEA_lca.loc[(site_year_combined_TEA_lca['hydrogen GHG emission difference (kg-CO2e/kg-H2)'] <= 0),'Multiplier']=0
        
        site_year_combined_TEA_lca['Hydrogen carbon abatement cost ($/tonne-CO2)'] = site_year_combined_TEA_lca['Multiplier']*(site_year_combined_TEA_lca['LCOH ($/kg)'] - lcoh_smr)/(site_year_combined_TEA_lca['SMR Total GHG Emissions (kg-CO2e/kg-H2)']-site_year_combined_TEA_lca['Hydrogen Total GHG Emissions (kg-CO2e/kg-H2)'])*1000
        site_year_combined_TEA_lca['Ammonia carbon abatement cost ($/tonne-CO2)'] = site_year_combined_TEA_lca['Multiplier']*(site_year_combined_TEA_lca['Ammonia price: Total ($/kg)']-lcoa_smr)/(site_year_combined_TEA_lca['Ammonia SMR Total GHG Emissions (kg-CO2e/kg-NH3)'] - site_year_combined_TEA_lca['Ammonia Total GHG Emissions (kg-CO2e/kg-NH3)'])*1000
        site_year_combined_TEA_lca['Steel carbon abatement cost ($/tonne-CO2)'] = site_year_combined_TEA_lca['Multiplier']*(site_year_combined_TEA_lca['Steel price: Total with Integration savings ($/tonne)']-lcos_smr)/(site_year_combined_TEA_lca['Steel SMR Total GHG Emissions (kg-CO2e/MT steel)'] - site_year_combined_TEA_lca['Steel Total GHG Emissions (kg-CO2e/MT steel)'])*1000
        
        # Plot carbon abatement cost with and without policy 
        # Plot hydrogen cost for all technologies
        hydrogen_cac_nopolicy = np.array(site_year_combined_TEA_lca.loc[site_year_combined_TEA_lca['Policy Option']=='no-policy','Hydrogen carbon abatement cost ($/tonne-CO2)'].values.tolist())
        hydrogen_cac_withpolicy = np.array(site_year_combined_TEA_lca.loc[site_year_combined_TEA_lca['Policy Option']=='max','Hydrogen carbon abatement cost ($/tonne-CO2)'].values.tolist())
        
        labels  = pd.unique(site_year_combined_TEA_lca['Label'].values.tolist())
        
        
        width = 0.4
        fig, ax = plt.subplots(1,1,figsize=(9,6), dpi= resolution)
        x = np.arange(len(labels))
        rects1 = ax.bar(x-0.2,hydrogen_cac_nopolicy,width,label = 'Without policy',edgecolor=['deepskyblue','goldenrod','darkorange','forestgreen','yellowgreen'],color=['deepskyblue','goldenrod','darkorange','darkgreen','yellowgreen'])
        rects2 = ax.bar(x+0.2,hydrogen_cac_withpolicy,width,label = 'With policy',color='white', edgecolor = ['deepskyblue','goldenrod','darkorange','forestgreen','yellowgreen'],hatch='.....')
        ax.set_xticks(x,labels)
        ax.axhline(y=0, color='k', linestyle='-',linewidth=1.5)
        

        # Decorations
        ax.set_title(scenario_title, fontsize=title_size)
        ax.spines[['left','top','right','bottom']].set_linewidth(1.5)
        ax.set_ylabel('Hydrogen Carbon Abatement Cost \n ($/tonne-CO2)', fontname = font, fontsize = axis_label_size)
        #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
        #ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},bbox_to_anchor = [0.01,0.5,0.5,0.5])
        ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper left')
        plt.legend(loc = 'upper left',ncol = 1)
        #max_y = np.max(barbottom)
        ax.set_ylim([-100,600])
        #ax.set_ylim([0,1.25*max_y])
        ax.tick_params(axis = 'y',labelsize = tickfontsize,direction = 'in',width=1.5)
        ax.tick_params(axis = 'x',labelsize = tickfontsize,direction = 'in',width=1.5,rotation=45)
        #ax2 = ax.twinx()
        #ax2.set_ylim([0,10])
        #plt.xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'hydrogen_CAC_'+file_name + '_'+retail_string+'_alltechnologies.png',pad_inches = 0.1)
        plt.close(fig = None) 
        