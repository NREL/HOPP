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
electrolysis_directory = 'examples/H2_Analysis/RODeO_financial_summary_Iowa_transmission_LRC'
#sensitivity_directory = 'examples/H2_Analysis/Financial_summary_distributed_sensitivity'
smr_directory = 'examples/H2_Analysis/SMR_results'
plot_directory = 'examples/H2_Analysis/Plots/'

# Retail price of interest ['retail-flat','wholesale']
retail_string = 'retail-flat'

plot_subdirectory = 'Stacked_Plots_all_technologies_' + 'Iowa_trans_LRC'

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

# # Open distributed case sensitivity
# # Read in the summary data from the electrolysis case database
# conn = sqlite3.connect(sensitivity_directory+'/Default_summary.db')
# financial_summary_electrolysis_distributed_sensitivity  = pd.read_sql_query("SELECT * From Summary",conn)

# conn.commit()
# conn.close()

# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']!='grid-only-wholesale') & (financial_summary_electrolysis['Grid Case']!='hybrid-grid-wholesale')]
elif retail_string == 'wholesale':
    financial_summary_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']!='grid-only-retail-flat') & (financial_summary_electrolysis['Grid Case']!='hybrid-grid-retail-flat')]
    
# Add labels for plotting
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='grid-only-'+retail_string,'Label']='Grid Only'
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='grid-only-'+retail_string,'Order']= 2
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='hybrid-grid-'+retail_string,'Label']='Grid + \n Renewables'
financial_summary_electrolysis.loc[financial_summary_electrolysis['Grid Case']=='hybrid-grid-'+retail_string,'Order']=3
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Centralized'),'Label']='Off Grid, \n Centralized EC'
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Centralized'),'Order']=4
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Distributed'),'Label']='Off Grid, \n Distributed EC'
financial_summary_electrolysis.loc[(financial_summary_electrolysis['Grid Case']=='off-grid') & (financial_summary_electrolysis['Electrolysis case']=='Distributed'),'Order']=5

financial_summary_smr.loc[financial_summary_smr['CCS Case']=='woCCS','Label']= 'SMR'
financial_summary_smr.loc[financial_summary_smr['CCS Case']=='woCCS','Order']= 0
financial_summary_smr.loc[financial_summary_smr['CCS Case']=='wCCS','Label']= 'SMR + CCS'
financial_summary_smr.loc[financial_summary_smr['CCS Case']=='wCCS','Order']= 1

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
            #'IN',
            #'TX',
            'IA',
            #'MS'
             ]
years = [
    '2020',
    '2025',
    '2030',
    '2035']

for site in locations:
    for atb_year in years:
        #site = 'IA'
        #atb_year = '2020'
        
        scenario_title = site + ', ' + atb_year
        file_name = site+'_' + atb_year
        
        # Limit to cases for specific site and year
        site_year_electrolysis = financial_summary_electrolysis.loc[(financial_summary_electrolysis['Site']==site) & (financial_summary_electrolysis['Year']==atb_year)]
        site_year_electrolysis['CCS Case'] = 'NA'
        site_year_smr = financial_summary_smr.loc[(financial_summary_smr['Site']==site) & (financial_summary_smr['Year']==atb_year)]
        site_year_smr['Electrolysis case']=  'NA'
        site_year_smr['Grid Case'] = 'NA'
        
        # Calculate o2/thermal integration savings
        site_year_electrolysis['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)']=site_year_electrolysis['Steel price: Total ($/tonne)'] - site_year_electrolysis['Steel Price with Integration ($/tonne)']
        site_year_smr['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)']=0
        
        #Calculate policy savings
        site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','LCOH: Policy savings ($/kg)'] = \
            site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','LCOH ($/kg)'].values - site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','LCOH ($/kg)'].values
        
        site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Steel price: Policy savings ($/tonne)'] = \
            site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','Steel price: Total ($/tonne)'].values
        
        site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Ammonia price: Policy savings ($/kg)'] = \
            site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - site_year_electrolysis.loc[site_year_electrolysis['Policy Option']=='max','Ammonia price: Total ($/kg)'].values
            
        site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','LCOH: Policy savings ($/kg)'] = \
            site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','LCOH ($/kg)'].values - site_year_smr.loc[site_year_smr['Policy Option']=='max','LCOH ($/kg)'].values
        
        site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','Steel price: Policy savings ($/tonne)'] = \
            site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - site_year_smr.loc[site_year_smr['Policy Option']=='max','Steel price: Total ($/tonne)'].values
            
        site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','Ammonia price: Policy savings ($/kg)'] = \
            site_year_smr.loc[site_year_smr['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - site_year_smr.loc[site_year_smr['Policy Option']=='max','Ammonia price: Total ($/kg)'].values
        
        site_year_electrolysis['Steel price: Integration Savings ($/tonne)']=site_year_electrolysis['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'] + site_year_electrolysis['Steel price: Labor savings ($/tonne)']
        site_year_smr['Steel price: Integration Savings ($/tonne)']=0
        
        site_year_combined = pd.concat([site_year_smr,site_year_electrolysis],join='inner',ignore_index=True) 
        
        
        # site_year_sensitivity = financial_summary_electrolysis_distributed_sensitivity.loc[(financial_summary_electrolysis_distributed_sensitivity['Site']==site) & (financial_summary_electrolysis_distributed_sensitivity['Year']==atb_year)]
        # site_year_sensitivity = site_year_sensitivity.loc[site_year_sensitivity['Policy Option']=='max']
        
        # hydrogen_error_low = site_year_combined.loc[(site_year_combined['Electrolysis case']=='Distributed') & (site_year_combined['Policy Option']=='max'),'LCOH ($/kg)'].values[0] - site_year_sensitivity['LCOH ($/kg)'].values[0]
        # steel_error_low = site_year_combined.loc[(site_year_combined['Electrolysis case']=='Distributed') & (site_year_combined['Policy Option']=='max'),'Steel price: Total ($/tonne)'].values[0] - site_year_sensitivity['Steel price: Total ($/tonne)'].values[0]
        # ammonia_error_low = site_year_combined.loc[(site_year_combined['Electrolysis case']=='Distributed') & (site_year_combined['Policy Option']=='max'),'Ammonia price: Total ($/kg)'].values[0] - site_year_sensitivity['Ammonia price: Total ($/kg)'].values[0]
         
        # hydrogen_error_low = site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='high','LCOH ($/kg)'].values[0] - site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='low','LCOH ($/kg)'].values[0]
        # #hydrogen_error_low = site_year_electrolysis.loc[(site_year_electrolysis['Label']=='Off Grid, \n Distributed EC') & (site_year_electrolysis['Policy Option']=='max'),'LCOH ($/kg)'].values[0] - site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='low','LCOH ($/kg)'].values[0]
        
        # steel_error_low = site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='high','Steel price: Total ($/tonne)'].values[0] - site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='low','Steel price: Total ($/tonne)'].values[0]
        # ammonia_error_low = site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='high','Ammonia price: Total ($/kg)'].values[0] - site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='low','Ammonia price: Total ($/kg)'].values[0]
        # #steel_error_high = site_year_sensitivity.loc[site_year_sensitivity['Sensitivity Case']=='high','Steel price: Total ($/tonne)'].values[0] - steel_price[-1]
        
        site_year_combined = site_year_combined.loc[site_year_combined['Policy Option']=='no-policy']
        site_year_combined = site_year_combined.sort_values(by='Order',ignore_index=True)
               
        #steel_price = site_year_combined['Steel price: Total ($/tonne)'].values 
    
        
        #site_year_combined['Steel price: Integration Savings ($/tonne)']=site_year_combined['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'] + site_year_combined['Steel price: Labor savings ($/tonne)']
        
        #site_year_combined['Steel price: Total Savings ($/tonne)']=site_year_combined['Steel price: Policy savings ($/tonne)']+site_year_combined['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'] + site_year_combined['Steel price: Labor savings ($/tonne)']
        
        labels  = site_year_combined['Label'].values.tolist()
        
        # error_low = []
        # error_high = []
        # for j in range(len(labels)-1):
        #     error_low.append(0)
        #     error_high.append(0)
        # error_low.append(max(0,hydrogen_error_low))
        # error_high.append(0)

        # error_high = np.array(error_high)
        # error_low = np.array(error_low)   
        
        # Plot hydrogen cost for all technologies
        lcoh_withpolicy = np.array(site_year_combined['LCOH ($/kg)'].values.tolist()) - np.array(site_year_combined['LCOH: Policy savings ($/kg)'].values.tolist())
        lcoh_policy_savings = np.array(site_year_combined['LCOH: Policy savings ($/kg)'].values.tolist())
        
        width = 0.5
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(1,1,figsize=(9,6), dpi= resolution)
        ax.bar(labels,lcoh_withpolicy,width,label='With Policy',edgecolor=['midnightblue','deepskyblue','goldenrod','darkorange','forestgreen','yellowgreen'],color=['midnightblue','deepskyblue','goldenrod','darkorange','darkgreen','yellowgreen'])
        #ax.bar(labels,lcoh_withpolicy,width,label='With Policy',edgecolor=['k','k','k','k','k','k'],color=['indigo','indigo','darkgoldenrod','darkorange','darkgreen','teal'])
        barbottom=lcoh_withpolicy
        # ax.errorbar(labels,lcoh_withpolicy,yerr=[error_low,error_high], fmt='none',elinewidth=[0,0,0,0,0,1],ecolor='none',capsize=6,markeredgewidth=1)  
        # ax.errorbar(labels[5],lcoh_withpolicy[5],yerr=[[error_low[5]],[error_high[5]]],fmt='none',elinewidth=1,capsize=6,markeredgewidth=1,ecolor='black')  
        ax.bar(labels,lcoh_policy_savings,width,bottom=barbottom,label = 'Without policy',color='white', edgecolor = ['midnightblue','deepskyblue','goldenrod','darkorange','forestgreen','yellowgreen'],hatch='.....')
        #ax.bar(labels,lcoh_policy_savings,width,bottom=barbottom,label = 'Without policy',color='none', edgecolor=['k','k','k','k','k','k'])
        ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1.5)
        barbottom = lcoh_withpolicy+lcoh_policy_savings

        # Decorations
        ax.set_title(scenario_title, fontsize=title_size)
        ax.spines[['left','top','right','bottom']].set_linewidth(1.5)
        ax.set_ylabel('Levelized  cost of hydrogen ($/kg)', fontname = font, fontsize = axis_label_size)
        #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
        ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper left')
        max_y = np.max(barbottom)
        #ax.set_ylim([0,6])
        ax.set_ylim([0,1.25*max_y])
        ax.tick_params(axis = 'y',labelsize = tickfontsize,direction = 'in',width=1.5)
        ax.tick_params(axis = 'x',labelsize = tickfontsize,direction = 'in',width=1.5,rotation=45)
        #ax2 = ax.twinx()
        #ax2.set_ylim([0,10])
        #plt.xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'lcoh_barchart_'+file_name + '_'+retail_string+'_IA_trans_LRC'+'_alltechnologies.png',pad_inches = 0.1)
        plt.close(fig = None) 
        
        
        # # Plot steel cost breakdown
        # error_low = []
        # error_high = []
        # for j in range(len(labels)-1):
        #     error_low.append(0)
        #     error_high.append(0)
        # error_low.append(max(0,steel_error_low))
        # error_high.append(0)

        # error_high = np.array(error_high)
        # error_low = np.array(error_low)   
        
        eaf_cap_cost = np.array(site_year_combined['Steel price: EAF and Casting CAPEX ($/tonne)'].values.tolist())
        shaftfurnace_cap_cost = np.array(site_year_combined['Steel price: Shaft Furnace CAPEX ($/tonne)'].values.tolist())
        oxsupply_cap_cost = np.array(site_year_combined['Steel price: Oxygen Supply CAPEX ($/tonne)'].values.tolist())
        h2preheat_cap_cost = np.array(site_year_combined['Steel price: H2 Pre-heating CAPEX ($/tonne)'].values.tolist())
        coolingtower_cap_cost = np.array(site_year_combined['Steel price: Cooling Tower CAPEX ($/tonne)'].values.tolist())
        piping_cap_cost = np.array(site_year_combined['Steel price: Piping CAPEX ($/tonne)'].values.tolist())
        elecinstr_cap_cost = np.array(site_year_combined['Steel price: Electrical & Instrumentation ($/tonne)'].values.tolist())
        buildingsstorwater_cap_cost = np.array(site_year_combined['Steel price: Buildings, Storage, Water Service CAPEX ($/tonne)'].values.tolist())
        misc_cap_cost = np.array(site_year_combined['Steel price: Miscellaneous CAPEX ($/tonne)'].values.tolist())
        installation_cost = np.array(site_year_combined['Steel price: Installation Cost ($/tonne)'].values.tolist())
        total_cap_cost = eaf_cap_cost+shaftfurnace_cap_cost+oxsupply_cap_cost+h2preheat_cap_cost+coolingtower_cap_cost\
            +piping_cap_cost+elecinstr_cap_cost+buildingsstorwater_cap_cost+misc_cap_cost+installation_cost\
            -np.array(site_year_combined['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'].values.tolist())
        
        annoplabor_cost = np.array(site_year_combined['Steel price: Annual Operating Labor Cost ($/tonne)'].values.tolist())
        maintenancelabor_cost = np.array(site_year_combined['Steel price: Maintenance Labor Cost ($/tonne)'].values.tolist())
        adminsupportlabor_cost = np.array(site_year_combined['Steel price: Administrative & Support Labor Cost ($/tonne)'].values.tolist())
        fixedom_cost = annoplabor_cost+maintenancelabor_cost+adminsupportlabor_cost - np.array(site_year_combined['Steel price: Labor savings ($/tonne)'].values.tolist())

        maintmaterials_cost = np.array(site_year_combined['Steel price: Maintenance Materials ($/tonne)'].values.tolist())
        water_cost = np.array(site_year_combined['Steel price: Raw Water Withdrawal ($/tonne)'].values.tolist())
        lime_cost = np.array(site_year_combined['Steel price: Lime ($/tonne)'].values.tolist())
        carbon_cost = np.array(site_year_combined['Steel price: Carbon ($/tonne)'].values.tolist())
        ironore_cost = np.array(site_year_combined['Steel price: Iron Ore ($/tonne)'].values.tolist())
        hydrogen_cost = np.array(site_year_combined['Steel price: Hydrogen ($/tonne)'].values.tolist()) - np.array(site_year_combined['Steel price: Policy savings ($/tonne)'].values.tolist())
        naturalgas_cost = np.array(site_year_combined['Steel price: Natural gas ($/tonne)'].values.tolist())
        electricity_cost = np.array(site_year_combined['Steel price: Electricity ($/tonne)'].values.tolist())
        slagdisposal_cost = np.array(site_year_combined['Steel price: Slag Disposal ($/tonne)'].values.tolist())
        
        other_feedstock_costs = maintmaterials_cost+water_cost+lime_cost+carbon_cost+naturalgas_cost+electricity_cost+slagdisposal_cost
        taxes_cost = np.array(site_year_combined['Steel price: Taxes ($/tonne)'].values.tolist())
        financial_cost = np.array(site_year_combined['Steel price: Financial ($/tonne)'].values.tolist())
        taxes_financial_costs = taxes_cost+financial_cost
        policy_savings = np.array(site_year_combined['Steel price: Policy savings ($/tonne)'].values.tolist())
        integration_savings= np.array(site_year_combined['Steel price: Integration Savings ($/tonne)'].values.tolist())
        
        width = 0.5
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(1,1,figsize=(9,6), dpi= resolution)
        ax.bar(labels,total_cap_cost,width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost
        ax.bar(labels,fixedom_cost,width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+fixedom_cost
        ax.bar(labels,ironore_cost,width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
        barbottom=barbottom+ironore_cost
        ax.bar(labels,hydrogen_cost,width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost
        ax.bar(labels,other_feedstock_costs,width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs
        ax.bar(labels,taxes_financial_costs,width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom=barbottom+taxes_financial_costs
        ax.bar(labels,policy_savings,width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings
        ax.bar(labels,integration_savings,width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
        barbottom = barbottom+integration_savings
        # ax.errorbar(labels,barbottom-integration_savings-policy_savings,yerr=[error_low,error_high], fmt='none',elinewidth=[0,0,0,0,0,1],ecolor='none',capsize=6,markeredgewidth=1)  
        # ax.errorbar(labels[5],barbottom[5]-integration_savings[5]-policy_savings[5],yerr=[[error_low[5]],[error_high[5]]],fmt='none',elinewidth=1,capsize=6,markeredgewidth=1,ecolor='black')                                        

        ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1.5)

        # Decorations
        ax.set_title(scenario_title, fontsize=title_size)
        ax.spines[['left','top','right','bottom']].set_linewidth(1.5)
        ax.set_ylabel('Breakeven price of steel ($/tonne steel)', fontname = font, fontsize = axis_label_size)
        #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
        ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper left')
        max_y = np.max(barbottom)
        ax.set_ylim([0,1800])
        #ax.set_ylim([0,1.4*max_y])
        ax.tick_params(axis = 'y',labelsize = tickfontsize,direction = 'in',width=1.5)
        ax.tick_params(axis = 'x',labelsize = tickfontsize,direction = 'in',width=1.5,rotation=45)
        #ax2 = ax.twinx()
        #ax2.set_ylim([0,10])
        #plt.xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'steelprice_barchart_'+file_name + '_'+retail_string+'_IA_trans_LRC'+'_alltechnologies.png',pad_inches = 0.1)
        plt.close(fig = None)
        
        # # Plot ammonia cost breakdown
        # error_low = []
        # error_high = []
        # for j in range(len(labels)-1):
        #     error_low.append(0)
        #     error_high.append(0)
        # error_low.append(max(0,ammonia_error_low))
        # error_high.append(0)
        

        # error_high = np.array(error_high)
        # error_low = np.array(error_low)

        airsep_cap_cost = np.array(site_year_combined['Ammonia price: Air Separation by Cryogenic ($/kg)'].values.tolist())
        haber_bosch_cap_cost = np.array(site_year_combined['Ammonia price: Haber Bosch ($/kg)'].values.tolist())
        boiler_steamturbine_cap_cost = np.array(site_year_combined['Ammonia price: Boiler and Steam Turbine ($/kg)'].values.tolist())
        cooling_tower_cap_cost = np.array(site_year_combined['Ammonia price: Cooling Tower ($/kg)'].values.tolist())
        depreciable_nonequipment_cost = np.array(site_year_combined['Ammonia price: Depreciable Nonequipment ($/kg)'].values.tolist())
        total_cap_cost_ammonia = airsep_cap_cost+haber_bosch_cap_cost+boiler_steamturbine_cap_cost+cooling_tower_cap_cost+depreciable_nonequipment_cost
        
        labor_cost = np.array(site_year_combined['Ammonia price: Labor Cost ($/kg)'].values.tolist())
        maintenance_cost = np.array(site_year_combined['Ammonia price: Maintenance Cost ($/kg)'].values.tolist())
        adminexpense_cost = np.array(site_year_combined['Ammonia price: Administrative Expense ($/kg)'].values.tolist())
        total_fixed_cost_ammonia = labor_cost+maintenance_cost+adminexpense_cost
        
        policy_savings_ammonia = np.array(site_year_combined['Ammonia price: Policy savings ($/kg)'].values.tolist())
        
        hydrogen_cost = np.array(site_year_combined['Ammonia price: Hydrogen ($/kg)'].values.tolist()) - policy_savings_ammonia
        electricity_cost = np.array(site_year_combined['Ammonia price: Electricity ($/kg)'].values.tolist())
        coolingwater_cost = np.array(site_year_combined['Ammonia price: Cooling water ($/kg)'].values.tolist())
        ironbasedcatalyst_cost = np.array(site_year_combined['Ammonia price: Iron based catalyst ($/kg)'].values.tolist())
        other_feedstock_costs_ammonia = electricity_cost+coolingwater_cost+ironbasedcatalyst_cost
        
        oxygenbyproduct_revenue = -1*np.array(site_year_combined['Ammonia price: Oxygen byproduct ($/kg)'].values.tolist())
        
        taxes_cost = np.array(site_year_combined['Ammonia price: Taxes ($/kg)'].values.tolist())
        financial_cost = np.array(site_year_combined['Ammonia price: Financial ($/kg)'].values.tolist())

        taxes_financial_costs_ammonia = taxes_cost+financial_cost
        
        width = 0.5
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(1,1,figsize=(9,6), dpi= resolution)
#        ax.bar(labels,oxygenbyproduct_revenue,width,label='Oxygen byproduct revenue')
        ax.bar(labels,total_cap_cost_ammonia,width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost_ammonia
        ax.bar(labels,total_fixed_cost_ammonia,width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+total_fixed_cost_ammonia
        ax.bar(labels,hydrogen_cost,width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost
        ax.bar(labels,other_feedstock_costs_ammonia,width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs_ammonia
        ax.bar(labels,taxes_financial_costs_ammonia,width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom = barbottom+taxes_financial_costs_ammonia
        ax.bar(labels,policy_savings_ammonia,width,bottom=barbottom,label = 'Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings_ammonia
        # ax.errorbar(labels,barbottom-policy_savings_ammonia,yerr=[error_low,error_high], fmt='none',elinewidth=[0,0,0,0,0,1],ecolor='none',capsize=6,markeredgewidth=1)                                        
        # ax.errorbar(labels[5],barbottom[5]-policy_savings_ammonia[5],yerr=[[error_low[5]],[error_high[5]]],fmt='none',elinewidth=1,capsize=6,markeredgewidth=1,ecolor='black')                                        
        ax.axhline(y=0.0, color='k', linestyle='-',linewidth=1.5)
        ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1.5)

        
        # Decorations
        ax.set_title(scenario_title, fontsize=title_size)
        ax.spines[['left','top','right','bottom']].set_linewidth(1.5)
        ax.set_ylabel('Breakeven price of ammonia ($/kg-NH3)', fontname = font, fontsize = axis_label_size)
        ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper left')
        min_y = np.min(oxygenbyproduct_revenue)
        max_y = np.max(barbottom+taxes_financial_costs_ammonia)
        #ax.set_ylim([-0.25,1.4*max_y])
        ax.set_ylim([-0.25,2.5])
        ax.tick_params(axis = 'y',labelsize = tickfontsize,direction = 'in',width=1.5)
        ax.tick_params(axis = 'x',labelsize = tickfontsize,direction = 'in',width=1.5,rotation = 45)
        plt.tight_layout()
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'ammoniaprice_barchart_'+file_name + '_'+retail_string+'_IA_trans_LRC'+'_alltechnologies.png',pad_inches = 0.1)
        plt.close(fig = None)