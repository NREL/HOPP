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
retail_string = 'retail-flat'
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
        #grid_case = 'hybrid-grid-retail-flat'
        
        fin_sum_usecase = financial_summary.loc[(financial_summary['Electrolysis case']==electrolysis_case) & (financial_summary['Grid Case']==grid_case)]
        
        fin_sum_usecase = fin_sum_usecase.rename(columns={'(-) Steel price: BOS savings ($/tonne)':'Steel price: Labor savings ($/tonne)'})
        
        #Calculate policy savings
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH: Policy savings ($/kg)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','LCOH ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','LCOH ($/kg)'].values
            
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Policy savings ($/tonne)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Steel price: Total ($/tonne)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Steel price: Total ($/tonne)'].values
        
        fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Policy savings ($/kg)'] = \
            fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy','Ammonia price: Total ($/kg)'].values - fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='max','Ammonia price: Total ($/kg)'].values
            
        # Calculate o2/thermal integration savings
        fin_sum_usecase['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)']= fin_sum_usecase['Steel price: Total ($/tonne)'] -  fin_sum_usecase['Steel Price with Integration ($/tonne)']
        
        fin_sum_usecase['Steel price: Integration Savings ($/tonne)']=fin_sum_usecase['Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'] + fin_sum_usecase['Steel price: Labor savings ($/tonne)']
            
        
        fin_sum_usecase = fin_sum_usecase.loc[fin_sum_usecase['Policy Option']=='no-policy']
        
        labels  = pd.unique(fin_sum_usecase['ATB Year']).astype(int).astype(str).tolist()    
        
#----------------------------- Preallocate dictionaries with LCOH categories and plot each one--------------------------------------------------------------------------------------------
        eaf_cap_cost = {}
        shaftfurnace_cap_cost = {}
        oxsupply_cap_cost = {}
        h2preheat_cap_cost = {}
        coolingtower_cap_cost = {}
        piping_cap_cost = {}
        elecinstr_cap_cost = {}
        buildingsstorwater_cap_cost = {}
        misc_cap_cost = {}
        installation_cost = {}
        total_cap_cost = {}
        annoplabor_cost = {}
        maintenancelabor_cost = {}
        adminsupportlabor_cost = {}
        fixedom_cost = {}
        maintmaterials_cost = {}
        water_cost = {}
        lime_cost = {}
        carbon_cost= {}
        ironore_cost = {}
        hydrogen_cost = {}
        naturalgas_cost = {}
        electricity_cost = {}
        slagdisposal_cost = {}
        other_feedstock_costs = {}
        taxes_cost = {}
        financial_cost = {}
        taxes_financial_costs = {}
        policy_savings = {}
        integration_savings = {}
        
        for site in locations:
            #site = 'IN'
            eaf_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: EAF and Casting CAPEX ($/tonne)'].values.tolist())
            shaftfurnace_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Shaft Furnace CAPEX ($/tonne)'].values.tolist())
            oxsupply_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Oxygen Supply CAPEX ($/tonne)'].values.tolist())
            h2preheat_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: H2 Pre-heating CAPEX ($/tonne)'].values.tolist())
            coolingtower_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Cooling Tower CAPEX ($/tonne)'].values.tolist())
            piping_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Piping CAPEX ($/tonne)'].values.tolist())
            elecinstr_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Electrical & Instrumentation ($/tonne)'].values.tolist())
            buildingsstorwater_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Buildings, Storage, Water Service CAPEX ($/tonne)'].values.tolist())
            misc_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Miscellaneous CAPEX ($/tonne)'].values.tolist())
            installation_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Installation Cost ($/tonne)'].values.tolist())
            total_cap_cost[site] = eaf_cap_cost[site]+shaftfurnace_cap_cost[site]+oxsupply_cap_cost[site]+h2preheat_cap_cost[site]+coolingtower_cap_cost[site]\
                +piping_cap_cost[site]+elecinstr_cap_cost[site]+buildingsstorwater_cap_cost[site]+misc_cap_cost[site]+installation_cost[site]\
                    -np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: O2 Sales & Thermal Integration Savings ($/tonne)'].values.tolist())
            
            annoplabor_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Annual Operating Labor Cost ($/tonne)'].values.tolist())
            maintenancelabor_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Maintenance Labor Cost ($/tonne)'].values.tolist())
            adminsupportlabor_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Administrative & Support Labor Cost ($/tonne)'].values.tolist())
            fixedom_cost[site] = annoplabor_cost[site]+maintenancelabor_cost[site]+adminsupportlabor_cost[site] - np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Labor savings ($/tonne)'].values.tolist())
    
            maintmaterials_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Maintenance Materials ($/tonne)'].values.tolist())
            water_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Raw Water Withdrawal ($/tonne)'].values.tolist())
            lime_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Lime ($/tonne)'].values.tolist())
            carbon_cost[site]= np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Carbon ($/tonne)'].values.tolist())
            ironore_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Iron Ore ($/tonne)'].values.tolist())
            hydrogen_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Hydrogen ($/tonne)'].values.tolist()) - np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Policy savings ($/tonne)'].values.tolist())
            naturalgas_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Natural gas ($/tonne)'].values.tolist())
            electricity_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Electricity ($/tonne)'].values.tolist())
            slagdisposal_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Slag Disposal ($/tonne)'].values.tolist())
            
            other_feedstock_costs[site] = maintmaterials_cost[site]+water_cost[site]+lime_cost[site]+carbon_cost[site]+naturalgas_cost[site]+electricity_cost[site]+slagdisposal_cost[site]
            taxes_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Taxes ($/tonne)'].values.tolist())
            financial_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Financial ($/tonne)'].values.tolist())
            taxes_financial_costs[site] = taxes_cost[site]+financial_cost[site] 
            policy_savings[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Policy savings ($/tonne)'].values.tolist())
            integration_savings[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Steel price: Integration Savings ($/tonne)'].values.tolist())
            
            width = 0.5
            resolution = 150
            #fig, ax = plt.subplots()
            fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
            ax.bar(labels,total_cap_cost[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
            barbottom=total_cap_cost[site]
            ax.bar(labels,fixedom_cost[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
            barbottom=barbottom+fixedom_cost[site]
            ax.bar(labels,ironore_cost[site],width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
            barbottom=barbottom+ironore_cost[site]
            ax.bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
            barbottom=barbottom+hydrogen_cost[site]
            ax.bar(labels,other_feedstock_costs[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
            barbottom=barbottom+other_feedstock_costs[site]
            ax.bar(labels,taxes_financial_costs[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
            barbottom=barbottom+taxes_financial_costs[site]
            ax.bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
            barbottom=barbottom+policy_savings[site]
            ax.bar(labels,integration_savings[site],width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
            barbottom = barbottom+integration_savings[site]
            # ax.errorbar(labels,barbottom-integration_savings-policy_savings,yerr=[error_low,error_high], fmt='none',elinewidth=[0,0,0,0,0,1],ecolor='none',capsize=6,markeredgewidth=1)  
            # ax.errorbar(labels[5],barbottom[5]-integration_savings[5]-policy_savings[5],yerr=[[error_low[5]],[error_high[5]]],fmt='none',elinewidth=1,capsize=6,markeredgewidth=1,ecolor='black')                                        

            # Decorations
            scenario_title = site + ', ' + electrolysis_case + ', ' + grid_case
            file_name = site + '_' + electrolysis_case + '_' + grid_case
            
            # Global Plot Settings
            font = 'Arial'
            title_size = 10
            axis_label_size = 10
            tickfontsize = 10
            legend_size = 8
            tick_size = 10
            
            ax.set_title(scenario_title, fontsize=title_size)
            #ax.spines[['left','top','right','bottom']].set_linewidth(1)
            ax.set_ylabel('Breakeven price of steel ($/tonne steel)', fontname = font, fontsize = axis_label_size)
            #ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
            ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':legend_size},loc='upper right')
            max_y = np.max(barbottom)
            ax.set_ylim([0,1800])
            #ax.set_ylim([0,1.4*max_y])
            ax.tick_params(axis = 'y',labelsize = tickfontsize,direction = 'in',width=1)
            ax.tick_params(axis = 'x',labelsize = tickfontsize,direction = 'in',width=1,rotation=45)
            #ax2 = ax.twinx()
            #ax2.set_ylim([0,10])
            #plt.xlim(x[0], x[-1])
            plt.tight_layout()
            plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'single_steelprice_barchart_'+file_name + '_'+ retail_string+'.png',pad_inches = 0.1)
            plt.close(fig = None)
            
            
            
#-------------------------- Plot steel price quad-plot-----------------------------------------------------------------------------------------------------------------------
        width = 0.5
        title_size_quad = 16
        axis_label_size_quad = 14
        legend_size_quad = 9
        tick_size = 10
        resolution = 150
        fig, ax = plt.subplots(2,2,figsize=(12,10), dpi= resolution)
        
        # Top left
        site = 'IN'
        ax[0,0].bar(labels,total_cap_cost[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost[site]
        ax[0,0].bar(labels,fixedom_cost[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+fixedom_cost[site]
        ax[0,0].bar(labels,ironore_cost[site],width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
        barbottom=barbottom+ironore_cost[site]
        ax[0,0].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[0,0].bar(labels,other_feedstock_costs[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs[site]
        ax[0,0].bar(labels,taxes_financial_costs[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom=barbottom+taxes_financial_costs[site]
        ax[0,0].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[0,0].bar(labels,integration_savings[site],width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
        barbottom = barbottom+integration_savings[site]
        ax[0,0].set_title('Indiana', fontsize=title_size_quad)
        ax[0,0].set_ylabel('Breakeven price of steel ($/tonne)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[0,0].set_ylim([0,1800])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[0,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 

        # Top right
        site = 'IA'
        ax[0,1].bar(labels,total_cap_cost[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost[site]
        ax[0,1].bar(labels,fixedom_cost[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+fixedom_cost[site]
        ax[0,1].bar(labels,ironore_cost[site],width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
        barbottom=barbottom+ironore_cost[site]
        ax[0,1].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[0,1].bar(labels,other_feedstock_costs[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs[site]
        ax[0,1].bar(labels,taxes_financial_costs[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom=barbottom+taxes_financial_costs[site]
        ax[0,1].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[0,1].bar(labels,integration_savings[site],width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
        barbottom = barbottom+integration_savings[site]
        ax[0,1].set_title('Iowa', fontsize=title_size_quad)
        ax[0,1].set_ylabel('Breakeven price of steel ($/tonne)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[0,1].set_ylim([0,1800])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[0,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        
        # Bottom left
        site = 'TX'
        ax[1,0].bar(labels,total_cap_cost[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost[site]
        ax[1,0].bar(labels,fixedom_cost[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+fixedom_cost[site]
        ax[1,0].bar(labels,ironore_cost[site],width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
        barbottom=barbottom+ironore_cost[site]
        ax[1,0].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[1,0].bar(labels,other_feedstock_costs[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs[site]
        ax[1,0].bar(labels,taxes_financial_costs[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom=barbottom+taxes_financial_costs[site]
        ax[1,0].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[1,0].bar(labels,integration_savings[site],width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
        barbottom = barbottom+integration_savings[site]
        ax[1,0].set_title('Texas', fontsize=title_size_quad)
        ax[1,0].set_ylabel('Breakeven price of steel ($/tonne)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[1,0].set_ylim([0,1800])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[1,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        
        # Bottom right
        site = 'MS'
        ax[1,1].bar(labels,total_cap_cost[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost[site]
        ax[1,1].bar(labels,fixedom_cost[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+fixedom_cost[site]
        ax[1,1].bar(labels,ironore_cost[site],width,bottom=barbottom,label='Iron Ore',edgecolor='black',color='navy')
        barbottom=barbottom+ironore_cost[site]
        ax[1,1].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[1,1].bar(labels,other_feedstock_costs[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs[site]
        ax[1,1].bar(labels,taxes_financial_costs[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom=barbottom+taxes_financial_costs[site]
        ax[1,1].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[1,1].bar(labels,integration_savings[site],width,bottom=barbottom,label = 'Integration Savings',color='white', edgecolor = 'darkgray',hatch='.....')
        barbottom = barbottom+integration_savings[site]
        ax[1,1].set_title('Iowa', fontsize=title_size_quad)
        ax[1,1].set_ylabel('Breakeven price of steel ($/tonne)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[1,1].set_ylim([0,1800])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[1,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        plt.tight_layout()
        file_name = electrolysis_case + '_' + grid_case
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'quad_steelprice_barchart_'+file_name + '_'+ retail_string+'.png',pad_inches = 0.1)
        plt.close(fig = None)