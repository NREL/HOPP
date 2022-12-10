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
        
#----------------------------- Preallocate dictionaries with LCOA categories and plot each one--------------------------------------------------------------------------------------------
        airsep_cap_cost = {}
        haber_bosch_cap_cost = {}
        boiler_steamturbine_cap_cost = {}
        cooling_tower_cap_cost = {}
        depreciable_nonequipment_cost = {}
        total_cap_cost_ammonia = {}
        labor_cost = {}
        maintenance_cost = {}
        adminexpense_cost = {}
        total_fixed_cost_ammonia = {}
        hydrogen_cost = {}
        electricity_cost = {}
        coolingwater_cost = {}
        ironbasedcatalyst_cost = {}
        other_feedstock_costs_ammonia = {}
        oxygenbyproduct_revenue = {}
        taxes_cost = {}
        financial_cost = {}
        taxes_financial_costs_ammonia = {}
        policy_savings = {}
        
        
        for site in locations:
            #site = 'IN'
            
            airsep_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Air Separation by Cryogenic ($/kg)'].values.tolist())
            haber_bosch_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Haber Bosch ($/kg)'].values.tolist())
            boiler_steamturbine_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Boiler and Steam Turbine ($/kg)'].values.tolist())
            cooling_tower_cap_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Cooling Tower ($/kg)'].values.tolist())
            depreciable_nonequipment_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Depreciable Nonequipment ($/kg)'].values.tolist())
            total_cap_cost_ammonia[site] = airsep_cap_cost[site]+haber_bosch_cap_cost[site]+boiler_steamturbine_cap_cost[site]+cooling_tower_cap_cost[site]+depreciable_nonequipment_cost[site]
            
            labor_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Labor Cost ($/kg)'].values.tolist())
            maintenance_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Maintenance Cost ($/kg)'].values.tolist())
            adminexpense_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Administrative Expense ($/kg)'].values.tolist())
            total_fixed_cost_ammonia[site] = labor_cost[site]+maintenance_cost[site]+adminexpense_cost[site]
            
            hydrogen_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Hydrogen ($/kg)'].values.tolist())-np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Policy savings ($/kg)'].values.tolist())
            electricity_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Electricity ($/kg)'].values.tolist())
            coolingwater_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Cooling water ($/kg)'].values.tolist())
            ironbasedcatalyst_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Iron based catalyst ($/kg)'].values.tolist())
            other_feedstock_costs_ammonia[site] = electricity_cost[site]+coolingwater_cost[site]+ironbasedcatalyst_cost[site]
            
            oxygenbyproduct_revenue[site] = -1*np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Oxygen byproduct ($/kg)'].values.tolist())
            
            taxes_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Taxes ($/kg)'].values.tolist())
            financial_cost[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Financial ($/kg)'].values.tolist())
            taxes_financial_costs_ammonia[site] = taxes_cost[site]+financial_cost[site]
            policy_savings[site] = np.array(fin_sum_usecase.loc[fin_sum_usecase['Site']==site,'Ammonia price: Policy savings ($/kg)'].values.tolist())
            
            width = 0.5
            resolution = 150
            #fig, ax = plt.subplots()
            fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
            ax.bar(labels,oxygenbyproduct_revenue[site],width,label='Oxygen byproduct revenue')
            ax.bar(labels,total_cap_cost_ammonia[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
            barbottom=total_cap_cost_ammonia[site]
            ax.bar(labels,total_fixed_cost_ammonia[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
            barbottom=barbottom+total_fixed_cost_ammonia[site]
            ax.bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
            barbottom=barbottom+hydrogen_cost[site]
            ax.bar(labels,other_feedstock_costs_ammonia[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
            barbottom=barbottom+other_feedstock_costs_ammonia[site]
            ax.bar(labels,taxes_financial_costs_ammonia[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
            barbottom = barbottom+taxes_financial_costs_ammonia[site]
            ax.bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
            barbottom=barbottom+policy_savings[site]
            ax.axhline(y=0.0, color='k', linestyle='-',linewidth=1)
            
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
            
            ax.set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size)
            ax.set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size)
            ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7},loc='upper right')
            #min_y = np.min(oxygenbyproduct_revenue)
            #max_y = np.max(barbottom+taxes_financial_costs_ammonia)
            #ax.set_ylim([-0.25,1.3*max_y])
            ax.set_ylim([-0.25,2.5])
            ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
            ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
            #ax2 = ax.twinx()
            #ax2.set_ylim([0,10])
            #plt.xlim(x[0], x[-1])
            plt.tight_layout()
            plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'single_ammoniaprice_barchart_'+file_name + '_'+ retail_string+'.png',pad_inches = 0.1)
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
        ax[0,0].bar(labels,oxygenbyproduct_revenue[site],width,label='Oxygen byproduct revenue')
        ax[0,0].bar(labels,total_cap_cost_ammonia[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost_ammonia[site]
        ax[0,0].bar(labels,total_fixed_cost_ammonia[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+total_fixed_cost_ammonia[site]
        ax[0,0].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[0,0].bar(labels,other_feedstock_costs_ammonia[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs_ammonia[site]
        ax[0,0].bar(labels,taxes_financial_costs_ammonia[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom = barbottom+taxes_financial_costs_ammonia[site]
        ax[0,0].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[0,0].axhline(y=0.0, color='k', linestyle='-',linewidth=1)
        ax[0,0].set_title('Indiana', fontsize=title_size_quad)
        ax[0,0].set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[0,0].set_ylim([-0.25,2.5])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[0,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 

        # Top right
        site = 'IA'
        ax[0,1].bar(labels,oxygenbyproduct_revenue[site],width,label='Oxygen byproduct revenue')
        ax[0,1].bar(labels,total_cap_cost_ammonia[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost_ammonia[site]
        ax[0,1].bar(labels,total_fixed_cost_ammonia[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+total_fixed_cost_ammonia[site]
        ax[0,1].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[0,1].bar(labels,other_feedstock_costs_ammonia[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs_ammonia[site]
        ax[0,1].bar(labels,taxes_financial_costs_ammonia[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom = barbottom+taxes_financial_costs_ammonia[site]
        ax[0,1].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[0,1].axhline(y=0.0, color='k', linestyle='-',linewidth=1)
        ax[0,1].set_title('Iowa', fontsize=title_size_quad)
        ax[0,1].set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[0,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[0,1].set_ylim([-0.25,2.5])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[0,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[0,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        
        # Bottom left
        site = 'TX'
        ax[1,0].bar(labels,oxygenbyproduct_revenue[site],width,label='Oxygen byproduct revenue')
        ax[1,0].bar(labels,total_cap_cost_ammonia[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost_ammonia[site]
        ax[1,0].bar(labels,total_fixed_cost_ammonia[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+total_fixed_cost_ammonia[site]
        ax[1,0].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[1,0].bar(labels,other_feedstock_costs_ammonia[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs_ammonia[site]
        ax[1,0].bar(labels,taxes_financial_costs_ammonia[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom = barbottom+taxes_financial_costs_ammonia[site]
        ax[1,0].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[1,0].axhline(y=0.0, color='k', linestyle='-',linewidth=1)
        ax[1,0].set_title('Texas', fontsize=title_size_quad)
        ax[1,0].set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,0].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[1,0].set_ylim([-0.25,2.5])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[1,0].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,0].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        
        # Bottom right
        site = 'MS'
        ax[1,1].bar(labels,oxygenbyproduct_revenue[site],width,label='Oxygen byproduct revenue')
        ax[1,1].bar(labels,total_cap_cost_ammonia[site],width,label='Total CAPEX',edgecolor='dimgray',color='dimgrey')
        barbottom=total_cap_cost_ammonia[site]
        ax[1,1].bar(labels,total_fixed_cost_ammonia[site],width,bottom=barbottom,label = 'Fixed O&M cost',edgecolor='steelblue',color='deepskyblue')
        barbottom=barbottom+total_fixed_cost_ammonia[site]
        ax[1,1].bar(labels,hydrogen_cost[site],width,bottom=barbottom,label='Hydrogen',edgecolor='cadetblue',color='lightseagreen')
        barbottom=barbottom+hydrogen_cost[site]
        ax[1,1].bar(labels,other_feedstock_costs_ammonia[site],width,bottom=barbottom,label='Other feedstocks',edgecolor='goldenrod',color='gold')
        barbottom=barbottom+other_feedstock_costs_ammonia[site]
        ax[1,1].bar(labels,taxes_financial_costs_ammonia[site],width,bottom=barbottom,label='Taxes and Finances',edgecolor='peru',color='darkorange')
        barbottom = barbottom+taxes_financial_costs_ammonia[site]
        ax[1,1].bar(labels,policy_savings[site],width,bottom=barbottom,label='Policy Savings',color='white', edgecolor = 'sandybrown',hatch='.....')
        barbottom=barbottom+policy_savings[site]
        ax[1,1].axhline(y=0.0, color='k', linestyle='-',linewidth=1)
        ax[1,1].set_title('Mississippi', fontsize=title_size_quad)
        ax[1,1].set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size_quad)
        #ax[0,0].set_xlabel('Technology Year', fontname = font, fontsize = axis_label_size_quad)
        ax[1,1].legend(fontsize = legend_size_quad, ncol = 2, prop = {'family':'Arial','size':legend_size_quad})
        max_y = np.max(barbottom)
        ax[1,1].set_ylim([-0.25,2.5])
        #ax[0,0].set_ylim([0,1.4*max_y])
        ax[1,1].tick_params(axis = 'y',labelsize = 12,direction = 'in')
        ax[1,1].tick_params(axis = 'x',labelsize = 12,direction = 'in',rotation = 45) 
        plt.tight_layout()
        file_name = electrolysis_case + '_' + grid_case
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'quad_ammoniaprice_barchart_'+file_name + '_'+ retail_string+'.png',pad_inches = 0.1)
        plt.close(fig = None)