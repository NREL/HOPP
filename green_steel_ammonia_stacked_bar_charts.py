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
output_directory = 'examples/H2_Analysis/financial_summary_results'
plot_directory = 'examples/H2_Analysis/Plots/'
plot_subdirectory = 'Stacked_Plots'
# Read in the summary data from the database
conn = sqlite3.connect(output_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Global Plot Settings
font = 'Arial'
title_size = 10
axis_label_size = 10
legend_size = 6
tick_size = 10
resolution = 150

# Loop iteration though scenarios

for i in financial_summary.index:
    
    #i=0
    # Scenario of Interest    
    # grid_case = RODeO_summary_outputs_optstorage.iloc[i, 127]
    # storage_case = RODeO_summary_outputs_optstorage.iloc[i, 126]
    hydrogen_model_case = financial_summary.loc[i,'Hydrogen model']
    location_case = financial_summary.loc[i, 'Site']
    policy_case = financial_summary.loc[i,'Policy Option'].replace(' ', '-')
 
    file_name = hydrogen_model_case + '_'  + location_case + '_policy-'+policy_case 
    scenario_title = hydrogen_model_case + ' H2 model, '  + location_case + ', '+'policy '+policy_case.replace('-',' ') 
    # Database Discretization

    if hydrogen_model_case == 'PyFAST':
        hydrogen_scenario = financial_summary[['ATB Year','Site','Hydrogen model','Policy Option','LCOH: Hydrogen Storage ($/kg)','LCOH: Compression ($/kg)','LCOH: Electrolyzer CAPEX ($/kg)','LCOH: Desalination CAPEX ($/kg)',
                                                       'LCOH: Electrolyzer FOM ($/kg)','LCOH:Desalination FOM ($/kg)','LCOH: Electrolyzer VOM ($/kg)','LCOH: Renewable Plant ($/kg)','LCOH: Renewable FOM ($/kg)',
                                                       'LCOH: Taxes ($/kg)','LCOH: Financial ($/kg)']].copy()
        
        hydrogen_scenario = hydrogen_scenario.rename(columns={'LCOH: Hydrogen Storage ($/kg)':'Hydrogen storage','LCOH: Compression ($/kg)':'Compression','LCOH: Electrolyzer CAPEX ($/kg)':'Electrolyzer CAPEX','LCOH: Desalination CAPEX ($/kg)':'Desalination CAPEX',\
                                                      'LCOH: Electrolyzer FOM ($/kg)':'Electrolyzer FOM','LCOH:Desalination FOM ($/kg)':'Desalination FOM','LCOH: Electrolyzer VOM ($/kg)':'Electrolyzer VOM','LCOH: Renewable Plant ($/kg)':'Renewable CAPEX',\
                                                      'LCOH: Renewable FOM ($/kg)':'Renewable FOM','LCOH: Taxes ($/kg)':'Taxes','LCOH: Financial ($/kg)':'Finances'})

        hydrogen_scenario = hydrogen_scenario[hydrogen_scenario['Hydrogen model'].isin([hydrogen_model_case])]
        hydrogen_scenario = hydrogen_scenario[hydrogen_scenario['Site'].isin([location_case])]
        hydrogen_scenario = hydrogen_scenario[hydrogen_scenario['Policy Option'].isin([policy_case.replace('-',' ')])]



        # Draw Plot and Annotate
        #fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)

        # columns = hydrogen_scenario.columns[4:]
        # lab = columns.values.tolist()
        # count = 0
        # for i in lab:
        #     lab[count] = i.replace(' (US$/kg)', '')
        #     count = count + 1
        # Manipulation data
        labels  = hydrogen_scenario['ATB Year'].astype(int).astype(str).values.tolist()
        storage_compression_cost = np.array(hydrogen_scenario['Hydrogen storage'].values.tolist())+np.array(hydrogen_scenario['Compression'].values.tolist())
        #compression_cost = np.array(hydrogen_scenario['Compression'].values.tolist())
        elec_cap_cost = np.array(hydrogen_scenario['Electrolyzer CAPEX'].values.tolist())
        desal_cap_cost = np.array(hydrogen_scenario['Desalination CAPEX'].values.tolist())
        elec_FOM = np.array(hydrogen_scenario['Electrolyzer FOM'].values.tolist())
        desal_FOM = np.array(hydrogen_scenario['Desalination FOM'].values.tolist())
        elec_VOM = np.array(hydrogen_scenario['Electrolyzer VOM'].values.tolist())
        renew_cap_cost = np.array(hydrogen_scenario['Renewable CAPEX'].values.tolist())
        renew_FOM = np.array(hydrogen_scenario['Renewable FOM'].values.tolist())
        taxes = np.array(hydrogen_scenario['Taxes'].values.tolist())
        financial_cost = np.array(hydrogen_scenario['Finances'].values.tolist())
        taxes_financial_cost= taxes+financial_cost
        #y = np.vstack([storage_cost, compression_cost, elec_cap_cost, desal_cap_cost,elec_FOM, desal_FOM, elec_VOM, renew_cap_cost, renew_FOM, taxes,financial_cost])
        #labels = columns.values.tolist()
        #ax = plt.gca()
        
        width = 0.5
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
        ax.bar(labels,storage_compression_cost, width, label='Hydrogen storage')
        #ax.bar(labels,compression_cost,width,bottom=storage_cost,label = 'Compression')
        ax.bar(labels,elec_cap_cost,width,bottom=storage_compression_cost,label = 'Electrolyzer CAPEX')
        ax.bar(labels,desal_cap_cost,width,bottom = storage_compression_cost+elec_cap_cost,label = 'Desalination CAPEX')
        ax.bar(labels,elec_FOM,width,bottom = storage_compression_cost+elec_cap_cost+desal_cap_cost,label = 'Electrolyzer FOM')
        ax.bar(labels,desal_FOM,width,bottom=storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM,label = 'Desalination FOM')
        ax.bar(labels,elec_VOM,width,bottom=storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM+desal_FOM,label = 'Electrolyzer VOM')
        ax.bar(labels,renew_cap_cost,width,bottom=storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM+desal_FOM+elec_VOM,label = 'Renewable CAPEX')
        ax.bar(labels,renew_FOM,width,bottom=storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM+desal_FOM+elec_VOM+renew_cap_cost,label = 'Renewable FOM')
        ax.bar(labels,taxes_financial_cost,width,bottom=storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM+desal_FOM+elec_VOM+renew_cap_cost+renew_FOM,label = 'Taxes and Finances')
        
        # Decorations
        ax.set_title(scenario_title, fontsize=title_size)

        ax.set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size)
        ax.set_xlabel('Year', fontname = font, fontsize = axis_label_size)
        ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
        max_y = np.max(storage_compression_cost+elec_cap_cost+desal_cap_cost+elec_FOM+desal_FOM+elec_VOM+renew_cap_cost+renew_FOM+taxes+financial_cost)
        ax.set_ylim([0,1.4*max_y])
        ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
        ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
        #ax2 = ax.twinx()
        #ax2.set_ylim([0,10])
        #plt.xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'lcoh_barchart_'+file_name + '.png',pad_inches = 0.1)
        plt.close(fig = None)
    
    steel_scenario = financial_summary[['ATB Year','Site','Hydrogen model','Policy Option','Steel price: EAF and Casting CAPEX ($/tonne)','Steel price: Shaft Furnace CAPEX ($/tonne)','Steel price: Oxygen Supply CAPEX ($/tonne)',
                                                'Steel price: H2 Pre-heating CAPEX ($/tonne)','Steel price: Cooling Tower CAPEX ($/tonne)','Steel price: Piping CAPEX ($/tonne)','Steel price: Electrical & Instrumentation ($/tonne)',\
                                                'Steel price: Buildings, Storage, Water Service CAPEX ($/tonne)','Steel price: Miscellaneous CAPEX ($/tonne)','Steel price: Annual Operating Labor Cost ($/tonne)',
                                                'Steel price: Maintenance Labor Cost ($/tonne)','Steel price: Administrative & Support Labor Cost ($/tonne)','Steel price: Installation Cost ($/tonne)','Steel price: Maintenance Materials ($/tonne)',
                                                'Steel price: Raw Water Withdrawal ($/tonne)','Steel price: Lime ($/tonne)','Steel price: Carbon ($/tonne)','Steel price: Iron Ore ($/tonne)','Steel price: Hydrogen ($/tonne)','Steel price: Natural gas ($/tonne)',
                                                'Steel price: Electricity ($/tonne)','Steel price: Slag Disposal ($/tonne)','Steel price: Taxes ($/tonne)','Steel price: Financial ($/tonne)']].copy()
                                                
    steel_scenario = steel_scenario[steel_scenario['Hydrogen model'].isin([hydrogen_model_case])]
    steel_scenario = steel_scenario[steel_scenario['Site'].isin([location_case])]
    steel_scenario = steel_scenario[steel_scenario['Policy Option'].isin([policy_case.replace('-',' ')])]
        
    # Manipulation data
    labels  = steel_scenario['ATB Year'].astype(int).astype(str).values.tolist()
    
    eaf_cap_cost = np.array(steel_scenario['Steel price: EAF and Casting CAPEX ($/tonne)'].values.tolist())
    shaftfurnace_cap_cost = np.array(steel_scenario['Steel price: Shaft Furnace CAPEX ($/tonne)'].values.tolist())
    oxsupply_cap_cost = np.array(steel_scenario['Steel price: Oxygen Supply CAPEX ($/tonne)'].values.tolist())
    h2preheat_cap_cost = np.array(steel_scenario['Steel price: H2 Pre-heating CAPEX ($/tonne)'].values.tolist())
    coolingtower_cap_cost = np.array(steel_scenario['Steel price: Cooling Tower CAPEX ($/tonne)'].values.tolist())
    piping_cap_cost = np.array(steel_scenario['Steel price: Piping CAPEX ($/tonne)'].values.tolist())
    elecinstr_cap_cost = np.array(steel_scenario['Steel price: Electrical & Instrumentation ($/tonne)'].values.tolist())
    buildingsstorwater_cap_cost = np.array(steel_scenario['Steel price: Buildings, Storage, Water Service CAPEX ($/tonne)'].values.tolist())
    misc_cap_cost = np.array(steel_scenario['Steel price: Miscellaneous CAPEX ($/tonne)'].values.tolist())
    installation_cost = np.array(steel_scenario['Steel price: Installation Cost ($/tonne)'].values.tolist())
    total_cap_cost = eaf_cap_cost+shaftfurnace_cap_cost+oxsupply_cap_cost+h2preheat_cap_cost+coolingtower_cap_cost\
        +piping_cap_cost+elecinstr_cap_cost+buildingsstorwater_cap_cost+misc_cap_cost+installation_cost
    
    annoplabor_cost = np.array(steel_scenario['Steel price: Annual Operating Labor Cost ($/tonne)'].values.tolist())
    maintenancelabor_cost = np.array(steel_scenario['Steel price: Maintenance Labor Cost ($/tonne)'].values.tolist())
    adminsupportlabor_cost = np.array(steel_scenario['Steel price: Administrative & Support Labor Cost ($/tonne)'].values.tolist())
    fixedom_cost = annoplabor_cost+maintenancelabor_cost+adminsupportlabor_cost

    maintmaterials_cost = np.array(steel_scenario['Steel price: Maintenance Materials ($/tonne)'].values.tolist())
    water_cost = np.array(steel_scenario['Steel price: Raw Water Withdrawal ($/tonne)'].values.tolist())
    lime_cost = np.array(steel_scenario['Steel price: Lime ($/tonne)'].values.tolist())
    carbon_cost = np.array(steel_scenario['Steel price: Carbon ($/tonne)'].values.tolist())
    ironore_cost = np.array(steel_scenario['Steel price: Iron Ore ($/tonne)'].values.tolist())
    hydrogen_cost = np.array(steel_scenario['Steel price: Hydrogen ($/tonne)'].values.tolist())
    naturalgas_cost = np.array(steel_scenario['Steel price: Natural gas ($/tonne)'].values.tolist())
    electricity_cost = np.array(steel_scenario['Steel price: Electricity ($/tonne)'].values.tolist())
    slagdisposal_cost = np.array(steel_scenario['Steel price: Slag Disposal ($/tonne)'].values.tolist())
    
    other_feedstock_costs = maintmaterials_cost+water_cost+lime_cost+carbon_cost+naturalgas_cost+electricity_cost+slagdisposal_cost
    taxes_cost = np.array(steel_scenario['Steel price: Taxes ($/tonne)'].values.tolist())
    financial_cost = np.array(steel_scenario['Steel price: Financial ($/tonne)'].values.tolist())
    taxes_financial_costs = taxes_cost+financial_cost


    width = 0.5
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
    ax.bar(labels,total_cap_cost,width,label='Total CAPEX')
    barbottom=total_cap_cost
    ax.bar(labels,fixedom_cost,width,bottom=barbottom,label = 'Fixed O&M cost')
    barbottom=barbottom+fixedom_cost
    ax.bar(labels,ironore_cost,width,bottom=barbottom,label='Iron Ore')
    barbottom=barbottom+ironore_cost
    ax.bar(labels,hydrogen_cost,width,bottom=barbottom,label='Hydrogen')
    barbottom=barbottom+hydrogen_cost
    ax.bar(labels,other_feedstock_costs,width,bottom=barbottom,label='Other feedstocks')
    barbottom=barbottom+other_feedstock_costs
    ax.bar(labels,taxes_financial_costs,width,bottom=barbottom,label='Taxes and Finances')

    
    # Decorations
    ax.set_title(scenario_title, fontsize=title_size)
    
    ax.set_ylabel('Breakeven price of steel ($/tonne)', fontname = font, fontsize = axis_label_size)
    ax.set_xlabel('Year', fontname = font, fontsize = axis_label_size)
    ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
    max_y = np.max(barbottom+taxes_financial_costs)
    ax.set_ylim([0,1.25*max_y])
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
    #ax2 = ax.twinx()
    #ax2.set_ylim([0,10])
    #plt.xlim(x[0], x[-1])
    plt.tight_layout()
    plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'steelprice_barchart_'+file_name + '.png',pad_inches = 0.1)
    plt.close(fig = None)
    
    # Plot ammonia bar charts
    ammonia_scenario = financial_summary[financial_summary['Hydrogen model'].isin([hydrogen_model_case])]
    ammonia_scenario = ammonia_scenario[ammonia_scenario['Site'].isin([location_case])]
    ammonia_scenario = ammonia_scenario[ammonia_scenario['Policy Option'].isin([policy_case.replace('-',' ')])]
    
    labels  = ammonia_scenario['ATB Year'].astype(int).astype(str).values.tolist()
    
    airsep_cap_cost = np.array(ammonia_scenario['Ammonia price: Air Separation by Cryogenic ($/kg)'].values.tolist())
    haber_bosch_cap_cost = np.array(ammonia_scenario['Ammonia price: Haber Bosch ($/kg)'].values.tolist())
    boiler_steamturbine_cap_cost = np.array(ammonia_scenario['Ammonia price: Boiler and Steam Turbine ($/kg)'].values.tolist())
    cooling_tower_cap_cost = np.array(ammonia_scenario['Ammonia price: Cooling Tower ($/kg)'].values.tolist())
    depreciable_nonequipment_cost = np.array(ammonia_scenario['Ammonia price: Depreciable Nonequipment ($/kg)'].values.tolist())
    total_cap_cost_ammonia = airsep_cap_cost+haber_bosch_cap_cost+boiler_steamturbine_cap_cost+cooling_tower_cap_cost+depreciable_nonequipment_cost
    
    labor_cost = np.array(ammonia_scenario['Ammonia price: Labor Cost ($/kg)'].values.tolist())
    maintenance_cost = np.array(ammonia_scenario['Ammonia price: Maintenance Cost ($/kg)'].values.tolist())
    adminexpense_cost = np.array(ammonia_scenario['Ammonia price: Administrative Expense ($/kg)'].values.tolist())
    total_fixed_cost_ammonia = labor_cost+maintenance_cost+adminexpense_cost
    
    hydrogen_cost = np.array(ammonia_scenario['Ammonia price: Hydrogen ($/kg)'].values.tolist())
    electricity_cost = np.array(ammonia_scenario['Ammonia price: Electricity ($/kg)'].values.tolist())
    coolingwater_cost = np.array(ammonia_scenario['Ammonia price: Cooling water ($/kg)'].values.tolist())
    ironbasedcatalyst_cost = np.array(ammonia_scenario['Ammonia price: Iron based catalyst ($/kg)'].values.tolist())
    other_feedstock_costs_ammonia = electricity_cost+coolingwater_cost+ironbasedcatalyst_cost
    
    oxygenbyproduct_revenue = np.array(-1*ammonia_scenario['Ammonia price: Oxygen byproduct ($/kg)'].values.tolist())
    
    taxes_cost = np.array(ammonia_scenario['Ammonia price: Taxes ($/kg)'].values.tolist())
    financial_cost = np.array(ammonia_scenario['Ammonia price: Financial ($/kg)'].values.tolist())
    taxes_financial_costs_ammonia = taxes_cost+financial_cost
    
    width = 0.5
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
    ax.bar(labels,total_cap_cost_ammonia,width,label='Total CAPEX')
    barbottom=total_cap_cost_ammonia
    ax.bar(labels,total_fixed_cost_ammonia,width,bottom=barbottom,label = 'Fixed O&M cost')
    barbottom=barbottom+total_fixed_cost_ammonia
    ax.bar(labels,hydrogen_cost,width,bottom=barbottom,label='Hydrogen')
    barbottom=barbottom+hydrogen_cost
    ax.bar(labels,other_feedstock_costs_ammonia,width,bottom=barbottom,label='Other feedstocks')
    barbottom=barbottom+other_feedstock_costs_ammonia
    ax.bar(labels,taxes_financial_costs_ammonia,width,bottom=barbottom,label='Taxes and Finances')

    
    # Decorations
    ax.set_title(scenario_title, fontsize=title_size)
    
    ax.set_ylabel('Breakeven price of ammonia ($/kg)', fontname = font, fontsize = axis_label_size)
    ax.set_xlabel('Year', fontname = font, fontsize = axis_label_size)
    ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
    max_y = np.max(barbottom+taxes_financial_costs_ammonia)
    ax.set_ylim([0,1.3*max_y])
    ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
    ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
    #ax2 = ax.twinx()
    #ax2.set_ylim([0,10])
    #plt.xlim(x[0], x[-1])
    plt.tight_layout()
    plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + 'ammoniaprice_barchart_'+file_name + '.png',pad_inches = 0.1)
    plt.close(fig = None)
   

        # Code for area plot, if we want it
        # fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
        # columns = hydrogen_scenario.columns[4:]
        # lab = columns.values.tolist()
        # count = 0
        # for i in lab:
        #     lab[count] = i.replace(' (US$/kg)', '')
        #     count = count + 1
        # # Manipulation data
        # x  = hydrogen_scenario['ATB Year'].values.tolist()
        # storage_cost = hydrogen_scenario['Hydrogen storage'].values.tolist()
        # compression_cost = hydrogen_scenario['Compression'].values.tolist()
        # elec_cap_cost = hydrogen_scenario['Electrolyzer CAPEX'].values.tolist()
        # desal_cap_cost = hydrogen_scenario['Desalination CAPEX'].values.tolist()
        # elec_FOM = hydrogen_scenario['Electrolyzer FOM'].values.tolist()
        # desal_FOM = hydrogen_scenario['Desalination FOM'].values.tolist()
        # elec_VOM = hydrogen_scenario['Electrolyzer VOM'].values.tolist()
        # renew_cap_cost = hydrogen_scenario['Renewable CAPEX'].values.tolist()
        # renew_FOM = hydrogen_scenario['Renewable FOM'].values.tolist()
        # taxes = hydrogen_scenario['Taxes'].values.tolist()
        # financial_cost = hydrogen_scenario['Finances'].values.tolist()
        # y = np.vstack([storage_cost, compression_cost, elec_cap_cost, desal_cap_cost,elec_FOM, desal_FOM, elec_VOM, renew_cap_cost, renew_FOM, taxes,financial_cost])
        # labels = columns.values.tolist()
        # ax = plt.gca()
        # ax.stackplot(x, y, labels=lab)
        # # Decorations
        # ax.set_title(scenario_title, fontsize=title_size)
        # ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
        # ax.set_ylabel('Levelised Cost of Hydrogen ($/kg)', fontname = font, fontsize = axis_label_size)
        # ax.set_xlabel('Year', fontname = font, fontsize = axis_label_size)
        # #ax.set_ylim([0,10])
        # ax.tick_params(axis = 'y',labelsize = 10,direction = 'in')
        # ax.tick_params(axis = 'x',labelsize = 10,direction = 'in',rotation = 45)
        # #ax2 = ax.twinx()
        # ax.set_ylim([0,1.4*max_y])
        # plt.xlim(x[0], x[-1])
        # plt.tight_layout()
        # plt.savefig(plot_directory +'/' + plot_subdirectory +'/' + file_name + '.png',pad_inches = 0.1)
        # plt.close(fig = None)
    

