# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:31:28 2022

@author: ereznic2
"""
# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')
import numpy as np
import pandas as pd
import numpy_financial as npf

sys.path.append('../PyFAST/')

import src.PyFAST as PyFAST

def run_pyfast_for_hydrogen(site_location,electrolyzer_size_mw,H2_Results,\
                            electrolyzer_system_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            capex_desal,opex_desal,plant_life,atb_year,water_cost,wind_size_mw,solar_size_mw,hybrid_plant,revised_renewable_cost,wind_om_cost_kw,
                            total_export_system_cost,
                            total_export_om_cost,
                            export_hvdc,
                            grid_connected_hopp,\
                                h2_ptc, wind_ptc):
    
    # plant_life=useful_life
    # electrolyzer_system_capex_kw = electrolyzer_capex_kw
    
    # Estimate average efficiency and water consumption
    electrolyzer_efficiency_while_running = []
    water_consumption_while_running = []
    hydrogen_production_while_running = []
    for j in range(len(H2_Results['electrolyzer_total_efficiency'])):
        if H2_Results['hydrogen_hourly_production'][j] > 0:
            electrolyzer_efficiency_while_running.append(H2_Results['electrolyzer_total_efficiency'][j])
            water_consumption_while_running.append(H2_Results['water_hourly_usage'][j])
            hydrogen_production_while_running.append(H2_Results['hydrogen_hourly_production'][j])
    
    electrolyzer_design_efficiency_HHV = np.max(electrolyzer_efficiency_while_running) # Should ideally be user input
    electrolyzer_average_efficiency_HHV = np.mean(electrolyzer_efficiency_while_running)
    water_consumption_avg_kgprhr = np.mean(water_consumption_while_running)
    
    water_consumption_avg_kgH2O_prkgH2 = water_consumption_avg_kgprhr/np.mean(hydrogen_production_while_running)
    
    water_consumption_avg_galH2O_prkgH2 = water_consumption_avg_kgH2O_prkgH2/3.79
    
    # Calculate average electricity consumption from average efficiency
    h2_HHV = 141.88
    elec_avg_consumption_kWhprkg = h2_HHV*1000/3600/electrolyzer_average_efficiency_HHV

    # Calculate electrolyzer production capacity
    electrolysis_plant_capacity_kgperday = electrolyzer_size_mw*electrolyzer_design_efficiency_HHV/h2_HHV*3600*24
    
    # Installed capital cost
    electrolyzer_installation_factor = 12/100  #[%] for stack cost 
    
    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land_cost = 250000   #[$]
    
    stack_replacement_cost = 15/100  #[% of installed capital cost]
    fixed_OM = 0.24     #[$/kg H2]    
    
    # Calculate electrolyzer installation cost
    total_direct_electrolyzer_cost_kw = (electrolyzer_system_capex_kw * (1+electrolyzer_installation_factor)) \
    
    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000
    
    electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)
    
    #electrolyzer_installation_cost = electrolyzer_system_capex_kw*stack_installation_factor*electrolyzer_size_mw\
    #                               + electrolyzer_indirect_cost                             
    
    compressor_capex_USDprkWe_of_electrolysis = 39
    
    # Renewables system size
    system_rating_mw = wind_size_mw + solar_size_mw
     
    # Calculate capital costs
    capex_electrolyzer_overnight = electrolyzer_total_installed_capex + electrolyzer_indirect_cost
    capex_storage_installed = hydrogen_storage_capacity_kg*hydrogen_storage_cost_USDprkg
    capex_compressor_installed = compressor_capex_USDprkWe_of_electrolysis*electrolyzer_size_mw*1000
    #capex_hybrid_installed = hybrid_plant.grid.total_installed_cost
    capex_hybrid_installed = revised_renewable_cost
    
    # Fixed and variable costs
    fixed_OM = 12.8 #[$/kW-y]
    fixed_cost_electrolysis_total = fixed_OM*electrolyzer_size_mw*1000
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    variable_OM = 1.30  #[$/MWh]
    
     # 
    if grid_connected_hopp == True:
        # If grid connected, conservatively assume electrolyzer runs with high CF
        # Or just take this straight from H2_Results if that works
        elec_cf = 0.97
    else:
        # If not grid connected, max DF will be relative to total renewable energy in
        elec_cf = H2_Results['cap_factor']

    # Amortized refurbishment expense [$/MWh]
    amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacement_cost)\
            *max(((plant_life*8760*elec_cf)/time_between_replacement-1),0)/plant_life/8760/elec_cf*1000

    total_variable_OM = variable_OM+amortized_refurbish_cost
    
    total_variable_OM_perkg = total_variable_OM*elec_avg_consumption_kWhprkg/1000
    
    fixed_cost_renewables = wind_om_cost_kw*system_rating_mw*1000

    # IRA Policy incentives
    h2_ptc_perkg = h2_ptc #[$/kg-H2]
    wind_ptc_perkg = wind_ptc * 55.5    #[$/kg-H2] (wind_ptc given as $/kWh. 55.5kWh/kg-H2.) This also assumes all electricity is used for H2 production

    
    # Set up PyFAST
    pf = PyFAST.PyFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.025 # based on 2022 ATB
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    # pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('capacity',H2_Results['hydrogen_annual_output']/365.0) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',atb_year+1)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)     #Based on 2022 atb baseline workbook
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    # pf.set_params('long term utilization',H2_Results['cap_factor'])    #Use 1 because capacity is accounted for in 'capacity' not H2_Results['cap_factor']
    pf.set_params('long term utilization',1)  #Use 1 because capacity is accounted for in 'capacity' not H2_Results['cap_factor']
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',property_tax_insurance)
    pf.set_params('admin expense percent',0)
    pf.set_params('total income tax rate',0.257) # 0.257 tax rate in 2022 atb baseline workbook. 0.21 current federal income tax rate, but proposed 2023 rate is 0.28. No state income tax in Texas
    pf.set_params('capital gains tax rate',0.15) # H2FAST default
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.10) # nominal return based on 2022 ATB baseline workbook
    pf.set_params('debt equity ratio of initial financing',(68.5/(100-68.5))) # 2022 ATB uses 68.5% debt
    pf.set_params('debt type','Revolving debt') # can be "Revolving debt" or "One time loan". Revolving debt is H2FAST default and leads to much lower LCOH
    pf.set_params('debt interest rate',0.037) # H2FAST default
    pf.set_params('cash onhand percent',1) # H2FAST default
    
    #----------------------------------- Add capital items to PyFAST ----------------
    pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name = "Renewable Plant",cost = capex_hybrid_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    pf.add_capital_item(name = "Export system",cost = total_export_system_cost,depr_type = "MACRS",depr_period = 5,refurb = [0])
    
    total_capex = capex_electrolyzer_overnight+capex_compressor_installed+capex_storage_installed+capex_desal+capex_hybrid_installed+total_export_system_cost

    capex_fraction = {'Electrolyzer':capex_electrolyzer_overnight/total_capex,
                      'Compression':capex_compressor_installed/total_capex,
                      'Hydrogen Storage':capex_storage_installed/total_capex,
                      'Desalination':capex_desal/total_capex,
                      'Renewable Plant':capex_hybrid_installed/total_capex,
                      'Export system': total_export_system_cost/total_capex}
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_electrolysis_total,escalation=gen_inflation)
    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)
    pf.add_fixed_cost(name="Renewable Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_renewables,escalation=gen_inflation)
    pf.add_fixed_cost(name="Export System Fixed O&M Cost",usage=1.0,unit='$/year',cost=total_export_om_cost,escalation=gen_inflation)
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    #pf.add_feedstock(name='Electricity',usage=elec_avg_consumption_kWhprkg,unit='kWh',cost=lcoe/100,escalation=gen_inflation)
    pf.add_feedstock(name='Water',usage=water_consumption_avg_galH2O_prkgH2,unit='gallon-water',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=gen_inflation)
    
     #------------------------------------ Add Incentives----------------------------------
    pf.add_incentive(name='H2 PTC',value=h2_ptc_perkg,decay=-gen_inflation,sunset_years=10,tax_credit=True)
    pf.add_incentive(name='Wind PTC',value=wind_ptc_perkg,decay=-gen_inflation,sunset_years=10,tax_credit=True)

        #------------------------------------- add incentives -----------------------------------
    # """ Note: units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced 
    #     Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # # add wind_itc (% of wind capex)
    # wind_itc_value_percent_wind_capex = incentive_dict["wind_itc"]
    # wind_capex_to_annual_h2_production_in_dollars_per_kg_h2 = capex_breakdown["wind"]/electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output']
    # wind_itc_in_dollars_per_kg_h2 = wind_itc_value_percent_wind_capex*wind_capex_to_annual_h2_production_in_dollars_per_kg_h2
    # pf.add_incentive(name='Wind ITC', value=wind_itc_in_dollars_per_kg_h2, decay=0, sunset_years=1, tax_credit=True)

    # # add wind_ptc ($/kW) 
    # # adjust from 1992 dollars to start year
    # wind_ptc_in_dollars_per_kw = -npf.fv(gen_inflation, plant_config["atb_year"]+round((orbit_project.installation_time/(365*24)))-1992, 0,  incentive_dict["wind_ptc"]) # given in 1992 dollars but adjust for inflation
    # kw_per_kg_h2 = sum(hopp_results["combined_pv_wind_power_production_hopp"])/electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output']
    # wind_ptc_in_dollars_per_kg_h2 = wind_ptc_in_dollars_per_kw*kw_per_kg_h2
    # pf.add_incentive(name='Wind PTC', value=wind_ptc_in_dollars_per_kg_h2, decay=-gen_inflation, sunset_years=10, tax_credit=True) #TODO check decay

    # # add h2_ptc ($/kg)
    # h2_ptc_inflation_adjusted = -npf.fv(gen_inflation, plant_config["atb_year"]+round((orbit_project.installation_time/(365*24)))-2022, 0,  incentive_dict["h2_ptc"])
    # pf.add_incentive(name='H2 PTC', value=h2_ptc_inflation_adjusted, decay=-gen_inflation, sunset_years=10, tax_credit=True) #TODO check decay    
    
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    price_breakdown = pf.get_cost_breakdown()
    
    # Calculate financial expense associated with equipment
    cap_expense = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
        
    # Calculate remaining financial expenses
    remaining_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
    
    # Calculate LCOH breakdown and assign capital expense to equipment costs
    price_breakdown_electrolyzer = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0] + cap_expense*capex_fraction['Electrolyzer']
    price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
    if hydrogen_storage_capacity_kg > 0:
        price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']
    else:
        price_breakdown_storage = 0
    if export_hvdc == True:
        price_breakdown_pipeline = 0
        price_breakdown_pipeline_FOM = 0
        price_breakdown_HVDC = price_breakdown.loc[price_breakdown['Name']=='Export system','NPV'].tolist()[0] +cap_expense*capex_fraction['Export system']
        price_breakdown_HVDC_FOM = price_breakdown.loc[price_breakdown['Name']=='Export System Fixed O&M Cost','NPV'].tolist()[0]
    else:
        price_breakdown_pipeline = price_breakdown.loc[price_breakdown['Name']=='Export system','NPV'].tolist()[0] +cap_expense*capex_fraction['Export system']
        price_breakdown_pipeline_FOM = price_breakdown.loc[price_breakdown['Name']=='Export System Fixed O&M Cost','NPV'].tolist()[0]
        price_breakdown_HVDC = 0
        price_breakdown_HVDC_FOM = 0
    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0] + cap_expense*capex_fraction['Desalination']
    price_breakdown_renewables = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Renewable Plant']
    price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_renewables_FOM = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant Fixed O&M Cost','NPV'].tolist()[0]
    if price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].empty:
        price_breakdown_taxes = 0 - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]
    else:    
        price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
            - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
            
    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

    price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]
       
    # price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]
        
    lcoh_check = price_breakdown_electrolyzer+price_breakdown_compression+price_breakdown_storage+price_breakdown_electrolysis_FOM\
        + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
            +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_taxes+price_breakdown_water+remaining_financial\
                +price_breakdown_pipeline+price_breakdown_pipeline_FOM+price_breakdown_HVDC+price_breakdown_HVDC_FOM
        
    lcoh_breakdown = {'LCOH: Compression & storage ($/kg)':price_breakdown_storage+price_breakdown_compression,\
                      'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                      'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                      'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,'LCOH: Renewable plant ($/kg)':price_breakdown_renewables,\
                      'LCOH: Renewable FOM ($/kg)':price_breakdown_renewables_FOM,'LCOH: Taxes ($/kg)':price_breakdown_taxes,'LCOH: Water consumption ($/kg)':price_breakdown_water,\
                      'LCOH: HVDC Export CAPEX ($/kg)':price_breakdown_HVDC,'LCOH: HVDC Export FOM ($/kg)':price_breakdown_HVDC_FOM, \
                      'LCOH: Pipeline Export CAPEX ($/kg)':price_breakdown_pipeline,'LCOH: Pipeline Export FOM ($/kg)':price_breakdown_pipeline_FOM,\
                      'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check}
    

    return(sol,summary,lcoh_breakdown,capex_electrolyzer_overnight/system_rating_mw/1000)