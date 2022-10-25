# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:31:28 2022

@author: ereznic2
"""
# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')

sys.path.append('../PyFAST/')

import src.PyFAST as PyFAST

def run_pyfast_for_hydrogen(site_location,electrolyzer_size_mw,electrolyzer_energy_kWh_per_kg,electrolyzer_capacity_factor,\
                            electrolyzer_capex_kw,storage_capacity_MWh,plant_life,water_consumption_galperkg,water_cost,lcoe):

# # Hard-coded inputs
# site_location = 'Site 1'
# lcoe = 0.07312156
# electrolyzer_size_mw = 1000
# electrolyzer_energy_kWh_per_kg = 55.5
# electrolyzer_capacity_factor = 0.41982
# electrolyzer_capex_kw = 1100
# storage_capacity_MWh = 378.5657893*electrolyzer_size_mw
# plant_life = 30
# water_consumption_galperkg = 3.78
# water_cost = 0.01

    # Calculate electrolyzer production capacity
    electrolysis_plant_capacity_kgperday = electrolyzer_size_mw*1000*24/electrolyzer_energy_kWh_per_kg
    
    # Calculate storage capacity in kg
    storage_capacity_kg = storage_capacity_MWh/0.033322222
    
    # Storage costs as a function of location
    if site_location == 'Site 1':
        h2_storage_cost_USDperkg =25
    elif site_location == 'Site 2':
        h2_storage_cost_USDperkg = 540
    elif site_location == 'Site 3':
        h2_storage_cost_USDperkg = 54
    elif site_location == 'Site 4':
        h2_storage_cost_USDperkg = 54
        
    #Capital costs provide by Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
    mechanical_bop_cost = 36  #[$/kW] for a compressor
    electrical_bop_cost = 82  #[$/kW] for a rectifier
    
    # Installed capital cost
    stack_installation_factor = 12/100  #[%] for stack cost 
    elec_installation_factor = 12/100   #[%] and electrical BOP 
    #mechanical BOP install cost = 0%
    
    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land_cost = 250000   #[$]
    
    stack_replacement_cost = 15/100  #[% of installed capital cost]
    fixed_OM = 0.24     #[$/kg H2]    
    
    
    # Calculate electrolyzer installation cost
    total_direct_electrolyzer_cost_kw = (electrolyzer_capex_kw * (1+stack_installation_factor)) \
        + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))
    
    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000
    
    electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)
    
    electrolyzer_installation_cost = electrolyzer_capex_kw*(stack_installation_factor + electrical_bop_cost*elec_installation_factor)\
                                   + electrolyzer_indirect_cost
    
    # Calculate capital costs
    capex_electrolyzer = electrolyzer_capex_kw*electrolyzer_size_mw*1000
    capex_storage = storage_capacity_kg*h2_storage_cost_USDperkg
    capex_mechanical = mechanical_bop_cost*electrolyzer_size_mw*1000
    capex_electrical = electrical_bop_cost*electrolyzer_size_mw*1000
    
    # Fixed and variable costs
    fixed_OM = 12.8 #[$/kW-y]
    fixed_cost_total = fixed_OM*electrolyzer_size_mw*1000
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    #variable_OM = 1.30  #[$/MWh]
    variable_OM_perkg = 0.0243774358475716
    
    # Set up PyFAST
    pf = PyFAST.PyFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.019
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":0.0})
    pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)
    pf.set_params('installation cost',{"value":electrolyzer_installation_cost,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',electrolyzer_capacity_factor)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',property_tax_insurance)
    pf.set_params('admin expense percent',0)
    pf.set_params('total income tax rate',0.27)
    pf.set_params('capital gains tax rate',0.15)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.1)
    pf.set_params('debt equity ratio of initial financing',0.5)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',0.06)
    pf.set_params('cash onhand percent',1)
    
    #----------------------------------- Add capital items to PyFAST ----------------
    pf.add_capital_item(name="Electrolyzer",cost=capex_electrolyzer,depr_type="MACRS",depr_period=20,refurb=[0])
    pf.add_capital_item(name="Compressor and mechanical BOP",cost=capex_mechanical,depr_type="MACRS",depr_period=20,refurb=[0])
    pf.add_capital_item(name="Electrical BOP",cost=capex_electrical,depr_type="MACRS",depr_period=20,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage,depr_type="MACRS",depr_period=20,refurb=[0])
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_total,escalation=gen_inflation)
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Electricity',usage=electrolyzer_energy_kWh_per_kg,unit='kWh',cost=lcoe,escalation=gen_inflation)
    pf.add_feedstock(name='Water',usage=water_consumption_galperkg,unit='gal',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=variable_OM_perkg,escalation=gen_inflation)
    
        
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    pf.plot_costs()
    
    return(sol,summary)