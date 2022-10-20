# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:13:58 2022

@author: ereznic2
"""

# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')

sys.path.append('../PyFAST/')

import src.PyFAST as PyFAST

def run_pyfast_for_steel(plant_capacity_mtpy,plant_capacity_factor,plant_life,levelized_cost_of_hydrogen,electricity_cost,natural_gas_cost):

# # Steel plant capacity in metric tonnes per year (eventually import to function)
    # plant_capacity_mtpy = 1162077
    # plant_capacity_factor = 0.9
    # plant_life = 30
    
    # steel_production_mtpy = plant_capacity_mtpy*plant_capacity_factor
    
    # # Hydrogen cost
    # levelized_cost_of_hydrogen = 7              # $/kg
    # natural_gas_cost = 4                        # $/MMBTU
    # electricity_cost = 48.92                    # $/MWh

    #--------------------- Capital costs and Total Plant Cost ---------------------

    capex_eaf_casting = 352191.5237*plant_capacity_mtpy**0.456
    capex_shaft_furnace = 489.68061*plant_capacity_mtpy**0.88741
    capex_oxygen_supply = 1715.21508*plant_capacity_mtpy**0.64574
    capex_h2_preheating = 45.69123*plant_capacity_mtpy**0.86564
    capex_cooling_tower = 2513.08314*plant_capacity_mtpy**0.63325
    capex_piping = 11815.72718*plant_capacity_mtpy**0.59983
    capex_elec_instr = 7877.15146*plant_capacity_mtpy**0.59983
    capex_buildings_storage_water = 1097.81876*plant_capacity_mtpy**0.8
    capex_misc = 7877.1546*plant_capacity_mtpy**0.59983
    
    total_plant_cost = capex_eaf_casting + capex_shaft_furnace + capex_oxygen_supply\
                     + capex_h2_preheating + capex_cooling_tower + capex_piping\
                     + capex_elec_instr + capex_buildings_storage_water + capex_misc
                     
    
    #-------------------------------Fixed O&M Costs------------------------------
    
    labor_cost_annual_operation = 69375996.9*((plant_capacity_mtpy/365*1000)**0.25242)\
                                /((1162077/365*1000)**0.25242)
    labor_cost_maintenance = 0.00863*total_plant_cost
    labor_cost_admin_support = 0.25*(labor_cost_annual_operation + labor_cost_maintenance)
    
    property_tax_insurance = 0.02*total_plant_cost
    
    total_fixed_operating_cost = labor_cost_annual_operation + labor_cost_maintenance\
                               + labor_cost_admin_support + property_tax_insurance
    
    #-------------------------- Feedstock and Waste Costs -------------------------
    
    maintenance_materials_unitcost = 7.72       # $/metric tonne of annual steel slab production at real CF
    raw_water_unitcost = 0.59289                # $/metrid tonne of raw water
    lime_unitcost = 100                         # $/metric tonne of lime
    carbon_unitcost = 190.39                    # $/metric tonne of Carbon
    slag_disposal_unitcost = 37.63              # $ metric tonne of Slag
    iron_ore_pellet_unitcost = 207.66           # $/metric tone of Ore 
    
    # ---------------Feedstock Consumtion and Waste/Emissions Production-----------
    
    iron_ore_consumption = 1.62927              # metric tonnes of iron ore/metric tonne of steel production
    raw_water_consumption = 0.80367             # metric tonnes of raw water/metric tonne of steel production
    lime_consumption = 0.01812                  # metric tonnes of lime/metric tonne of steel production
    carbon_consumption = 0.0538                 # metric tonnes of carbon/metric tonne of steel production
    hydrogen_consumption = 0.06596              # metric tonnes of hydrogen/metric tonne of steel production
    natural_gas_consumption = 0.71657           # GJ-LHV/metric tonne of steel production
    electricity_consumption = 0.5502            # MWh/metric tonne of steel production
    
    co2_fuel_emissions = 0.03929                # metric tonnes of CO2/metric tonne of steel production
    co2_carbon_emissions = 0.017466             # metric tonnes of CO2/metridc tonne of steel production
    slag_production = 0.17433                   # metric tonnes of slag/metric tonne of steel production
    surface_water_discharge = 0.42113           # metridc tonnes of surface water discharge/metridc tonne of steel production
    
    #---------------------- Owner's (Installation) Costs --------------------------
    labor_cost_fivemonth = 5/12*(labor_cost_annual_operation + labor_cost_maintenance \
                               + labor_cost_admin_support)
    
    maintenance_materials_onemonth = maintenance_materials_unitcost*plant_capacity_mtpy/12
    non_fuel_consumables_onemonth = plant_capacity_mtpy*(raw_water_consumption*raw_water_unitcost\
                                  + lime_consumption*lime_unitcost + carbon_consumption*carbon_unitcost\
                                  + iron_ore_consumption*iron_ore_pellet_unitcost)/12
        
    waste_disposal_onemonth = plant_capacity_mtpy*slag_disposal_unitcost*slag_production/12
    
    one_month_energy_cost_25percent = 0.25*plant_capacity_mtpy*(hydrogen_consumption*levelized_cost_of_hydrogen*1000\
                                    + natural_gas_consumption*natural_gas_cost/1.05505585\
                                    + electricity_consumption*electricity_cost)/12
    two_percent_tpc = 0.02*total_plant_cost
    
    fuel_consumables_60day_supply_cost = plant_capacity_mtpy*(raw_water_consumption*raw_water_unitcost\
                                  + lime_consumption*lime_unitcost + carbon_consumption*carbon_unitcost\
                                  + iron_ore_consumption*iron_ore_pellet_unitcost)/365*60
        
    spare_parts_cost = 0.005*total_plant_cost
    
    land_cost = 0.775*plant_capacity_mtpy
    misc_owners_costs = 0.15*total_plant_cost
    
    installation_cost = labor_cost_fivemonth + two_percent_tpc\
                       + fuel_consumables_60day_supply_cost + spare_parts_cost\
                       + misc_owners_costs
                       
    #total_overnight_capital_cost = total_plant_cost + total_owners_cost
    

        
    # Set up PyFAST
    pf = PyFAST.PyFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.019
    pf.set_params('commodity',{"name":'Steel',"unit":"metric tonnes","initial price":1000,"escalation":gen_inflation})
    pf.set_params('capacity',plant_capacity_mtpy/365) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',20)
    pf.set_params('installation cost',{"value":installation_cost,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',5.3)
    pf.set_params('long term utilization',plant_capacity_factor)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',0)
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
    pf.add_capital_item(name="EAF & Casting",cost=capex_eaf_casting,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Shaft Furnace",cost=capex_shaft_furnace,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Oxygen Supply",cost=capex_oxygen_supply,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="H2 Pre-heating",cost=capex_h2_preheating,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Cooling Tower",cost=capex_cooling_tower,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Piping",cost=capex_piping,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Electrical & Instrumentation",cost=capex_elec_instr,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Buildings, Storage, Water Service",cost=capex_buildings_storage_water,depr_type="MACRS",depr_period=10,refurb=[0])
    pf.add_capital_item(name="Other Miscellaneous Costs",cost=capex_misc,depr_type="MACRS",depr_period=10,refurb=[0])
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Annual Operating Labor Cost",usage=1,unit='$/year',cost=labor_cost_annual_operation,escalation=gen_inflation)
    pf.add_fixed_cost(name="Maintenance Labor Cost",usage=1,unit='$/year',cost=labor_cost_maintenance,escalation=gen_inflation)
    pf.add_fixed_cost(name="Administrative & Support Labor Cost",usage=1,unit='$/year',cost=labor_cost_admin_support,escalation=gen_inflation)
    pf.add_fixed_cost(name="Property tax and insurance",usage=1,unit='$/year',cost=0.02*total_plant_cost,escalation=gen_inflation) 
    # Putting property tax and insurance here to zero out depcreciation/escalation. Could instead put it in set_params if
    # we think that is more accurate
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Maintenance Materials',usage=1.0,unit='Units per metric tonne of steel',cost=maintenance_materials_unitcost,escalation=gen_inflation)
    pf.add_feedstock(name='Raw Water Withdrawal',usage=raw_water_consumption,unit='metric tonnes of water per metric tonne of steel',cost=maintenance_materials_unitcost,escalation=gen_inflation)
    pf.add_feedstock(name='Lime',usage=lime_consumption,unit='metric tonnes of lime per metric tonne of steel',cost=lime_unitcost,escalation=gen_inflation)
    pf.add_feedstock(name='Carbon',usage=carbon_consumption,unit='metric tonnes of carbon per metric tonne of steel',cost=carbon_unitcost,escalation=gen_inflation)
    pf.add_feedstock(name='Iron Ore',usage=iron_ore_consumption,unit='metric tonnes of iron ore per metric tonne of steel',cost=iron_ore_pellet_unitcost,escalation=gen_inflation)
    pf.add_feedstock(name='Hydrogen',usage=hydrogen_consumption,unit='metric tonnes of hydrogen per metric tonne of steel',cost=levelized_cost_of_hydrogen*1000,escalation=gen_inflation)
    pf.add_feedstock(name='Natural Gas',usage=natural_gas_consumption,unit='GJ-LHV per metric tonne of steel',cost=natural_gas_cost/1.05505585,escalation=gen_inflation)
    pf.add_feedstock(name='Electricity',usage=electricity_consumption,unit='MWh per metric tonne of steel',cost=electricity_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Slag Disposal',usage=slag_production,unit='metric tonnes of slag per metric tonne of steel',cost=slag_disposal_unitcost,escalation=gen_inflation)
    
    #------------------------------ Sovle for breakeven price ---------------------------
    
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    return(sol,summary)






