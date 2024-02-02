"""
Function to call profast for ammonia model
Written by Abhineet Gupta
"""

import ProFAST
import pandas as pd

# # Add location of PyFAST code 
# import sys
# sys.path.append('../PyFAST/')

# import src.PyFAST as PyFAST

# Implement equations from Ammonia model received
def run_profast_for_ammonia(plant_capacity_kgpy,plant_capacity_factor,plant_life,levelized_cost_of_hydrogen, electricity_cost,grid_prices_interpolated_USDperMWh,cooling_water_cost,iron_based_catalyst_cost,oxygen_price):
    # Inputs:
    # plant_capacity_kgpy = 462323016 ##KgNH3/year
    # plant_capacity_factor = 0.9
    # plant_life = 40 #years
    # # Costs from original model (for reference)
    # levelized_cost_of_hydrogen = 4.83       # $/kg
    # electricity_cost = 69.83                # $/MWh
    # cooling_water_cost = 0.000113349938601175 # $/Gal
    # iron_based_catalyist_cost = 23.19977341 # $/kg
    # oxygen_price = 0.0285210891617726       # $/kg

    model_year_CEPCI = 596.2
    equation_year_CEPCI = 541.7
    
    ammonia_production_kgpy = plant_capacity_kgpy*plant_capacity_factor #=  416,090,714      
    
    # scale with respect to a baseline plant
    scaling_ratio = plant_capacity_kgpy/(365.0*1266638.4)
    
    # CapEx
    scaling_factor_equipment = 0.6
    capex_scale_factor = scaling_ratio**scaling_factor_equipment
    capex_air_separation_crygenic = model_year_CEPCI/equation_year_CEPCI*22506100 * capex_scale_factor
    capex_haber_bosch = model_year_CEPCI/equation_year_CEPCI*18642800 * capex_scale_factor
    capex_boiler = model_year_CEPCI/equation_year_CEPCI*7069100 * capex_scale_factor
    capex_cooling_tower = model_year_CEPCI/equation_year_CEPCI*4799200 * capex_scale_factor
    capex_direct = capex_air_separation_crygenic + capex_haber_bosch\
        + capex_boiler + capex_cooling_tower
    capex_depreciable_nonequipment = capex_direct*0.42 + \
       4112701.84103543*scaling_ratio
    capex_total = capex_direct + capex_depreciable_nonequipment
    
    land_cost = capex_depreciable_nonequipment
    
    # O&M Cost
    scaling_factor_labor = 0.25
    labor_cost = 57 * 50 * 2080 * scaling_ratio**scaling_factor_labor
    general_administration_cost = labor_cost * 0.2
    property_tax_insurance = capex_total * 0.02
    maintenance_cost = capex_direct * 0.005 * \
        scaling_ratio**scaling_factor_equipment
    land_cost = 2500000*capex_scale_factor
    fixed_O_and_M_cost = land_cost + labor_cost + \
        general_administration_cost + \
        property_tax_insurance + maintenance_cost
    
    # Feedstock
    H2_consumption = 0.197284403 # kg_H2/ kg_NH3
    H2_cost_in_startup_year = levelized_cost_of_hydrogen * H2_consumption\
         * plant_capacity_kgpy * plant_capacity_factor
    
    electricity_usage = 0.530645243/1000 # mWh/kg_NH3
    energy_cost_in_startup_year = electricity_cost * electricity_usage\
        * plant_capacity_kgpy * plant_capacity_factor # 
    
    cooling_water_usage = 0.049236824 # Gal/kg_NH3
    iron_based_catalyst_usage = 0.000091295354067341 # kg/kg_NH3
    non_energy_cost_in_startup_year = \
        ((cooling_water_cost * cooling_water_usage) + \
            (iron_based_catalyst_cost*iron_based_catalyst_usage)) * \
                plant_capacity_kgpy * plant_capacity_factor
    
    variable_cost_in_startup_year = energy_cost_in_startup_year\
        + non_energy_cost_in_startup_year

    # By-product
    oxygen_byproduct = 0.29405077250145     # kg/kg_NH#
    credits_byproduct = oxygen_price*oxygen_byproduct * \
        plant_capacity_kgpy * plant_capacity_factor

    financial_assumptions = pd.read_csv('H2_Analysis/financial_inputs.csv',index_col=None,header=0)
    financial_assumptions.set_index(["Parameter"], inplace = True)
    financial_assumptions = financial_assumptions['Hydrogen/Steel/Ammonia']

     # Set up ProFAST
    pf = ProFAST.ProFAST()

    install_years = 3
    analysis_start = list(grid_prices_interpolated_USDperMWh.keys())[0] - install_years
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params('commodity',{"name":'Ammonia',"unit":"kg","initial price":1000,"escalation":gen_inflation})
    pf.set_params('capacity',plant_capacity_kgpy/365) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',analysis_start)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',12*install_years)
    pf.set_params('installation cost',{"value":fixed_O_and_M_cost,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',plant_capacity_factor)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance',0)
    pf.set_params('admin expense',0)
    pf.set_params('total income tax rate',financial_assumptions['total income tax rate'])
    pf.set_params('capital gains tax rate',financial_assumptions['capital gains tax rate'])
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',financial_assumptions['leverage after tax nominal discount rate'])
    pf.set_params('debt equity ratio of initial financing',financial_assumptions['debt equity ratio of initial financing'])
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',financial_assumptions['debt interest rate'])
    pf.set_params('cash onhand',1)
    
    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="Air Separation by Cryogenic",cost=capex_air_separation_crygenic,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Haber Bosch",cost=capex_haber_bosch,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Boiler and Steam Turbine",cost=capex_boiler,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Cooling Tower",cost=capex_cooling_tower,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Depreciable Nonequipment",cost=capex_depreciable_nonequipment,depr_type="MACRS",depr_period=7,refurb=[0])

    total_capex = capex_air_separation_crygenic+capex_haber_bosch+capex_boiler+capex_cooling_tower+capex_depreciable_nonequipment
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Labor Cost",usage=1,unit='$/year',cost=labor_cost,escalation=gen_inflation)
    pf.add_fixed_cost(name="Maintenance Cost",usage=1,unit='$/year',cost=maintenance_cost,escalation=gen_inflation)
    pf.add_fixed_cost(name="Administrative Expense",usage=1,unit='$/year',cost=general_administration_cost,escalation=gen_inflation)
    pf.add_fixed_cost(name="Property tax and insurance",usage=1,unit='$/year',cost=property_tax_insurance,escalation=0.0)
    #pf.add_fixed_cost(name="Land cost",cost=2500000*capex_scale_factor,depr_type="MACRS",depr_period=20,refurb=[0])
     
    # Putting property tax and insurance here to zero out depcreciation/escalation. Could instead put it in set_params if
    # we think that is more accurate
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Hydrogen',usage=H2_consumption,unit='kilogram of hydrogen per kilogram of ammonia',cost=levelized_cost_of_hydrogen,escalation=gen_inflation)
    pf.add_feedstock(name='Electricity',usage=electricity_usage,unit='MWh per kilogram of ammonia',cost=grid_prices_interpolated_USDperMWh,escalation=gen_inflation)
    pf.add_feedstock(name='Cooling water',usage=cooling_water_usage,unit='Gallon per kilogram of ammonia',cost=cooling_water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Iron based catalyst',usage=iron_based_catalyst_usage,unit='kilogram of catalyst per kilogram of ammonia',cost=iron_based_catalyst_cost,escalation=gen_inflation)
    pf.add_coproduct(name='Oxygen byproduct',usage=oxygen_byproduct,unit='kilogram of oxygen per kilogram of ammonia',cost=oxygen_price,escalation=gen_inflation)
    
    #------------------------------ Sovle for breakeven price ---------------------------
    
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    price_breakdown = pf.get_cost_breakdown()
    
    price_breakdown_air_separation_by_cryogenic = price_breakdown.loc[price_breakdown['Name']=='Air Separation by Cryogenic','NPV'].tolist()[0]
    price_breakdown_Haber_Bosch = price_breakdown.loc[price_breakdown['Name']=='Haber Bosch','NPV'].tolist()[0]
    price_breakdown_boiler_and_steam_turbine = price_breakdown.loc[price_breakdown['Name']=='Boiler and Steam Turbine','NPV'].tolist()[0]
    price_breakdown_cooling_tower = price_breakdown.loc[price_breakdown['Name']=='Cooling Tower','NPV'].tolist()[0]
    price_breakdown_depreciable_nonequipment = price_breakdown.loc[price_breakdown['Name']=='Depreciable Nonequipment','NPV'].tolist()[0]
    price_breakdown_installation = price_breakdown.loc[price_breakdown['Name']=='Installation cost','NPV'].tolist()[0]
 

    price_breakdown_labor_cost_annual = price_breakdown.loc[price_breakdown['Name']=='Labor Cost','NPV'].tolist()[0]  
    price_breakdown_maintenance_cost = price_breakdown.loc[price_breakdown['Name']=='Maintenance Cost','NPV'].tolist()[0]  
    price_breakdown_administrative_expense = price_breakdown.loc[price_breakdown['Name']=='Administrative Expense','NPV'].tolist()[0]  
    price_breakdown_property_tax_and_insurance = price_breakdown.loc[price_breakdown['Name']=='Property tax and insurance','NPV'].tolist()[0]
    #price_breakdown_land_cost = price_breakdown.loc[price_breakdown['Name']=='Land cost','NPV'].tolist()[0]  

    if levelized_cost_of_hydrogen < 0:
        price_breakdown_hydrogen = -1*price_breakdown.loc[price_breakdown['Name']=='Hydrogen','NPV'].tolist()[0] 
    else:
        price_breakdown_hydrogen = price_breakdown.loc[price_breakdown['Name']=='Hydrogen','NPV'].tolist()[0] 
    price_breakdown_electricity = price_breakdown.loc[price_breakdown['Name']=='Electricity','NPV'].tolist()[0] 
    price_breakdown_cooling_water = price_breakdown.loc[price_breakdown['Name']=='Cooling water','NPV'].tolist()[0]
    price_breakdown_iron_based_catalyst = price_breakdown.loc[price_breakdown['Name']=='Iron based catalyst','NPV'].tolist()[0]
    price_breakdown_oxygen_byproduct = -price_breakdown.loc[price_breakdown['Name']=='Oxygen byproduct','NPV'].tolist()[0]

    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
        
    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]
        
    # price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Property tax and insurance','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]

        # Calculate financial expense associated with equipment
    price_breakdown_financial_equipment = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
        
    # Calculate remaining financial expenses
    price_breakdown_financial_remaining = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property tax and insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
        
    price_check = price_breakdown_air_separation_by_cryogenic+price_breakdown_Haber_Bosch+price_breakdown_boiler_and_steam_turbine+price_breakdown_cooling_tower+price_breakdown_depreciable_nonequipment\
        +price_breakdown_installation+price_breakdown_labor_cost_annual+price_breakdown_maintenance_cost+price_breakdown_administrative_expense\
        +price_breakdown_hydrogen+price_breakdown_electricity+price_breakdown_cooling_water+price_breakdown_iron_based_catalyst-price_breakdown_oxygen_byproduct\
        +price_breakdown_taxes+price_breakdown_financial_equipment+price_breakdown_financial_remaining
        
    ammonia_price_breakdown = {'Ammonia price: Air Separation by Cryogenic ($/kg)':price_breakdown_air_separation_by_cryogenic,
                               'Ammonia price: Haber Bosch ($/kg)':price_breakdown_Haber_Bosch,'Ammonia price: Boiler and Steam Turbine ($/kg)':price_breakdown_boiler_and_steam_turbine,
                               'Ammonia price: Cooling Tower ($/kg)':price_breakdown_cooling_tower,'Ammonia price: Depreciable Nonequipment ($/kg)':price_breakdown_depreciable_nonequipment,
                               'Ammonia price: Labor Cost ($/kg)':price_breakdown_labor_cost_annual,'Ammonia price: Maintenance Cost ($/kg)':price_breakdown_maintenance_cost,
                               'Ammonia price: Administrative Expense ($/kg)':price_breakdown_administrative_expense,
                               'Ammonia price: Hydrogen ($/kg)':price_breakdown_hydrogen,'Ammonia price: Electricity ($/kg)':price_breakdown_electricity,
                               'Ammonia price: Cooling water ($/kg)':price_breakdown_cooling_water,'Ammonia price: Iron based catalyst ($/kg)':price_breakdown_iron_based_catalyst,
                               'Ammonia price: Oxygen byproduct ($/kg)':price_breakdown_oxygen_byproduct,'Ammonia price: Taxes ($/kg)':price_breakdown_taxes,
                               'Ammonia price: Equipment Financing ($/kg)':price_breakdown_financial_equipment,
                               'Ammonia price: Remaining Financial ($/kg)':price_breakdown_financial_remaining,'Ammonia price: Total ($/kg)':price_check}
    
    price_breakdown = price_breakdown.drop(columns=['index','Amount'])
    
    return(sol,summary,price_breakdown,ammonia_production_kgpy,ammonia_price_breakdown,total_capex)