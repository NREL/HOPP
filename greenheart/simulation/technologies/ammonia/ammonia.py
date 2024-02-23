"""
Author: Abhineet Gupta and Kaitlin Brunik
Created: 02/22/2024
Institution: National Renewable Energy Lab
Description: This file outputs 
Costs are in 2018 USD

Sources:
    - [1] 
"""
from typing import Dict, Union
import ProFAST

import pandas as pd
from attrs import define, Factory


@define
class Feedstocks:
    """Feedstock consumption and related costs."""
    # electricity cost, $/MWh of electricty
    electricity_cost: float

    # hydrogen cost, $/kg of hydrogen
    hydrogen_cost: float

    # cooling water cost, $/Gal of water
    cooling_water_cost: float

    # iron based catalyst cost, $/kg of iron based catalyst
    iron_based_catalyst_cost: float

    # oxygen cost, $/kg of oxygen
    oxygen_cost: float

    # electricity consumption, MWh of electricity/kg of ammonia production
    electricity_consumption: float = 0.1207/1000

    # hydrogen consumption, kg of hydrogen/kg of ammonia production
    hydrogen_consumption = 0.197284403 

    # cooling water consumption, gallons of water/kg of ammonia production
    cooling_water_consumption = 0.049236824 

    # iron based catalyst consumption, kg of iron based catylyst/kg of ammonia production
    iron_based_catalyst_consumption = 0.000091295354067341 

    # oxygen byproduct, kg of oxygen/kg of ammonia production
    oxygen_byproduct = 0.29405077250145

@define
class AmmoniaCostModelConfig:
    """Ammonia cost model inputs."""
    # plant capacity, kg per year
    plant_capacity_kgpy: float

    # plant capacity factor
    plant_capacity_factor: float

    feedstocks: Feedstocks

@define
class AmmoniaCosts:
    """Calculated ammonia plant costs."""
    # CapEx
    capex_air_separation_crygenic: float
    capex_haber_bosch: float
    capex_boiler: float
    capex_cooling_tower: float
    capex_direct: float
    capex_depreciable_nonequipment: float
    land_cost: float

    #Fixed OpEx
    labor_cost: float
    general_administration_cost: float
    property_tax_insurance: float
    maintenance_cost: float

    #Feedstock and Byproduct costs
    H2_cost_in_startup_year: float
    energy_cost_in_startup_year: float
    non_energy_cost_in_startup_year: float
    variable_cost_in_startup_year: float
    credits_byproduct: float

@define
class AmmoniaCostModelOutputs(AmmoniaCosts):
    """Calculated ammonia costs."""
    #CapEx
    capex_total: float

    #Fixed OpEx
    total_fixed_operating_cost:float

def run_ammonia_model(plant_capacity_kgpy: float, plant_capacity_factor: float) -> float:
    ammonia_production_kgpy = plant_capacity_kgpy*plant_capacity_factor 
    
    return ammonia_production_kgpy

def run_ammonia_costs(config: AmmoniaCostModelConfig) -> AmmoniaCosts:
    feedstocks = config.feedstocks

    model_year_CEPCI = 596.2 #TODO: what year
    equation_year_CEPCI = 541.7 #TODO: what year

    # scale with respect to a baseline plant (What is this?)
    scaling_ratio = config.plant_capacity_kgpy/(365.0*1266638.4)

   # -------------------------------CapEx Costs------------------------------
    scaling_factor_equipment = 0.6

    capex_scale_factor = scaling_ratio**scaling_factor_equipment
    capex_air_separation_crygenic = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 22506100 
        * capex_scale_factor
    )
    capex_haber_bosch = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 18642800 
        * capex_scale_factor
    )
    capex_boiler = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 7069100 
        * capex_scale_factor
    )
    capex_cooling_tower = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 4799200 
        * capex_scale_factor
    )
    capex_direct = (
        capex_air_separation_crygenic 
        + capex_haber_bosch
        + capex_boiler
        + capex_cooling_tower
    )
    capex_depreciable_nonequipment = (
        capex_direct * 0.42
        + 4112701.84103543
        * scaling_ratio
    )
    capex_total = (
        capex_direct 
        + capex_depreciable_nonequipment
    )
    land_cost = capex_depreciable_nonequipment  #TODO: determine if this is the right method or the one in Fixed O&M costs

    # -------------------------------Fixed O&M Costs------------------------------
    scaling_factor_labor = 0.25
    labor_cost = (
        57 * 50 * 2080 
        * scaling_ratio**scaling_factor_labor
    )
    general_administration_cost = (
        labor_cost * 0.2
    )
    property_tax_insurance = (
        capex_total * 0.02
    )
    maintenance_cost = (
        capex_direct * 0.005
        * scaling_ratio**scaling_factor_equipment
    )
    land_cost = (
        2500000
        *capex_scale_factor
    )
    total_fixed_operating_cost = (
        land_cost 
        + labor_cost 
        + general_administration_cost 
        + property_tax_insurance 
        + maintenance_cost
    )

    # -------------------------------Feedstock Costs------------------------------
    H2_cost_in_startup_year = (
        feedstocks.hydrogen_cost 
        * feedstocks.hydrogen_consumption
        * config.plant_capacity_kgpy 
        * config.plant_capacity_factor
    )
    energy_cost_in_startup_year = (
        feedstocks.electricity_cost 
        * feedstocks.electricity_consumption
        * config.plant_capacity_kgpy 
        * config.plant_capacity_factor
    )
    non_energy_cost_in_startup_year = (
        ((feedstocks.cooling_water_cost 
        * feedstocks.cooling_water_consumption) 
        + (feedstocks.iron_based_catalyst_cost
        * feedstocks.iron_based_catalyst_consumption)) 
        * config.plant_capacity_kgpy 
        * config.plant_capacity_factor
    )
    variable_cost_in_startup_year = (
        energy_cost_in_startup_year
        + non_energy_cost_in_startup_year
    )
    # -------------------------------Byproduct Costs------------------------------
    credits_byproduct = (
        feedstocks.oxygen_cost
        * feedstocks.oxygen_byproduct 
        * config.plant_capacity_kgpy 
        * config.plant_capacity_factor
    )

    return AmmoniaCostModelOutputs(
        #Capex
        capex_air_separation_crygenic=capex_air_separation_crygenic,
        capex_haber_bosch=capex_haber_bosch,
        capex_boiler=capex_boiler,
        capex_cooling_tower=capex_cooling_tower,
        capex_direct=capex_direct,
        capex_depreciable_nonequipment=capex_depreciable_nonequipment,
        capex_total=capex_total,
        land_cost=land_cost,
        #Fixed OpEx
        labor_cost=labor_cost,
        general_administration_cost=general_administration_cost,
        property_tax_insurance=property_tax_insurance,
        maintenance_cost=maintenance_cost,
        total_fixed_operating_cost=total_fixed_operating_cost,
        #Feedstock & Byproducts
        H2_cost_in_startup_year=H2_cost_in_startup_year,
        energy_cost_in_startup_year=energy_cost_in_startup_year,
        non_energy_cost_in_startup_year=non_energy_cost_in_startup_year,
        variable_cost_in_startup_year=variable_cost_in_startup_year,
        credits_byproduct=credits_byproduct
    )

@define
class AmmoniaFinanceModelConfig:
    plant_life: int
    plant_capacity_kgpy: float
    plant_capacity_factor: float

    grid_prices: Dict[int, float]

    # raw cost inputs
    feedstocks: Feedstocks

    # calculated CapEx/OpEx costs
    costs: Union[AmmoniaCosts, AmmoniaCostModelOutputs]

    financial_assumptions: Dict[str, float] = Factory(dict)

    install_years: int = 3

@define
class AmmoniaFinanceModelOutputs:
    sol: dict
    summary: dict
    price_breakdown: pd.DataFrame
    ammonia_price_breakdown: dict

def run_ammonia_finance(config: AmmoniaFinanceModelConfig) -> AmmoniaFinanceModelOutputs:
    feedstocks = config.feedstocks
    costs = config.costs
    
     # Set up ProFAST
    pf = ProFAST.ProFAST('blank')

    # apply all params passed through from config
    for param, val in config.financial_assumptions.items():
        pf.set_params(param, val)

    analysis_start = int(list(config.grid_prices.keys())[0]) - config.install_years
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params(
        'commodity',
        {
            "name":'Ammonia',
            "unit":"kg",
            "initial price":1000,
            "escalation":gen_inflation
            },
    )
    pf.set_params('capacity',config.plant_capacity_kgpy/365) #units/day
    pf.set_params('maintenance',
        {
            "value":0,
            "escalation": gen_inflation
         },
    )
    pf.set_params('analysis start year', analysis_start)
    pf.set_params('operating life', config.plant_life)
    pf.set_params('installation months',12*config.install_years)
    pf.set_params('installation cost',
        {
            "value":costs.total_fixed_operating_cost,
            "depr type":"Straight line",
            "depr period":4,
            "depreciable":False
        }
    )
    pf.set_params('non depr assets',costs.land_cost)
    pf.set_params(
        'end of proj sale non depr assets',
        costs.land_cost*(1+gen_inflation)**config.plant_life
    )
    pf.set_params('demand rampup',0)
    pf.set_params(
        'long term utilization',
        config.plant_capacity_factor
    )
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',
        {
            'value':00,
            'escalation':gen_inflation
        }
    )
    pf.set_params('rent',
        {
            'value':0,
            'escalation':gen_inflation
        }
    )
    pf.set_params('property tax and insurance',0)
    pf.set_params('admin expense',0)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('cash onhand',1)
    
    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(
        name="Air Separation by Cryogenic",
        cost=costs.capex_air_separation_crygenic,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0]
    )
    pf.add_capital_item(
        name="Haber Bosch",
        cost=costs.capex_haber_bosch,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0]
    )
    pf.add_capital_item(
        name="Boiler and Steam Turbine",
        cost=costs.capex_boiler,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0]
    )
    pf.add_capital_item(
        name="Cooling Tower",
        cost=costs.capex_cooling_tower,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0]
    )
    pf.add_capital_item(
        name="Depreciable Nonequipment",
        cost=costs.capex_depreciable_nonequipment,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0]
    )

    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(
        name="Labor Cost",
        usage=1,
        unit='$/year',
        cost=costs.labor_cost,
        escalation=gen_inflation
    )
    pf.add_fixed_cost(
        name="Maintenance Cost",
        usage=1,
        unit='$/year',
        cost=costs.maintenance_cost,
        escalation=gen_inflation
    )
    pf.add_fixed_cost(
        name="Administrative Expense",
        usage=1,
        unit='$/year',
        cost=costs.general_administration_cost,
        escalation=gen_inflation
    )
    pf.add_fixed_cost(
        name="Property tax and insurance",
        usage=1,
        unit='$/year',
        cost=costs.property_tax_insurance,
        escalation=0.0
    )

    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(
        name='Hydrogen',
        usage=feedstocks.hydrogen_consumption,
        unit='kilogram of hydrogen per kilogram of ammonia',
        cost=feedstocks.hydrogen_cost,
        escalation=gen_inflation
    )

    pf.add_feedstock(
        name='Electricity',
        usage=feedstocks.electricity_consumption,
        unit='MWh per kilogram of ammonia',
        cost=config.grid_prices,
        escalation=gen_inflation
    )
    pf.add_feedstock(
        name='Cooling water',
        usage=feedstocks.cooling_water_consumption,
        unit='Gallon per kilogram of ammonia',
        cost=feedstocks.cooling_water_cost,
        escalation=gen_inflation
    )
    pf.add_feedstock(
        name='Iron based catalyst',
        usage=feedstocks.iron_based_catalyst_consumption,
        unit='kilogram of catalyst per kilogram of ammonia',
        cost=feedstocks.iron_based_catalyst_cost,
        escalation=gen_inflation
    )
    pf.add_coproduct(
        name='Oxygen byproduct',
        usage=feedstocks.oxygen_byproduct,
        unit='kilogram of oxygen per kilogram of ammonia',
        cost=feedstocks.oxygen_cost,
        escalation=gen_inflation
    )
    
    #------------------------------ Sovle for breakeven price ---------------------------
    
    sol = pf.solve_price()

    summary = pf.get_summary_vals()
    
    price_breakdown = pf.get_cost_breakdown()
    
    price_breakdown_air_separation_by_cryogenic = price_breakdown.loc[
            price_breakdown['Name']=='Air Separation by Cryogenic','NPV'
        ].tolist()[0]
    price_breakdown_Haber_Bosch = price_breakdown.loc[
            price_breakdown['Name']=='Haber Bosch','NPV'
        ].tolist()[0]
    price_breakdown_boiler_and_steam_turbine = price_breakdown.loc[
            price_breakdown['Name']=='Boiler and Steam Turbine','NPV'
        ].tolist()[0]
    price_breakdown_cooling_tower = price_breakdown.loc[
            price_breakdown['Name']=='Cooling Tower','NPV'
        ].tolist()[0]
    price_breakdown_depreciable_nonequipment = price_breakdown.loc[
            price_breakdown['Name']=='Depreciable Nonequipment','NPV'
        ].tolist()[0]
    price_breakdown_installation = price_breakdown.loc[
            price_breakdown['Name']=='Installation cost','NPV'
        ].tolist()[0]
 

    price_breakdown_labor_cost_annual = price_breakdown.loc[
            price_breakdown['Name']=='Labor Cost','NPV'
        ].tolist()[0]  
    price_breakdown_maintenance_cost = price_breakdown.loc[
            price_breakdown['Name']=='Maintenance Cost','NPV'
        ].tolist()[0]  
    price_breakdown_administrative_expense = price_breakdown.loc[
            price_breakdown['Name']=='Administrative Expense','NPV'
        ].tolist()[0]  
    price_breakdown_property_tax_and_insurance = price_breakdown.loc[
            price_breakdown['Name']=='Property tax and insurance','NPV'
        ].tolist()[0]

    if feedstocks.hydrogen_cost < 0:
        price_breakdown_hydrogen = -1*price_breakdown.loc[
            price_breakdown['Name']=='Hydrogen','NPV'
        ].tolist()[0] 
    else:
        price_breakdown_hydrogen = price_breakdown.loc[
            price_breakdown['Name']=='Hydrogen','NPV'
        ].tolist()[0] 
    price_breakdown_electricity = price_breakdown.loc[
        price_breakdown['Name']=='Electricity','NPV'
        ].tolist()[0] 
    price_breakdown_cooling_water = price_breakdown.loc[
            price_breakdown['Name']=='Cooling water','NPV'
        ].tolist()[0]
    price_breakdown_iron_based_catalyst = price_breakdown.loc[
            price_breakdown['Name']=='Iron based catalyst','NPV'
        ].tolist()[0]
    price_breakdown_oxygen_byproduct = -price_breakdown.loc[
            price_breakdown['Name']=='Oxygen byproduct','NPV'
        ].tolist()[0]

    price_breakdown_taxes = (price_breakdown.loc[
            price_breakdown['Name']=='Income taxes payable','NPV'
        ].tolist()[0] 
        - price_breakdown.loc[
            price_breakdown['Name'] == 'Monetized tax losses','NPV'
        ].tolist()[0]
    )
    if gen_inflation > 0:
        price_breakdown_taxes = (
            price_breakdown_taxes 
            + price_breakdown.loc[
                price_breakdown['Name']=='Capital gains taxes payable','NPV'
                ].tolist()[0]
        )
        # Calculate financial expense associated with equipment
    price_breakdown_financial_equipment = (
        price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
    )
    # Calculate remaining financial expenses
    price_breakdown_financial_remaining = (
        price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]
        + price_breakdown.loc[price_breakdown['Name']=='Property tax and insurance','NPV'].tolist()[0]
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
    )
    price_check = (
        price_breakdown_air_separation_by_cryogenic
        +price_breakdown_Haber_Bosch
        +price_breakdown_boiler_and_steam_turbine
        +price_breakdown_cooling_tower
        +price_breakdown_depreciable_nonequipment
        +price_breakdown_installation
        +price_breakdown_labor_cost_annual
        +price_breakdown_maintenance_cost
        +price_breakdown_administrative_expense
        +price_breakdown_hydrogen
        +price_breakdown_electricity
        +price_breakdown_cooling_water
        +price_breakdown_iron_based_catalyst
        -price_breakdown_oxygen_byproduct
        +price_breakdown_taxes
        +price_breakdown_financial_equipment
        +price_breakdown_financial_remaining
    )
    ammonia_price_breakdown = {
        'Ammonia price: Air Separation by Cryogenic ($/kg)':price_breakdown_air_separation_by_cryogenic,
        'Ammonia price: Haber Bosch ($/kg)':price_breakdown_Haber_Bosch,
        'Ammonia price: Boiler and Steam Turbine ($/kg)':price_breakdown_boiler_and_steam_turbine,
        'Ammonia price: Cooling Tower ($/kg)':price_breakdown_cooling_tower,
        'Ammonia price: Depreciable Nonequipment ($/kg)':price_breakdown_depreciable_nonequipment,
        'Ammonia price: Labor Cost ($/kg)':price_breakdown_labor_cost_annual,
        'Ammonia price: Maintenance Cost ($/kg)':price_breakdown_maintenance_cost,
        'Ammonia price: Administrative Expense ($/kg)':price_breakdown_administrative_expense,
        'Ammonia price: Hydrogen ($/kg)':price_breakdown_hydrogen,
        'Ammonia price: Electricity ($/kg)':price_breakdown_electricity,
        'Ammonia price: Cooling water ($/kg)':price_breakdown_cooling_water,
        'Ammonia price: Iron based catalyst ($/kg)':price_breakdown_iron_based_catalyst,
        'Ammonia price: Oxygen byproduct ($/kg)':price_breakdown_oxygen_byproduct,
        'Ammonia price: Taxes ($/kg)':price_breakdown_taxes,
        'Ammonia price: Equipment Financing ($/kg)':price_breakdown_financial_equipment,
        'Ammonia price: Remaining Financial ($/kg)':price_breakdown_financial_remaining,
        'Ammonia price: Total ($/kg)':price_check
    }

    price_breakdown = price_breakdown.drop(columns=['Amount'])

    return AmmoniaFinanceModelOutputs(
        sol=sol,
        summary=summary,
        price_breakdown=price_breakdown,
        ammonia_price_breakdown=ammonia_price_breakdown
    )