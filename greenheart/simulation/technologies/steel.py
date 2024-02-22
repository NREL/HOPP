from typing import Dict, Union
import ProFAST

import pandas as pd
from attrs import define


@define
class Feedstocks:
    """Feedstock consumption and related costs."""
    # electricity cost, $/metric tonne of steel production
    electricity_cost: float

    # natural gas costs, indexed by year
    naturalgas_prices: Dict[int, float]

    # $/metric tonne of lime
    lime_unitcost: float

    # $/metric tonne of Carbon
    carbon_unitcost: float

    # $/metric tone of Ore
    iron_ore_pellet_unitcost: float

    # kgO2
    excess_oxygen: float

    # $/kgO2
    # NOTE: should be 0 when o2_heat_integration == False, handle in finance code?
    oxygen_market_price: float = 0.03 

    # $/metric tonne of raw water
    raw_water_unitcost: float = 0.59289

    # metric tonnes of iron ore/metric tonne of steel production
    iron_ore_consumption: float = 1.62927

    # metric tonnes of raw water/metric tonne of steel production
    raw_water_consumption: float = 0.80367

    # metric tonnes of lime/metric tonne of steel production
    lime_consumption: float = 0.01812

    # metric tonnes of carbon/metric tonne of steel production
    carbon_consumption: float = 0.0538

    # metric tonnes of hydrogen/metric tonne of steel production
    hydrogen_consumption: float = 0.06596

    # GJ-LHV/metric tonne of steel production
    natural_gas_consumption: float = 0.71657

    # MWh/metric tonne of steel production
    electricity_consumption: float = 0.5502

    # $ metric tonne of Slag
    slag_disposal_unitcost: float = 37.63

    # metric tonnes of slag/metric tonne of steel production
    slag_production: float = 0.17433

    # $/metric tonne of annual steel slab production at real CF
    maintenance_materials_unitcost: float = 7.72


@define
class SteelCostModelConfig:
    """Steel cost model inputs."""
    operational_year: int

    # plant capacity, Mt per year
    plant_capacity_mtpy: float

    # levelized cost of hydrogen
    lcoh: float

    feedstocks: Feedstocks

    # O2 heat integration, affects preheating CapEx, cooling CapEx, O2 sales
    o2_heat_integration: bool

    # metric tonnes of CO2/metric tonne of steel production
    co2_fuel_emissions: float = 0.03929

    # metric tonnes of CO2/metric tonne of steel production
    co2_carbon_emissions: float = 0.17466

    # metric tonnes of surface water discharge/metridc tonne of steel production
    surface_water_discharge: float = 0.42113


@define
class SteelCosts:
    """
    Calculated steel costs shared between steel cost model and steel finance model.
    """
    # CapEx
    capex_eaf_casting: float
    capex_shaft_furnace: float
    capex_oxygen_supply: float
    capex_h2_preheating: float
    capex_cooling_tower: float
    capex_piping: float
    capex_elec_instr: float
    capex_buildings_storage_water: float
    capex_misc: float

    # OpEx
    labor_cost_annual_operation: float
    labor_cost_maintenance: float
    labor_cost_admin_support: float
    property_tax_insurance: float

    # Owner installation
    land_cost: float
    installation_cost: float


@define
class SteelCostModelOutputs(SteelCosts):
    """
    Calculated steel costs.
    """

    # CapEx
    total_plant_cost: float

    # OpEx
    total_fixed_operating_cost: float
    
    # Owner installation
    labor_cost_fivemonth: float
    maintenance_materials_onemonth: float
    non_fuel_consumables_onemonth: float
    waste_disposal_onemonth: float
    monthly_energy_cost: float
    spare_parts_cost: float
    misc_owners_costs: float


def run_steel_model(plant_capacity_mtpy: float, plant_capacity_factor: float) -> float:
    steel_production_mtpy = plant_capacity_mtpy * plant_capacity_factor

    return steel_production_mtpy


def run_steel_costs(config: SteelCostModelConfig) -> SteelCosts:
    feedstocks = config.feedstocks

    model_year_CEPCI = 596.2
    equation_year_CEPCI = 708.8

    capex_eaf_casting = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 352191.5237
        * config.plant_capacity_mtpy**0.456
    )
    capex_shaft_furnace = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 489.68061
        * config.plant_capacity_mtpy**0.88741
    )
    capex_oxygen_supply = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 1715.21508
        * config.plant_capacity_mtpy**0.64574
    )
    if config.o2_heat_integration:
        capex_h2_preheating = (
            model_year_CEPCI
            / equation_year_CEPCI
            * (1 - 0.4)
            * (45.69123 * config.plant_capacity_mtpy**0.86564)
        )  # Optimistic ballpark estimate of 60% reduction in preheating
        capex_cooling_tower = (
            model_year_CEPCI
            / equation_year_CEPCI
            * (1 - 0.3)
            * (2513.08314 * config.plant_capacity_mtpy**0.63325)
        )  # Optimistic ballpark estimate of 30% reduction in cooling
    else:
        capex_h2_preheating = (
            model_year_CEPCI
            / equation_year_CEPCI
            * 45.69123
            * config.plant_capacity_mtpy**0.86564
        )
        capex_cooling_tower = (
            model_year_CEPCI
            / equation_year_CEPCI
            * 2513.08314
            * config.plant_capacity_mtpy**0.63325
        )
    capex_piping = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 11815.72718
        * config.plant_capacity_mtpy**0.59983
    )
    capex_elec_instr = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 7877.15146
        * config.plant_capacity_mtpy**0.59983
    )
    capex_buildings_storage_water = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 1097.81876
        * config.plant_capacity_mtpy**0.8
    )
    capex_misc = (
        model_year_CEPCI
        / equation_year_CEPCI
        * 7877.1546
        * config.plant_capacity_mtpy**0.59983
    )

    total_plant_cost = (
        capex_eaf_casting
        + capex_shaft_furnace
        + capex_oxygen_supply
        + capex_h2_preheating
        + capex_cooling_tower
        + capex_piping
        + capex_elec_instr
        + capex_buildings_storage_water
        + capex_misc
    )

    # -------------------------------Fixed O&M Costs------------------------------

    labor_cost_annual_operation = (
        69375996.9
        * ((config.plant_capacity_mtpy / 365 * 1000) ** 0.25242)
        / ((1162077 / 365 * 1000) ** 0.25242)
    )
    labor_cost_maintenance = 0.00863 * total_plant_cost
    labor_cost_admin_support = 0.25 * (
        labor_cost_annual_operation + labor_cost_maintenance
    )

    property_tax_insurance = 0.02 * total_plant_cost

    total_fixed_operating_cost = (
        labor_cost_annual_operation
        + labor_cost_maintenance
        + labor_cost_admin_support
        + property_tax_insurance
    )

    # ---------------------- Owner's (Installation) Costs --------------------------
    labor_cost_fivemonth = (
        5
        / 12
        * (
            labor_cost_annual_operation
            + labor_cost_maintenance
            + labor_cost_admin_support
        )
    )

    maintenance_materials_onemonth = (
        feedstocks.maintenance_materials_unitcost * config.plant_capacity_mtpy / 12
    )
    non_fuel_consumables_onemonth = (
        config.plant_capacity_mtpy
        * (
            feedstocks.raw_water_consumption * feedstocks.raw_water_unitcost
            + feedstocks.lime_consumption * feedstocks.lime_unitcost
            + feedstocks.carbon_consumption * feedstocks.carbon_unitcost
            + feedstocks.iron_ore_consumption * feedstocks.iron_ore_pellet_unitcost
        )
        / 12
    )

    waste_disposal_onemonth = (
        config.plant_capacity_mtpy
        * feedstocks.slag_disposal_unitcost
        * feedstocks.slag_production
        / 12
    )

    monthly_energy_cost = (
        config.plant_capacity_mtpy
        * (
            feedstocks.hydrogen_consumption * config.lcoh * 1000
            + feedstocks.natural_gas_consumption
            * feedstocks.naturalgas_prices[config.operational_year]
            + feedstocks.electricity_consumption * feedstocks.electricity_cost
        )
        / 12
    )
    two_percent_tpc = 0.02 * total_plant_cost

    fuel_consumables_60day_supply_cost = (
        config.plant_capacity_mtpy
        * (
            feedstocks.raw_water_consumption * feedstocks.raw_water_unitcost
            + feedstocks.lime_consumption * feedstocks.lime_unitcost
            + feedstocks.carbon_consumption * feedstocks.carbon_unitcost
            + feedstocks.iron_ore_consumption * feedstocks.iron_ore_pellet_unitcost
        )
        / 365
        * 60
    )

    spare_parts_cost = 0.005 * total_plant_cost
    land_cost = 0.775 * config.plant_capacity_mtpy
    misc_owners_costs = 0.15 * total_plant_cost

    installation_cost = (
        labor_cost_fivemonth
        + two_percent_tpc
        + fuel_consumables_60day_supply_cost
        + spare_parts_cost
        + misc_owners_costs
    )

    return SteelCostModelOutputs(
        # CapEx
        capex_eaf_casting=capex_eaf_casting,
        capex_shaft_furnace=capex_shaft_furnace,
        capex_oxygen_supply=capex_oxygen_supply,
        capex_h2_preheating=capex_h2_preheating,
        capex_cooling_tower=capex_cooling_tower,
        capex_piping=capex_piping,
        capex_elec_instr=capex_elec_instr,
        capex_buildings_storage_water=capex_buildings_storage_water,
        capex_misc=capex_misc,
        total_plant_cost=total_plant_cost,
        # Fixed OpEx
        labor_cost_annual_operation=labor_cost_annual_operation,
        labor_cost_maintenance=labor_cost_maintenance,
        labor_cost_admin_support=labor_cost_admin_support,
        property_tax_insurance=property_tax_insurance,
        total_fixed_operating_cost=total_fixed_operating_cost,
        # Owner's Installation costs
        labor_cost_fivemonth=labor_cost_fivemonth,
        maintenance_materials_onemonth=maintenance_materials_onemonth,
        non_fuel_consumables_onemonth=non_fuel_consumables_onemonth,
        waste_disposal_onemonth=waste_disposal_onemonth,
        monthly_energy_cost=monthly_energy_cost,
        spare_parts_cost=spare_parts_cost,
        land_cost=land_cost,
        misc_owners_costs=misc_owners_costs,
        installation_cost=installation_cost,
    )


@define
class SteelFinanceModelConfig:
    plant_life: int
    plant_capacity_mtpy: float
    plant_capacity_factor: float
    steel_production_mtpy: float
    o2_heat_integration: bool

    lcoh: float
    financial_assumptions: str
    grid_prices: dict

    # raw cost inputs
    feedstocks: Feedstocks

    # calculated CapEx/OpEx costs
    costs: Union[SteelCosts, SteelCostModelOutputs]


@define
class SteelFinanceModelOutputs:
    sol: dict
    summary: dict
    price_breakdown: pd.DataFrame
    steel_price_breakdown: dict


def run_steel_finance(config: SteelFinanceModelConfig) -> SteelFinanceModelOutputs:
    feedstocks = config.feedstocks
    costs = config.costs

    financial_assumptions = pd.read_csv(
        config.financial_assumptions
        # "H2_Analysis/financial_inputs.csv", index_col=None, header=0
    )
    financial_assumptions.set_index(["Parameter"], inplace=True)
    financial_assumptions = financial_assumptions["Hydrogen/Steel/Ammonia"]

    # Set up ProFAST
    pf = ProFAST.ProFAST("blank")

    install_years = 3
    analysis_start = list(config.grid_prices.keys())[0] - install_years

    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params(
        "commodity",
        {
            "name": "Steel",
            "unit": "metric tonnes",
            "initial price": 1000,
            "escalation": gen_inflation,
        },
    )
    pf.set_params("capacity", config.plant_capacity_mtpy / 365)  # units/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params("analysis start year", analysis_start)
    pf.set_params("operating life", config.plant_life)
    pf.set_params("installation months", 12 * install_years)
    pf.set_params(
        "installation cost",
        {
            "value": costs.installation_cost,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    pf.set_params("non depr assets", costs.land_cost)
    pf.set_params(
        "end of proj sale non depr assets",
        costs.land_cost * (1 + gen_inflation) ** config.plant_life,
    )
    pf.set_params("demand rampup", 5.3)
    pf.set_params("long term utilization", config.plant_capacity_factor)
    pf.set_params("credit card fees", 0)
    pf.set_params("sales tax", 0)
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    pf.set_params("property tax and insurance", 0)
    pf.set_params("admin expense", 0)
    pf.set_params(
        "total income tax rate", financial_assumptions["total income tax rate"]
    )
    pf.set_params(
        "capital gains tax rate", financial_assumptions["capital gains tax rate"]
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        financial_assumptions["leverage after tax nominal discount rate"],
    )
    pf.set_params(
        "debt equity ratio of initial financing",
        financial_assumptions["debt equity ratio of initial financing"],
    )
    pf.set_params("debt type", "Revolving debt")
    pf.set_params("debt interest rate", financial_assumptions["debt interest rate"])
    pf.set_params("cash onhand", 1)

    # ----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(
        name="EAF & Casting",
        cost=costs.capex_eaf_casting,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Shaft Furnace",
        cost=costs.capex_shaft_furnace,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Oxygen Supply",
        cost=costs.capex_oxygen_supply,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="H2 Pre-heating",
        cost=costs.capex_h2_preheating,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Cooling Tower",
        cost=costs.capex_cooling_tower,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Piping",
        cost=costs.capex_piping,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Electrical & Instrumentation",
        cost=costs.capex_elec_instr,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Buildings, Storage, Water Service",
        cost=costs.capex_buildings_storage_water,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Other Miscellaneous Costs",
        cost=costs.capex_misc,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )

    # -------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(
        name="Annual Operating Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost_annual_operation,
        escalation=gen_inflation,
    )
    pf.add_fixed_cost(
        name="Maintenance Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost_maintenance,
        escalation=gen_inflation,
    )
    pf.add_fixed_cost(
        name="Administrative & Support Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost_admin_support,
        escalation=gen_inflation,
    )
    pf.add_fixed_cost(
        name="Property tax and insurance",
        usage=1,
        unit="$/year",
        cost=costs.property_tax_insurance,
        escalation=0.0,
    )
    # Putting property tax and insurance here to zero out depcreciation/escalation. Could instead put it in set_params if
    # we think that is more accurate

    # ---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(
        name="Maintenance Materials",
        usage=1.0,
        unit="Units per metric tonne of steel",
        cost=feedstocks.maintenance_materials_unitcost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Raw Water Withdrawal",
        usage=feedstocks.raw_water_consumption,
        unit="metric tonnes of water per metric tonne of steel",
        cost=feedstocks.raw_water_unitcost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Lime",
        usage=feedstocks.lime_consumption,
        unit="metric tonnes of lime per metric tonne of steel",
        cost=feedstocks.lime_unitcost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Carbon",
        usage=feedstocks.carbon_consumption,
        unit="metric tonnes of carbon per metric tonne of steel",
        cost=feedstocks.carbon_unitcost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Iron Ore",
        usage=feedstocks.iron_ore_consumption,
        unit="metric tonnes of iron ore per metric tonne of steel",
        cost=feedstocks.iron_ore_pellet_unitcost,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Hydrogen",
        usage=feedstocks.hydrogen_consumption,
        unit="metric tonnes of hydrogen per metric tonne of steel",
        cost=config.lcoh * 1000,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Natural Gas",
        usage=feedstocks.natural_gas_consumption,
        unit="GJ-LHV per metric tonne of steel",
        cost=feedstocks.naturalgas_prices,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Electricity",
        usage=feedstocks.electricity_consumption,
        unit="MWh per metric tonne of steel",
        cost=config.grid_prices,
        escalation=gen_inflation,
    )
    pf.add_feedstock(
        name="Slag Disposal",
        usage=feedstocks.slag_production,
        unit="metric tonnes of slag per metric tonne of steel",
        cost=feedstocks.slag_disposal_unitcost,
        escalation=gen_inflation,
    )

    pf.add_coproduct(
        name="Oxygen sales",
        usage=feedstocks.excess_oxygen,
        unit="kg O2 per metric tonne of steel",
        cost=feedstocks.oxygen_market_price,
        escalation=gen_inflation,
    )

    # ------------------------------ Solve for breakeven price ---------------------------

    sol = pf.solve_price()

    summary = pf.get_summary_vals()

    price_breakdown = pf.get_cost_breakdown()

    price_breakdown_eaf_casting = price_breakdown.loc[
        price_breakdown["Name"] == "EAF & Casting", "NPV"
    ].tolist()[0]
    price_breakdown_shaft_furnace = price_breakdown.loc[
        price_breakdown["Name"] == "Shaft Furnace", "NPV"
    ].tolist()[0]
    price_breakdown_oxygen_supply = price_breakdown.loc[
        price_breakdown["Name"] == "Oxygen Supply", "NPV"
    ].tolist()[0]
    price_breakdown_h2_preheating = price_breakdown.loc[
        price_breakdown["Name"] == "H2 Pre-heating", "NPV"
    ].tolist()[0]
    price_breakdown_cooling_tower = price_breakdown.loc[
        price_breakdown["Name"] == "Cooling Tower", "NPV"
    ].tolist()[0]
    price_breakdown_piping = price_breakdown.loc[
        price_breakdown["Name"] == "Piping", "NPV"
    ].tolist()[0]
    price_breakdown_elec_instr = price_breakdown.loc[
        price_breakdown["Name"] == "Electrical & Instrumentation", "NPV"
    ].tolist()[0]
    price_breakdown_buildings_storage_water = price_breakdown.loc[
        price_breakdown["Name"] == "Buildings, Storage, Water Service", "NPV"
    ].tolist()[0]
    price_breakdown_misc = price_breakdown.loc[
        price_breakdown["Name"] == "Other Miscellaneous Costs", "NPV"
    ].tolist()[0]
    price_breakdown_installation = price_breakdown.loc[
        price_breakdown["Name"] == "Installation cost", "NPV"
    ].tolist()[0]
    price_breakdown_labor_cost_annual = price_breakdown.loc[
        price_breakdown["Name"] == "Annual Operating Labor Cost", "NPV"
    ].tolist()[0]
    price_breakdown_labor_cost_maintenance = price_breakdown.loc[
        price_breakdown["Name"] == "Maintenance Labor Cost", "NPV"
    ].tolist()[0]
    price_breakdown_labor_cost_admin_support = price_breakdown.loc[
        price_breakdown["Name"] == "Administrative & Support Labor Cost", "NPV"
    ].tolist()[0]
    price_breakdown_maintenance_materials = price_breakdown.loc[
        price_breakdown["Name"] == "Maintenance Materials", "NPV"
    ].tolist()[0]
    price_breakdown_water_withdrawal = price_breakdown.loc[
        price_breakdown["Name"] == "Raw Water Withdrawal", "NPV"
    ].tolist()[0]
    price_breakdown_lime = price_breakdown.loc[
        price_breakdown["Name"] == "Lime", "NPV"
    ].tolist()[0]
    price_breakdown_carbon = price_breakdown.loc[
        price_breakdown["Name"] == "Carbon", "NPV"
    ].tolist()[0]
    price_breakdown_iron_ore = price_breakdown.loc[
        price_breakdown["Name"] == "Iron Ore", "NPV"
    ].tolist()[0]
    if config.lcoh < 0:
        price_breakdown_hydrogen = (
            -1
            * price_breakdown.loc[
                price_breakdown["Name"] == "Hydrogen", "NPV"
            ].tolist()[0]
        )
    else:
        price_breakdown_hydrogen = price_breakdown.loc[
            price_breakdown["Name"] == "Hydrogen", "NPV"
        ].tolist()[0]
    price_breakdown_natural_gas = price_breakdown.loc[
        price_breakdown["Name"] == "Natural Gas", "NPV"
    ].tolist()[0]
    price_breakdown_electricity = price_breakdown.loc[
        price_breakdown["Name"] == "Electricity", "NPV"
    ].tolist()[0]
    price_breakdown_slag = price_breakdown.loc[
        price_breakdown["Name"] == "Slag Disposal", "NPV"
    ].tolist()[0]
    price_breakdown_taxes = (
        price_breakdown.loc[
            price_breakdown["Name"] == "Income taxes payable", "NPV"
        ].tolist()[0]
        - price_breakdown.loc[
            price_breakdown["Name"] == "Monetized tax losses", "NPV"
        ].tolist()[0]
    )
    if config.o2_heat_integration:
        price_breakdown_O2sales = price_breakdown.loc[
            price_breakdown["Name"] == "Oxygen sales", "NPV"
        ].tolist()[0]
    else:
        price_breakdown_O2sales = 0

    if gen_inflation > 0:
        price_breakdown_taxes = (
            price_breakdown_taxes
            + price_breakdown.loc[
                price_breakdown["Name"] == "Capital gains taxes payable", "NPV"
            ].tolist()[0]
        )

    # Calculate financial expense associated with equipment
    price_breakdown_financial_equipment = (
        price_breakdown.loc[
            price_breakdown["Name"] == "Repayment of debt", "NPV"
        ].tolist()[0]
        + price_breakdown.loc[
            price_breakdown["Name"] == "Interest expense", "NPV"
        ].tolist()[0]
        + price_breakdown.loc[
            price_breakdown["Name"] == "Dividends paid", "NPV"
        ].tolist()[0]
        - price_breakdown.loc[
            price_breakdown["Name"] == "Inflow of debt", "NPV"
        ].tolist()[0]
        - price_breakdown.loc[
            price_breakdown["Name"] == "Inflow of equity", "NPV"
        ].tolist()[0]
    )

    # Calculate remaining financial expenses
    price_breakdown_financial_remaining = (
        price_breakdown.loc[
            price_breakdown["Name"] == "Non-depreciable assets", "NPV"
        ].tolist()[0]
        + price_breakdown.loc[
            price_breakdown["Name"] == "Cash on hand reserve", "NPV"
        ].tolist()[0]
        + price_breakdown.loc[
            price_breakdown["Name"] == "Property tax and insurance", "NPV"
        ].tolist()[0]
        - price_breakdown.loc[
            price_breakdown["Name"] == "Sale of non-depreciable assets", "NPV"
        ].tolist()[0]
        - price_breakdown.loc[
            price_breakdown["Name"] == "Cash on hand recovery", "NPV"
        ].tolist()[0]
    )

    price_breakdown_check = (
        price_breakdown_eaf_casting
        + price_breakdown_shaft_furnace
        + price_breakdown_oxygen_supply
        + price_breakdown_h2_preheating
        + price_breakdown_cooling_tower
        + price_breakdown_piping
        + price_breakdown_elec_instr
        + price_breakdown_buildings_storage_water
        + price_breakdown_misc
        + price_breakdown_installation
        + price_breakdown_labor_cost_annual
        + price_breakdown_labor_cost_maintenance
        + price_breakdown_labor_cost_admin_support
        + price_breakdown_maintenance_materials
        + price_breakdown_water_withdrawal
        + price_breakdown_lime
        + price_breakdown_carbon
        + price_breakdown_iron_ore
        + price_breakdown_hydrogen
        + price_breakdown_natural_gas
        + price_breakdown_electricity
        + price_breakdown_slag
        + price_breakdown_taxes
        + price_breakdown_financial_equipment
        + price_breakdown_financial_remaining
        + price_breakdown_O2sales
    )  # a neater way to implement is add to price_breakdowns but I am not sure if ProFAST can handle negative costs

    bos_savings = (price_breakdown_labor_cost_admin_support) * 0.3
    steel_price_breakdown = {
        "Steel price: EAF and Casting CAPEX ($/tonne)": price_breakdown_eaf_casting,
        "Steel price: Shaft Furnace CAPEX ($/tonne)": price_breakdown_shaft_furnace,
        "Steel price: Oxygen Supply CAPEX ($/tonne)": price_breakdown_oxygen_supply,
        "Steel price: H2 Pre-heating CAPEX ($/tonne)": price_breakdown_h2_preheating,
        "Steel price: Cooling Tower CAPEX ($/tonne)": price_breakdown_cooling_tower,
        "Steel price: Piping CAPEX ($/tonne)": price_breakdown_piping,
        "Steel price: Electrical & Instrumentation ($/tonne)": price_breakdown_elec_instr,
        "Steel price: Buildings, Storage, Water Service CAPEX ($/tonne)": price_breakdown_buildings_storage_water,
        "Steel price: Miscellaneous CAPEX ($/tonne)": price_breakdown_misc,
        "Steel price: Annual Operating Labor Cost ($/tonne)": price_breakdown_labor_cost_annual,
        "Steel price: Maintenance Labor Cost ($/tonne)": price_breakdown_labor_cost_maintenance,
        "Steel price: Administrative & Support Labor Cost ($/tonne)": price_breakdown_labor_cost_admin_support,
        "Steel price: Installation Cost ($/tonne)": price_breakdown_installation,
        "Steel price: Maintenance Materials ($/tonne)": price_breakdown_maintenance_materials,
        "Steel price: Raw Water Withdrawal ($/tonne)": price_breakdown_water_withdrawal,
        "Steel price: Lime ($/tonne)": price_breakdown_lime,
        "Steel price: Carbon ($/tonne)": price_breakdown_carbon,
        "Steel price: Iron Ore ($/tonne)": price_breakdown_iron_ore,
        "Steel price: Hydrogen ($/tonne)": price_breakdown_hydrogen,
        "Steel price: Natural gas ($/tonne)": price_breakdown_natural_gas,
        "Steel price: Electricity ($/tonne)": price_breakdown_electricity,
        "Steel price: Slag Disposal ($/tonne)": price_breakdown_slag,
        "Steel price: Taxes ($/tonne)": price_breakdown_taxes,
        "Steel price: Equipment Financing ($/tonne)": price_breakdown_financial_equipment,
        "Steel price: Remaining Financial ($/tonne)": price_breakdown_financial_remaining,
        "Steel price: Oxygen sales ($/tonne)": price_breakdown_O2sales,
        "Steel price: Total ($/tonne)": price_breakdown_check,
        "(-) Steel price: BOS savings ($/tonne)": bos_savings,
    }

    price_breakdown = price_breakdown.drop(columns=["Amount"])

    return SteelFinanceModelOutputs(
        sol=sol,
        summary=summary,
        price_breakdown=price_breakdown,
        steel_price_breakdown=steel_price_breakdown,
    )
