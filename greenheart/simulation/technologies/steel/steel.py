import copy
from typing import Dict, Union, Optional, Tuple

import ProFAST
import pandas as pd
from attrs import define, Factory, field

@define
class Feedstocks:
    """
    Represents the consumption rates and costs of various feedstocks used in steel
    production.

    Attributes:
        natural_gas_prices (Dict[str, float]):
            Natural gas costs, indexed by year ($/GJ).
        excess_oxygen (float): Excess oxygen produced (kgO2), default = 395.
        lime_unitcost (float): Cost per metric tonne of lime ($/metric tonne).
        carbon_unitcost (float): Cost per metric tonne of carbon ($/metric tonne).
        electricity_cost (float):
            Electricity cost per metric tonne of steel production ($/metric tonne).
        iron_ore_pellet_unitcost (float):
            Cost per metric tonne of iron ore ($/metric tonne).
        oxygen_market_price (float):
            Market price per kg of oxygen ($/kgO2).
        raw_water_unitcost (float):
            Cost per metric tonne of raw water ($/metric tonne).
        iron_ore_consumption (float):
            Iron ore consumption per metric tonne of steel production (metric tonnes).
        raw_water_consumption (float):
            Raw water consumption per metric tonne of steel production (metric tonnes).
        lime_consumption (float):
            Lime consumption per metric tonne of steel production (metric tonnes).
        carbon_consumption (float):
            Carbon consumption per metric tonne of steel production (metric tonnes).
        hydrogen_consumption (float):
            Hydrogen consumption per metric tonne of steel production (metric tonnes).
        natural_gas_consumption (float):
            Natural gas consumption per metric tonne of steel production (GJ-LHV).
        electricity_consumption (float):
            Electricity consumption per metric tonne of steel production (MWh).
        slag_disposal_unitcost (float):
            Cost per metric tonne of slag disposal ($/metric tonne).
        slag_production (float):
            Slag production per metric tonne of steel production (metric tonnes).
        maintenance_materials_unitcost (float):
            Cost per metric tonne of annual steel slab production at real capacity
            factor ($/metric tonne).
    """

    natural_gas_prices: Dict[str, float]
    excess_oxygen: float = 395
    lime_unitcost: float = 122.1
    carbon_unitcost: float = 236.97
    electricity_cost: float = 48.92
    iron_ore_pellet_unitcost: float = 207.35
    oxygen_market_price: float = 0.03
    raw_water_unitcost: float = 0.59289
    iron_ore_consumption: float = 1.62927
    raw_water_consumption: float = 0.80367
    lime_consumption: float = 0.01812
    carbon_consumption: float = 0.0538
    hydrogen_consumption: float = 0.06596
    natural_gas_consumption: float = 0.71657
    electricity_consumption: float = 0.5502
    slag_disposal_unitcost: float = 37.63
    slag_production: float = 0.17433
    maintenance_materials_unitcost: float = 7.72


@define
class SteelCostModelConfig:
    """
    Configuration for the steel cost model, including operational parameters and
    feedstock costs.

    Attributes:
        operational_year (int): The year of operation for cost estimation.
        plant_capacity_mtpy (float): Plant capacity in metric tons per year.
        lcoh (float): Levelized cost of hydrogen ($/kg).
        feedstocks (Feedstocks):
            An instance of the Feedstocks class containing feedstock consumption
            rates and costs.
        o2_heat_integration (bool):
            Indicates whether oxygen and heat integration is used, affecting preheating
            CapEx, cooling CapEx, and oxygen sales. Default is True.
        co2_fuel_emissions (float):
            CO2 emissions from fuel per metric tonne of steel production.
        co2_carbon_emissions (float):
            CO2 emissions from carbon per metric tonne of steel production.
        surface_water_discharge (float):
            Surface water discharge per metric tonne of steel production.
    """

    operational_year: int
    plant_capacity_mtpy: float
    lcoh: float
    feedstocks: Feedstocks
    o2_heat_integration: bool = True
    co2_fuel_emissions: float = 0.03929
    co2_carbon_emissions: float = 0.17466
    surface_water_discharge: float = 0.42113


@define
class SteelCosts:
    """
    Base dataclass for calculated steel costs.

    Attributes:
        capex_eaf_casting (float):
            Capital expenditure for electric arc furnace and casting.
        capex_shaft_furnace (float): Capital expenditure for shaft furnace.
        capex_oxygen_supply (float): Capital expenditure for oxygen supply.
        capex_h2_preheating (float): Capital expenditure for hydrogen preheating.
        capex_cooling_tower (float): Capital expenditure for cooling tower.
        capex_piping (float): Capital expenditure for piping.
        capex_elec_instr (float):
            Capital expenditure for electrical and instrumentation.
        capex_buildings_storage_water (float):
            Capital expenditure for buildings, storage, and water service.
        capex_misc (float):
            Capital expenditure for miscellaneous items.
        labor_cost_annual_operation (float): Annual operating labor cost.
        labor_cost_maintenance (float): Maintenance labor cost.
        labor_cost_admin_support (float): Administrative and support labor cost.
        property_tax_insurance (float): Cost for property tax and insurance.
        land_cost (float): Cost of land.
        installation_cost (float): Cost of installation.

    Note:
        These represent the minimum set of required cost data for
        `run_steel_finance_model`, as well as base data for `SteelCostModelOutputs`.
    """

    capex_eaf_casting: float
    capex_shaft_furnace: float
    capex_oxygen_supply: float
    capex_h2_preheating: float
    capex_cooling_tower: float
    capex_piping: float
    capex_elec_instr: float
    capex_buildings_storage_water: float
    capex_misc: float
    labor_cost_annual_operation: float
    labor_cost_maintenance: float
    labor_cost_admin_support: float
    property_tax_insurance: float
    land_cost: float
    installation_cost: float


@define
class SteelCostModelOutputs(SteelCosts):
    """
    Outputs of the steel cost model, extending the SteelCosts data with total
    cost calculations and specific cost components related to the operation and
    installation of a steel production plant.

    Attributes:
        total_plant_cost (float):
            The total capital expenditure (CapEx) for the steel plant.
        total_fixed_operating_cost (float):
            The total annual operating expenditure (OpEx), including labor,
            maintenance, administrative support, and property tax/insurance.
        labor_cost_fivemonth (float):
            Cost of labor for the first five months of operation, often used in startup
            cost calculations.
        maintenance_materials_onemonth (float):
            Cost of maintenance materials for one month of operation.
        non_fuel_consumables_onemonth (float):
            Cost of non-fuel consumables for one month of operation.
        waste_disposal_onemonth (float):
            Cost of waste disposal for one month of operation.
        monthly_energy_cost (float):
            Cost of energy (electricity, natural gas, etc.) for one month of operation.
        spare_parts_cost (float):
            Cost of spare parts as part of the initial investment.
        misc_owners_costs (float):
            Miscellaneous costs incurred by the owner, including but not limited to,
            initial supply stock, safety equipment, and initial training programs.
    """

    total_plant_cost: float
    total_fixed_operating_cost: float
    labor_cost_fivemonth: float
    maintenance_materials_onemonth: float
    non_fuel_consumables_onemonth: float
    waste_disposal_onemonth: float
    monthly_energy_cost: float
    spare_parts_cost: float
    misc_owners_costs: float

@define
class SteelCapacityModelConfig:
    """
    Configuration inputs for the steel capacity sizing model, including plant capacity and
    feedstock details.

    Attributes:
        hydrogen_amount_kgpy Optional (float): The amount of hydrogen available in kilograms 
            per year to make steel.
        desired_steel_mtpy Optional (float): The amount of desired steel production in
            metric tonnes per year.
        input_capacity_factor_estimate (float): The estimated steel plant capacity factor.
        feedstocks (Feedstocks): An instance of the `Feedstocks` class detailing the
            costs and consumption rates of resources used in production.
    """
    input_capacity_factor_estimate: float
    feedstocks: Feedstocks
    hydrogen_amount_kgpy: Optional[float] = field(default=None)
    desired_steel_mtpy: Optional[float] = field(default=None)


    def __attrs_post_init__(self):
        if self.hydrogen_amount_kgpy is None and self.desired_steel_mtpy is None:
            raise ValueError("`hydrogen_amount_kgpy` or `desired_steel_mtpy` is a required input.")

        if self.hydrogen_amount_kgpy and self.desired_steel_mtpy:
            raise ValueError("can only select one input: `hydrogen_amount_kgpy` or `desired_steel_mtpy`.")

@define
class SteelCapacityModelOutputs:
    """
    Outputs from the steel size model.

    Attributes:
        steel_plant_size_mtpy (float): If amount of hydrogen in kilograms per year is input, 
            the size of the steel plant in metric tonnes per year is output.
        hydrogen_amount_kgpy (float): If amount of steel production in metric tonnes per year is input, 
            the amount of necessary hydrogen feedstock in kilograms per year is output.
    """
    steel_plant_capacity_mtpy: float
    hydrogen_amount_kgpy: float


def run_size_steel_plant_capacity(config: SteelCapacityModelConfig) -> SteelCapacityModelOutputs:
    """
    Calculates either the annual steel production in metric tons based on plant capacity and
    available hydrogen or the amount of required hydrogen based on a desired steel production.

    Args:
        config (SteelCapacityModelConfig):
            Configuration object containing all necessary parameters for the capacity sizing,
            including capacity factor estimate and feedstock costs.

    Returns:
        SteelCapacityModelOutputs: An object containing steel plant capacity in metric tons
        per year and amount of hydrogen required in kilograms per year.

    """

    if config.hydrogen_amount_kgpy:
        steel_plant_capacity_mtpy = (config.hydrogen_amount_kgpy 
            / 1000
            / config.feedstocks.hydrogen_consumption 
            * config.input_capacity_factor_estimate
        )
        hydrogen_amount_kgpy = config.hydrogen_amount_kgpy

    if config.desired_steel_mtpy:
        hydrogen_amount_kgpy = (config.desired_steel_mtpy 
            * 1000
            * config.feedstocks.hydrogen_consumption
            / config.input_capacity_factor_estimate
        )
        steel_plant_capacity_mtpy = (config.desired_steel_mtpy 
            / config.input_capacity_factor_estimate
        )

    return SteelCapacityModelOutputs(
        steel_plant_capacity_mtpy=steel_plant_capacity_mtpy,
        hydrogen_amount_kgpy=hydrogen_amount_kgpy
    )


def run_steel_model(plant_capacity_mtpy: float, plant_capacity_factor: float) -> float:
    """
    Calculates the annual steel production in metric tons based on plant capacity and
    capacity factor.

    Args:
        plant_capacity_mtpy (float):
            The plant's annual capacity in metric tons per year.
        plant_capacity_factor (float):
            The capacity factor of the plant.

    Returns:
        float: The calculated annual steel production in metric tons per year.
    """
    steel_production_mtpy = plant_capacity_mtpy * plant_capacity_factor

    return steel_production_mtpy


def run_steel_cost_model(config: SteelCostModelConfig) -> SteelCostModelOutputs:
    """
    Calculates the capital expenditure (CapEx) and operating expenditure (OpEx) for
    a steel manufacturing plant based on the provided configuration.

    Args:
        config (SteelCostModelConfig):
            Configuration object containing all necessary parameters for the cost
            model, including plant capacity, feedstock costs, and integration options
            for oxygen and heat.

    Returns:
        SteelCostModelOutputs: An object containing detailed breakdowns of capital and
        operating costs, as well as total plant cost and other financial metrics.

    Note:
        The calculation includes various cost components such as electric arc furnace
        (EAF) casting, shaft furnace, oxygen supply, hydrogen preheating, cooling tower,
        and more, adjusted based on the Chemical Engineering Plant Cost Index (CEPCI).
    """
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
            * feedstocks.natural_gas_prices[str(config.operational_year)]
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
    """
    Configuration for the steel finance model, including plant characteristics, financial assumptions, and cost inputs.

    Attributes:
        plant_life (int): The operational lifetime of the plant in years.
        plant_capacity_mtpy (float): Plant capacity in metric tons per year.
        plant_capacity_factor (float):
            The fraction of the year the plant operates at full capacity.
        steel_production_mtpy (float): Annual steel production in metric tons.
        lcoh (float): Levelized cost of hydrogen.
        grid_prices (Dict[str, float]): Electricity prices per unit.
        feedstocks (Feedstocks):
            The feedstocks required for steel production, including types and costs.
        costs (Union[SteelCosts, SteelCostModelOutputs]):
            Calculated CapEx and OpEx costs.
        o2_heat_integration (bool): Indicates if oxygen and heat integration is used.
        financial_assumptions (Dict[str, float]):
            Financial assumptions for model calculations.
        install_years (int): The number of years over which the plant is installed.
        gen_inflation (float): General inflation rate.
    """

    plant_life: int
    plant_capacity_mtpy: float
    plant_capacity_factor: float
    steel_production_mtpy: float
    lcoh: float
    grid_prices: Dict[str, float]
    feedstocks: Feedstocks
    costs: Union[SteelCosts, SteelCostModelOutputs]
    o2_heat_integration: bool = True
    financial_assumptions: Dict[str, float] = Factory(dict)
    install_years: int = 3
    gen_inflation: float = 0.00


@define
class SteelFinanceModelOutputs:
    """
    Represents the outputs of the steel finance model, encapsulating the results of financial analysis for steel production.

    Attributes:
        sol (dict):
            A dictionary containing the solution to the financial model, including key
            financial indicators such as NPV (Net Present Value), IRR (Internal Rate of
            Return), and breakeven price.
        summary (dict):
            A summary of key results from the financial analysis, providing a
            high-level overview of financial metrics and performance indicators.
        price_breakdown (pd.DataFrame):
            A Pandas DataFrame detailing the cost breakdown for producing steel,
            including both capital and operating expenses, as well as the impact of
            various cost factors on the overall price of steel.
    """

    sol: dict
    summary: dict
    price_breakdown: pd.DataFrame


def run_steel_finance_model(
    config: SteelFinanceModelConfig,
) -> SteelFinanceModelOutputs:
    """
    Executes the financial model for steel production, calculating the breakeven price
    of steel and other financial metrics based on the provided configuration and cost
    models.

    This function integrates various cost components, including capital expenditures
    (CapEx), operating expenses (OpEx), and owner's costs. It leverages the ProFAST
    financial analysis software framework.

    Args:
        config (SteelFinanceModelConfig):
            Configuration object containing all necessary parameters and assumptions
            for the financial model, including plant characteristics, cost inputs,
            financial assumptions, and grid prices.

    Returns:
        SteelFinanceModelOutputs:
            Object containing detailed financial analysis results, including solution
            metrics, summary values, price breakdown, and steel price breakdown per
            tonne. This output is instrumental in assessing the financial performance
            and breakeven price for the steel production facility.
    """

    feedstocks = config.feedstocks
    costs = config.costs

    # Set up ProFAST
    pf = ProFAST.ProFAST("blank")

    # apply all params passed through from config
    for param, val in config.financial_assumptions.items():
        pf.set_params(param, val)

    analysis_start = int(list(config.grid_prices.keys())[0]) - config.install_years

    # Fill these in - can have most of them as 0 also
    pf.set_params(
        "commodity",
        {
            "name": "Steel",
            "unit": "metric tonnes",
            "initial price": 1000,
            "escalation": config.gen_inflation,
        },
    )
    pf.set_params("capacity", config.plant_capacity_mtpy / 365)  # units/day
    pf.set_params("maintenance", {"value": 0, "escalation": config.gen_inflation})
    pf.set_params("analysis start year", analysis_start)
    pf.set_params("operating life", config.plant_life)
    pf.set_params("installation months", 12 * config.install_years)
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
        costs.land_cost * (1 + config.gen_inflation) ** config.plant_life,
    )
    pf.set_params("demand rampup", 5.3)
    pf.set_params("long term utilization", config.plant_capacity_factor)
    pf.set_params("credit card fees", 0)
    pf.set_params("sales tax", 0)
    pf.set_params(
        "license and permit", {"value": 00, "escalation": config.gen_inflation}
    )
    pf.set_params("rent", {"value": 0, "escalation": config.gen_inflation})
    pf.set_params("property tax and insurance", 0)
    pf.set_params("admin expense", 0)
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", config.gen_inflation)
    pf.set_params("debt type", "Revolving debt")
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
        escalation=config.gen_inflation,
    )
    pf.add_fixed_cost(
        name="Maintenance Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost_maintenance,
        escalation=config.gen_inflation,
    )
    pf.add_fixed_cost(
        name="Administrative & Support Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost_admin_support,
        escalation=config.gen_inflation,
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
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Raw Water Withdrawal",
        usage=feedstocks.raw_water_consumption,
        unit="metric tonnes of water per metric tonne of steel",
        cost=feedstocks.raw_water_unitcost,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Lime",
        usage=feedstocks.lime_consumption,
        unit="metric tonnes of lime per metric tonne of steel",
        cost=feedstocks.lime_unitcost,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Carbon",
        usage=feedstocks.carbon_consumption,
        unit="metric tonnes of carbon per metric tonne of steel",
        cost=feedstocks.carbon_unitcost,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Iron Ore",
        usage=feedstocks.iron_ore_consumption,
        unit="metric tonnes of iron ore per metric tonne of steel",
        cost=feedstocks.iron_ore_pellet_unitcost,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Hydrogen",
        usage=feedstocks.hydrogen_consumption,
        unit="metric tonnes of hydrogen per metric tonne of steel",
        cost=config.lcoh * 1000,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Natural Gas",
        usage=feedstocks.natural_gas_consumption,
        unit="GJ-LHV per metric tonne of steel",
        cost=feedstocks.natural_gas_prices,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Electricity",
        usage=feedstocks.electricity_consumption,
        unit="MWh per metric tonne of steel",
        cost=config.grid_prices,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Slag Disposal",
        usage=feedstocks.slag_production,
        unit="metric tonnes of slag per metric tonne of steel",
        cost=feedstocks.slag_disposal_unitcost,
        escalation=config.gen_inflation,
    )

    pf.add_coproduct(
        name="Oxygen sales",
        usage=feedstocks.excess_oxygen,
        unit="kg O2 per metric tonne of steel",
        cost=feedstocks.oxygen_market_price,
        escalation=config.gen_inflation,
    )

    # ------------------------------ Set up outputs ---------------------------

    sol = pf.solve_price()
    summary = pf.get_summary_vals()
    price_breakdown = pf.get_cost_breakdown()

    return SteelFinanceModelOutputs(
        sol=sol,
        summary=summary,
        price_breakdown=price_breakdown,
    )


def run_steel_full_model(greenheart_config: dict) -> Tuple[SteelCapacityModelOutputs, SteelCostModelOutputs, SteelFinanceModelOutputs]:
    """
    Runs the full steel model, including capacity, cost, and finance models.

    Args:
        greenheart_config (dict): The configuration for the greenheart model.

    Returns:
        Tuple[SteelCapacityModelOutputs, SteelCostModelOutputs, SteelFinanceModelOutputs]:
            A tuple containing the outputs of the steel capacity, cost, and finance models.
    """
    # this is likely to change as we refactor to use config dataclasses, but for now
    # we'll just copy the config and modify it as needed
    config = copy.deepcopy(greenheart_config)

    steel_costs = config["steel"]["costs"]
    steel_capacity = config["steel"]["capacity"]
    feedstocks = Feedstocks(**steel_costs["feedstocks"])

    # run steel capacity model to get steel plant size
    # uses hydrogen amount from electrolyzer physics model
    capacity_config = SteelCapacityModelConfig(
        feedstocks=feedstocks,
        **steel_capacity
    )
    steel_capacity = run_size_steel_plant_capacity(capacity_config)

    # run steel cost model
    steel_costs["feedstocks"] = feedstocks
    steel_cost_config = SteelCostModelConfig(
        plant_capacity_mtpy=steel_capacity.steel_plant_capacity_mtpy,
        **steel_costs
    )
    steel_cost_config.plant_capacity_mtpy = steel_capacity.steel_plant_capacity_mtpy
    steel_costs = run_steel_cost_model(steel_cost_config)

    # run steel finance model
    steel_finance = config["steel"]["finances"]
    steel_finance["feedstocks"] = feedstocks

    steel_finance_config = SteelFinanceModelConfig(
        plant_capacity_mtpy=steel_capacity.steel_plant_capacity_mtpy,
        plant_capacity_factor=capacity_config.input_capacity_factor_estimate,
        steel_production_mtpy=run_steel_model(
            steel_capacity.steel_plant_capacity_mtpy,
            capacity_config.input_capacity_factor_estimate,
        ),
        costs=steel_costs,
        **steel_finance
    )
    steel_finance = run_steel_finance_model(steel_finance_config)

    return (
        steel_capacity,
        steel_costs,
        steel_finance
    )
