import copy
from typing import Dict, Union, Optional, Tuple
import ProFAST

import pandas as pd
from attrs import define, Factory, field

import os

@define
class Feedstocks:
    """
    Represents the costs and consumption rates of various feedstocks and resources
    used in ammonia production.

    Attributes:
        electricity_cost (float): Cost per MWh of electricity.
        hydrogen_cost (float): Cost per kg of hydrogen.
        cooling_water_cost (float): Cost per gallon of cooling water.
        iron_based_catalyst_cost (float): Cost per kg of iron-based catalyst.
        oxygen_cost (float): Cost per kg of oxygen.
        electricity_consumption (float): Electricity consumption in MWh per kg of
            ammonia production, default is 0.1207 / 1000.
        hydrogen_consumption (float): Hydrogen consumption in kg per kg of ammonia
            production, default is 0.197284403.
        cooling_water_consumption (float): Cooling water consumption in gallons per
            kg of ammonia production, default is 0.049236824.
        iron_based_catalyst_consumption (float): Iron-based catalyst consumption in kg
            per kg of ammonia production, default is 0.000091295354067341.
        oxygen_byproduct (float): Oxygen byproduct in kg per kg of ammonia production,
            default is 0.29405077250145.
    """

    electricity_cost: float
    hydrogen_cost: float
    cooling_water_cost: float
    iron_based_catalyst_cost: float
    oxygen_cost: float
    electricity_consumption: float = 0.1207 / 1000
    hydrogen_consumption = 0.197284403
    cooling_water_consumption = 0.049236824
    iron_based_catalyst_consumption = 0.000091295354067341
    oxygen_byproduct = 0.29405077250145


@define
class AmmoniaCostModelConfig:
    """
    Configuration inputs for the ammonia cost model, including plant capacity and
    feedstock details.

    Attributes:
        plant_capacity_kgpy (float): Annual production capacity of the plant in kg.
        plant_capacity_factor (float): The ratio of actual production to maximum
            possible production over a year.
        feedstocks (Feedstocks): An instance of the `Feedstocks` class detailing the
            costs and consumption rates of resources used in production.
    """

    plant_capacity_kgpy: float
    plant_capacity_factor: float
    feedstocks: Feedstocks


@define
class AmmoniaCosts:
    """
    Base dataclass for calculated costs related to ammonia production, including
    capital expenditures (CapEx) and operating expenditures (OpEx).

    Attributes:
        capex_air_separation_crygenic (float): Capital cost for air separation.
        capex_haber_bosch (float): Capital cost for the Haber-Bosch process.
        capex_boiler (float): Capital cost for boilers.
        capex_cooling_tower (float): Capital cost for cooling towers.
        capex_direct (float): Direct capital costs.
        capex_depreciable_nonequipment (float): Depreciable non-equipment capital costs.
        land_cost (float): Cost of land.
        labor_cost (float): Annual labor cost.
        general_administration_cost (float): Annual general and administrative cost.
        property_tax_insurance (float): Annual property tax and insurance cost.
        maintenance_cost (float): Annual maintenance cost.
        total_fixed_operating_cost (float): Total annual fixed operating cost.
        H2_cost_in_startup_year (float): Hydrogen cost in the startup year.
        energy_cost_in_startup_year (float): Energy cost in the startup year.
        non_energy_cost_in_startup_year (float): Non-energy cost in the startup year.
        variable_cost_in_startup_year (float): Variable cost in the startup year.
        credits_byproduct (float): Credits from byproducts.
    """

    # CapEx
    capex_air_separation_crygenic: float
    capex_haber_bosch: float
    capex_boiler: float
    capex_cooling_tower: float
    capex_direct: float
    capex_depreciable_nonequipment: float
    land_cost: float

    # Fixed OpEx
    labor_cost: float
    general_administration_cost: float
    property_tax_insurance: float
    maintenance_cost: float
    total_fixed_operating_cost: float

    # Feedstock and Byproduct costs
    H2_cost_in_startup_year: float
    energy_cost_in_startup_year: float
    non_energy_cost_in_startup_year: float
    variable_cost_in_startup_year: float
    credits_byproduct: float


@define
class AmmoniaCostModelOutputs(AmmoniaCosts):
    """
    Outputs from the ammonia cost model, extending `AmmoniaCosts` with total capital
    expenditure calculations.

    Attributes:
        capex_total (float): The total capital expenditure for the ammonia plant.
    """

    # CapEx
    capex_total: float


def run_ammonia_model(
    plant_capacity_kgpy: float, plant_capacity_factor: float
) -> float:
    """
    Calculates the annual ammonia production in kilograms based on the plant's
    capacity and its capacity factor.

    Args:
        plant_capacity_kgpy (float): The plant's annual capacity in kilograms per year.
        plant_capacity_factor (float): The capacity factor of the plant, a ratio of
            its actual output over a period of time to its potential output if it
            were possible for it to operate at full capacity continuously over the
            same period.

    Returns:
        float: The calculated annual ammonia production in kilograms per year.
    """
    ammonia_production_kgpy = plant_capacity_kgpy * plant_capacity_factor

    return ammonia_production_kgpy

@define
class AmmoniaCapacityModelConfig:
    """
    Configuration inputs for the ammonia capacity sizing model, including plant capacity and
    feedstock details.

    Attributes:
        hydrogen_amount_kgpy Optional (float): The amount of hydrogen available in kilograms 
            per year to make ammonia.
        desired_ammonia_kgpy Optional (float): The amount of desired ammonia production in
            kilograms per year.
        input_capacity_factor_estimate (float): The estimated ammonia plant capacity factor.
        feedstocks (Feedstocks): An instance of the `Feedstocks` class detailing the
            costs and consumption rates of resources used in production.
    """
    input_capacity_factor_estimate: float
    feedstocks: Feedstocks
    hydrogen_amount_kgpy: Optional[float] = field(default=None)
    desired_ammonia_kgpy: Optional[float] = field(default=None)


    def __attrs_post_init__(self):
        if self.hydrogen_amount_kgpy is None and self.desired_ammonia_kgpy is None:
            raise ValueError("`hydrogen_amount_kgpy` or `desired_ammonia_kgpy` is a required input.")

        if self.hydrogen_amount_kgpy and self.desired_ammonia_kgpy:
            raise ValueError("can only select one input: `hydrogen_amount_kgpy` or `desired_ammonia_kgpy`.")

@define
class AmmoniaCapacityModelOutputs:
    """
    Outputs from the ammonia plant capacity size model.

    Attributes:
        ammonia_plant_capacity_kgpy (float): If amount of hydrogen in kilograms per year is input, 
            the size of the ammonia plant in kilograms per year is output.
        hydrogen_amount_kgpy (float): If amount of ammonia production in kilograms per year is input, 
            the amount of necessary hydrogen feedstock in kilograms per year is output.
    """
    ammonia_plant_capacity_kgpy: float
    hydrogen_amount_kgpy: float

def run_size_ammonia_plant_capacity(config: AmmoniaCapacityModelConfig) -> AmmoniaCapacityModelOutputs:
    """
    Calculates either the annual ammonia production in kilograms based on plant capacity and
    available hydrogen or the amount of required hydrogen based on a desired ammonia production.

    Args:
        config (AmmoniaCapacityModelConfig):
            Configuration object containing all necessary parameters for the capacity sizing,
            including capacity factor estimate and feedstock costs.

    Returns:
        AmmoniaCapacityModelOutputs: An object containing ammonia plant capacity in kilograms
        per year and amount of hydrogen required in kilograms per year.

    """
    if config.hydrogen_amount_kgpy:
        ammonia_plant_capacity_kgpy = (config.hydrogen_amount_kgpy 
            / config.feedstocks.hydrogen_consumption 
            * config.input_capacity_factor_estimate
        )
        hydrogen_amount_kgpy = config.hydrogen_amount_kgpy

    if config.desired_ammonia_kgpy:
        hydrogen_amount_kgpy = (config.desired_ammonia_kgpy
            * config.feedstocks.hydrogen_consumption
            / config.input_capacity_factor_estimate
        )
        ammonia_plant_capacity_kgpy = (config.desired_ammonia_kgpy 
            / config.input_capacity_factor_estimate
        )

    return AmmoniaCapacityModelOutputs(
        ammonia_plant_capacity_kgpy=ammonia_plant_capacity_kgpy,
        hydrogen_amount_kgpy=hydrogen_amount_kgpy
    )

def run_ammonia_cost_model(config: AmmoniaCostModelConfig) -> AmmoniaCostModelOutputs:
    """
    Calculates the various costs associated with ammonia production, including
    capital expenditures (CapEx), operating expenditures (OpEx), and credits from
    byproducts, based on the provided configuration settings.

    Args:
        config (AmmoniaCostModelConfig): Configuration object containing all necessary
            parameters for the cost calculation, including plant capacity, capacity
            factor, and feedstock costs.

    Returns:
        AmmoniaCostModelOutputs: Object containing detailed breakdowns of calculated
            costs, including total capital expenditure, operating costs, and credits
            from byproducts.
    """
    feedstocks = config.feedstocks

    model_year_CEPCI = 596.2  # TODO: what year
    equation_year_CEPCI = 541.7  # TODO: what year

    # scale with respect to a baseline plant (What is this?)
    scaling_ratio = config.plant_capacity_kgpy / (365.0 * 1266638.4)

    # -------------------------------CapEx Costs------------------------------
    scaling_factor_equipment = 0.6

    capex_scale_factor = scaling_ratio**scaling_factor_equipment
    capex_air_separation_crygenic = (
        model_year_CEPCI / equation_year_CEPCI * 22506100 * capex_scale_factor
    )
    capex_haber_bosch = (
        model_year_CEPCI / equation_year_CEPCI * 18642800 * capex_scale_factor
    )
    capex_boiler = model_year_CEPCI / equation_year_CEPCI * 7069100 * capex_scale_factor
    capex_cooling_tower = (
        model_year_CEPCI / equation_year_CEPCI * 4799200 * capex_scale_factor
    )
    capex_direct = (
        capex_air_separation_crygenic
        + capex_haber_bosch
        + capex_boiler
        + capex_cooling_tower
    )
    capex_depreciable_nonequipment = (
        capex_direct * 0.42 + 4112701.84103543 * scaling_ratio
    )
    capex_total = capex_direct + capex_depreciable_nonequipment
    land_cost = capex_depreciable_nonequipment  # TODO: determine if this is the right method or the one in Fixed O&M costs

    # -------------------------------Fixed O&M Costs------------------------------
    scaling_factor_labor = 0.25
    labor_cost = 57 * 50 * 2080 * scaling_ratio**scaling_factor_labor
    general_administration_cost = labor_cost * 0.2
    property_tax_insurance = capex_total * 0.02
    maintenance_cost = capex_direct * 0.005 * scaling_ratio**scaling_factor_equipment
    land_cost = 2500000 * capex_scale_factor
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
        (
            (feedstocks.cooling_water_cost * feedstocks.cooling_water_consumption)
            + (
                feedstocks.iron_based_catalyst_cost
                * feedstocks.iron_based_catalyst_consumption
            )
        )
        * config.plant_capacity_kgpy
        * config.plant_capacity_factor
    )
    variable_cost_in_startup_year = (
        energy_cost_in_startup_year + non_energy_cost_in_startup_year
    )
    # -------------------------------Byproduct Costs------------------------------
    credits_byproduct = (
        feedstocks.oxygen_cost
        * feedstocks.oxygen_byproduct
        * config.plant_capacity_kgpy
        * config.plant_capacity_factor
    )

    return AmmoniaCostModelOutputs(
        # Capex
        capex_air_separation_crygenic=capex_air_separation_crygenic,
        capex_haber_bosch=capex_haber_bosch,
        capex_boiler=capex_boiler,
        capex_cooling_tower=capex_cooling_tower,
        capex_direct=capex_direct,
        capex_depreciable_nonequipment=capex_depreciable_nonequipment,
        capex_total=capex_total,
        land_cost=land_cost,
        # Fixed OpEx
        labor_cost=labor_cost,
        general_administration_cost=general_administration_cost,
        property_tax_insurance=property_tax_insurance,
        maintenance_cost=maintenance_cost,
        total_fixed_operating_cost=total_fixed_operating_cost,
        # Feedstock & Byproducts
        H2_cost_in_startup_year=H2_cost_in_startup_year,
        energy_cost_in_startup_year=energy_cost_in_startup_year,
        non_energy_cost_in_startup_year=non_energy_cost_in_startup_year,
        variable_cost_in_startup_year=variable_cost_in_startup_year,
        credits_byproduct=credits_byproduct,
    )


@define
class AmmoniaFinanceModelConfig:
    """
    Configuration for the financial model of an ammonia production plant, including
    operational parameters, cost inputs, and financial assumptions.

    Attributes:
        plant_life (int): Expected operational life of the plant in years.
        plant_capacity_kgpy (float): Annual production capacity of the plant in kilograms.
        plant_capacity_factor (float): The fraction of the year that the plant operates
            at full capacity.
        grid_prices (Dict[str, float]): Electricity prices per kWh, indexed by year.
        feedstocks (Feedstocks): Instance of `Feedstocks` detailing costs and consumption
            rates of inputs.
        costs (Union[AmmoniaCosts, AmmoniaCostModelOutputs]): Pre-calculated capital and
            operating costs for the plant.
        financial_assumptions (Dict[str, float]): Key financial metrics and assumptions
            for the model, such as discount rate and inflation rate. Default is an
            empty dict but should be populated with relevant values.
        install_years (int): Number of years over which the plant is installed and
            ramped up to full production, default is 3 years.
        gen_inflation (float): General inflation rate, default is 0.0.
        save_plots (bool): select whether or not to save output plots
        show_plots (bool): select whether or not to show output plots during run
        output_dir (str): where to store any saved plots or data
        design_scenario_id (int): what design scenario the plots correspond to
    """

    plant_life: int
    plant_capacity_kgpy: float
    plant_capacity_factor: float
    grid_prices: Dict[str, float]
    feedstocks: Feedstocks
    costs: Union[AmmoniaCosts, AmmoniaCostModelOutputs]
    financial_assumptions: Dict[str, float] = Factory(dict)
    install_years: int = 3
    gen_inflation: float = 0.0
    save_plots: bool = False
    show_plots: bool = False
    output_dir: str = "./output/"
    design_scenario_id: int = 0


@define
class AmmoniaFinanceModelOutputs:
    """
    Outputs from the financial model of an ammonia production plant, providing detailed
    financial analysis and projections.

    Attributes:
        sol (dict): Solution to the financial model, containing key performance indicators
            like Net Present Value (NPV), Internal Rate of Return (IRR), and payback
            period.
        summary (dict): Summary of the financial analysis, providing a high-level overview
            of the plant's financial viability.
        price_breakdown (pd.DataFrame): Detailed breakdown of costs contributing to the
            production price of ammonia.
        ammonia_price_breakdown (dict): Breakdown of the ammonia production cost into
            component costs, showing the contribution of each cost element to the
            overall production cost.
    """

    sol: dict
    summary: dict
    price_breakdown: pd.DataFrame


def run_ammonia_finance_model(
    config: AmmoniaFinanceModelConfig,
) -> AmmoniaFinanceModelOutputs:
    """
    Executes the financial analysis for an ammonia production plant based on the
    provided configuration settings. This analysis includes calculating the Net
    Present Value (NPV), Internal Rate of Return (IRR), payback period, and
    providing a detailed cost breakdown for producing ammonia.

    This function leverages the configuration specified in `AmmoniaFinanceModelConfig`,
    including plant operational parameters, grid prices, feedstock costs, pre-calculated
    CapEx and OpEx, and financial assumptions to evaluate the financial performance of
    the ammonia production facility.

    Args:
        config (AmmoniaFinanceModelConfig): Configuration object containing all the
            necessary parameters for the financial analysis, including assumptions
            and pre-calculated cost inputs.

    Returns:
        AmmoniaFinanceModelOutputs: An object containing the results of the financial
            analysis. This includes a solution dictionary with key financial metrics,
            a summary of the financial viability, a price breakdown of ammonia
            production costs, and a detailed breakdown of how each cost component
            contributes to the overall cost of ammonia.
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
            "name": "Ammonia",
            "unit": "kg",
            "initial price": 1000,
            "escalation": config.gen_inflation,
        },
    )
    pf.set_params("capacity", config.plant_capacity_kgpy / 365)  # units/day
    pf.set_params(
        "maintenance",
        {"value": 0, "escalation": config.gen_inflation},
    )
    pf.set_params("analysis start year", analysis_start)
    pf.set_params("operating life", config.plant_life)
    pf.set_params("installation months", 12 * config.install_years)
    pf.set_params(
        "installation cost",
        {
            "value": costs.total_fixed_operating_cost,
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
    pf.set_params("demand rampup", 0)
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
        name="Air Separation by Cryogenic",
        cost=costs.capex_air_separation_crygenic,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Haber Bosch",
        cost=costs.capex_haber_bosch,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )
    pf.add_capital_item(
        name="Boiler and Steam Turbine",
        cost=costs.capex_boiler,
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
        name="Depreciable Nonequipment",
        cost=costs.capex_depreciable_nonequipment,
        depr_type="MACRS",
        depr_period=7,
        refurb=[0],
    )

    # -------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(
        name="Labor Cost",
        usage=1,
        unit="$/year",
        cost=costs.labor_cost,
        escalation=config.gen_inflation,
    )
    pf.add_fixed_cost(
        name="Maintenance Cost",
        usage=1,
        unit="$/year",
        cost=costs.maintenance_cost,
        escalation=config.gen_inflation,
    )
    pf.add_fixed_cost(
        name="Administrative Expense",
        usage=1,
        unit="$/year",
        cost=costs.general_administration_cost,
        escalation=config.gen_inflation,
    )
    pf.add_fixed_cost(
        name="Property tax and insurance",
        usage=1,
        unit="$/year",
        cost=costs.property_tax_insurance,
        escalation=0.0,
    )

    # ---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(
        name="Hydrogen",
        usage=feedstocks.hydrogen_consumption,
        unit="kilogram of hydrogen per kilogram of ammonia",
        cost=feedstocks.hydrogen_cost,
        escalation=config.gen_inflation,
    )

    pf.add_feedstock(
        name="Electricity",
        usage=feedstocks.electricity_consumption,
        unit="MWh per kilogram of ammonia",
        cost=config.grid_prices,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Cooling water",
        usage=feedstocks.cooling_water_consumption,
        unit="Gallon per kilogram of ammonia",
        cost=feedstocks.cooling_water_cost,
        escalation=config.gen_inflation,
    )
    pf.add_feedstock(
        name="Iron based catalyst",
        usage=feedstocks.iron_based_catalyst_consumption,
        unit="kilogram of catalyst per kilogram of ammonia",
        cost=feedstocks.iron_based_catalyst_cost,
        escalation=config.gen_inflation,
    )
    pf.add_coproduct(
        name="Oxygen byproduct",
        usage=feedstocks.oxygen_byproduct,
        unit="kilogram of oxygen per kilogram of ammonia",
        cost=feedstocks.oxygen_cost,
        escalation=config.gen_inflation,
    )

    # ------------------------------ Set up outputs ---------------------------

    sol = pf.solve_price()
    summary = pf.get_summary_vals()
    price_breakdown = pf.get_cost_breakdown()

    if config.save_plots or config.show_plots:
        savepaths = [
            config.output_dir + "figures/capex/",
            config.output_dir + "figures/annual_cash_flow/",
            config.output_dir + "figures/lcoa_breakdown/",
            config.output_dir + "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        pf.plot_capital_expenses(
            fileout=savepaths[0] + "ammonia_capital_expense_%i.pdf" % (config.design_scenario_id),
            show_plot=config.show_plots,
        )
        pf.plot_cashflow(
            fileout=savepaths[1] + "ammonia_cash_flow_%i.png"
            % (config.design_scenario_id),
            show_plot=config.show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out).to_csv(
            savepaths[3] + "ammonia_cash_flow_%i.csv" % (config.design_scenario_id)
        )

        pf.plot_costs(
            savepaths[2] + "lcoa_%i" % (config.design_scenario_id),
            show_plot=config.show_plots,
        )

    return AmmoniaFinanceModelOutputs(
        sol=sol,
        summary=summary,
        price_breakdown=price_breakdown,
    )

def run_ammonia_full_model(greenheart_config: dict, save_plots=False, show_plots=False, output_dir="./output/", design_scenario_id=0) -> Tuple[AmmoniaCapacityModelOutputs, AmmoniaCostModelOutputs, AmmoniaFinanceModelOutputs]:
    """
    Runs the full ammonia production model, including capacity sizing, cost calculation,

    Args:
        greenheart_config (dict): Configuration settings for the ammonia production model,
            including capacity, costs, and financial assumptions.

    Returns:
        Tuple[AmmoniaCapacityModelOutputs, AmmoniaCostModelOutputs, AmmoniaFinanceModelOutputs]:
            A tuple containing the outputs of the ammonia capacity model, ammonia cost
            model, and ammonia finance model.
    """
    # this is likely to change as we refactor to use config dataclasses, but for now
    # we'll just copy the config and modify it as needed
    config = copy.deepcopy(greenheart_config)

    ammonia_costs = config["ammonia"]["costs"]
    ammonia_capacity = config["ammonia"]["capacity"]
    feedstocks = Feedstocks(**ammonia_costs["feedstocks"])

    # run ammonia capacity model to get ammonia plant size
    capacity_config = AmmoniaCapacityModelConfig(
        feedstocks=feedstocks,
        **ammonia_capacity
    )
    ammonia_capacity = run_size_ammonia_plant_capacity(capacity_config)

    # run ammonia cost model
    ammonia_costs["feedstocks"] = feedstocks
    ammonia_cost_config = AmmoniaCostModelConfig(
        plant_capacity_factor=capacity_config.input_capacity_factor_estimate,
        plant_capacity_kgpy=ammonia_capacity.ammonia_plant_capacity_kgpy,
        **ammonia_costs
    )
    ammonia_cost_config.plant_capacity_kgpy = (
        ammonia_capacity.ammonia_plant_capacity_kgpy
    )
    ammonia_costs = run_ammonia_cost_model(ammonia_cost_config)

    # run ammonia finance model
    ammonia_finance = config["ammonia"]["finances"]
    ammonia_finance["feedstocks"] = feedstocks

    ammonia_finance_config = AmmoniaFinanceModelConfig(
        plant_capacity_kgpy=ammonia_capacity.ammonia_plant_capacity_kgpy,
        plant_capacity_factor=capacity_config.input_capacity_factor_estimate,
        costs=ammonia_costs,
        show_plots=show_plots, 
        save_plots=save_plots,
        output_dir=output_dir,
        design_scenario_id=design_scenario_id,
        **ammonia_finance
    )
    ammonia_finance = run_ammonia_finance_model(ammonia_finance_config)

    return (
        ammonia_capacity,
        ammonia_costs,
        ammonia_finance,
    )