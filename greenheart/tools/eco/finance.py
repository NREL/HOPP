import os.path

import numpy as np
import numpy_financial as npf

import ProFAST  # system financial model
from ORBIT import ProjectManager
import pandas as pd

# Function to run orbit from provided inputs - this is just for wind costs
def run_orbit(orbit_config, verbose=False, weather=None, orbit_hybrid_electrical_export_config={}):
    # set up ORBIT
    project = ProjectManager(orbit_config, weather=weather)

    # run ORBIT
    project.run(availability=orbit_config["installation_availability"])

    # run ORBIT for hybrid substation if applicable
    if orbit_hybrid_electrical_export_config == {}:
        hybrid_substation_project = None
    else:
        hybrid_substation_project = ProjectManager(orbit_hybrid_electrical_export_config, weather=weather)
        hybrid_substation_project.run(availability=orbit_config["installation_availability"])

    # print results if desired
    if verbose:
        print(f"Installation CapEx:  {project.installation_capex/1e6:.0f} M")
        print(f"System CapEx:        {project.system_capex/1e6:.0f} M")
        print(f"Turbine CapEx:       {project.turbine_capex/1e6:.0f} M")
        print(f"Soft CapEx:          {project.soft_capex/1e6:.0f} M")
        print(f"Total CapEx:        {project.total_capex/1e6:.0f} M")
        print(f"Annual OpEx Rate:        {max(project.monthly_opex.values())*12:.0f} ")
        print(f"\nInstallation Time: {project.installation_time:.0f} h")
        print("\nN Substations: ", (project.phases["ElectricalDesign"].num_substations))
        print("N cables: ", (project.phases["ElectricalDesign"].num_cables))
        print("\n")

        # cable cost breakdown
        print("Cable specific costs")
        print(
            "Export cable installation CAPEX: %.2f M USD"
            % (project.phases["ExportCableInstallation"].installation_capex * 1e-6)
        )
        print("\n")

    return project, hybrid_substation_project


def adjust_orbit_costs(orbit_project, eco_config):

    if ("expected_plant_cost" in eco_config["finance_parameters"]["wind"]) and (eco_config["finance_parameters"]["wind"]["expected_plant_cost"] != 'none'):
        wind_capex_multiplier = (eco_config["finance_parameters"]["wind"]["expected_plant_cost"]*1E9)/orbit_project.total_capex
    else:
        wind_capex_multiplier = 1.0

    wind_total_capex = orbit_project.total_capex*wind_capex_multiplier
    wind_capex_breakdown = orbit_project.capex_breakdown
    for key in wind_capex_breakdown.keys():
        wind_capex_breakdown[key] *= wind_capex_multiplier

    return wind_total_capex, wind_capex_breakdown, wind_capex_multiplier

def breakout_export_costs_from_orbit_results(orbit_project, eco_config, design_scenario):
    # adjust wind capex to meet expectations
    wind_total_capex, wind_capex_breakdown, wind_capex_multiplier = adjust_orbit_costs(orbit_project=orbit_project, eco_config=eco_config)
    
    # onshore substation cost is not included in ORBIT costs by default, so we have to add it separately
    total_wind_installed_costs_with_export = wind_total_capex

    # breakout export system costs
    array_cable_equipment_cost = wind_capex_breakdown["Array System"]
    array_cable_installation_cost = wind_capex_breakdown[
        "Array System Installation"
    ]
    total_array_cable_system_capex = (
        array_cable_equipment_cost + array_cable_installation_cost
    )

    export_cable_equipment_cost = wind_capex_breakdown["Export System"] # this should include the onshore substation
    export_cable_installation_cost = wind_capex_breakdown[
        "Export System Installation"
    ]
    substation_equipment_cost = wind_capex_breakdown["Offshore Substation"]
    substation_installation_cost = wind_capex_breakdown[
        "Offshore Substation Installation"
    ]
    total_export_cable_system_capex = (
        export_cable_equipment_cost + export_cable_installation_cost
    )

    total_offshore_substation_capex = substation_equipment_cost + substation_installation_cost

    total_electrical_export_system_cost = (
        total_array_cable_system_capex
        + total_offshore_substation_capex
        + total_export_cable_system_capex
    )

    ## adjust wind cost to remove export
    if design_scenario["transportation"] == "hvdc+pipeline":
        unused_export_system_cost = 0.0
    elif (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "turbine"
    ):
        unused_export_system_cost = (
            total_array_cable_system_capex
            + total_export_cable_system_capex
            + total_offshore_substation_capex
        )
    elif (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "platform"
    ):
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    elif (
        design_scenario["electrolyzer_location"] == "platform"
        and design_scenario["h2_storage_location"] == "platform"
    ):
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    elif (
        (design_scenario["electrolyzer_location"] == "platform" or
         design_scenario["electrolyzer_location"] == "turbine")
        and design_scenario["h2_storage_location"] == "onshore"
    ):
        unused_export_system_cost = (
            total_export_cable_system_capex  # TODO check assumptions here
        )
    else:
        unused_export_system_cost = 0.0

    total_used_export_system_costs = (
        total_electrical_export_system_cost - unused_export_system_cost
    )

    total_wind_cost_no_export = (
        total_wind_installed_costs_with_export - total_used_export_system_costs
    )

    return total_wind_cost_no_export, total_used_export_system_costs

def run_capex(
    hopp_results,
    orbit_project,
    orbit_hybrid_electrical_export_project,
    electrolyzer_cost_results,
    h2_pipe_array_results,
    h2_transport_compressor_results,
    h2_transport_pipe_results,
    h2_storage_results,
    hopp_config,
    eco_config,
    orbit_config,
    design_scenario,
    desal_results,
    platform_results,
    verbose=False,
):
    
    total_wind_cost_no_export, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_project, eco_config, design_scenario)
    
    if orbit_hybrid_electrical_export_project is not None:
        _, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_project, eco_config, design_scenario)

    # wave capex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs

        wcapex = cost_dict['structural_assembly_cost_modeled']+\
            cost_dict['power_takeoff_system_cost_modeled']+\
            cost_dict['mooring_found_substruc_cost_modeled']
        wbos = cost_dict['development_cost_modeled']+\
            cost_dict['eng_and_mgmt_cost_modeled']+\
            cost_dict['plant_commissioning_cost_modeled']+\
            cost_dict['site_access_port_staging_cost_modeled']+\
            cost_dict['assembly_and_install_cost_modeled']+\
            cost_dict['other_infrastructure_cost_modeled']
        welec_infrastruc_costs = cost_dict['array_cable_system_cost_modeled']+\
            cost_dict['export_cable_system_cost_modeled']+\
            cost_dict['other_elec_infra_cost_modeled'] # +\
            # cost_dict['onshore_substation_cost_modeled']+\
            # cost_dict['offshore_substation_cost_modeled']
        # financial = cost_dict['project_contingency']+\
            # cost_dict['insurance_during_construction']+\
            # cost_dict['reserve_accounts']
        wave_capex = wcapex + wbos + welec_infrastruc_costs
    else:
        wave_capex = 0.0

    # solar capex
    if "pv" in hopp_config["technologies"].keys():
        solar_capex = hopp_results["hybrid_plant"].pv.total_installed_cost
    else:
        solar_capex = 0.0

    # battery capex
    if "battery" in hopp_config["technologies"].keys():
        battery_capex = hopp_results["hybrid_plant"].battery.total_installed_cost
    else:
        battery_capex = 0.0

    # TODO bos capex
    # bos_capex = hopp_results["hybrid_plant"].bos.total_installed_cost

    ## desal capex
    if desal_results != None:
        desal_capex = desal_results["desal_capex_usd"]
    else:
        desal_capex = 0.0

    ## electrolyzer capex
    electrolyzer_total_capital_cost = electrolyzer_cost_results[
        "electrolyzer_total_capital_cost"
    ]

    if (
        design_scenario["electrolyzer_location"] == "platform"
        or design_scenario["h2_storage_location"] == "platform"
        or hopp_config["site"]["solar"]
    ):
        platform_costs = platform_results["capex"]
    else:
        platform_costs = 0.0

    # h2 transport
    h2_transport_compressor_capex = h2_transport_compressor_results["compressor_capex"]
    h2_transport_pipe_capex = h2_transport_pipe_results["total capital cost [$]"][0]

    ## h2 storage
    if eco_config["h2_storage"]["type"] == "none":
        h2_storage_capex = 0.0
    elif (
        eco_config["h2_storage"]["type"] == "pipe"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        eco_config["h2_storage"]["type"] == "turbine"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        eco_config["h2_storage"]["type"] == "pressure_vessel"
    ):  # pressure vessel storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        eco_config["h2_storage"]["type"] == "salt_cavern"
    ):  # salt cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        eco_config["h2_storage"]["type"] == "lined_rock_cavern"
    ):  # lined rock cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    else:
        raise NotImplementedError("the storage type you have indicated (%s) has not been implemented." % eco_config["h2_storage"]["type"])

    # store capex component breakdown
    capex_breakdown = {
        "wind": total_wind_cost_no_export,
        "wave": wave_capex,
        "solar": solar_capex,
        "battery": battery_capex,
        "platform": platform_costs,
        "electrical_export_system": total_used_export_system_costs,
        "desal": desal_capex,
        "electrolyzer": electrolyzer_total_capital_cost,
        "h2_pipe_array": h2_pipe_array_results["capex"],
        "h2_transport_compressor": h2_transport_compressor_capex,
        "h2_transport_pipeline": h2_transport_pipe_capex,
        "h2_storage": h2_storage_capex,
    }

    # discount capex to appropriate year for unified costing
    for key in capex_breakdown.keys():
        if key == "h2_storage":
            # if design_scenario["h2_storage_location"] == "turbine" and eco_config["h2_storage"]["type"] == "turbine":
            #     cost_year = eco_config["finance_parameters"]["discount_years"][key][
            #         design_scenario["h2_storage_location"]
            #     ]
            # else:
            cost_year = eco_config["finance_parameters"]["discount_years"][key][
                eco_config["h2_storage"]["type"]
            ]
        else:
            cost_year = eco_config["finance_parameters"]["discount_years"][key]

        periods = orbit_config["cost_year"] - cost_year

        capex_breakdown[key] = -npf.fv(
            eco_config["finance_parameters"]["general_inflation"],
            periods,
            0.0,
            capex_breakdown[key],
        )

    total_system_installed_cost = sum(
        capex_breakdown[key] for key in capex_breakdown.keys()
    )

    if verbose:
        print("\nCAPEX Breakdown")
        for key in capex_breakdown.keys():
            print(key, "%.2f" % (capex_breakdown[key] * 1e-6), " M")

        print(
            "\nTotal system CAPEX: ",
            "$%.2f" % (total_system_installed_cost * 1e-9),
            " B",
        )

    return total_system_installed_cost, capex_breakdown

def run_opex(
    hopp_results,
    orbit_project,
    orbit_hybrid_electrical_export_project,
    electrolyzer_cost_results,
    h2_pipe_array_results,
    h2_transport_compressor_results,
    h2_transport_pipe_results,
    h2_storage_results,
    hopp_config,
    eco_config,
    orbit_config,
    desal_results,
    platform_results,
    verbose=False,
    total_export_system_cost=0,
):
    # WIND ONLY Total O&M expenses including fixed, variable, and capacity-based, $/year
    # use values from hybrid substation if a hybrid plant
    if orbit_hybrid_electrical_export_project is None:
        
        annual_operating_cost_wind = (
            max(orbit_project.monthly_opex.values()) * 12
        )  # np.average(hopp_results["hybrid_plant"].wind.om_total_expense)

    else:
        
        annual_operating_cost_wind = (
            max(orbit_hybrid_electrical_export_project.monthly_opex.values()) * 12
        )

    # wave opex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs
        wave_opex = cost_dict["maintenance_cost"] + cost_dict["operations_cost"]
    else:
        wave_opex = 0.0

    # solar opex
    if "solar" in hopp_config["technologies"].keys():
        solar_opex = hopp_results["hybrid_plant"].pv.om_fixed + np.sum(hopp_results["hybrid_plant"].pv.om_variable)
    else:
        solar_opex = 0.0

    # battery opex
    if "battery" in hopp_config["technologies"].keys():
        battery_opex = hopp_results["hybrid_plant"].battery.om_fixed + np.sum(hopp_results["hybrid_plant"].battery.om_variable)
    else:
        battery_opex = 0.0

    # H2 OPEX
    platform_operating_costs = platform_results["opex"]  # TODO update this

    annual_operating_cost_h2 = electrolyzer_cost_results["electrolyzer_OM_cost_annual"]

    h2_transport_compressor_opex = h2_transport_compressor_results[
        "compressor_opex"
    ]  # annual

    h2_transport_pipeline_opex = h2_transport_pipe_results["annual operating cost [$]"][
        0
    ]  # annual

    storage_opex = h2_storage_results["storage_opex"]
    # desal OPEX
    if desal_results != None:
        desal_opex = desal_results["desal_opex_usd_per_year"]
    else:
        desal_opex = 0.0
    annual_operating_cost_desal = desal_opex

    # store opex component breakdown
    opex_breakdown_annual = {
        "wind_and_electrical": annual_operating_cost_wind,
        "platform": platform_operating_costs,
        #   "electrical_export_system": total_export_om_cost,
        "wave": wave_opex,
        "solar": solar_opex,
        "battery": battery_opex,
        "desal": annual_operating_cost_desal,
        "electrolyzer": annual_operating_cost_h2,
        "h2_pipe_array": h2_pipe_array_results["opex"],
        "h2_transport_compressor": h2_transport_compressor_opex,
        "h2_transport_pipeline": h2_transport_pipeline_opex,
        "h2_storage": storage_opex,
    }

    # discount opex to appropriate year for unified costing
    for key in opex_breakdown_annual.keys():
        if key == "h2_storage":
            cost_year = eco_config["finance_parameters"]["discount_years"][key][
                eco_config["h2_storage"]["type"]
            ]
        else:
            cost_year = eco_config["finance_parameters"]["discount_years"][key]

        periods = orbit_config["cost_year"] - cost_year
        opex_breakdown_annual[key] = -npf.fv(
            eco_config["finance_parameters"]["general_inflation"],
            periods,
            0.0,
            opex_breakdown_annual[key],
        )

    # Calculate the total annual OPEX of the installed system
    total_annual_operating_costs = sum(opex_breakdown_annual.values())

    if verbose:
        print("\nAnnual OPEX Breakdown")
        for key in opex_breakdown_annual.keys():
            print(key, "%.2f" % (opex_breakdown_annual[key] * 1e-6), " M")

        print(
            "\nTotal Annual OPEX: ",
            "$%.2f" % (total_annual_operating_costs * 1e-6),
            " M",
        )
        print(opex_breakdown_annual)
    return total_annual_operating_costs, opex_breakdown_annual


def run_profast_lcoe(
    eco_config,
    orbit_config,
    orbit_project,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    verbose=False,
    show_plots=False,
    save_plots=False,
):
    gen_inflation = eco_config["finance_parameters"]["general_inflation"]

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST()
    pf.set_params(
        "commodity",
        {
            "name": "electricity",
            "unit": "kWh",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )
    pf.set_params(
        "capacity", np.sum(hopp_results["combined_hybrid_power_production_hopp"]) / 365.0
    )  # kWh/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params("analysis start year", orbit_config["atb_year"] + 1)
    pf.set_params(
        "operating life", orbit_config["project_parameters"]["project_lifetime"]
    )
    pf.set_params(
        "installation months",
        (orbit_project.installation_time / (365 * 24)) * (12.0 / 1.0),
    )
    pf.set_params(
        "installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    if land_cost > 0:
        pf.set_params("non depr assets", land_cost)
        pf.set_params(
            "end of proj sale non depr assets",
            land_cost
            * (1 + gen_inflation) ** orbit_config["project_parameters"]["project_lifetime"],
        )
    pf.set_params("demand rampup", 0)
    pf.set_params("long term utilization", 1)
    pf.set_params("credit card fees", 0)
    pf.set_params("sales tax", eco_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    pf.set_params(
        "property tax and insurance",
        eco_config["finance_parameters"]["property_tax"]
        + eco_config["finance_parameters"]["property_insurance"],
    )
    pf.set_params(
        "admin expense",
        eco_config["finance_parameters"]["administrative_expense_percent_of_sales"],
    )
    pf.set_params(
        "total income tax rate",
        eco_config["finance_parameters"]["total_income_tax_rate"],
    )
    pf.set_params(
        "capital gains tax rate",
        eco_config["finance_parameters"]["capital_gains_tax_rate"],
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        eco_config["finance_parameters"]["discount_rate"],
    )
    pf.set_params(
        "debt equity ratio of initial financing",
        (
            eco_config["finance_parameters"]["debt_equity_split"]
            / (100 - eco_config["finance_parameters"]["debt_equity_split"])
        ),
    )
    pf.set_params("debt type", eco_config["finance_parameters"]["debt_type"])
    pf.set_params(
        "loan period if used", eco_config["finance_parameters"]["loan_period"]
    )
    pf.set_params(
        "debt interest rate", eco_config["finance_parameters"]["debt_interest_rate"]
    )
    pf.set_params(
        "cash onhand", eco_config["finance_parameters"]["cash_onhand_months"]
    )

    # ----------------------------------- Add capital items to ProFAST ----------------
    if "wind" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wind System",
            cost=capex_breakdown["wind"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "wave" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wave System",
            cost=capex_breakdown["wave"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
        
    if "solar" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Solar System",
            cost=capex_breakdown["solar"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if "battery" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Battery System",
            cost=capex_breakdown["battery"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if (design_scenario["transportation"] == "hvdc+pipeline" or not 
        (design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "turbine")
    ):
        pf.add_capital_item(
            name="Electrical Export system",
            cost=capex_breakdown["electrical_export_system"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    # -------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(
        name="Wind and Electrical Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["wind_and_electrical"],
        escalation=gen_inflation,
    )

    if "wave" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Wave O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["wave"],
            escalation=gen_inflation,
        )

    if "solar" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Solar O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["solar"],
            escalation=gen_inflation,
        )

    if "battery" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Battery O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["battery"],
            escalation=gen_inflation,
        )
    

    # ------------------------------------- add incentives -----------------------------------
    """ Note: ptc units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = eco_config["policy_parameters"]["option%s" % (incentive_option)]
    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    wind_ptc_in_dollars_per_kw = -npf.fv(
        gen_inflation,
        orbit_config["atb_year"]
        + round((orbit_project.installation_time / (365 * 24)))
        - 1992,
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation
    
    pf.add_incentive(
        name="Electricity PTC",
        value=wind_ptc_in_dollars_per_kw,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    sol = pf.solve_price()

    lcoe = sol["price"]

    if verbose:
        print("\nProFAST LCOE: ", "%.2f" % (lcoe * 1e3), "$/MWh")

    if show_plots or save_plots:
        if not os.path.exists("figures/wind_only"):
            os.makedirs("figures/wind_only")
        pf.plot_costs_yearly(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout="figures/wind_only/annual_cash_flow_wind_only_%i.png"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs_yearly2(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout="figures/wind_only/annual_cash_flow_wind_only_%i.html"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_capital_expenses(
            fileout="figures/wind_only/capital_expense_only_%i.png"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout="figures/wind_only/cash_flow_wind_only_%i.png"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs(
            fileout="figures/wind_only/cost_breakdown_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )

    return lcoe, pf


def run_profast_grid_only(
    eco_config, 
    orbit_config,
    orbit_project,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    design_scenario,
    total_accessory_power_renewable_kw,
    total_accessory_power_grid_kw,
    verbose=False,
    show_plots=False,
    save_plots=False,
):
    gen_inflation = eco_config["finance_parameters"]["general_inflation"]

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST()
    pf.set_params(
        "commodity",
        {
            "name": "Hydrogen",
            "unit": "kg",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )
    pf.set_params(
        "capacity",
        electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"] / 365.0,
    )  # kg/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params("analysis start year", orbit_config["atb_year"] + 1)
    pf.set_params(
        "operating life", orbit_config["project_parameters"]["project_lifetime"]
    )
    # pf.set_params('installation months', (orbit_project.installation_time/(365*24))*(12.0/1.0))
    pf.set_params(
        "installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    if land_cost > 0:
        pf.set_params("non depr assets", land_cost)
        pf.set_params(
            "end of proj sale non depr assets",
            land_cost
            * (1 + gen_inflation) ** orbit_config["project_parameters"]["project_lifetime"],
        )
    pf.set_params("demand rampup", 0)
    pf.set_params("long term utilization", 1)
    pf.set_params("credit card fees", 0)
    pf.set_params("sales tax", eco_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    pf.set_params(
        "property tax and insurance",
        eco_config["finance_parameters"]["property_tax"]
        + eco_config["finance_parameters"]["property_insurance"],
    )
    pf.set_params(
        "admin expense",
        eco_config["finance_parameters"]["administrative_expense_percent_of_sales"],
    )
    pf.set_params(
        "total income tax rate",
        eco_config["finance_parameters"]["total_income_tax_rate"],
    )
    pf.set_params(
        "capital gains tax rate",
        eco_config["finance_parameters"]["capital_gains_tax_rate"],
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        eco_config["finance_parameters"]["discount_rate"],
    )
    pf.set_params(
        "debt equity ratio of initial financing",
        (
            eco_config["finance_parameters"]["debt_equity_split"]
            / (100 - eco_config["finance_parameters"]["debt_equity_split"])
        ),
    )
    pf.set_params("debt type", eco_config["finance_parameters"]["debt_type"])
    pf.set_params(
        "loan period if used", eco_config["finance_parameters"]["loan_period"]
    )
    pf.set_params(
        "debt interest rate", eco_config["finance_parameters"]["debt_interest_rate"]
    )
    pf.set_params(
        "cash onhand", eco_config["finance_parameters"]["cash_onhand_months"]
    )

    # ----------------------------------- Add capital items to ProFAST ----------------
    # pf.add_capital_item(name="Wind System",cost=capex_breakdown["wind"], depr_type=eco_config["finance_parameters"]["depreciation_method"], depr_period=eco_config["finance_parameters"]["depreciation_period"],refurb=[0])
    # pf.add_capital_item(
    #     name="Electrical Export system",
    #     cost=capex_breakdown["electrical_export_system"],
    #     depr_type=eco_config["finance_parameters"]["depreciation_method"],
    #     depr_period=eco_config["finance_parameters"]["depreciation_period"],
    #     refurb=[0],
    # )

    electrolyzer_refurbishment_schedule = np.zeros(
        orbit_config["project_parameters"]["project_lifetime"]
    )
    refurb_period = round(
        eco_config["electrolyzer"]["time_between_replacement"] / (24 * 365)
    )
    electrolyzer_refurbishment_schedule[
        refurb_period : orbit_config["project_parameters"][
            "project_lifetime"
        ] : refurb_period
    ] = eco_config["electrolyzer"]["replacement_cost_percent"]
    
    pf.add_capital_item(
        name="Electrolysis System",
        cost=capex_breakdown["electrolyzer"],
        depr_type=eco_config["finance_parameters"]["depreciation_method"],
        depr_period=eco_config["finance_parameters"][
            "depreciation_period_electrolyzer"
        ],
        refurb=list(electrolyzer_refurbishment_schedule),
    )

    pf.add_capital_item(
        name="Hydrogen Storage System",
        cost=capex_breakdown["h2_storage"],
        depr_type=eco_config["finance_parameters"]["depreciation_method"],
        depr_period=eco_config["finance_parameters"][
            "depreciation_period_electrolyzer"
        ],
        refurb=[0],
    )
    # pf.add_capital_item(name ="Desalination system",cost=capex_breakdown["desal"], depr_type=eco_config["finance_parameters"]["depreciation_method"],depr_period=eco_config["finance_parameters"]["depreciation_period"],refurb=[0])

    # -------------------------------------- Add fixed costs--------------------------------
    # pf.add_fixed_cost(name="Wind Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["wind"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Electrical Export Fixed O&M Cost", usage=1.0,unit='$/year',cost=opex_breakdown["electrical_export_system"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["desal"],escalation=gen_inflation)
    pf.add_fixed_cost(
        name="Electrolyzer Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["electrolyzer"],
        escalation=gen_inflation,
    )
    pf.add_fixed_cost(
        name="Hydrogen Storage Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["h2_storage"],
        escalation=gen_inflation,
    )

    # ---------------------- Add feedstocks, note the various cost options-------------------
    galperkg = 3.785411784
    pf.add_feedstock(
        name="Water",
        usage=electrolyzer_physics_results["H2_Results"]["water_annual_usage"]
        * galperkg
        / electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"],
        unit="gal",
        cost="US Average",
        escalation=gen_inflation,
    )

    # if eco_config["project_parameters"]["grid_connection"]:

    energy_purchase = 365*24*eco_config["electrolyzer"]["rating"] * 1e3 + total_accessory_power_renewable_kw + total_accessory_power_grid_kw

    pf.add_fixed_cost(
        name="Electricity from grid",
        usage=1.0,
        unit="$/year",
        cost=energy_purchase * eco_config["project_parameters"]["ppa_price"],
        escalation=gen_inflation,
    )

    sol = pf.solve_price()

    lcoh = sol["price"]
    if verbose:
        print("\nLCOH grid only: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST grid only NPV: ", "%.2f" % (sol["NPV"]))
        print("ProFAST grid only IRR: ", "%.5f" % (max(sol["irr"])))
        print("ProFAST grid only LCO: ", "%.2f" % (sol["lco"]), "$/kg")
        print("ProFAST grid only Profit Index: ", "%.2f" % (sol["profit index"]))
        print("ProFAST grid only payback period: ", sol["investor payback period"])

    if save_plots or show_plots:
        if not os.path.exists("figures"):
            os.mkdir("figures")
            os.mkdir("figures/lcoh_breakdown")
            os.mkdir("figures/capex")
            os.mkdir("figures/annual_cash_flow")
        savepaths = [
            "figures/capex/",
            "figures/annual_cash_flow/",
            "figures/lcoh_breakdown/",
            "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.mkdir(savepath)

        pf.plot_capital_expenses(
            fileout="figures/capex/capital_expense_grid_only_%i.pdf" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout="figures/annual_cash_flow/cash_flow_grid_only_%i.png"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out, orient='index').to_csv("data/cash_flow_grid_only_%i.csv" % (design_scenario["id"]))

        pf.plot_costs(
            "figures/lcoh_breakdown/lcoh_grid_only_%i" % (design_scenario["id"]),
            show_plot=show_plots,
        )
    return lcoh, pf


def run_profast_full_plant_model(
    eco_config,
    orbit_config,
    orbit_project,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    total_accessory_power_renewable_kw,
    total_accessory_power_grid_kw,
    verbose=False,
    show_plots=False,
    save_plots=False,
):
    gen_inflation = eco_config["finance_parameters"]["general_inflation"]

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST()
    pf.set_params(
        "commodity",
        {
            "name": "Hydrogen",
            "unit": "kg",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )
    pf.set_params(
        "capacity",
        electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"] / 365.0,
    )  # kg/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params("analysis start year", orbit_config["atb_year"] + 1)
    pf.set_params(
        "operating life", orbit_config["project_parameters"]["project_lifetime"]
    )
    pf.set_params(
        "installation months",
        (orbit_project.installation_time / (365 * 24)) * (12.0 / 1.0),
    )  # convert from hours to months
    pf.set_params(
        "installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    if land_cost > 0:
        pf.set_params("non depr assets", land_cost)
        pf.set_params(
            "end of proj sale non depr assets",
            land_cost
            * (1 + gen_inflation) ** orbit_config["project_parameters"]["project_lifetime"],
        )
    pf.set_params("demand rampup", 0)
    pf.set_params("long term utilization", 1)  # TODO should use utilization
    pf.set_params("credit card fees", 0)
    pf.set_params("sales tax", eco_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    # TODO how to handle property tax and insurance for fully offshore?
    pf.set_params(
        "property tax and insurance",
        eco_config["finance_parameters"]["property_tax"]
        + eco_config["finance_parameters"]["property_insurance"],
    )
    pf.set_params(
        "admin expense",
        eco_config["finance_parameters"]["administrative_expense_percent_of_sales"],
    )
    pf.set_params(
        "total income tax rate",
        eco_config["finance_parameters"]["total_income_tax_rate"],
    )
    pf.set_params(
        "capital gains tax rate",
        eco_config["finance_parameters"]["capital_gains_tax_rate"],
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        eco_config["finance_parameters"]["discount_rate"],
    )
    pf.set_params(
        "debt equity ratio of initial financing",
        (
            eco_config["finance_parameters"]["debt_equity_split"]
            / (100 - eco_config["finance_parameters"]["debt_equity_split"])
        ),
    )  # TODO this may not be put in right
    pf.set_params("debt type", eco_config["finance_parameters"]["debt_type"])
    pf.set_params(
        "loan period if used", eco_config["finance_parameters"]["loan_period"]
    )
    pf.set_params(
        "debt interest rate", eco_config["finance_parameters"]["debt_interest_rate"]
    )
    pf.set_params(
        "cash onhand", eco_config["finance_parameters"]["cash_onhand_months"]
    )

    # ----------------------------------- Add capital and fixed items to ProFAST ----------------
    if "wind" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wind System",
            cost=capex_breakdown["wind"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "wave" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wave System",
            cost=capex_breakdown["wave"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "solar" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Solar System",
            cost=capex_breakdown["solar"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if "battery" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Battery System",
            cost=capex_breakdown["battery"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
        
    if "platform" in capex_breakdown.keys() and capex_breakdown["platform"] > 0:
        pf.add_capital_item(
            name="Equipment Platform",
            cost=capex_breakdown["platform"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Equipment Platform O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["platform"],
            escalation=gen_inflation,
        )

    pf.add_fixed_cost(
        name="Wind and Electrical Export Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["wind_and_electrical"],
        escalation=gen_inflation,
    )
    if "wave" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Wave O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["wave"],
            escalation=gen_inflation,
        )

    if "solar" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Solar O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["solar"],
            escalation=gen_inflation,
        )

    if "battery" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Battery O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["battery"],
            escalation=gen_inflation,
        )

    if design_scenario["transportation"] == "hvdc+pipeline" or not (
        design_scenario["electrolyzer_location"] == "turbine"
        and design_scenario["h2_storage_location"] == "turbine"
    ):
        pf.add_capital_item(
            name="Electrical Export system",
            cost=capex_breakdown["electrical_export_system"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
        # TODO assess if this makes sense (electrical export O&M included in wind O&M)

    electrolyzer_refurbishment_schedule = np.zeros(
        orbit_config["project_parameters"]["project_lifetime"]
    )
    refurb_period = round(
        eco_config["electrolyzer"]["time_between_replacement"] / (24 * 365)
    )
    electrolyzer_refurbishment_schedule[
        refurb_period : orbit_config["project_parameters"][
            "project_lifetime"
        ] : refurb_period
    ] = eco_config["electrolyzer"]["replacement_cost_percent"]

    pf.add_capital_item(
        name="Electrolysis System",
        cost=capex_breakdown["electrolyzer"],
        depr_type=eco_config["finance_parameters"]["depreciation_method"],
        depr_period=eco_config["finance_parameters"][
            "depreciation_period_electrolyzer"
        ],
        refurb=list(electrolyzer_refurbishment_schedule),
    )
    pf.add_fixed_cost(
        name="Electrolysis System Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["electrolyzer"],
        escalation=gen_inflation,
    )

    if design_scenario["electrolyzer_location"] == "turbine":
        pf.add_capital_item(
            name="H2 Pipe Array System",
            cost=capex_breakdown["h2_pipe_array"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="H2 Pipe Array Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_pipe_array"],
            escalation=gen_inflation,
        )

    if (
        design_scenario["h2_storage_location"] == "onshore"
        and design_scenario["electrolyzer_location"] != "onshore"
    ) or (
        design_scenario["h2_storage_location"] != "onshore"
        and design_scenario["electrolyzer_location"] == "onshore"
    ) or (
        design_scenario["transportation"] == "hvdc+pipeline"
    ):
        pf.add_capital_item(
            name="H2 Transport Compressor System",
            cost=capex_breakdown["h2_transport_compressor"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_capital_item(
            name="H2 Transport Pipeline System",
            cost=capex_breakdown["h2_transport_pipeline"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )

        pf.add_fixed_cost(
            name="H2 Transport Compression Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_transport_compressor"],
            escalation=gen_inflation,
        )
        pf.add_fixed_cost(
            name="H2 Transport Pipeline Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_transport_pipeline"],
            escalation=gen_inflation,
        )

    if eco_config["h2_storage"]["type"] != "none":
        pf.add_capital_item(
            name="Hydrogen Storage System",
            cost=capex_breakdown["h2_storage"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Hydrogen Storage Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_storage"],
            escalation=gen_inflation,
        )

    # ---------------------- Add feedstocks, note the various cost options-------------------
    if design_scenario["electrolyzer_location"] == "onshore":
        galperkg = 3.785411784
        pf.add_feedstock(
            name="Water",
            usage=electrolyzer_physics_results["H2_Results"]["water_annual_usage"]
            * galperkg
            / electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"],
            unit="gal",
            cost="US Average",
            escalation=gen_inflation,
        )
    else:
        pf.add_capital_item(
            name="Desal System",
            cost=capex_breakdown["desal"],
            depr_type=eco_config["finance_parameters"]["depreciation_method"],
            depr_period=eco_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Desal Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["desal"],
            escalation=gen_inflation,
        )

    if eco_config["project_parameters"]["grid_connection"] or total_accessory_power_grid_kw > 0:

        energy_purchase = total_accessory_power_grid_kw*365*24

        if eco_config["project_parameters"]["grid_connection"]:
            annual_energy_shortfall = np.sum(hopp_results["energy_shortfall_hopp"])
            energy_purchase += annual_energy_shortfall

        pf.add_fixed_cost(
            name="Electricity from grid",
            usage=1.0,
            unit="$/year",
            cost=energy_purchase * eco_config["project_parameters"]["ppa_price"],
            escalation=gen_inflation,
        )

    # ------------------------------------- add incentives -----------------------------------
    """ Note: units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = eco_config["policy_parameters"]["option%s" % (incentive_option)]

    # add wind_itc (% of wind capex)
    electricity_itc_value_percent_wind_capex = incentive_dict["electricity_itc"]
    electricity_itc_value_dollars = electricity_itc_value_percent_wind_capex * (
        capex_breakdown["wind"] + capex_breakdown["electrical_export_system"]
    )
    pf.set_params(
        "one time cap inct",
        {
            "value": electricity_itc_value_dollars,
            "depr type": eco_config["finance_parameters"]["depreciation_method"],
            "depr period": eco_config["finance_parameters"]["depreciation_period"],
            "depreciable": True,
        },
    )

    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    electricity_ptc_in_dollars_per_kw = -npf.fv(
        gen_inflation,
        orbit_config["atb_year"]
        + round((orbit_project.installation_time / (365 * 24)))
        - 1992,
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation
    kw_per_kg_h2 = (
        sum(hopp_results["combined_hybrid_power_production_hopp"])
        / electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"]
    )
    electricity_ptc_in_dollars_per_kg_h2 = electricity_ptc_in_dollars_per_kw * kw_per_kg_h2
    pf.add_incentive(
        name="Electricity PTC",
        value=electricity_ptc_in_dollars_per_kg_h2,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    # add h2_ptc ($/kg)
    h2_ptc_inflation_adjusted = -npf.fv(
        gen_inflation,
        orbit_config["atb_year"]
        + round((orbit_project.installation_time / (365 * 24)))
        - 2022,
        0,
        incentive_dict["h2_ptc"],
    )
    pf.add_incentive(
        name="H2 PTC",
        value=h2_ptc_inflation_adjusted,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    # ------------------------------------ solve and post-process -----------------------------

    sol = pf.solve_price()

    df = pf.cash_flow_out

    lcoh = sol["price"]

    if verbose:
        print("\nProFAST LCOH: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST NPV: ", "%.2f" % (sol["NPV"]))
        print("ProFAST IRR: ", "%.5f" % (max(sol["irr"])))
        print("ProFAST LCO: ", "%.2f" % (sol["lco"]), "$/kg")
        print("ProFAST Profit Index: ", "%.2f" % (sol["profit index"]))
        print("ProFAST payback period: ", sol["investor payback period"])

        MIRR = npf.mirr(
            df["Investor cash flow"],
            eco_config["finance_parameters"]["debt_interest_rate"],
            eco_config["finance_parameters"]["discount_rate"],
        )  # TODO probably ignore MIRR
        NPV = npf.npv(
            eco_config["finance_parameters"]["general_inflation"],
            df["Investor cash flow"],
        )
        ROI = np.sum(df["Investor cash flow"]) / abs(
            np.sum(df["Investor cash flow"][df["Investor cash flow"] < 0])
        )  # ROI is not a good way of thinking about the value of the project

        # TODO project level IRR - capex and operating cash flow

        # note: hurdle rate typically 20% IRR before investing in it due to typically optimistic assumptions
        # note: negative retained earnings (keeping debt, paying down equity) - to get around it, do another line for retained earnings and watch dividends paid by the rpoject (net income/equity should stay positive this way)

        print("Investor NPV: ", np.round(NPV * 1e-6, 2), "M USD")
        print("Investor MIRR: ", np.round(MIRR, 5), "")
        print("Investor ROI: ", np.round(ROI, 5), "")

    if save_plots or show_plots:
        if not os.path.exists("figures"):
            os.mkdir("figures")
            os.mkdir("figures/lcoh_breakdown")
            os.mkdir("figures/capex")
            os.mkdir("figures/annual_cash_flow")
        savepaths = [
            "figures/capex/",
            "figures/annual_cash_flow/",
            "figures/lcoh_breakdown/",
            "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.mkdir(savepath)

        pf.plot_capital_expenses(
            fileout="figures/capex/capital_expense_%i.pdf" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout="figures/annual_cash_flow/cash_flow_%i.png"
            % (design_scenario["id"]),
            show_plot=show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out).to_csv("data/cash_flow_%i.csv" % (design_scenario["id"]))

        pf.plot_costs(
            "figures/lcoh_breakdown/lcoh_%i" % (design_scenario["id"]),
            show_plot=show_plots,
        )

    return lcoh, pf
