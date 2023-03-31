import numpy as np
import matplotlib.pyplot as plt

import examples.hopp_tools as hopp_tools

from hopp.hydrogen.desal.desal_model import RO_desal
from hopp.hydrogen.electrolysis.pem_mass_and_footprint import (
    mass as run_electrolyzer_mass,
)
from hopp.hydrogen.electrolysis.pem_mass_and_footprint import (
    footprint as run_electrolyzer_footprint,
)
from hopp.hydrogen.electrolysis.H2_cost_model import basic_H2_cost_model


def run_electrolyzer_physics(
    hopp_results,
    hopp_scenario,
    hopp_h2_args,
    plant_config,
    wind_resource,
    design_scenario,
    show_plots=False,
    save_plots=False,
    verbose=False,
):
    # parse inputs to provide to hopp_tools call
    hybrid_plant = hopp_results["hybrid_plant"]

    if plant_config["project_parameters"]["grid_connection"]:
        # print(np.ones(365*24)*(hopp_h2_args["electrolyzer_size"]*1E3))
        energy_to_electrolyzer_kw = np.ones(365 * 24) * (
            hopp_h2_args["electrolyzer_size"] * 1e3
        )
    else:
        energy_to_electrolyzer_kw = hopp_results[
            "combined_pv_wind_power_production_hopp"
        ]

    scenario = hopp_scenario
    wind_size_mw = hopp_h2_args["wind_size_mw"]
    solar_size_mw = hopp_h2_args["solar_size_mw"]
    electrolyzer_size_mw = hopp_h2_args["electrolyzer_size"]
    kw_continuous = hopp_h2_args["kw_continuous"]
    electrolyzer_capex_kw = plant_config["electrolyzer"]["electrolyzer_capex"]
    lcoe = hopp_results["lcoe"]

    # run electrolyzer model
    H2_Results, _, electrical_generation_timeseries = hopp_tools.run_H2_PEM_sim(
        hybrid_plant,
        energy_to_electrolyzer_kw,
        scenario,
        wind_size_mw,
        solar_size_mw,
        electrolyzer_size_mw,
        kw_continuous,
        electrolyzer_capex_kw,
        lcoe,
    )

    # calculate utilization rate
    energy_capacity = hopp_h2_args["electrolyzer_size"] * 365 * 24  # MWh
    energy_available = sum(energy_to_electrolyzer_kw) * 1e-3  # MWh
    capacity_factor_electrolyzer = energy_available / energy_capacity

    # calculate mass and foorprint of system
    mass_kg = run_electrolyzer_mass(electrolyzer_size_mw)
    footprint_m2 = run_electrolyzer_footprint(electrolyzer_size_mw)

    # store results for return
    electrolyzer_physics_results = {
        "H2_Results": H2_Results,
        "electrical_generation_timeseries": electrical_generation_timeseries,
        "capacity_factor": capacity_factor_electrolyzer,
        "equipment_mass_kg": mass_kg,
        "equipment_footprint_m2": footprint_m2,
        "energy_to_electrolyzer_kw": energy_to_electrolyzer_kw,
    }

    if verbose:
        print("\nElectrolyzer Physics:")  # 61837444.34555772 145297297.29729727
        print("H2 Produced Annually: ", H2_Results["hydrogen_annual_output"])
        print(
            "Max H2 hourly (tonnes): ",
            max(H2_Results["hydrogen_hourly_production"]) * 1e-3,
        )
        print(
            "Max H2 daily (tonnes): ",
            max(
                np.convolve(
                    H2_Results["hydrogen_hourly_production"], np.ones(24), mode="valid"
                )
            )
            * 1e-3,
        )
        prodrate = 1.0 / 50.0  # kg/kWh
        roughest = electrical_generation_timeseries * prodrate
        print("Energy to electrolyzer (kWh): ", sum(energy_to_electrolyzer_kw))
        print(
            "Energy per kg (kWh/kg): ",
            energy_available * 1e3 / H2_Results["hydrogen_annual_output"],
        )
        print("Max hourly based on est kg/kWh (kg): ", max(roughest))
        print(
            "Max daily rough est (tonnes): ",
            max(np.convolve(roughest, np.ones(24), mode="valid")) * 1e-3,
        )
        print("Capacity Factor Electrolyzer: ", H2_Results["cap_factor"])

    if save_plots or show_plots:
        N = 24 * 7 * 4
        fig, ax = plt.subplots(3, 2, sharex=True, sharey="row")

        wind_speed = [W[2] for W in wind_resource._data["data"]]

        # plt.title("4-week running average")
        pad = 5
        ax[0, 0].annotate(
            "Hourly",
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )
        ax[0, 1].annotate(
            "4-week running average",
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

        ax[0, 0].plot(wind_speed)
        ave_x = range(4 * 7 * 24 - 1, len(H2_Results["hydrogen_hourly_production"]) + 1)
        print(len(ave_x))
        ax[0, 1].plot(ave_x, np.convolve(wind_speed, np.ones(N) / (N), mode="valid"))
        ax[0, 0].set(ylabel="Wind\n(m/s)", ylim=[0, 30], xlim=[0, len(wind_speed)])

        ax[1, 0].plot(electrical_generation_timeseries * 1e-3)
        ax[1, 0].axhline(y=400, color="r", linestyle="--", label="Nameplate Capacity")
        ax[1, 1].plot(
            ave_x[:-1],
            np.convolve(
                electrical_generation_timeseries * 1e-3, np.ones(N) / (N), mode="valid"
            ),
        )
        ax[1, 1].axhline(y=400, color="r", linestyle="--", label="Nameplate Capacity")
        ax[1, 0].set(ylabel="Power\n(MW)", ylim=[0, 500], xlim=[0, len(wind_speed)])
        # ax[1].legend(frameon=False, loc="best")

        ax[2, 0].plot(H2_Results["hydrogen_hourly_production"])
        ax[2, 1].plot(
            ave_x[:-1],
            np.convolve(
                H2_Results["hydrogen_hourly_production"], np.ones(N) / (N), mode="valid"
            ),
        )
        ax[2, 0].set(
            xlabel="Hour",
            ylabel="Hydrogen\n(kg/hr)",
            ylim=[0, 6000],
            xlim=[0, len(H2_Results["hydrogen_hourly_production"])],
        )
        ax[2, 1].set(
            xlabel="Hour",
            ylim=[0, 6000],
            xlim=[
                4 * 7 * 24 - 1,
                len(H2_Results["hydrogen_hourly_production"] + 4 * 7 * 24 + 2),
            ],
        )

        plt.tight_layout()
        if save_plots:
            plt.savefig(
                "figures/production/production_overview_%i.png"
                % (design_scenario["id"]),
                transparent=True,
            )
        if show_plots:
            plt.show()

    return electrolyzer_physics_results


def run_electrolyzer_cost(
    electrolyzer_physics_results,
    hopp_scenario,
    plant_config,
    design_scenario,
    verbose=False,
):
    # unpack inputs
    H2_Results = electrolyzer_physics_results["H2_Results"]
    electrolyzer_size_mw = plant_config["electrolyzer"]["rating"]
    useful_life = plant_config["project_parameters"]["project_lifetime"]
    atb_year = plant_config["atb_year"]
    electrical_generation_timeseries = electrolyzer_physics_results[
        "electrical_generation_timeseries"
    ]
    nturbines = plant_config["plant"]["num_turbines"]

    # run hydrogen production cost model - from hopp examples
    if design_scenario["h2_location"] == "onshore":
        offshore = 0
    else:
        offshore = 1

    if design_scenario["h2_location"] == "turbine":
        per_turb_electrolyzer_size_mw = electrolyzer_size_mw / nturbines
        per_turb_h2_annual_output = H2_Results["hydrogen_annual_output"] / nturbines
        per_turb_electrical_generation_timeseries = (
            electrical_generation_timeseries / nturbines
        )

        (
            cf_h2_annuals,
            per_turb_electrolyzer_total_capital_cost,
            per_turb_electrolyzer_OM_cost,
            per_turb_electrolyzer_capex_kw,
            time_between_replacement,
            h2_tax_credit,
            h2_itc,
        ) = basic_H2_cost_model(
            plant_config["electrolyzer"]["electrolyzer_capex"],
            plant_config["electrolyzer"]["time_between_replacement"],
            per_turb_electrolyzer_size_mw,
            useful_life,
            atb_year,
            per_turb_electrical_generation_timeseries,
            per_turb_h2_annual_output,
            hopp_scenario["H2 PTC"],
            hopp_scenario["Wind ITC"],
            include_refurb_in_opex=False,
            offshore=offshore,
        )

        electrolyzer_total_capital_cost = (
            per_turb_electrolyzer_total_capital_cost * nturbines
        )
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost * nturbines

    else:
        (
            cf_h2_annuals,
            electrolyzer_total_capital_cost,
            electrolyzer_OM_cost,
            electrolyzer_capex_kw,
            time_between_replacement,
            h2_tax_credit,
            h2_itc,
        ) = basic_H2_cost_model(
            plant_config["electrolyzer"]["electrolyzer_capex"],
            plant_config["electrolyzer"]["time_between_replacement"],
            electrolyzer_size_mw,
            useful_life,
            atb_year,
            electrical_generation_timeseries,
            H2_Results["hydrogen_annual_output"],
            hopp_scenario["H2 PTC"],
            hopp_scenario["Wind ITC"],
            include_refurb_in_opex=False,
            offshore=offshore,
        )

    # package outputs for return
    electrolyzer_cost_results = {
        "electrolyzer_total_capital_cost": electrolyzer_total_capital_cost,
        "electrolyzer_OM_cost_annual": electrolyzer_OM_cost,
    }

    # print some results if desired
    if verbose:
        print("\nHydrogen Cost Results:")
        print(
            "Electrolyzer Total CAPEX $/kW: ",
            electrolyzer_total_capital_cost / (electrolyzer_size_mw * 1e3),
        )
        print(
            "Electrolyzer O&M $/kW: ",
            electrolyzer_OM_cost / (electrolyzer_size_mw * 1e3),
        )
        print(
            "Electrolyzer O&M $/kg: ",
            electrolyzer_OM_cost / H2_Results["hydrogen_annual_output"],
        )

    return electrolyzer_cost_results


def run_desal(
    plant_config, electrolyzer_physics_results, design_scenario, verbose=False
):
    if verbose:
        print("\n")
        print("Desal Results")

    if design_scenario["h2_location"] == "onshore":
        desal_results = {
            "feed_water_flowrat_m3perhr": 0,
            "desal_capex_usd": 0,
            "desal_opex_usd_per_year": 0,
            "power_for_desal_kw": 0,
            "fresh_water_capacity_m3_per_hour": 0,
            "equipment_mass_kg": 0,
            "equipment_footprint_m2": 0,
        }
    else:
        freshwater_kg_per_hr = electrolyzer_physics_results["H2_Results"][
            "water_annual_usage"
        ] / (
            365 * 24
        )  # convert from kg/yr to kg/hr

        if design_scenario["h2_location"] == "platform":
            (
                desal_capacity_m3_per_hour,
                feedwater_m3_per_hr,
                desal_power,
                desal_capex,
                desal_opex,
                desal_mass_kg,
                desal_size_m2,
            ) = RO_desal(freshwater_kg_per_hr, salinity="Seawater")

            # package outputs
            desal_results = {
                "fresh_water_flowrate_m3perhr": desal_capacity_m3_per_hour,
                "feed_water_flowrat_m3perhr": feedwater_m3_per_hr,
                "desal_capex_usd": desal_capex,
                "desal_opex_usd_per_year": desal_opex,
                "power_for_desal_kw": desal_power,
                "equipment_mass_kg": desal_mass_kg,
                "equipment_footprint_m2": desal_size_m2,
            }

            if verbose:
                print("Fresh water needed (m^3/hr): ", desal_capacity_m3_per_hour)

        elif design_scenario["h2_location"] == "turbine":
            nturbines = plant_config["plant"]["num_turbines"]

            # size for per-turbine desal #TODO consider using individual power generation time series from each turbine
            in_turb_freshwater_kg_per_hr = freshwater_kg_per_hr / nturbines

            (
                per_turb_desal_capacity_m3_per_hour,
                per_turb_feedwater_m3_per_hr,
                per_turb_desal_power,
                per_turb_desal_capex,
                per_turb_desal_opex,
                per_turb_desal_mass_kg,
                per_turb_desal_size_m2,
            ) = RO_desal(in_turb_freshwater_kg_per_hr, salinity="Seawater")

            fresh_water_flowrate = nturbines * per_turb_desal_capacity_m3_per_hour
            feed_water_flowrate = nturbines * per_turb_feedwater_m3_per_hr
            desal_capex = nturbines * per_turb_desal_capex
            desal_opex = nturbines * per_turb_desal_opex
            power_for_desal = nturbines * per_turb_desal_power

            # package outputs
            desal_results = {
                "fresh_water_flowrate_m3perhr": fresh_water_flowrate,
                "feed_water_flowrat_m3perhr": feed_water_flowrate,
                "desal_capex_usd": desal_capex,
                "desal_opex_usd_per_year": desal_opex,
                "power_for_desal_kw": power_for_desal,
                "per_turb_equipment_mass_kg": per_turb_desal_mass_kg,
                "per_turb_equipment_footprint_m2": per_turb_desal_size_m2,
            }

        if verbose:
            print(
                "Fresh water needed (m^3/hr): ",
                desal_results["fresh_water_flowrate_m3perhr"],
            )
            print("Requested fresh water (m^3/hr):", freshwater_kg_per_hr / 997)

    if verbose:
        for key in desal_results.keys():
            print("Average", key, " ", np.average(desal_results[key]))
        print("\n")

    return desal_results
