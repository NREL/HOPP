# general imports
import os
from typing import Optional

import numpy as np
import pandas as pd
from greenheart.simulation.technologies.ammonia.ammonia import (
    run_ammonia_full_model,
)

from greenheart.simulation.technologies.steel.steel import (
    run_steel_full_model,
)

pd.options.mode.chained_assignment = None  # default='warn'

# visualization imports
import matplotlib.pyplot as plt

# HOPP imports
import greenheart.tools.eco.electrolysis as he_elec
import greenheart.tools.eco.finance as he_fin
import greenheart.tools.eco.hopp_mgmt as he_hopp
import greenheart.tools.eco.utilities as he_util
import greenheart.tools.eco.hydrogen_mgmt as he_h2


# set up function to run base line case
def run_simulation(
    filename_hopp_config: str,
    filename_greenheart_config: str,
    filename_turbine_config: str,
    filename_orbit_config: str,
    filename_floris_config: str,
    electrolyzer_rating_mw: Optional[float] = None,
    solar_rating: Optional[float] = None,
    battery_capacity_kw: Optional[float] = None,
    battery_capacity_kwh: Optional[float] = None,
    wind_rating: Optional[float] = None,
    verbose: bool = False,
    show_plots: bool = False,
    save_plots: bool = False,
    use_profast: bool = True,
    post_processing: bool = True,
    storage_type: Optional[str] = None,
    incentive_option: int = 1,
    plant_design_scenario: int = 1,
    output_level: int = 1,
    grid_connection: Optional[bool] = None,
):

    # load inputs as needed
    (
        hopp_config,
        greenheart_config,
        orbit_config,
        turbine_config,
        floris_config,
        orbit_hybrid_electrical_export_config,
    ) = he_util.get_inputs(
        filename_hopp_config,
        filename_greenheart_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        filename_turbine_config=filename_turbine_config,
        verbose=verbose,
        show_plots=show_plots,
        save_plots=save_plots,
    )

    if electrolyzer_rating_mw != None:
        greenheart_config["electrolyzer"]["flag"] = True
        greenheart_config["electrolyzer"]["rating"] = electrolyzer_rating_mw

    if solar_rating != None:
        hopp_config["site"]["solar"] = True
        hopp_config["technologies"]["pv"]["system_capacity_kw"] = solar_rating

    if battery_capacity_kw != None:
        hopp_config["site"]["battery"]["flag"] = True
        hopp_config["technologies"]["battery"][
            "system_capacity_kw"
        ] = battery_capacity_kw

    if battery_capacity_kwh != None:
        hopp_config["site"]["battery"]["flag"] = True
        hopp_config["technologies"]["battery"][
            "system_capacity_kwh"
        ] = battery_capacity_kwh

    if storage_type != None:
        greenheart_config["h2_storage"]["type"] = storage_type

    if wind_rating != None:
        orbit_config["plant"]["capacity"] = int(wind_rating * 1e-3)
        orbit_config["plant"]["num_turbines"] = int(
            wind_rating * 1e-3 / turbine_config["turbine_rating"]
        )
        hopp_config["technologies"]["wind"]["num_turbines"] = orbit_config["plant"][
            "num_turbines"
        ]

    if grid_connection != None:
        greenheart_config["project_parameters"]["grid_connection"] = grid_connection
        if grid_connection:
            hopp_config["technologies"]["grid"]["interconnect_kw"] = (
                orbit_config["plant"]["capacity"] * 1e6
            )

    # 7 scenarios, 3 discrete variables
    design_scenario = greenheart_config["plant_design"][
        "scenario%s" % (plant_design_scenario)
    ]
    design_scenario["id"] = plant_design_scenario

    # if design_scenario["h2_storage_location"] == "turbine":
    #     plant_config["h2_storage"]["type"] = "turbine"

    # run orbit for wind plant construction and other costs
    ## TODO get correct weather (wind, wave) inputs for ORBIT input (possibly via ERA5)
    orbit_project, orbit_hybrid_electrical_export_project = he_fin.run_orbit(
        orbit_config,
        weather=None,
        verbose=verbose,
        orbit_hybrid_electrical_export_config=orbit_hybrid_electrical_export_config,
    )

    # setup HOPP model
    # hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args = he_hopp.setup_hopp(hopp_config, greenheart_config, orbit_config, turbine_config, orbit_project, floris_config, show_plots=show_plots, save_plots=save_plots)
    hopp_config, hopp_site = he_hopp.setup_hopp(
        hopp_config,
        greenheart_config,
        orbit_config,
        turbine_config,
        orbit_project,
        floris_config,
        show_plots=show_plots,
        save_plots=save_plots,
    )

    # run HOPP model
    # hopp_results = he_hopp.run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=verbose)
    hopp_results = he_hopp.run_hopp(
        hopp_config,
        hopp_site,
        project_lifetime=orbit_config["project_parameters"]["project_lifetime"],
        verbose=verbose,
    )

    # this portion of the system is inside a function so we can use a solver to determine the correct energy availability for h2 production
    def energy_internals(
        hopp_results=hopp_results,
        orbit_project=orbit_project,
        design_scenario=design_scenario,
        orbit_config=orbit_config,
        hopp_config=hopp_config,
        greenheart_config=greenheart_config,
        turbine_config=turbine_config,
        wind_resource=hopp_site.wind_resource,
        verbose=verbose,
        show_plots=show_plots,
        save_plots=save_plots,
        solver=True,
        power_for_peripherals_kw_in=0.0,
        breakdown=False,
    ):

        hopp_results_internal = dict(hopp_results)

        # set energy input profile
        ### subtract peripheral power from supply to get what is left for electrolyzer
        remaining_power_profile_in = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )

        high_count = sum(
            np.asarray(hopp_results["combined_hybrid_power_production_hopp"])
            >= power_for_peripherals_kw_in
        )
        total_peripheral_energy = power_for_peripherals_kw_in * 365 * 24
        distributed_peripheral_power = total_peripheral_energy / high_count
        for i in range(len(hopp_results["combined_hybrid_power_production_hopp"])):
            r = (
                hopp_results["combined_hybrid_power_production_hopp"][i]
                - distributed_peripheral_power
            )
            if r > 0:
                remaining_power_profile_in[i] = r

        hopp_results_internal["combined_hybrid_power_production_hopp"] = tuple(
            remaining_power_profile_in
        )

        # run electrolyzer physics model
        electrolyzer_physics_results = he_elec.run_electrolyzer_physics(
            hopp_results_internal,
            orbit_config["project_parameters"]["project_lifetime"],
            greenheart_config,
            wind_resource,
            design_scenario,
            show_plots=show_plots,
            save_plots=save_plots,
            verbose=verbose,
        )

        # run electrolyzer cost model
        electrolyzer_cost_results = he_elec.run_electrolyzer_cost(
            electrolyzer_physics_results,
            orbit_config,
            hopp_config,
            greenheart_config,
            design_scenario,
            verbose=verbose,
        )

        desal_results = he_elec.run_desal(
            orbit_config, electrolyzer_physics_results, design_scenario, verbose
        )

        # run array system model
        h2_pipe_array_results = he_h2.run_h2_pipe_array(
            orbit_config,
            orbit_project,
            electrolyzer_physics_results,
            design_scenario,
            verbose,
        )

        # compressor #TODO size correctly
        h2_transport_compressor, h2_transport_compressor_results = (
            he_h2.run_h2_transport_compressor(
                greenheart_config,
                electrolyzer_physics_results,
                design_scenario,
                verbose=verbose,
            )
        )

        # transport pipeline
        h2_transport_pipe_results = he_h2.run_h2_transport_pipe(
            orbit_config,
            greenheart_config,
            electrolyzer_physics_results,
            design_scenario,
            verbose=verbose,
        )

        # pressure vessel storage
        pipe_storage, h2_storage_results = he_h2.run_h2_storage(
            orbit_config,
            greenheart_config,
            turbine_config,
            electrolyzer_physics_results,
            design_scenario,
            verbose=verbose,
        )

        total_energy_available = np.sum(
            hopp_results["combined_hybrid_power_production_hopp"]
        )

        ### get all energy non-electrolyzer usage in kw
        desal_power_kw = desal_results["power_for_desal_kw"]

        h2_transport_compressor_power_kw = h2_transport_compressor_results[
            "compressor_power"
        ]  # kW

        h2_storage_energy_kwh = h2_storage_results["storage_energy"]
        h2_storage_power_kw = h2_storage_energy_kwh * (1.0 / (365 * 24))

        # if transport is not HVDC and h2 storage is on shore, then power the storage from the grid
        if (design_scenario["transportation"] == "pipeline") and (
            design_scenario["h2_storage_location"] == "onshore"
        ):
            total_accessory_power_renewable_kw = (
                desal_power_kw + h2_transport_compressor_power_kw
            )
            total_accessory_power_grid_kw = h2_storage_power_kw
        else:
            total_accessory_power_renewable_kw = (
                desal_power_kw + h2_transport_compressor_power_kw + h2_storage_power_kw
            )
            total_accessory_power_grid_kw = 0.0

        ### subtract peripheral power from supply to get what is left for electrolyzer and also get grid power
        remaining_power_profile = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )
        grid_power_profile = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )
        for i in range(len(hopp_results["combined_hybrid_power_production_hopp"])):
            r = (
                hopp_results["combined_hybrid_power_production_hopp"][i]
                - total_accessory_power_renewable_kw
            )
            grid_power_profile[i] = total_accessory_power_grid_kw
            if r > 0:
                remaining_power_profile[i] = r

        if verbose and not solver:
            print("\nEnergy/Power Results:")
            print("Supply (MWh): ", total_energy_available)
            print("Desal (kW): ", desal_power_kw)
            print("Transport compressor (kW): ", h2_transport_compressor_power_kw)
            print("Storage compression, refrigeration, etc (kW): ", h2_storage_power_kw)
            # print("Difference: ", total_energy_available/(365*24) - np.sum(remaining_power_profile)/(365*24) - total_accessory_power_renewable_kw)

        if (show_plots or save_plots) and not solver:
            fig, ax = plt.subplots(1)
            plt.plot(
                np.asarray(hopp_results["combined_hybrid_power_production_hopp"])
                * 1e-6,
                label="Total Energy Available",
            )
            plt.plot(
                remaining_power_profile * 1e-6,
                label="Energy Available for Electrolysis",
            )
            plt.xlabel("Hour")
            plt.ylabel("Power (GW)")
            plt.tight_layout()
            if save_plots:
                savepath = "figures/power_series/"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                plt.savefig(
                    savepath + "power_%i.png" % (design_scenario["id"]),
                    transparent=True,
                )
            if show_plots:
                plt.show()
        if solver:
            if breakdown:
                return (
                    total_accessory_power_renewable_kw,
                    total_accessory_power_grid_kw,
                    desal_power_kw,
                    h2_transport_compressor_power_kw,
                    h2_storage_power_kw,
                    remaining_power_profile,
                )
            else:
                return total_accessory_power_renewable_kw
        else:
            return (
                electrolyzer_physics_results,
                electrolyzer_cost_results,
                desal_results,
                h2_pipe_array_results,
                h2_transport_compressor,
                h2_transport_compressor_results,
                h2_transport_pipe_results,
                pipe_storage,
                h2_storage_results,
                total_accessory_power_renewable_kw,
                total_accessory_power_grid_kw,
                remaining_power_profile,
            )

    # define function to provide to the brent solver
    def energy_residual_function(power_for_peripherals_kw_in):

        # get results for current design
        power_for_peripherals_kw_out = energy_internals(
            power_for_peripherals_kw_in=power_for_peripherals_kw_in,
            solver=True,
            verbose=False,
        )

        # collect residual
        power_residual = power_for_peripherals_kw_out - power_for_peripherals_kw_in

        return power_residual

    def simple_solver(initial_guess=0.0):

        # get results for current design
        (
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            desal_power_kw,
            h2_transport_compressor_power_kw,
            h2_storage_power_kw,
            remaining_power_profile,
        ) = energy_internals(
            power_for_peripherals_kw_in=initial_guess,
            solver=True,
            verbose=False,
            breakdown=True,
        )

        return (
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            desal_power_kw,
            h2_transport_compressor_power_kw,
            h2_storage_power_kw,
        )

    #################### solving for energy needed for non-electrolyzer components ####################################
    # this approach either exactly over over-estimates the energy needed for non-electrolyzer components
    solver_results = simple_solver(0)
    solver_result = solver_results[0]

    # # this is a check on the simple solver
    # print("\nsolver result: ", solver_result)
    # residual = energy_residual_function(solver_result)
    # print("\nresidual: ", residual)

    # this approach exactly sizes the energy needed for the non-electrolyzer components (according to the current models anyway)
    # solver_result = optimize.brentq(energy_residual_function, -10, 20000, rtol=1E-5)
    # OptimizeResult = optimize.root(energy_residual_function, 11E3, tol=1)
    # solver_result = OptimizeResult.x
    # solver_results = simple_solver(solver_result)
    # solver_result = solver_results[0]
    # print(solver_result)

    ##################################################################################################################

    # get results for final design
    (
        electrolyzer_physics_results,
        electrolyzer_cost_results,
        desal_results,
        h2_pipe_array_results,
        h2_transport_compressor,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        pipe_storage,
        h2_storage_results,
        total_accessory_power_renewable_kw,
        total_accessory_power_grid_kw,
        remaining_power_profile,
    ) = energy_internals(solver=False, power_for_peripherals_kw_in=solver_result)

    ## end solver loop here
    platform_results = he_h2.run_equipment_platform(
        hopp_config,
        greenheart_config,
        orbit_config,
        design_scenario,
        hopp_results,
        electrolyzer_physics_results,
        h2_storage_results,
        desal_results,
        verbose=verbose,
    )

    ################# OSW intermediate calculations" aka final financial calculations
    # does LCOE even make sense if we are only selling the H2? I think in this case LCOE should not be used, rather LCOH should be used. Or, we could use LCOE based on the electricity actually used for h2
    # I think LCOE is just being used to estimate the cost of the electricity used, but in this case we should just use the cost of the electricity generating plant since we are not selling to the grid. We
    # could build in a grid connection later such that we use LCOE for any purchased electricity and sell any excess electricity after H2 production
    # actually, I think this is what OSW is doing for LCOH

    # TODO double check full-system CAPEX
    capex, capex_breakdown = he_fin.run_capex(
        hopp_results,
        orbit_project,
        orbit_hybrid_electrical_export_project,
        electrolyzer_cost_results,
        h2_pipe_array_results,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        h2_storage_results,
        hopp_config,
        greenheart_config,
        orbit_config,
        design_scenario,
        desal_results,
        platform_results,
        verbose=verbose,
    )

    # TODO double check full-system OPEX
    opex_annual, opex_breakdown_annual = he_fin.run_opex(
        hopp_results,
        orbit_project,
        orbit_hybrid_electrical_export_project,
        electrolyzer_cost_results,
        h2_pipe_array_results,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        h2_storage_results,
        hopp_config,
        greenheart_config,
        orbit_config,
        desal_results,
        platform_results,
        verbose=verbose,
        total_export_system_cost=capex_breakdown["electrical_export_system"],
    )

    if verbose:
        print(
            "hybrid plant capacity factor: ",
            np.sum(hopp_results["combined_hybrid_power_production_hopp"])
            / (hopp_results["hybrid_plant"].system_capacity_kw.hybrid * 365 * 24),
        )

    steel_finance = None
    ammonia_finance = None

    if use_profast:
        lcoe, pf_lcoe = he_fin.run_profast_lcoe(
            greenheart_config,
            orbit_config,
            orbit_project,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            incentive_option,
            design_scenario,
            verbose=verbose,
            show_plots=show_plots,
            save_plots=save_plots,
        )
        lcoh_grid_only, pf_grid_only = he_fin.run_profast_grid_only(
            greenheart_config,
            orbit_config,
            orbit_project,
            electrolyzer_physics_results,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            design_scenario,
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            verbose=verbose,
            show_plots=show_plots,
            save_plots=save_plots,
        )
        lcoh, pf_lcoh = he_fin.run_profast_full_plant_model(
            greenheart_config,
            orbit_config,
            orbit_project,
            electrolyzer_physics_results,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            incentive_option,
            design_scenario,
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            verbose=verbose,
            show_plots=show_plots,
            save_plots=save_plots,
        )

        hydrogen_amount_kgpy = electrolyzer_physics_results["H2_Results"][
            "hydrogen_annual_output"
        ]

        if "steel" in greenheart_config:
            if verbose:
                print("Running steel\n")

            # use lcoh from the electrolyzer model if it is not already in the config
            if "lcoh" not in greenheart_config["steel"]["finances"]:
                greenheart_config["steel"]["finances"]["lcoh"] = lcoh

            # use lcoh from the electrolyzer model if it is not already in the config
            if "lcoh" not in greenheart_config["steel"]["costs"]:
                greenheart_config["steel"]["costs"]["lcoh"] = lcoh

            # use the hydrogen amount from the electrolyzer physics model if it is not already in the config
            if "hydrogen_amount_kgpy" not in greenheart_config["steel"]["capacity"]:
                greenheart_config["steel"]["capacity"][
                    "hydrogen_amount_kgpy"
                ] = hydrogen_amount_kgpy

            _, _, steel_finance = run_steel_full_model(greenheart_config)

        if "ammonia" in greenheart_config:
            if verbose:
                print("Running ammonia\n")

            # use the hydrogen amount from the electrolyzer physics model if it is not already in the config
            if "hydrogen_amount_kgpy" not in greenheart_config["ammonia"]["capacity"]:
                greenheart_config["ammonia"]["capacity"][
                    "hydrogen_amount_kgpy"
                ] = hydrogen_amount_kgpy

            _, _, ammonia_finance = run_ammonia_full_model(greenheart_config)

    ################# end OSW intermediate calculations
    if post_processing:
        power_breakdown = he_util.post_process_simulation(
            lcoe,
            lcoh,
            pf_lcoh,
            pf_lcoe,
            hopp_results,
            electrolyzer_physics_results,
            hopp_config,
            greenheart_config,
            orbit_config,
            h2_storage_results,
            capex_breakdown,
            opex_breakdown_annual,
            orbit_project,
            platform_results,
            desal_results,
            design_scenario,
            plant_design_scenario,
            incentive_option,
            solver_results=solver_results,
            show_plots=show_plots,
            save_plots=save_plots,
            verbose=verbose,
        )  # , lcoe, lcoh, lcoh_with_grid, lcoh_grid_only)

    # return
    if output_level == 0:
        return 0
    elif output_level == 1:
        return lcoh
    elif output_level == 2:
        return (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf_lcoh,
            electrolyzer_physics_results,
        )
    elif output_level == 3:
        return (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf_lcoh,
            electrolyzer_physics_results,
            pf_lcoe,
            power_breakdown,
        )
    elif output_level == 4:
        return lcoe, lcoh, lcoh_grid_only
    elif output_level == 5:
        return lcoe, lcoh, steel_finance, ammonia_finance
    elif output_level == 6:
        return hopp_results, electrolyzer_physics_results, remaining_power_profile


def run_sweeps(simulate=False, verbose=True, show_plots=True, use_profast=True):

    if simulate:
        verbose = False
        show_plots = False
    if simulate:
        storage_types = ["none", "pressure_vessel", "pipe", "salt_cavern"]
        wind_ratings = [400]  # , 800, 1200] #[200, 400, 600, 800]

        for wind_rating in wind_ratings:
            ratings = np.linspace(
                round(0.2 * wind_rating, ndigits=0), 2 * wind_rating + 1, 50
            )
            for storage_type in storage_types:
                lcoh_array = np.zeros(len(ratings))
                for z in np.arange(0, len(ratings)):
                    lcoh_array[z] = run_simulation(
                        electrolyzer_rating_mw=ratings[z],
                        wind_rating=wind_rating,
                        verbose=verbose,
                        show_plots=show_plots,
                        use_profast=use_profast,
                        storage_type=storage_type,
                    )
                    print(lcoh_array)
                np.savetxt(
                    "data/lcoh_vs_rating_%s_storage_%sMWwindplant.txt"
                    % (storage_type, wind_rating),
                    np.c_[ratings, lcoh_array],
                )

    if show_plots:

        wind_ratings = [400, 800, 1200]  # [200, 400, 600, 800]
        indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 6))

        for i in np.arange(0, len(wind_ratings)):
            wind_rating = wind_ratings[i]
            data_no_storage = np.loadtxt(
                "data/lcoh_vs_rating_none_storage_%sMWwindplant.txt" % (wind_rating)
            )
            data_pressure_vessel = np.loadtxt(
                "data/lcoh_vs_rating_pressure_vessel_storage_%sMWwindplant.txt"
                % (wind_rating)
            )
            data_salt_cavern = np.loadtxt(
                "data/lcoh_vs_rating_salt_cavern_storage_%sMWwindplant.txt"
                % (wind_rating)
            )
            data_pipe = np.loadtxt(
                "data/lcoh_vs_rating_pipe_storage_%sMWwindplant.txt" % (wind_rating)
            )

            ax[indexes[i]].plot(
                data_pressure_vessel[:, 0] / wind_rating,
                data_pressure_vessel[:, 1],
                label="Pressure Vessel",
            )
            ax[indexes[i]].plot(
                data_pipe[:, 0] / wind_rating, data_pipe[:, 1], label="Underground Pipe"
            )
            ax[indexes[i]].plot(
                data_salt_cavern[:, 0] / wind_rating,
                data_salt_cavern[:, 1],
                label="Salt Cavern",
            )
            ax[indexes[i]].plot(
                data_no_storage[:, 0] / wind_rating,
                data_no_storage[:, 1],
                "--k",
                label="No Storage",
            )

            ax[indexes[i]].scatter(
                data_pressure_vessel[np.argmin(data_pressure_vessel[:, 1]), 0]
                / wind_rating,
                np.min(data_pressure_vessel[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_pipe[np.argmin(data_pipe[:, 1]), 0] / wind_rating,
                np.min(data_pipe[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_salt_cavern[np.argmin(data_salt_cavern[:, 1]), 0] / wind_rating,
                np.min(data_salt_cavern[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_no_storage[np.argmin(data_no_storage[:, 1]), 0] / wind_rating,
                np.min(data_no_storage[:, 1]),
                color="k",
                label="Optimal ratio",
            )

            ax[indexes[i]].legend(frameon=False, loc="best")

            ax[indexes[i]].set_xlim([0.2, 2.0])
            ax[indexes[i]].set_ylim([0, 25])

            ax[indexes[i]].annotate("%s MW Wind Plant" % (wind_rating), (0.6, 1.0))

        ax[1, 0].set_xlabel("Electrolyzer/Wind Plant Rating Ratio")
        ax[1, 1].set_xlabel("Electrolyzer/Wind Plant Rating Ratio")
        ax[0, 0].set_ylabel("LCOH ($/kg)")
        ax[1, 0].set_ylabel("LCOH ($/kg)")

        plt.tight_layout()
        plt.savefig("lcoh_vs_rating_ratio.pdf", transparent=True)
        plt.show()

    return 0


def run_policy_options_storage_types(
    verbose=True, show_plots=False, save_plots=False, use_profast=True
):

    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]
    policy_options = [1, 2, 3, 4, 5, 6, 7]

    lcoh_array = np.zeros((len(storage_types), len(policy_options)))
    for i, storage_type in enumerate(storage_types):
        for j, poption in enumerate(policy_options):
            lcoh_array[i, j] = run_simulation(
                storage_type=storage_type,
                incentive_option=poption,
                verbose=verbose,
                show_plots=show_plots,
                use_profast=use_profast,
            )
        print(lcoh_array)

    savepath = "results/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.savetxt(
        savepath + "lcoh-with-policy.txt",
        np.c_[np.round(lcoh_array, decimals=2)],
        header="rows: %s, columns: %s"
        % ("".join(storage_types), "".join(str(p) for p in policy_options)),
        fmt="%.2f",
    )

    return 0


def run_policy_storage_design_options(
    verbose=False, show_plots=False, save_plots=False, use_profast=True
):

    design_scenarios = [1, 2, 3, 4, 5, 6, 7]
    policy_options = [1, 2, 3, 4, 5, 6, 7]
    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]

    design_series = []
    policy_series = []
    storage_series = []
    lcoh_series = []
    lcoe_series = []
    electrolyzer_capacity_factor_series = []
    annual_energy_breakdown_series = {
        "design": [],
        "policy": [],
        "storage": [],
        "wind_kwh": [],
        "renewable_kwh": [],
        "grid_power_kwh": [],
        "electrolyzer_kwh": [],
        "desal_kwh": [],
        "h2_transport_compressor_power_kwh": [],
        "h2_storage_power_kwh": [],
    }

    lcoh_array = np.zeros((len(design_scenarios), len(policy_options)))
    for i, design in enumerate(design_scenarios):
        for j, policy in enumerate(policy_options):
            for storage in storage_types:
                if storage != "pressure_vessel":  # and storage != "none"):
                    if design != 1 and design != 5 and design != 7:
                        print("skipping: ", design, " ", policy, " ", storage)
                        continue
                design_series.append(design)
                policy_series.append(policy)
                storage_series.append(storage)
                (
                    lcoh,
                    lcoe,
                    capex_breakdown,
                    opex_breakdown_annual,
                    pf_lcoh,
                    electrolyzer_physics_results,
                    pf_lcoe,
                    annual_energy_breakdown,
                ) = run_simulation(
                    storage_type=storage,
                    plant_design_scenario=design,
                    incentive_option=policy,
                    verbose=verbose,
                    show_plots=show_plots,
                    use_profast=use_profast,
                    output_level=3,
                )
                lcoh_series.append(lcoh)
                lcoe_series.append(lcoe)
                electrolyzer_capacity_factor_series.append(
                    electrolyzer_physics_results["capacity_factor"]
                )

                annual_energy_breakdown_series["design"].append(design)
                annual_energy_breakdown_series["policy"].append(policy)
                annual_energy_breakdown_series["storage"].append(storage)
                for key in annual_energy_breakdown.keys():
                    annual_energy_breakdown_series[key].append(
                        annual_energy_breakdown[key]
                    )

    savepath = "data/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df = pd.DataFrame.from_dict(
        {
            "Design": design_series,
            "Storage": storage_series,
            "Policy": policy_series,
            "LCOH [$/kg]": lcoh_series,
            "LCOE [$/kWh]": lcoe_series,
            "Electrolyzer capacity factor": electrolyzer_capacity_factor_series,
        }
    )
    df.to_csv(savepath + "design-storage-policy-lcoh.csv")

    df_energy = pd.DataFrame.from_dict(annual_energy_breakdown_series)
    df_energy.to_csv(savepath + "annual_energy_breakdown.csv")
    return 0


def run_design_options(
    verbose=False, show_plots=False, save_plots=False, incentive_option=1
):

    design_options = range(1, 8)  # 8
    scenario_lcoh = []
    scenario_lcoe = []
    scenario_capex_breakdown = []
    scenario_opex_breakdown_annual = []
    scenario_pf = []
    scenario_electrolyzer_physics = []

    for design in design_options:
        (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf,
            electrolyzer_physics_results,
        ) = run_simulation(
            verbose=verbose,
            show_plots=show_plots,
            use_profast=True,
            incentive_option=incentive_option,
            plant_design_scenario=design,
            output_level=2,
        )
        scenario_lcoh.append(lcoh)
        scenario_lcoe.append(lcoe)
        scenario_capex_breakdown.append(capex_breakdown)
        scenario_opex_breakdown_annual.append(opex_breakdown_annual)
        scenario_pf.append(pf)
        scenario_electrolyzer_physics.append(electrolyzer_physics_results)
    df_aggregate = pd.DataFrame.from_dict(
        {
            "Design": [int(x) for x in design_options],
            "LCOH [$/kg]": scenario_lcoh,
            "LCOE [$/kWh]": scenario_lcoe,
        }
    )
    df_capex = pd.DataFrame(scenario_capex_breakdown)
    df_opex = pd.DataFrame(scenario_opex_breakdown_annual)

    df_capex.insert(0, "Design", design_options)
    df_opex.insert(0, "Design", design_options)

    # df_aggregate = df_aggregate.transpose()
    df_capex = df_capex.transpose()
    df_opex = df_opex.transpose()

    results_path = "./combined_results/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    df_aggregate.to_csv(results_path + "metrics.csv")
    df_capex.to_csv(results_path + "capex.csv")
    df_opex.to_csv(results_path + "opex.csv")
    return 0


def run_storage_options():
    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]
    lcoe_list = []
    lcoh_list = []
    lcoh_with_grid_list = []
    lcoh_grid_only_list = []
    for storage_type in storage_types:
        lcoe, lcoh, _ = run_simulation(
            verbose=False,
            show_plots=False,
            save_plots=False,
            use_profast=True,
            incentive_option=1,
            plant_design_scenario=1,
            storage_type=storage_type,
            output_level=4,
            grid_connection=False,
        )
        lcoe_list.append(lcoe)
        lcoh_list.append(lcoh)

        # with grid
        _, lcoh_with_grid, lcoh_grid_only = run_simulation(
            verbose=False,
            show_plots=False,
            save_plots=False,
            use_profast=True,
            incentive_option=1,
            plant_design_scenario=1,
            storage_type=storage_type,
            output_level=4,
            grid_connection=True,
        )
        lcoh_with_grid_list.append(lcoh_with_grid)
        lcoh_grid_only_list.append(lcoh_grid_only)

    data_dict = {
        "Storage Type": storage_types,
        "LCOE [$/MW]": np.asarray(lcoe_list) * 1e3,
        "LCOH [$/kg]": lcoh_list,
        "LCOH with Grid [$/kg]": lcoh_with_grid_list,
        "LCOH Grid Only [$/kg]": lcoh_grid_only_list,
    }
    df = pd.DataFrame.from_dict(data_dict)

    savepath = "data/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df.to_csv(savepath + "storage-types-and-matrics.csv")
    return 0
