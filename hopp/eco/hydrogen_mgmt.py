import numpy as np
import pandas as pd

from hopp.hydrogen.h2_transport.h2_compression import Compressor
from hopp.hydrogen.h2_storage.pressure_vessel import PressureVessel
from hopp.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import (
    Underground_Pipe_Storage,
)
from hopp.hydrogen.h2_storage.on_turbine.on_turbine_hydrogen_storage import (
    PressurizedTower,
)
from hopp.hydrogen.h2_transport.h2_export_pipe import run_pipe_analysis
from hopp.hydrogen.h2_transport.h2_pipe_array import run_pipe_array_const_diam
from hopp.offshore.fixed_platform import (
    install_platform,
    calc_platform_opex,
    calc_substructure_mass_and_cost,
)


def run_h2_pipe_array(
    plant_config, orbit_project, electrolyzer_physics_results, design_scenario, verbose
):
    if (
        design_scenario["electrolyzer_location"] == "turbine"
        and not design_scenario["h2_storage_location"] == "turbine"
    ):
        # get pipe lengths from ORBIT using cable lengths (horizontal only)
        pipe_lengths = orbit_project.phases["ArraySystemDesign"].sections_distance

        turbine_h2_flowrate = (
            max(
                electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"]
            )
            * ((1.0 / 60.0) ** 2)
            / plant_config["plant"]["num_turbines"]
        )
        m_dot = (
            np.ones_like(pipe_lengths) * turbine_h2_flowrate
        )  # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
        p_inlet = (
            31  # Inlet pressure [bar] - assumed outlet pressure from electrolyzer model
        )
        p_outlet = 10  # Outlet pressure [bar] - about 20 bar drop
        depth = plant_config["site"]["depth"]  # depth of pipe [m]

        capex, opex = run_pipe_array_const_diam(
            pipe_lengths, depth, p_inlet, p_outlet, m_dot
        )

        h2_pipe_array_results = {"capex": capex, "opex": opex}
    else:
        h2_pipe_array_results = {"capex": 0.0, "opex": 0.0}

    return h2_pipe_array_results


def run_h2_transport_compressor(
    plant_config, electrolyzer_physics_results, design_scenario, verbose=False
):
    if design_scenario["transportation"] == "pipeline" or (
        design_scenario["h2_storage_location"] != "onshore"
        and design_scenario["electrolyzer_location"] == "onshore"
    ):
        ########## compressor model from Jamie Kee based on HDSAM
        flow_rate_kg_per_hr = max(
            electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"]
        )  # kg/hr
        number_of_compressors = 2  # a third will be added as backup in the code
        p_inlet = 20  # bar
        p_outlet = plant_config["h2_transport_compressor"]["outlet_pressure"]  # bar
        flow_rate_kg_d = flow_rate_kg_per_hr * 24.0

        compressor = Compressor(
            p_outlet,
            flow_rate_kg_d,
            p_inlet=p_inlet,
            n_compressors=number_of_compressors,
        )
        compressor.compressor_power()
        system_power_kw = compressor.compressor_system_power()
        total_capex, total_OM = compressor.compressor_costs()  # 2016$ , 2016$/y

        print(f"CAPEX: {round(total_capex,2)} $")
        print(f"Annual operating expense: {round(total_OM,2)} $/yr")

        h2_transport_compressor_results = {
            "compressor_power": system_power_kw,
            "compressor_capex": total_capex,
            "compressor_opex": total_OM,
        }

    else:
        compressor = None
        h2_transport_compressor_results = {
            "compressor_power": 0.0,
            "compressor_capex": 0.0,
            "compressor_opex": 0.0,
        }
        flow_rate_kg_per_hr = 0.0

    if verbose:
        print("\nCompressor Results:")
        print("Total H2 Flowrate (kg/hr): ", flow_rate_kg_per_hr)
        print(
            "Compressor_power (kW): ",
            h2_transport_compressor_results["compressor_power"],
        )
        print(
            "Compressor capex [USD]: ",
            h2_transport_compressor_results["compressor_capex"],
        )
        print(
            "Compressor opex [USD/yr]: ",
            h2_transport_compressor_results["compressor_opex"],
        )  # annual

    return compressor, h2_transport_compressor_results


def run_h2_transport_pipe(
    plant_config, electrolyzer_physics_results, design_scenario, verbose=False
):
    # prepare inputs
    export_pipe_length = plant_config["site"]["distance_to_landfall"]  # Length [km]
    mass_flow_rate = max(
        electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"]
    ) * (
        (1.0 / 60.0) ** 2
    )  # from [kg/hr] to mass flow rate in [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = plant_config["h2_transport_compressor"][
        "outlet_pressure"
    ]  # Inlet pressure [bar]
    p_outlet = plant_config["h2_transport_pipe"][
        "outlet_pressure"
    ]  # Outlet pressure [bar]
    depth = plant_config["site"]["depth"]  # depth of pipe [m]

    # run model
    if (design_scenario["transportation"] == "pipeline") or (
        design_scenario["h2_storage_location"] != "onshore"
        and design_scenario["electrolyzer_location"] == "onshore"
    ):
        h2_transport_pipe_results = run_pipe_analysis(
            export_pipe_length, mass_flow_rate, p_inlet, p_outlet, depth
        )
    else:
        h2_transport_pipe_results = pd.DataFrame.from_dict(
            {
                "index": 0,
                "Grade": ["none"],
                "Outer diameter (mm)": [0 * 141.3],
                "Inner Diameter (mm)": [0 * 134.5],
                "Schedule": ["none"],
                "Thickness (mm)": [0 * 3.4],
                "volume [m3]": [0 * 30.969133941093407],
                "weight [kg]": [0 * 242798.01009817232],
                "mat cost [$]": [0 * 534155.6222159792],
                "labor cost [$]": [0 * 2974375.749734022],
                "misc cost [$]": [0 * 970181.8962542458],
                "ROW cost [$]": [0 * 954576.9166912301],
                "total capital cost [$]": [0 * 5433290.0184895478],
                "annual operating cost [$]": [0.0],
            }
        )
    if verbose:
        print("\nH2 Transport Pipe Results")
        for col in h2_transport_pipe_results.columns:
            if col == "index":
                continue
            print(col, h2_transport_pipe_results[col][0])
        print("\n")

    return h2_transport_pipe_results


def run_h2_storage(
    plant_config,
    turbine_config,
    electrolyzer_physics_results,
    design_scenario,
    verbose=False,
):
    nturbines = plant_config["plant"]["num_turbines"]

    if design_scenario["h2_storage_location"] == "platform":
        if (
            plant_config["h2_storage"]["type"] != "pressure_vessel"
            and plant_config["h2_storage"]["type"] != "none"
        ):
            raise ValueError(
                "Only pressure vessel storage can be used on the off shore platform"
            )

    # initialize output dictionary
    h2_storage_results = dict()

    storage_hours = plant_config["h2_storage"]["days"] * 24
    storage_max_fill_rate = np.max(
        electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"]
    )

    ##################### get storage capacity from turbine storage model
    if plant_config["h2_storage"]["capacity_from_max_on_turbine_storage"]:
        turbine = {
            "tower_length": turbine_config["tower"]["length"],
            "section_diameters": turbine_config["tower"]["section_diameters"],
            "section_heights": turbine_config["tower"]["section_heights"],
        }

        h2_storage = PressurizedTower(plant_config["atb_year"], turbine)
        h2_storage.run()

        h2_storage_capacity_single_turbine = h2_storage.get_capacity_H2()  # kg

        h2_capacity = nturbines * h2_storage_capacity_single_turbine  # in kg
    ###################################
    else:
        h2_capacity = round(storage_hours * storage_max_fill_rate)

    if plant_config["h2_storage"]["type"] == "none":
        h2_storage_results["h2_capacity"] = 0.0
    else:
        h2_storage_results["h2_capacity"] = h2_capacity

    # if storage_hours == 0:
    if (
        plant_config["h2_storage"]["type"] == "none"
        or design_scenario["h2_storage_location"] == "none"
    ):
        h2_storage_results["storage_capex"] = 0.0
        h2_storage_results["storage_opex"] = 0.0
        h2_storage_results["storage_energy"] = 0.0

        h2_storage = None

    elif design_scenario["h2_storage_location"] == "turbine":
        if plant_config["h2_storage"]["type"] == "turbine":
            turbine = {
                "tower_length": turbine_config["tower"]["length"],
                "section_diameters": turbine_config["tower"]["section_diameters"],
                "section_heights": turbine_config["tower"]["section_heights"],
            }

            h2_storage = PressurizedTower(plant_config["atb_year"], turbine)
            h2_storage.run()

            h2_storage_results["storage_capex"] = nturbines * h2_storage.get_capex()
            h2_storage_results["storage_opex"] = nturbines * h2_storage.get_opex()

            if verbose:
                print("On-turbine H2 storage:")
                print("mass empty (single turbine): ", h2_storage.get_mass_empty())
                print(
                    "H2 capacity (kg) - single turbine: ", h2_storage.get_capacity_H2()
                )
                print("storage pressure: ", h2_storage.get_pressure_H2())

            h2_storage_results[
                "storage_energy"
            ] = 0.0  # low pressure, so no additional compression needed beyond electolyzer

        elif plant_config["h2_storage"]["type"] == "pressure_vessel":
            
            energy_cost = 0.0

            h2_storage = PressureVessel(Energy_cost=energy_cost)
            h2_storage.run()

            (
                capex_dist_total,
                opex_dist_total,
                energy,
                area_site,
                mass_tank_empty_site,
                _,
            ) = h2_storage.distributed_storage_vessels(h2_capacity, nturbines)
            # capex, opex, energy = h2_storage.calculate_from_fit(h2_capacity)

            h2_storage_results["storage_capex"] = capex_dist_total
            h2_storage_results["storage_opex"] = opex_dist_total
            h2_storage_results["storage_energy"] = (
                energy
                * electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"]
            )  # total in kWh
            h2_storage_results["tank_mass_full_kg"] = (
                h2_storage.get_tank_mass(h2_capacity)[1] + h2_capacity
            )
            h2_storage_results["tank_footprint_m2"] = h2_storage.get_tank_footprint(
                h2_capacity, upright=True
            )[1]
            h2_storage_results[
                "tank volume (m^3)"
            ] = h2_storage.compressed_gas_function.Vtank
            h2_storage_results["Number of tanks"] = h2_storage.get_tanks(h2_capacity)
            if verbose:
                print("ENERGY FOR STORAGE: ", energy * 1e-3 / (365 * 24), " MW")
                print("Tank volume (M^3): ", h2_storage_results["tank volume (m^3)"])
                print(
                    "Single Tank capacity (kg): ",
                    h2_storage.compressed_gas_function.single_tank_h2_capacity_kg,
                )
                print("N Tanks: ", h2_storage_results["Number of tanks"])

        else:
            ValueError(
                "with storage location set to tower, only 'pressure_vessel' and 'tower' types are implemented."
            )

    elif plant_config["h2_storage"]["type"] == "pipe":
        # for more information, see https://www.nrel.gov/docs/fy14osti/58564.pdf
        # initialize dictionary for pipe storage parameters
        storage_input = dict()

        # pull parameters from plat_config file
        storage_input["H2_storage_kg"] = h2_capacity
        storage_input["compressor_output_pressure"] = plant_config[
            "h2_storage_compressor"
        ]["output_pressure"]

        # run pipe storage model
        h2_storage = Underground_Pipe_Storage(storage_input, h2_storage_results)

        h2_storage.pipe_storage_costs()

        h2_storage_results["storage_capex"] = h2_storage_results["pipe_storage_capex"]
        h2_storage_results["storage_opex"] = h2_storage_results["pipe_storage_opex"]
        h2_storage_results["storage_energy"] = 0.0

    elif plant_config["h2_storage"]["type"] == "pressure_vessel":
        if plant_config["project_parameters"]["grid_connection"]:
            energy_cost = plant_config["project_parameters"]["ppa_price"]
        else:
            energy_cost = 0.0

        h2_storage = PressureVessel(Energy_cost=energy_cost)
        h2_storage.run()

        capex, opex, energy = h2_storage.calculate_from_fit(h2_capacity)

        h2_storage_results["storage_capex"] = capex
        h2_storage_results["storage_opex"] = opex
        h2_storage_results["storage_energy"] = (
            energy
            * electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"]
        )  # total in kWh
        h2_storage_results["tank_mass_full_kg"] = (
            h2_storage.get_tank_mass(h2_capacity)[1] + h2_capacity
        )
        h2_storage_results["tank_footprint_m2"] = h2_storage.get_tank_footprint(
            h2_capacity, upright=True
        )[1]
        h2_storage_results[
            "tank volume (m^3)"
        ] = h2_storage.compressed_gas_function.Vtank
        h2_storage_results[
            "Number of tanks"
        ] = h2_storage.compressed_gas_function.number_of_tanks
        if verbose:
            print("ENERGY FOR STORAGE: ", energy * 1e-3 / (365 * 24), " MW")
            print("Tank volume (M^3): ", h2_storage_results["tank volume (m^3)"])
            print(
                "Single Tank capacity (kg): ",
                h2_storage.compressed_gas_function.single_tank_h2_capacity_kg,
            )
            print("N Tanks: ", h2_storage_results["Number of tanks"])

    elif plant_config["h2_storage"]["type"] == "salt_cavern":
        # TODO replace this rough estimate with real numbers
        h2_storage = None
        capex = 36.0 * h2_capacity  # based on Papadias 2021 table 7
        opex = (
            0.021 * capex
        )  # based on https://www.pnnl.gov/sites/default/files/media/file/Hydrogen_Methodology.pdf

        h2_storage_results["storage_capex"] = capex
        h2_storage_results["storage_opex"] = opex
        h2_storage_results["storage_energy"] = 0.0
    else:
        raise (
            ValueError(
                "H2 storage type %s was given, but must be one of ['none', 'pipe', 'pressure_vessel', 'salt_cavern]"
            )
        )

    if verbose:
        print("\nH2 Storage Results:")
        print("H2 storage capex: ${0:,.0f}".format(h2_storage_results["storage_capex"]))
        print(
            "H2 storage annual opex: ${0:,.0f}/yr".format(
                h2_storage_results["storage_opex"]
            )
        )
        print("H2 storage capacity (tonnes): ", h2_storage_results["h2_capacity"]/1000)
        if h2_storage_results["h2_capacity"] > 0:
            print("H2 storage cost $/kg of H2: ", h2_storage_results["storage_capex"]/h2_storage_results["h2_capacity"])
        

    return h2_storage, h2_storage_results

def run_equipment_platform(
    plant_config,
    design_scenario,
    electrolyzer_physics_results,
    h2_storage_results,
    desal_results,
    verbose=False,
):
    topmass = 0.0  # tonnes
    toparea = 0.0  # m^2

    if (
        design_scenario["electrolyzer_location"] == "platform"
        or design_scenario["h2_storage_location"] == "platform"
    ):
        """ "equipment_mass_kg": desal_mass_kg,
        "equipment_footprint_m2": desal_size_m2"""

        if design_scenario["electrolyzer_location"] == "platform":
            topmass += (
                electrolyzer_physics_results["equipment_mass_kg"] * 1e-3
            )  # from kg to tonnes
            topmass += desal_results["equipment_mass_kg"] * 1e-3  # from kg to tonnes
            toparea += electrolyzer_physics_results["equipment_footprint_m2"]
            toparea += desal_results["equipment_footprint_m2"]

        if (
            design_scenario["h2_storage_location"] == "platform"
            and plant_config["h2_storage"]["type"] != "none"
        ):
            topmass += (
                h2_storage_results["tank_mass_full_kg"] * 1e-3
            )  # from kg to tonnes
            toparea += h2_storage_results["tank_footprint_m2"]

        distance = plant_config["site"]["distance_to_landfall"]

        installation_cost = install_platform(
            topmass,
            toparea,
            distance,
            install_duration=plant_config["platform"]["installation_days"],
        )

        depth = plant_config["site"]["depth"]  # depth of pipe [m]

        capex, platform_mass = calc_substructure_mass_and_cost(topmass, toparea, depth)

        opex_rate = plant_config["platform"]["opex_rate"]
        total_opex = calc_platform_opex(capex, opex_rate)

        total_capex = capex + installation_cost

    else:
        platform_mass = 0.0
        total_capex = 0.0
        total_opex = 0.0

    platform_results = {
        "topmass_kg": topmass,
        "toparea_m2": toparea,
        "platform_mass_tonnes": platform_mass,
        "capex": total_capex,
        "opex": total_opex,
    }
    if verbose:
        print("\nPlatform Results")
        for key in platform_results.keys():
            print(key, "%.2f" % (platform_results[key]))

    return platform_results
