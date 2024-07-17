import numpy as np
import pandas as pd
import warnings
import copy

from ORBIT import ProjectManager, load_config
from ORBIT.core import Vessel
from ORBIT.core.library import initialize_library
from ORBIT.phases.design import DesignPhase
from ORBIT.phases.install import InstallPhase

from greenheart.simulation.technologies.hydrogen.h2_transport.h2_compression import (
    Compressor,
)
from greenheart.simulation.technologies.hydrogen.h2_storage.pressure_vessel.compressed_gas_storage_model_20221021.Compressed_all import (
    PressureVessel,
)
from greenheart.simulation.technologies.hydrogen.h2_storage.pipe_storage import (
    UndergroundPipeStorage,
)

from greenheart.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern.lined_rock_cavern import (
    LinedRockCavernStorage,
)
from greenheart.simulation.technologies.hydrogen.h2_storage.salt_cavern.salt_cavern import (
    SaltCavernStorage,
)
from greenheart.simulation.technologies.hydrogen.h2_storage.on_turbine.on_turbine_hydrogen_storage import (
    PressurizedTower,
)

from greenheart.simulation.technologies.hydrogen.h2_transport.h2_export_pipe import (
    run_pipe_analysis,
)
from greenheart.simulation.technologies.hydrogen.h2_transport.h2_pipe_array import (
    run_pipe_array_const_diam,
)
from greenheart.simulation.technologies.offshore.fixed_platform import (
    FixedPlatformDesign,
    FixedPlatformInstallation,
)
from greenheart.simulation.technologies.offshore.floating_platform import (
    FloatingPlatformDesign,
    FloatingPlatformInstallation,
)
from greenheart.simulation.technologies.offshore.all_platforms import calc_platform_opex

from greenheart.simulation.technologies.hydrogen.h2_storage.storage_sizing import (
    hydrogen_storage_capacity,
)


def run_h2_pipe_array(
    greenheart_config,
    hopp_config,
    turbine_config,
    wind_cost_results,
    electrolyzer_physics_results,
    design_scenario,
    verbose,
):
    if design_scenario["transportation"] == "hvdc+pipeline" or (
        design_scenario["electrolyzer_location"] == "turbine"
        and not design_scenario["h2_storage_location"] == "turbine"
    ):
        # get pipe lengths from ORBIT using cable lengths (horizontal only)
        if design_scenario["wind_location"] == "offshore":
            pipe_lengths = wind_cost_results.orbit_project.phases[
                "ArraySystemDesign"
            ].sections_distance

        if (
            design_scenario["wind_location"] == "onshore"
        ):  # TODO: improve pipe length estimate
            raise NotImplementedError(
                "distributed electrolysis is not implemented for onshore wind."
            )
            # np.ones_like(hopp_config["technologies"]["wind"]["num_turbines"])*(
            #     greenheart_config["site"]["wind_layout"]["turbine_spacing"]
            #     *turbine_config['rotor_diameter']
            # )

        turbine_h2_flowrate = (
            max(
                electrolyzer_physics_results["H2_Results"][
                    "Hydrogen Hourly Production [kg/hr]"
                ]
            )
            * ((1.0 / 60.0) ** 2)
            / hopp_config["technologies"]["wind"]["num_turbines"]
        )
        m_dot = (
            np.ones_like(pipe_lengths) * turbine_h2_flowrate
        )  # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
        p_inlet = (
            31  # Inlet pressure [bar] - assumed outlet pressure from electrolyzer model
        )
        p_outlet = 10  # Outlet pressure [bar] - about 20 bar drop
        depth = greenheart_config["site"]["depth"]  # depth of pipe [m]

        capex, opex = run_pipe_array_const_diam(
            pipe_lengths, depth, p_inlet, p_outlet, m_dot
        )

        h2_pipe_array_results = {"capex": capex, "opex": opex}
    else:
        h2_pipe_array_results = {"capex": 0.0, "opex": 0.0}

    return h2_pipe_array_results


def run_h2_transport_compressor(
    greenheart_config, electrolyzer_physics_results, design_scenario, verbose=False
):
    if (
        design_scenario["transportation"] == "pipeline"
        or design_scenario["transportation"] == "hvdc+pipeline"
        or (
            design_scenario["h2_storage_location"] != "onshore"
            and design_scenario["electrolyzer_location"] == "onshore"
        )
    ):
        ########## compressor model from Jamie Kee based on HDSAM
        flow_rate_kg_per_hr = max(
            electrolyzer_physics_results["H2_Results"][
                "Hydrogen Hourly Production [kg/hr]"
            ]
        )  # kg/hr
        number_of_compressors = 2  # a third will be added as backup in the code
        p_inlet = 20  # bar
        p_outlet = greenheart_config["h2_transport_compressor"][
            "outlet_pressure"
        ]  # bar
        flow_rate_kg_d = flow_rate_kg_per_hr * 24.0

        compressor = Compressor(
            p_outlet,
            flow_rate_kg_d,
            p_inlet=p_inlet,
            n_compressors=number_of_compressors,
        )
        compressor.compressor_power()
        motor_rating, system_power_kw = compressor.compressor_system_power()
        total_capex, total_OM = compressor.compressor_costs()  # 2016$ , 2016$/y

        # print(f"CAPEX: {round(total_capex,2)} $")
        # print(f"Annual operating expense: {round(total_OM,2)} $/yr")

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
    orbit_config,
    greenheart_config,
    electrolyzer_physics_results,
    design_scenario,
    verbose=False,
):
    # prepare inputs
    export_pipe_length = orbit_config["site"]["distance_to_landfall"]  # Length [km]
    mass_flow_rate = max(
        electrolyzer_physics_results["H2_Results"]["Hydrogen Hourly Production [kg/hr]"]
    ) * (
        (1.0 / 60.0) ** 2
    )  # from [kg/hr] to mass flow rate in [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = greenheart_config["h2_transport_compressor"][
        "outlet_pressure"
    ]  # Inlet pressure [bar]
    p_outlet = greenheart_config["h2_transport_pipe"][
        "outlet_pressure"
    ]  # Outlet pressure [bar]
    depth = greenheart_config["site"]["depth"]  # depth of pipe [m]

    # run model
    if (
        design_scenario["transportation"] == "pipeline"
        or design_scenario["transportation"] == "hvdc+pipeline"
    ) or (
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
    hopp_config,
    greenheart_config,
    turbine_config,
    electrolyzer_physics_results,
    design_scenario,
    verbose=False,
):

    if design_scenario["h2_storage_location"] == "platform":
        if (
            greenheart_config["h2_storage"]["type"] != "pressure_vessel"
            and greenheart_config["h2_storage"]["type"] != "none"
        ):
            raise ValueError(
                "Only pressure vessel storage can be used on the off shore platform"
            )

    if design_scenario["h2_storage_location"] == "turbine":
        if (
            greenheart_config["h2_storage"]["type"] != "turbine"
            and greenheart_config["h2_storage"]["type"] != "pressure_vessel"
            and greenheart_config["h2_storage"]["type"] != "none"
        ):
            raise ValueError(
                "Only turbine or pressure vessel storage can be used for turbine hydrogen storage location"
            )
    ########### initialize output dictionary ###########
    h2_storage_results = dict()

    storage_max_fill_rate = np.max(
        electrolyzer_physics_results["H2_Results"]["Hydrogen Hourly Production [kg/hr]"]
    )

    ########### get hydrogen storage size in kilograms ###########
    ##################### no hydrogen storage
    if greenheart_config["h2_storage"]["type"] == "none":
        h2_storage_capacity_kg = 0.0
        storage_max_fill_rate = 0.0

    ##################### get storage capacity from turbine storage model
    elif greenheart_config["h2_storage"]["capacity_from_max_on_turbine_storage"]:
        nturbines = hopp_config["technologies"]["wind"]["num_turbines"]
        turbine = {
            "tower_length": turbine_config["tower"]["length"],
            "section_diameters": turbine_config["tower"]["section_diameters"],
            "section_heights": turbine_config["tower"]["section_heights"],
        }

        h2_storage = PressurizedTower(
            greenheart_config["project_parameters"]["atb_year"], turbine
        )
        h2_storage.run()

        h2_storage_capacity_single_turbine = h2_storage.get_capacity_H2()  # kg

        h2_storage_capacity_kg = nturbines * h2_storage_capacity_single_turbine  # in kg

    ##################### get storage capacity from hydrogen storage demand
    elif greenheart_config["h2_storage"]["size_capacity_from_demand"]["flag"]:
        hydrogen_storage_demand = np.mean(
            electrolyzer_physics_results["H2_Results"][
                "Hydrogen Hourly Production [kg/hr]"
            ]
        )  # TODO: update demand based on end-use needs
        (
            hydrogen_storage_capacity_kg,
            hydrogen_storage_duration_hr,
            hydrogen_storage_soc,
        ) = hydrogen_storage_capacity(
            electrolyzer_physics_results["H2_Results"],
            greenheart_config["electrolyzer"]["rating"],
            hydrogen_storage_demand,
        )
        h2_storage_capacity_kg = hydrogen_storage_capacity_kg
        h2_storage_results["hydrogen_storage_duration_hr"] = (
            hydrogen_storage_duration_hr
        )
        h2_storage_results["hydrogen_storage_soc"] = hydrogen_storage_soc

    ##################### get storage capacity based on storage days in config
    else:
        storage_hours = greenheart_config["h2_storage"]["days"] * 24
        h2_storage_capacity_kg = round(storage_hours * storage_max_fill_rate)

    h2_storage_results["h2_storage_capacity_kg"] = h2_storage_capacity_kg
    h2_storage_results["h2_storage_max_fill_rate_kg_hr"] = storage_max_fill_rate

    ########### run specific hydrogen storage models for costs and energy use ###########
    if (
        greenheart_config["h2_storage"]["type"] == "none"
        or design_scenario["h2_storage_location"] == "none"
    ):
        h2_storage_results["storage_capex"] = 0.0
        h2_storage_results["storage_opex"] = 0.0
        h2_storage_results["storage_energy"] = 0.0

        h2_storage = None

    elif greenheart_config["h2_storage"]["type"] == "turbine":
        if design_scenario["h2_storage_location"] == "turbine":
            turbine = {
                "tower_length": turbine_config["tower"]["length"],
                "section_diameters": turbine_config["tower"]["section_diameters"],
                "section_heights": turbine_config["tower"]["section_heights"],
            }

            h2_storage = PressurizedTower(
                greenheart_config["project_parameters"]["atb_year"], turbine
            )
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

            h2_storage_results["storage_energy"] = (
                0.0  # low pressure, so no additional compression needed beyond electolyzer
            )
        else:
            raise ValueError(
                "`h2_storage_location` must be set to 'turbine' to use 'turbine' for h2 storage type."
            )

    elif greenheart_config["h2_storage"]["type"] == "pipe":
        # for more information, see https://www.nrel.gov/docs/fy14osti/58564.pdf
        # initialize dictionary for pipe storage parameters
        storage_input = dict()

        # pull parameters from plat_config file
        storage_input["h2_storage_kg"] = h2_storage_capacity_kg
        storage_input["compressor_output_pressure"] = greenheart_config[
            "h2_storage_compressor"
        ]["output_pressure"]
        storage_input["system_flow_rate"] = storage_max_fill_rate
        storage_input["model"] = "papadias"

        # run pipe storage model
        h2_storage = UndergroundPipeStorage(storage_input)

        h2_storage.pipe_storage_capex()
        h2_storage.pipe_storage_opex()

        h2_storage_results["storage_capex"] = h2_storage.output_dict[
            "pipe_storage_capex"
        ]
        h2_storage_results["storage_opex"] = h2_storage.output_dict["pipe_storage_opex"]
        h2_storage_results["storage_energy"] = 0.0

    elif greenheart_config["h2_storage"]["type"] == "pressure_vessel":
        if design_scenario["h2_storage_location"] == "turbine":
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
            ) = h2_storage.distributed_storage_vessels(h2_storage_capacity_kg, 1)
            # ) = h2_storage.distributed_storage_vessels(h2_capacity, nturbines)
            # capex, opex, energy = h2_storage.calculate_from_fit(h2_capacity)

            h2_storage_results["storage_capex"] = capex_dist_total
            h2_storage_results["storage_opex"] = opex_dist_total
            h2_storage_results["storage_energy"] = (
                energy
                * electrolyzer_physics_results["H2_Results"][
                    "Life: Annual H2 production [kg/year]"
                ]
            )  # total in kWh
            h2_storage_results["tank_mass_full_kg"] = (
                h2_storage.get_tank_mass(h2_storage_capacity_kg)[1]
                + h2_storage_capacity_kg
            )
            h2_storage_results["tank_footprint_m2"] = h2_storage.get_tank_footprint(
                h2_storage_capacity_kg, upright=True
            )[1]
            h2_storage_results["tank volume (m^3)"] = (
                h2_storage.compressed_gas_function.Vtank
            )
            h2_storage_results["Number of tanks"] = h2_storage.get_tanks(
                h2_storage_capacity_kg
            )
            if verbose:
                print("ENERGY FOR STORAGE: ", energy * 1e-3 / (365 * 24), " MW")
                print("Tank volume (M^3): ", h2_storage_results["tank volume (m^3)"])
                print(
                    "Single Tank capacity (kg): ",
                    h2_storage.compressed_gas_function.single_tank_h2_capacity_kg,
                )
                print("N Tanks: ", h2_storage_results["Number of tanks"])

        else:
            # if plant_config["project_parameters"]["grid_connection"]:
            #     energy_cost = plant_config["project_parameters"]["ppa_price"]
            # else:
            #     energy_cost = 0.0
            energy_cost = 0.0  # energy cost is now handled outside the storage model

            h2_storage = PressureVessel(Energy_cost=energy_cost)
            h2_storage.run()

            capex, opex, energy = h2_storage.calculate_from_fit(h2_storage_capacity_kg)

            h2_storage_results["storage_capex"] = capex
            h2_storage_results["storage_opex"] = opex
            h2_storage_results["storage_energy"] = (
                energy
                * electrolyzer_physics_results["H2_Results"][
                    "Life: Annual H2 production [kg/year]"
                ]
            )  # total in kWh
            h2_storage_results["tank_mass_full_kg"] = (
                h2_storage.get_tank_mass(h2_storage_capacity_kg)[1]
                + h2_storage_capacity_kg
            )
            h2_storage_results["tank_footprint_m2"] = h2_storage.get_tank_footprint(
                h2_storage_capacity_kg, upright=True
            )[1]
            h2_storage_results["tank volume (m^3)"] = (
                h2_storage.compressed_gas_function.Vtank
            )
            h2_storage_results["Number of tanks"] = (
                h2_storage.compressed_gas_function.number_of_tanks
            )
            if verbose:
                print("ENERGY FOR STORAGE: ", energy * 1e-3 / (365 * 24), " MW")
                print("Tank volume (M^3): ", h2_storage_results["tank volume (m^3)"])
                print(
                    "Single Tank capacity (kg): ",
                    h2_storage.compressed_gas_function.single_tank_h2_capacity_kg,
                )
                print("N Tanks: ", h2_storage_results["Number of tanks"])

    elif greenheart_config["h2_storage"]["type"] == "salt_cavern":
        # initialize dictionary for salt cavern storage parameters
        storage_input = dict()

        # pull parameters from plant_config file
        storage_input["h2_storage_kg"] = h2_storage_capacity_kg
        storage_input["system_flow_rate"] = storage_max_fill_rate
        storage_input["model"] = "papadias"

        # run salt cavern storage model
        h2_storage = SaltCavernStorage(storage_input)

        h2_storage.salt_cavern_capex()
        h2_storage.salt_cavern_opex()

        h2_storage_results["storage_capex"] = h2_storage.output_dict[
            "salt_cavern_storage_capex"
        ]
        h2_storage_results["storage_opex"] = h2_storage.output_dict[
            "salt_cavern_storage_opex"
        ]
        h2_storage_results["storage_energy"] = 0.0

    elif greenheart_config["h2_storage"]["type"] == "lined_rock_cavern":
        # initialize dictionary for salt cavern storage parameters
        storage_input = dict()

        # pull parameters from plat_config file
        storage_input["h2_storage_kg"] = h2_storage_capacity_kg
        storage_input["system_flow_rate"] = storage_max_fill_rate
        storage_input["model"] = "papadias"

        # run salt cavern storage model
        h2_storage = LinedRockCavernStorage(storage_input)

        h2_storage.lined_rock_cavern_capex()
        h2_storage.lined_rock_cavern_opex()

        h2_storage_results["storage_capex"] = h2_storage.output_dict[
            "lined_rock_cavern_storage_capex"
        ]
        h2_storage_results["storage_opex"] = h2_storage.output_dict[
            "lined_rock_cavern_storage_opex"
        ]
        h2_storage_results["storage_energy"] = 0.0
    else:
        raise (
            ValueError(
                "H2 storage type %s was given, but must be one of ['none', 'turbine', 'pipe', 'pressure_vessel', 'salt_cavern', 'lined_rock_cavern']"
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
        print(
            "H2 storage capacity (tonnes): ",
            h2_storage_results["h2_storage_capacity_kg"] / 1000,
        )
        if h2_storage_results["h2_storage_capacity_kg"] > 0:
            print(
                "H2 storage cost $/kg of H2: ",
                h2_storage_results["storage_capex"]
                / h2_storage_results["h2_storage_capacity_kg"],
            )

    return h2_storage, h2_storage_results


def run_equipment_platform(
    hopp_config,
    greenheart_config,
    orbit_config,
    design_scenario,
    hopp_results,
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
        or design_scenario["pv_location"] == "platform"
        or design_scenario["battery_location"] == "platform"
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
            and greenheart_config["h2_storage"]["type"] != "none"
        ):
            topmass += (
                h2_storage_results["tank_mass_full_kg"] * 1e-3
            )  # from kg to tonnes
            toparea += h2_storage_results["tank_footprint_m2"]

        if (
            "battery" in hopp_config["technologies"].keys()
            and design_scenario["battery_location"] == "platform"
        ):
            battery_area = hopp_results["hybrid_plant"].battery.footprint_area
            battery_mass = hopp_results["hybrid_plant"].battery.system_mass

            topmass += battery_mass
            toparea += battery_area

        if (
            hopp_config["site"]["solar"]
            and design_scenario["pv_location"] == "platform"
        ):
            pv_area = hopp_results["hybrid_plant"].pv.footprint_area
            solar_mass = hopp_results["hybrid_plant"].pv.system_mass

            if pv_area > toparea:
                warnings.warn(
                    f"Solar area ({pv_area} m^2) must be smaller than platform area ({toparea} m^2)",
                    UserWarning,
                )
            topmass += solar_mass

        #### initialize
        if (
            greenheart_config["platform"]["design_phases"][0]
            == "FloatingPlatformDesign"
        ):
            if not ProjectManager.find_key_match("FloatingPlatformDesign"):
                ProjectManager.register_design_phase(FloatingPlatformDesign)
            if not ProjectManager.find_key_match("FloatingPlatformInstallation"):
                ProjectManager.register_install_phase(FloatingPlatformInstallation)
        else:
            if not ProjectManager.find_key_match("FixedPlatformDesign"):
                ProjectManager.register_design_phase(FixedPlatformDesign)
            if not ProjectManager.find_key_match("FixedPlatformInstallation"):
                ProjectManager.register_install_phase(FixedPlatformInstallation)

        platform_config = copy.deepcopy(greenheart_config["platform"])

        # assign site parameters
        if platform_config["site"]["depth"] == -1:
            platform_config["site"]["depth"] = greenheart_config["site"]["depth"]
        if platform_config["site"]["distance"] == -1:
            platform_config["site"]["distance"] = orbit_config["site"]["distance"]
        # assign equipment values
        
        if platform_config["equipment"]["tech_combined_mass"] == -1:
            platform_config["equipment"]["tech_combined_mass"] = topmass
        if platform_config["equipment"]["tech_required_area"] == -1:
            platform_config["equipment"]["tech_required_area"] = toparea
        platform = ProjectManager(platform_config)
        platform.run()

        design_capex = platform.design_results["platform_design"]["total_cost"]
        install_capex = platform.installation_capex
        total_capex = design_capex + install_capex

        total_opex = calc_platform_opex(total_capex, platform_config["opex_rate"])

        platform_mass = platform.design_results["platform_design"]["mass"]
        platform_area = platform.design_results["platform_design"]["area"]

    else:
        platform_mass = 0.0
        platform_area = 0.0
        total_capex = 0.0
        total_opex = 0.0

    platform_results = {
        "topmass_kg": topmass,
        "toparea_m2": platform_area,
        "platform_mass_tonnes": platform_mass,
        "capex": total_capex,
        "opex": total_opex,
    }
    
    if verbose:
        print("\nPlatform Results")
        for key in platform_results.keys():
            print(key, "%.2f" % (platform_results[key]))

    return platform_results
