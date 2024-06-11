import os
import os.path
import yaml
import copy

import numpy as np
import numpy_financial as npf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

import ORBIT as orbit

from hopp.simulation.technologies.resource.wind_resource import WindResource

from hopp.simulation import HoppInterface

from hopp.utilities import load_yaml

from hopp.simulation.technologies.dispatch import plot_tools

from .finance import adjust_orbit_costs

"""
This function returns the ceiling of a/b (rounded to the nearest greater integer). 
The function was copied from https://stackoverflow.com/a/17511341/5128616
"""
def ceildiv(a, b):
    return -(a // -b)

# Function to load inputs
def get_inputs(
    filename_hopp_config,
    filename_greenheart_config,
    filename_orbit_config,
    filename_turbine_config,
    filename_floris_config=None,
    verbose=False,
    show_plots=False,
    save_plots=False,
):

    ############### load turbine inputs from yaml

    # load turbine inputs
    turbine_config = load_yaml(filename_turbine_config)

    # load hopp inputs
    hopp_config = load_yaml(filename_hopp_config)

    # load eco inputs
    greenheart_config = load_yaml(filename_greenheart_config)

    ################ load plant inputs from yaml
    if filename_orbit_config != None:
        orbit_config = orbit.load_config(filename_orbit_config)

        # print plant inputs if desired
        if verbose:
            print("\nPlant configuration:")
            for key in orbit_config.keys():
                print(key, ": ", orbit_config[key])

        # check that orbit and hopp inputs are compatible
        if (
            orbit_config["plant"]["capacity"]
            != hopp_config["technologies"]["wind"]["num_turbines"]
            * hopp_config["technologies"]["wind"]["turbine_rating_kw"]
            * 1e-3
        ):
            raise (
                ValueError("Provided ORBIT and HOPP wind plant capacities do not match")
            )

    # update floris_config file with correct input from other files
    # load floris inputs
    if (
        hopp_config["technologies"]["wind"]["model_name"] == "floris"
    ):  # TODO replace elements of the file
        if filename_floris_config is None:
            raise (ValueError("floris input file must be specified."))
        else:
            floris_config = load_yaml(filename_floris_config)
            floris_config.update({"farm": {"turbine_type": turbine_config}})
    else:
        floris_config = None

    # print turbine inputs if desired
    if verbose:
        print("\nTurbine configuration:")
        for key in turbine_config.keys():
            print(key, ": ", turbine_config[key])

    ############## provide custom layout for ORBIT and FLORIS if desired
    if filename_orbit_config != None:
        if orbit_config["plant"]["layout"] == "custom":
            # generate ORBIT config from floris layout
            for i, x in enumerate(floris_config["farm"]["layout_x"]):
                floris_config["farm"]["layout_x"][i] = x + 400

            layout_config, layout_data_location = convert_layout_from_floris_for_orbit(
                floris_config["farm"]["layout_x"],
                floris_config["farm"]["layout_y"],
                save_config=True,
            )

            # update orbit_config with custom layout
            # orbit_config = orbit.core.library.extract_library_data(orbit_config, additional_keys=layout_config)
            orbit_config["array_system_design"]["location_data"] = layout_data_location

    # if hybrid plant, adjust hybrid plant capacity to include all technologies
    total_hybrid_plant_capacity_mw = 0.0
    for tech in hopp_config["technologies"].keys():
        if tech == "grid":
            continue
        elif tech == "wind":
            total_hybrid_plant_capacity_mw += (
                hopp_config["technologies"][tech]["num_turbines"]
                * hopp_config["technologies"][tech]["turbine_rating_kw"]
                * 1e-3
            )
        elif tech == "pv":
            total_hybrid_plant_capacity_mw += (
                hopp_config["technologies"][tech]["system_capacity_kw"] * 1e-3
            )
        elif tech == "wave":
            total_hybrid_plant_capacity_mw += (
                hopp_config["technologies"][tech]["num_devices"]
                * hopp_config["technologies"][tech]["device_rating_kw"]
                * 1e-3
            )

    # initialize dict for hybrid plant
    if filename_orbit_config != None:
        if total_hybrid_plant_capacity_mw != orbit_config["plant"]["capacity"]:
            orbit_hybrid_electrical_export_config = copy.deepcopy(orbit_config)
            orbit_hybrid_electrical_export_config["plant"][
                "capacity"
            ] = total_hybrid_plant_capacity_mw
            orbit_hybrid_electrical_export_config["plant"].pop(
                "num_turbines"
            )  # allow orbit to set num_turbines later based on the new hybrid capacity and turbine rating
        else:
            orbit_hybrid_electrical_export_config = {}

    if verbose:
        print(
            f"Total hybrid plant rating calculated: {total_hybrid_plant_capacity_mw} MW"
        )

    if filename_orbit_config is None:
        orbit_config = None
        orbit_hybrid_electrical_export_config = {}

    ############## return all inputs

    return (
        hopp_config,
        greenheart_config,
        orbit_config,
        turbine_config,
        floris_config,
        orbit_hybrid_electrical_export_config,
    )


def convert_layout_from_floris_for_orbit(turbine_x, turbine_y, save_config=False):

    turbine_x_km = (np.array(turbine_x) * 1e-3).tolist()
    turbine_y_km = (np.array(turbine_y) * 1e-3).tolist()

    # initialize dict with data for turbines
    turbine_dict = {
        "id": list(range(0, len(turbine_x))),
        "substation_id": ["OSS"] * len(turbine_x),
        "name": list(range(0, len(turbine_x))),
        "longitude": turbine_x_km,
        "latitude": turbine_y_km,
        "string": [0] * len(turbine_x),  # can be left empty
        "order": [0] * len(turbine_x),  # can be left empty
        "cable_length": [0] * len(turbine_x),
        "bury_speed": [0] * len(turbine_x),
    }
    string_counter = -1
    order_counter = 0
    for i in range(0, len(turbine_x)):
        if turbine_x[i] - 400 == 0:
            string_counter += 1
            order_counter = 0

        turbine_dict["order"][i] = order_counter
        turbine_dict["string"][i] = string_counter

        order_counter += 1

    # initialize dict with substation information
    substation_dict = {
        "id": "OSS",
        "substation_id": "OSS",
        "name": "OSS",
        "longitude": np.min(turbine_x_km) - 200 * 1e-3,
        "latitude": np.average(turbine_y_km),
        "string": "",  # can be left empty
        "order": "",  # can be left empty
        "cable_length": "",
        "bury_speed": "",
    }

    # combine turbine and substation dicts
    for key in turbine_dict.keys():
        # turbine_dict[key].append(substation_dict[key])
        turbine_dict[key].insert(0, substation_dict[key])

    # add location data
    file_name = "osw_cable_layout"
    save_location = "./input/project/plant/"
    # turbine_dict["array_system_design"]["location_data"] = data_location
    if save_config:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        # create pandas data frame
        df = pd.DataFrame.from_dict(turbine_dict)

        # df.drop("index")
        df.set_index("id")

        # save to csv
        df.to_csv(save_location + file_name + ".csv", index=False)

    return turbine_dict, file_name


def visualize_plant(
    hopp_config,
    greenheart_config,
    turbine_config,
    wind_cost_outputs,
    hopp_results,
    platform_results,
    desal_results,
    h2_storage_results,
    electrolyzer_physics_results,
    design_scenario,
    colors,
    plant_design_number,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):
    # save plant sizing to dict
    component_areas = {}

    plt.rcParams.update({"font.size": 7})

    if hopp_config["technologies"]["wind"]["model_name"] != "floris":
        raise (
            NotImplementedError(
                f"`visualize_plant()` only works with the 'floris' wind model, `model_name` \
                                  {hopp_config['technologies']['wind']['model_name']} has been specified"
            )
        )

    # set colors
    turbine_rotor_color = colors[0]
    turbine_tower_color = colors[1]
    pipe_color = colors[2]
    cable_color = colors[8]
    electrolyzer_color = colors[4]
    desal_color = colors[9]
    h2_storage_color = colors[6]
    substation_color = colors[7]
    equipment_platform_color = colors[1]
    compressor_color = colors[0]
    if hopp_config["site"]["solar"]:
        solar_color = colors[2]
    if hopp_config["site"]["wave"]:
        wave_color = colors[8]
    battery_color = colors[8]

    # set hatches
    solar_hatch = "//"
    wave_hatch = "\\\\"
    battery_hatch = "+"
    electrolyzer_hatch = "///"
    desalinator_hatch = "xxxx"

    # Views
    # offshore plant, onshore plant, offshore platform, offshore turbine

    # get plant location

    # get shore location

    # get cable/pipe locations
    if design_scenario["wind_location"] == "offshore":
        cable_array_points = (
            wind_cost_outputs.orbit_project.phases["ArraySystemDesign"].coordinates
            * 1e3
        )  # ORBIT gives coordinates in km, convert to m
        pipe_array_points = (
            wind_cost_outputs.orbit_project.phases["ArraySystemDesign"].coordinates
            * 1e3
        )  # ORBIT gives coordinates in km, convert to m

        # get turbine tower base diameter
        tower_base_diameter = wind_cost_outputs.orbit_project.config["turbine"][
            "tower"
        ]["section_diameters"][
            0
        ]  # in m
        tower_base_radius = tower_base_diameter / 2.0

        # get turbine locations
        turbine_x = (
            wind_cost_outputs.orbit_project.phases[
                "ArraySystemDesign"
            ].turbines_x.flatten()
            * 1e3
        )  # ORBIT gives coordinates in km, convert to m
        turbine_x = turbine_x[~np.isnan(turbine_x)]
        turbine_y = (
            wind_cost_outputs.orbit_project.phases[
                "ArraySystemDesign"
            ].turbines_y.flatten()
            * 1e3
        )  # ORBIT gives coordinates in km, convert to m
        turbine_y = turbine_y[~np.isnan(turbine_y)]

        # get offshore substation location and dimensions
        substation_x = (
            wind_cost_outputs.orbit_project.phases["ArraySystemDesign"].oss_x * 1e3
        )  # ORBIT gives coordinates in km, convert to m (treated as center)
        substation_y = (
            wind_cost_outputs.orbit_project.phases["ArraySystemDesign"].oss_y * 1e3
        )  # ORBIT gives coordinates in km, convert to m (treated as center)
        substation_side_length = 20  # [m] just based on a large substation (https://www.windpowerengineering.com/making-modern-offshore-substation/) since the dimensions are not available in ORBIT

        # get equipment platform location and dimensions
        equipment_platform_area = platform_results["toparea_m2"]
        equipment_platform_side_length = np.sqrt(equipment_platform_area)
        equipment_platform_x = (
            substation_x - substation_side_length - equipment_platform_side_length / 2
        )  # [m] (treated as center)
        equipment_platform_y = substation_y  # [m] (treated as center)

        # get platform equipment dimensions
        if design_scenario["electrolyzer_location"] == "turbine":
            desal_equipment_area = desal_results[
                "per_turb_equipment_footprint_m2"
            ]  # equipment_footprint_m2
        elif design_scenario["electrolyzer_location"] == "platform":
            desal_equipment_area = desal_results["equipment_footprint_m2"]
        else:
            desal_equipment_area = 0

        desal_equipment_side = np.sqrt(desal_equipment_area)

        # get pipe points
        pipe_x = np.array([substation_x - 1000, substation_x])
        pipe_y = np.array([substation_y, substation_y])

        # get cable points
        cable_x = pipe_x
        cable_y = pipe_y

    else:
        turbine_x = np.array(
            hopp_config["technologies"]["wind"]["floris_config"]["farm"]["layout_x"]
        )
        turbine_y = np.array(
            hopp_config["technologies"]["wind"]["floris_config"]["farm"]["layout_y"]
        )
        cable_array_points = []
    
    # wind farm area
    turbine_length_x = np.max(turbine_x)-np.min(turbine_x)
    turbine_length_y = np.max(turbine_y)-np.min(turbine_y)
    turbine_area = turbine_length_x * turbine_length_y

    # compressor side # not sized
    compressor_area = 25
    compressor_side = np.sqrt(compressor_area)
    ## create figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    # get turbine rotor diameter
    rotor_diameter = turbine_config["rotor_diameter"]  # in m
    rotor_radius = rotor_diameter / 2.0

    # set onshore substation dimensions
    onshore_substation_x_side_length = 127.25  # [m] based on 1 acre area https://www.power-technology.com/features/making-space-for-power-how-much-land-must-renewables-use/
    onshore_substation_y_side_length = 31.8  # [m] based on 1 acre area https://www.power-technology.com/features/making-space-for-power-how-much-land-must-renewables-use/
    onshore_substation_area = onshore_substation_x_side_length * onshore_substation_y_side_length

    if greenheart_config["h2_storage"]["type"] == "pressure_vessel":
        h2_storage_area = h2_storage_results["tank_footprint_m2"]
        h2_storage_side = np.sqrt(h2_storage_area)
    else:
        h2_storage_side = 0
        h2_storage_area = 0

    electrolyzer_area = electrolyzer_physics_results["equipment_footprint_m2"]
    if design_scenario["electrolyzer_location"] == "turbine":
        electrolyzer_area /= hopp_config["technologies"]["wind"]["num_turbines"]

    electrolyzer_side = np.sqrt(electrolyzer_area)

    # set onshore origin
    onshorex = 50
    onshorey = 50

    wind_buffer = np.min(turbine_x) - (
        onshorey + 2 * rotor_diameter + electrolyzer_side
    )
    if "pv" in hopp_config["technologies"].keys():
        wind_buffer -= np.sqrt(hopp_results["hybrid_plant"].pv.footprint_area)
    if "battery" in hopp_config["technologies"].keys():
        wind_buffer -= np.sqrt(hopp_results["hybrid_plant"].battery.footprint_area)
    if wind_buffer < 50:
        onshorey += wind_buffer - 50

    if design_scenario["wind_location"] == "offshore":
        origin_x = substation_x
        origin_y = substation_y
    else:
        origin_x = 0.0
        origin_y = 0.0

    ## create figure
    if design_scenario["wind_location"] == "offshore":
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax_index_plant = (0, 0)
        ax_index_detail = (1, 0)
        ax_index_wind_plant = (0, 1)
        ax_index_turbine_detail = (1, 1)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax_index_plant = 0
        ax_index_wind_plant = 0
        ax_index_detail = 1
        ax_index_turbine_detail = False

    # plot the stuff

    # onshore plant | offshore plant
    # platform/substation | turbine

    ## add turbines
    def add_turbines(ax, turbine_x, turbine_y, radius, color):
        i = 0
        for x, y in zip(turbine_x, turbine_y):
            if i == 0:
                rlabel = "Wind Turbine Rotor"
                tlabel = "Wind Turbine Tower"
                i += 1
            else:
                rlabel = None
                tlabel = None
            turbine_patch = patches.Circle(
                (x, y), radius=radius, color=color, fill=False, label=rlabel, zorder=10,
            )
            ax.add_patch(turbine_patch)

    add_turbines(
        ax[ax_index_wind_plant], turbine_x, turbine_y, rotor_radius, turbine_rotor_color
    )
    component_areas["turbine_area_m2"] = turbine_area
    # turbine_patch01_tower = patches.Circle((x, y), radius=tower_base_radius, color=turbine_tower_color, fill=False, label=tlabel, zorder=10)
    # ax[0, 1].add_patch(turbine_patch01_tower)
    if design_scenario["wind_location"] == "onshore":
        add_turbines(
            ax[ax_index_detail], turbine_x, turbine_y, rotor_radius, turbine_rotor_color
        )

    if ax_index_turbine_detail:
        # turbine_patch11_rotor = patches.Circle((turbine_x[0], turbine_y[0]), radius=rotor_radius, color=turbine_rotor_color, fill=False, label=None, zorder=10)
        tlabel = "Wind Turbine Tower"
        turbine_patch11_tower = patches.Circle(
            (turbine_x[0], turbine_y[0]),
            radius=tower_base_radius,
            color=turbine_tower_color,
            fill=False,
            label=tlabel,
            zorder=10,
        )
        # ax[1, 1].add_patch(turbine_patch11_rotor)
        ax[ax_index_turbine_detail].add_patch(turbine_patch11_tower)

    # add pipe array
    if design_scenario["transportation"] == "hvdc+pipeline" or (
        design_scenario["h2_storage_location"] != "turbine"
        and design_scenario["electrolyzer_location"] == "turbine"
    ):
        i = 0
        for point_string in pipe_array_points:
            if i == 0:
                label = "Array Pipes"
                i += 1
            else:
                label = None
            ax[0, 1].plot(
                point_string[:, 0],
                point_string[:, 1] - substation_side_length / 2,
                ":",
                color=pipe_color,
                zorder=0,
                linewidth=1,
                label=label,
            )
            ax[1, 0].plot(
                point_string[:, 0],
                point_string[:, 1] - substation_side_length / 2,
                ":",
                color=pipe_color,
                zorder=0,
                linewidth=1,
                label=label,
            )
            ax[1, 1].plot(
                point_string[:, 0],
                point_string[:, 1] - substation_side_length / 2,
                ":",
                color=pipe_color,
                zorder=0,
                linewidth=1,
                label=label,
            )

    ## add cables
    if (len(cable_array_points) > 1) and (
        design_scenario["h2_storage_location"] != "turbine"
        or design_scenario["transportation"] == "hvdc+pipeline"
    ):
        i = 0
        for point_string in cable_array_points:
            if i == 0:
                label = "Array Cables"
                i += 1
            else:
                label = None
            ax[0, 1].plot(
                point_string[:, 0],
                point_string[:, 1] + substation_side_length / 2,
                "-",
                color=cable_color,
                zorder=0,
                linewidth=1,
                label=label,
            )
            ax[1, 0].plot(
                point_string[:, 0],
                point_string[:, 1] + substation_side_length / 2,
                "-",
                color=cable_color,
                zorder=0,
                linewidth=1,
                label=label,
            )
            ax[1, 1].plot(
                point_string[:, 0],
                point_string[:, 1] + substation_side_length / 2,
                "-",
                color=cable_color,
                zorder=0,
                linewidth=1,
                label=label,
            )

    ## add offshore substation
    if design_scenario["wind_location"] == "offshore" and (
        design_scenario["h2_storage_location"] != "turbine"
        or design_scenario["transportation"] == "hvdc+pipeline"
    ):
        substation_patch01 = patches.Rectangle(
            (
                substation_x - substation_side_length,
                substation_y - substation_side_length / 2,
            ),
            substation_side_length,
            substation_side_length,
            fill=True,
            color=substation_color,
            label="Substation*",
            zorder=11,
        )
        substation_patch10 = patches.Rectangle(
            (
                substation_x - substation_side_length,
                substation_y - substation_side_length / 2,
            ),
            substation_side_length,
            substation_side_length,
            fill=True,
            color=substation_color,
            label="Substation*",
            zorder=11,
        )
        ax[0, 1].add_patch(substation_patch01)
        ax[1, 0].add_patch(substation_patch10)

        component_areas['offshore_substation_area_m2'] = substation_side_length ** 2

    ## add equipment platform
    if design_scenario["wind_location"] == "offshore" and (
        design_scenario["h2_storage_location"] == "platform"
        or design_scenario["electrolyzer_location"] == "platform"
    ):  # or design_scenario["transportation"] == "pipeline":
        equipment_platform_patch01 = patches.Rectangle(
            (
                equipment_platform_x - equipment_platform_side_length / 2,
                equipment_platform_y - equipment_platform_side_length / 2,
            ),
            equipment_platform_side_length,
            equipment_platform_side_length,
            color=equipment_platform_color,
            fill=True,
            label="Equipment Platform",
            zorder=1,
        )
        equipment_platform_patch10 = patches.Rectangle(
            (
                equipment_platform_x - equipment_platform_side_length / 2,
                equipment_platform_y - equipment_platform_side_length / 2,
            ),
            equipment_platform_side_length,
            equipment_platform_side_length,
            color=equipment_platform_color,
            fill=True,
            label="Equipment Platform",
            zorder=1,
        )
        ax[0, 1].add_patch(equipment_platform_patch01)
        ax[1, 0].add_patch(equipment_platform_patch10)

        component_areas['equipment_platform_area_m2'] = equipment_platform_area

    ## add hvdc cable
    if (
        design_scenario["transportation"] == "hvdc"
        or design_scenario["transportation"] == "hvdc+pipeline"
    ):
        ax[0, 0].plot(
            [onshorex + onshore_substation_x_side_length, 1000],
            [48, 48],
            "--",
            color=cable_color,
            label="HVDC Cable",
        )
        ax[0, 1].plot(
            [-5000, substation_x],
            [substation_y - 100, substation_y - 100],
            "--",
            color=cable_color,
            label="HVDC Cable",
            zorder=0,
        )
        ax[1, 0].plot(
            [-5000, substation_x],
            [substation_y - 2, substation_y - 2],
            "--",
            color=cable_color,
            label="HVDC Cable",
            zorder=0,
        )

    ## add onshore substation
    if (
        design_scenario["transportation"] == "hvdc"
        or design_scenario["transportation"] == "hvdc+pipeline"
    ):
        onshore_substation_patch00 = patches.Rectangle(
            (
                onshorex + 0.2 * onshore_substation_y_side_length,
                onshorey - onshore_substation_y_side_length * 1.2,
            ),
            onshore_substation_x_side_length,
            onshore_substation_y_side_length,
            fill=True,
            color=substation_color,
            label="Substation*",
            zorder=11,
        )
        ax[0, 0].add_patch(onshore_substation_patch00)

        component_areas['onshore_substation_area_m2'] = onshore_substation_area

    ## add transport pipeline
    if design_scenario["transportation"] == "colocated":
        # add hydrogen pipeline to end use
        linetype = "-."
        label = "Pipeline to Storage/End-Use"
        linewidth = 1.0

        ax[ax_index_plant].plot(
            [onshorex, -10000],
            [onshorey, onshorey],
            linetype,
            color=pipe_color,
            label=label,
            linewidth=linewidth,
            zorder=0,
        )

        ax[ax_index_detail].plot(
            [onshorex, -10000],
            [onshorey, onshorey],
            linetype,
            color=pipe_color,
            label=label,
            linewidth=linewidth,
            zorder=0,
        )
    if (
        design_scenario["transportation"] == "pipeline"
        or design_scenario["transportation"] == "hvdc+pipeline"
        or (
            design_scenario["transportation"] == "hvdc"
            and design_scenario["h2_storage_location"] == "platform"
        )
    ):
        linetype = "-."
        label = "Transport Pipeline"
        linewidth = 1.0

        ax[ax_index_plant].plot(
            [onshorex, 1000],
            [onshorey + 2, onshorey + 2],
            linetype,
            color=pipe_color,
            label=label,
            linewidth=linewidth,
            zorder=0,
        )

        if design_scenario["wind_location"] == "offshore":
            ax[ax_index_wind_plant].plot(
                [-5000, substation_x],
                [substation_y + 100, substation_y + 100],
                linetype,
                linewidth=linewidth,
                color=pipe_color,
                label=label,
                zorder=0,
            )
            ax[ax_index_detail].plot(
                [-5000, substation_x],
                [substation_y + 2, substation_y + 2],
                linetype,
                linewidth=linewidth,
                color=pipe_color,
                label=label,
                zorder=0,
            )

            if (
                design_scenario["transportation"] == "hvdc"
                or design_scenario["transportation"] == "hvdc+pipeline"
            ) and design_scenario["h2_storage_location"] == "platform":
                h2cx = onshorex - compressor_side
                h2cy = onshorey - compressor_side + 2
                h2cax = ax[ax_index_plant]
            else:
                h2cx = substation_x - substation_side_length
                h2cy = substation_y
                h2cax = ax[ax_index_detail]

        if design_scenario["wind_location"] == "onshore":
            compressor_patch01 = patches.Rectangle(
                (origin_x, origin_y),
                compressor_side,
                compressor_side,
                color=compressor_color,
                fill=None,
                label="Transport Compressor*",
                hatch="+++",
                zorder=20,
            )
            ax[ax_index_plant].add_patch(compressor_patch01)

        compressor_patch10 = patches.Rectangle(
            (h2cx, h2cy),
            compressor_side,
            compressor_side,
            color=compressor_color,
            fill=None,
            label="Transport Compressor*",
            hatch="+++",
            zorder=20,
        )
        h2cax.add_patch(compressor_patch10)

        component_areas['compressor_area_m2'] = compressor_area

    ## add plant components
    if design_scenario["electrolyzer_location"] == "onshore":
        electrolyzer_x = onshorex
        electrolyzer_y = onshorey
        electrolyzer_patch = patches.Rectangle(
            (electrolyzer_x, electrolyzer_y),
            electrolyzer_side,
            electrolyzer_side,
            color=electrolyzer_color,
            fill=None,
            label="Electrolyzer",
            zorder=20,
            hatch=electrolyzer_hatch,
        )
        ax[ax_index_plant].add_patch(electrolyzer_patch)
        component_areas['electrolyzer_area_m2'] = electrolyzer_area

        if design_scenario["wind_location"] == "onshore":
            electrolyzer_patch = patches.Rectangle(
                (onshorex - h2_storage_side, onshorey + 4),
                electrolyzer_side,
                electrolyzer_side,
                color=electrolyzer_color,
                fill=None,
                label="Electrolyzer",
                zorder=20,
                hatch=electrolyzer_hatch,
            )
            ax[ax_index_detail].add_patch(electrolyzer_patch)

    elif design_scenario["electrolyzer_location"] == "platform":
        dx = equipment_platform_x - equipment_platform_side_length / 2
        dy = equipment_platform_y - equipment_platform_side_length / 2
        e_side_y = equipment_platform_side_length
        e_side_x = electrolyzer_area / e_side_y
        d_side_y = equipment_platform_side_length
        d_side_x = desal_equipment_area / d_side_y
        electrolyzer_x = dx + d_side_x
        electrolyzer_y = dy

        electrolyzer_patch = patches.Rectangle(
            (electrolyzer_x, electrolyzer_y),
            e_side_x,
            e_side_y,
            color=electrolyzer_color,
            fill=None,
            zorder=20,
            label="Electrolyzer",
            hatch=electrolyzer_hatch,
        )
        ax[ax_index_detail].add_patch(electrolyzer_patch)
        desal_patch = patches.Rectangle(
            (dx, dy),
            d_side_x,
            d_side_y,
            color=desal_color,
            zorder=21,
            fill=None,
            label="Desalinator",
            hatch=desalinator_hatch,
        )
        ax[ax_index_detail].add_patch(desal_patch)
        component_areas['desalination_area_m2'] = desal_equipment_area

    elif design_scenario["electrolyzer_location"] == "turbine":
        electrolyzer_patch11 = patches.Rectangle(
            (turbine_x[0], turbine_y[0] + tower_base_radius),
            electrolyzer_side,
            electrolyzer_side,
            color=electrolyzer_color,
            fill=None,
            zorder=20,
            label="Electrolyzer",
            hatch=electrolyzer_hatch,
        )
        ax[ax_index_turbine_detail].add_patch(electrolyzer_patch11)
        desal_patch11 = patches.Rectangle(
            (turbine_x[0] - desal_equipment_side, turbine_y[0] + tower_base_radius),
            desal_equipment_side,
            desal_equipment_side,
            color=desal_color,
            zorder=21,
            fill=None,
            label="Desalinator",
            hatch=desalinator_hatch,
        )
        ax[ax_index_turbine_detail].add_patch(desal_patch11)
        component_areas['desalination_area_m2'] = desal_equipment_area
        i = 0
        for x, y in zip(turbine_x, turbine_y):
            if i == 0:
                elable = "Electrolyzer"
                dlabel = "Desalinator"
            else:
                elable = None
                dlabel = None
            electrolyzer_patch01 = patches.Rectangle(
                (x, y + tower_base_radius),
                electrolyzer_side,
                electrolyzer_side,
                color=electrolyzer_color,
                fill=None,
                zorder=20,
                label=elable,
                hatch=electrolyzer_hatch,
            )
            desal_patch01 = patches.Rectangle(
                (x - desal_equipment_side, y + tower_base_radius),
                desal_equipment_side,
                desal_equipment_side,
                color=desal_color,
                zorder=21,
                fill=None,
                label=dlabel,
                hatch=desalinator_hatch,
            )
            ax[ax_index_wind_plant].add_patch(electrolyzer_patch01)
            ax[ax_index_wind_plant].add_patch(desal_patch01)
            i += 1

    h2_storage_hatch = "\\\\\\"
    if design_scenario["h2_storage_location"] == "onshore" and (
        greenheart_config["h2_storage"]["type"] != "none"
    ):
        h2_storage_patch = patches.Rectangle(
            (onshorex - h2_storage_side, onshorey - h2_storage_side - 2),
            h2_storage_side,
            h2_storage_side,
            color=h2_storage_color,
            fill=None,
            label="H$_2$ Storage",
            hatch=h2_storage_hatch,
        )
        ax[ax_index_plant].add_patch(h2_storage_patch)
        component_areas["h2_storage_area_m2"] = h2_storage_area

        if design_scenario["wind_location"] == "onshore":
            h2_storage_patch = patches.Rectangle(
                (onshorex - h2_storage_side, onshorey - h2_storage_side - 2),
                h2_storage_side,
                h2_storage_side,
                color=h2_storage_color,
                fill=None,
                label="H$_2$ Storage",
                hatch=h2_storage_hatch,
            )
            ax[ax_index_detail].add_patch(h2_storage_patch)
            component_areas["h2_storage_area_m2"] = h2_storage_area
    elif design_scenario["h2_storage_location"] == "platform" and (
        greenheart_config["h2_storage"]["type"] != "none"
    ):
        s_side_y = equipment_platform_side_length
        s_side_x = h2_storage_area / s_side_y
        sx = equipment_platform_x - equipment_platform_side_length / 2
        sy = equipment_platform_y - equipment_platform_side_length / 2
        if design_scenario["electrolyzer_location"] == "platform":
            sx += equipment_platform_side_length - s_side_x

        h2_storage_patch = patches.Rectangle(
            (sx, sy),
            s_side_x,
            s_side_y,
            color=h2_storage_color,
            fill=None,
            label="H$_2$ Storage",
            hatch=h2_storage_hatch,
        )
        ax[ax_index_detail].add_patch(h2_storage_patch)
        component_areas["h2_storage_area_m2"] = h2_storage_area

    elif design_scenario["h2_storage_location"] == "turbine":

        if greenheart_config["h2_storage"]["type"] == "turbine":
            h2_storage_patch = patches.Circle(
                (turbine_x[0], turbine_y[0]),
                radius=tower_base_diameter / 2,
                color=h2_storage_color,
                fill=None,
                label="H$_2$ Storage",
                hatch=h2_storage_hatch,
            )
            ax[ax_index_turbine_detail].add_patch(h2_storage_patch)
            component_areas["h2_storage_area_m2"] = h2_storage_area
            i = 0
            for x, y in zip(turbine_x, turbine_y):
                if i == 0:
                    slable = "H$_2$ Storage"
                else:
                    slable = None
                h2_storage_patch = patches.Circle(
                    (x, y),
                    radius=tower_base_diameter / 2,
                    color=h2_storage_color,
                    fill=None,
                    label=None,
                    hatch=h2_storage_hatch,
                )
                ax[ax_index_wind_plant].add_patch(h2_storage_patch)
        elif greenheart_config["h2_storage"]["type"] == "pressure_vessel":
            h2_storage_side = np.sqrt(
                h2_storage_area / greenheart_config["plant"]["num_turbines"]
            )
            h2_storage_patch = patches.Rectangle(
                (
                    turbine_x[0] - h2_storage_side - desal_equipment_side,
                    turbine_y[0] + tower_base_radius,
                ),
                width=h2_storage_side,
                height=h2_storage_side,
                color=h2_storage_color,
                fill=None,
                label="H$_2$ Storage",
                hatch=h2_storage_hatch,
            )
            ax[ax_index_turbine_detail].add_patch(h2_storage_patch)
            component_areas["h2_storage_area_m2"] = h2_storage_area
            i = 0
            for x, y in zip(turbine_x, turbine_y):
                if i == 0:
                    slable = "H$_2$ Storage"
                else:
                    slable = None
                h2_storage_patch = patches.Rectangle(
                    (
                        turbine_x[i] - h2_storage_side - desal_equipment_side,
                        turbine_y[i] + tower_base_radius,
                    ),
                    width=h2_storage_side,
                    height=h2_storage_side,
                    color=h2_storage_color,
                    fill=None,
                    label=slable,
                    hatch=h2_storage_hatch,
                )
                ax[ax_index_wind_plant].add_patch(h2_storage_patch)
                i += 1

    ## add battery
    if "battery" in hopp_config["technologies"].keys():
        component_areas['battery_area_m2'] = hopp_results["hybrid_plant"].battery.footprint_area
        if design_scenario["battery_location"] == "onshore":
            battery_side_y = np.sqrt(
                hopp_results["hybrid_plant"].battery.footprint_area
            )
            battery_side_x = battery_side_y

            batteryx = electrolyzer_x

            batteryy = electrolyzer_y + electrolyzer_side + 10

            battery_patch = patches.Rectangle(
                (batteryx, batteryy),
                battery_side_x,
                battery_side_y,
                color=battery_color,
                fill=None,
                label="Battery Array",
                hatch=battery_hatch,
            )
            ax[ax_index_plant].add_patch(battery_patch)

            if design_scenario["wind_location"] == "onshore":

                battery_patch = patches.Rectangle(
                    (batteryx, batteryy),
                    battery_side_x,
                    battery_side_y,
                    color=battery_color,
                    fill=None,
                    label="Battery Array",
                    hatch=battery_hatch,
                )
                ax[ax_index_detail].add_patch(battery_patch)

        elif design_scenario["battery_location"] == "platform":
            battery_side_y = equipment_platform_side_length
            battery_side_x = (
                hopp_results["hybrid_plant"].battery.footprint_area / battery_side_y
            )

            batteryx = equipment_platform_x - equipment_platform_side_length / 2
            batteryy = equipment_platform_y - equipment_platform_side_length / 2

            battery_patch = patches.Rectangle(
                (batteryx, batteryy),
                battery_side_x,
                battery_side_y,
                color=battery_color,
                fill=None,
                label="Battery Array",
                hatch=battery_hatch,
            )
            ax[ax_index_detail].add_patch(battery_patch)   

    else:
        battery_side_y = 0.0
        battery_side_x = 0.0   
    
    ## add solar
    if hopp_config["site"]["solar"]:
        component_areas['pv_area_m2'] = hopp_results["hybrid_plant"].pv.footprint_area
        if design_scenario["pv_location"] == "offshore":
            solar_side_y = equipment_platform_side_length
            solar_side_x = hopp_results["hybrid_plant"].pv.footprint_area / solar_side_y

            solarx = equipment_platform_x - equipment_platform_side_length / 2
            solary = equipment_platform_y - equipment_platform_side_length / 2

            solar_patch = patches.Rectangle(
                (solarx, solary),
                solar_side_x,
                solar_side_y,
                color=solar_color,
                fill=None,
                label="Solar Array",
                hatch=solar_hatch,
            )
            ax[ax_index_detail].add_patch(solar_patch)
        else:
            solar_side_y = np.sqrt(hopp_results["hybrid_plant"].pv.footprint_area)
            solar_side_x = hopp_results["hybrid_plant"].pv.footprint_area / solar_side_y

            solarx = electrolyzer_x

            solary = electrolyzer_y + electrolyzer_side + 10

            if "battery" in hopp_config["technologies"].keys():
                solary += battery_side_y + 10

            solar_patch = patches.Rectangle(
                (solarx, solary),
                solar_side_x,
                solar_side_y,
                color=solar_color,
                fill=None,
                label="Solar Array",
                hatch=solar_hatch,
            )

            ax[ax_index_plant].add_patch(solar_patch)

            solar_patch = patches.Rectangle(
                (solarx, solary),
                solar_side_x,
                solar_side_y,
                color=solar_color,
                fill=None,
                label="Solar Array",
                hatch=solar_hatch,
            )

            ax[ax_index_detail].add_patch(solar_patch)
    else:
        solar_side_x = 0.0
        solar_side_y = 0.0

    ## add wave
    if hopp_config["site"]["wave"]:
        # get wave generation area geometry
        num_devices = hopp_config["technologies"]["wave"]["num_devices"]
        distance_to_shore = (
            hopp_config["technologies"]["wave"]["cost_inputs"]["distance_to_shore"]
            * 1e3
        )
        number_rows = hopp_config["technologies"]["wave"]["cost_inputs"]["number_rows"]
        device_spacing = hopp_config["technologies"]["wave"]["cost_inputs"][
            "device_spacing"
        ]
        row_spacing = hopp_config["technologies"]["wave"]["cost_inputs"]["row_spacing"]

        # calculate wave generation area dimenstions
        wave_side_y = device_spacing * np.ceil(num_devices / number_rows)
        wave_side_x = row_spacing * (number_rows)
        wave_area = wave_side_x * wave_side_y
        component_areas['wave_area_m2'] = wave_area

        # generate wave generation patch
        wavex = substation_x - wave_side_x
        wavey = substation_y + distance_to_shore
        wave_patch = patches.Rectangle(
            (wavex, wavey),
            wave_side_x,
            wave_side_y,
            color=wave_color,
            fill=None,
            label="Wave Array",
            hatch=wave_hatch,
            zorder=1,
        )
        ax[ax_index_wind_plant].add_patch(wave_patch)

        # add electrical transmission for wave
        wave_export_cable_coords_x = [substation_x, substation_x]
        wave_export_cable_coords_y = [substation_y, substation_y + distance_to_shore]

        ax[ax_index_wind_plant].plot(
            wave_export_cable_coords_x,
            wave_export_cable_coords_y,
            cable_color,
            zorder=0,
        )
        ax[ax_index_detail].plot(
            wave_export_cable_coords_x,
            wave_export_cable_coords_y,
            cable_color,
            zorder=0,
        )

    if design_scenario["wind_location"] == "offshore":
        allpoints = cable_array_points.flatten()
    else:
        allpoints = turbine_x

    allpoints = allpoints[~np.isnan(allpoints)]

    if design_scenario["wind_location"] == "offshore":
        roundto = -2
        ax[ax_index_plant].set(
            xlim=[
                round(np.min(onshorex - 100), ndigits=roundto),
                round(
                    np.max(
                        onshorex
                        + onshore_substation_x_side_length
                        + electrolyzer_side
                        + 200
                    ),
                    ndigits=roundto,
                ),
            ],
            ylim=[
                round(np.min(onshorey - 100), ndigits=roundto),
                round(
                    np.max(
                        onshorey
                        + battery_side_y
                        + electrolyzer_side
                        + solar_side_y
                        + 100
                    ),
                    ndigits=roundto,
                ),
            ],
        )
        ax[ax_index_plant].set(aspect="equal")
    else:
        roundto = -3
        ax[ax_index_plant].set(
            xlim=[
                round(np.min(allpoints - 6000), ndigits=roundto),
                round(np.max(allpoints + 6000), ndigits=roundto),
            ],
            ylim=[
                round(np.min(onshorey - 1000), ndigits=roundto),
                round(np.max(turbine_y + 4000), ndigits=roundto),
            ],
        )
        ax[ax_index_plant].autoscale()
        ax[ax_index_plant].set(aspect="equal")
        ax[ax_index_plant].xaxis.set_major_locator(ticker.MultipleLocator(2000))
        ax[ax_index_plant].yaxis.set_major_locator(ticker.MultipleLocator(1000))

    roundto = -3
    ax[ax_index_wind_plant].set(
        xlim=[
            round(np.min(allpoints - 6000), ndigits=roundto),
            round(np.max(allpoints + 6000), ndigits=roundto),
        ],
        ylim=[
            round((np.min([np.min(turbine_y), onshorey]) - 1000), ndigits=roundto),
            round(np.max(turbine_y + 4000), ndigits=roundto),
        ],
    )
    # ax[ax_index_wind_plant].autoscale()
    ax[ax_index_wind_plant].set(aspect="equal")
    ax[ax_index_wind_plant].xaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax[ax_index_wind_plant].yaxis.set_major_locator(ticker.MultipleLocator(1000))

    if design_scenario["wind_location"] == "offshore":
        roundto = -2
        ax[ax_index_detail].set(
            xlim=[
                round(origin_x - 400, ndigits=roundto),
                round(origin_x + 100, ndigits=roundto),
            ],
            ylim=[
                round(origin_y - 200, ndigits=roundto),
                round(origin_y + 200, ndigits=roundto),
            ],
        )
        ax[ax_index_detail].set(aspect="equal")
    else:
        roundto = -2

        if "pv" in hopp_config["technologies"].keys():
            xmax = round(
                np.max([onshorex + 510, solarx + solar_side_x + 100]), ndigits=roundto
            )
            ymax = round(solary + solar_side_y + 100, ndigits=roundto)
        else:
            xmax = round(np.max([onshorex + 510, 100]), ndigits=roundto)
            ymax = round(100, ndigits=roundto)
        ax[ax_index_detail].set(
            xlim=[round(onshorex - 10, ndigits=roundto), xmax,],
            ylim=[round(onshorey - 200, ndigits=roundto), ymax,],
        )
        ax[ax_index_detail].set(aspect="equal")

    if design_scenario["wind_location"] == "offshore":
        tower_buffer0 = 10
        tower_buffer1 = 10
        roundto = -1
        ax[ax_index_turbine_detail].set(
            xlim=[
                round(
                    turbine_x[0] - tower_base_radius - tower_buffer0 - 50,
                    ndigits=roundto,
                ),
                round(
                    turbine_x[0] + tower_base_radius + 3 * tower_buffer1,
                    ndigits=roundto,
                ),
            ],
            ylim=[
                round(
                    turbine_y[0] - tower_base_radius - 2 * tower_buffer0,
                    ndigits=roundto,
                ),
                round(
                    turbine_y[0] + tower_base_radius + 4 * tower_buffer1,
                    ndigits=roundto,
                ),
            ],
        )
        ax[ax_index_turbine_detail].set(aspect="equal")
        ax[ax_index_turbine_detail].xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax[ax_index_turbine_detail].yaxis.set_major_locator(ticker.MultipleLocator(10))
        # ax[0,1].legend(frameon=False)
        # ax[0,1].axis('off')

    if design_scenario["wind_location"] == "offshore":
        labels = [
            "(a) Onshore plant",
            "(b) Offshore plant",
            "(c) Equipment platform and substation",
            "(d) NW-most wind turbine",
        ]
    else:
        labels = ["(a) Full plant", "(b) Non-wind plant detail"]
    for axi, label in zip(ax.flatten(), labels):
        axi.legend(frameon=False, ncol=2)  # , ncol=2, loc="best")
        axi.set(xlabel="Easting (m)", ylabel="Northing (m)")
        axi.set_title(label, loc="left")
        # axi.spines[['right', 'top']].set_visible(False)

    ## save the plot
    plt.tight_layout()
    savepaths = [
            output_dir + "figures/layout/",
            output_dir + "data/",
        ]
    if save_plots:
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
        plt.savefig(
            savepaths[0] + "plant_layout_%i.png" % (plant_design_number), transparent=True
        )
        
        df = pd.DataFrame([component_areas])
        df.to_csv(savepaths[1] + "component_areas_layout_%i.csv" % (plant_design_number), index=False)

    if show_plots:
        plt.show()
    return 0


def save_energy_flows(
    hybrid_plant: HoppInterface.system, electrolyzer_physics_results, solver_results, hours, ax=None, simulation_length=8760, output_dir="./output/",
):

    

    if ax == None:
        fig, ax = plt.subplots(1)

    output = {}
    if hybrid_plant.pv:
        solar_plant_power = np.array(
            hybrid_plant.pv.generation_profile[0:simulation_length]
        )
        output.update({"pv generation [kW]": solar_plant_power})
    if hybrid_plant.wind:
        wind_plant_power = np.array(
            hybrid_plant.wind.generation_profile[0:simulation_length]
        )
        output.update({"wind generation [kW]": wind_plant_power})
    if hybrid_plant.wave:
        wave_plant_power = np.array(
            hybrid_plant.wave.generation_profile[0:simulation_length]
        )
        output.update({"wave generation [kW]": wave_plant_power})
    if hybrid_plant.battery:
        battery_power_out_mw = hybrid_plant.battery.outputs.P 
        output.update({"battery discharge [kW]": [(int(p>0))*p*1E3 for p in battery_power_out_mw]}) # convert from MW to kW and extract only discharging
        output.update({"battery charge [kW]": [-(int(p<0))*p*1E3 for p in battery_power_out_mw]}) # convert from MW to kW and extract only charging
        output.update({"battery state of charge [%]": hybrid_plant.battery.outputs.dispatch_SOC})

    output.update({"total renewable energy production hourly [kW]": [solver_results[0]]*simulation_length})
    output.update({"grid energy usage hourly [kW]": [solver_results[1]]*simulation_length})
    output.update({"desal energy hourly [kW]": [solver_results[2]]*simulation_length})
    output.update({"electrolyzer energy hourly [kW]": electrolyzer_physics_results["power_to_electrolyzer_kw"]})
    output.update({"transport compressor energy hourly [kW]": [solver_results[3]]*simulation_length})
    output.update({"storage energy hourly [kW]": [solver_results[4]]*simulation_length})
    output.update({"h2 production hourly [kg]": electrolyzer_physics_results["H2_Results"]["Hydrogen Hourly Production [kg/hr]"]})
    
    df = pd.DataFrame.from_dict(output)

    filepath = os.path.abspath(output_dir + "data/production/")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, "energy_flows.csv"))

    return output


# set up function to post-process HOPP results
def post_process_simulation(
    lcoe,
    lcoh,
    pf_lcoh,
    pf_lcoe,
    hopp_results,
    electrolyzer_physics_results,
    hopp_config,
    greenheart_config,
    orbit_config,
    turbine_config,
    h2_storage_results,
    capex_breakdown,
    opex_breakdown,
    wind_cost_results,
    platform_results,
    desal_results,
    design_scenario,
    plant_design_number,
    incentive_option,
    solver_results=[],
    show_plots=False,
    save_plots=False,
    verbose=False,
    output_dir="./output/",
):  # , lcoe, lcoh, lcoh_with_grid, lcoh_grid_only):
    # colors (official NREL color palette https://brand.nrel.gov/content/index/guid/color_palette?parent=61)
    colors = [
        "#0079C2",
        "#00A4E4",
        "#F7A11A",
        "#FFC423",
        "#5D9732",
        "#8CC63F",
        "#5E6A71",
        "#D1D5D8",
        "#933C06",
        "#D9531E",
    ]
    # load saved results

    # post process results
    if verbose:
        print("LCOE: ", round(lcoe * 1e3, 2), "$/MWh")
        print("LCOH: ", round(lcoh, 2), "$/kg")
        print(
            "hybrid electricity plant capacity factor: ",
            round(
                np.sum(hopp_results["combined_hybrid_power_production_hopp"])
                / (hopp_results["hybrid_plant"].system_capacity_kw.hybrid * 365 * 24),
                2,
            ),
        )
        print(
            "electrolyzer capacity factor: ",
            round(
                np.sum(electrolyzer_physics_results["power_to_electrolyzer_kw"])
                * 1e-3
                / (greenheart_config["electrolyzer"]["rating"] * 365 * 24),
                2,
            ),
        )
        print(
            "Electorlyzer CAPEX installed $/kW: ",
            round(
                capex_breakdown["electrolyzer"]
                / (greenheart_config["electrolyzer"]["rating"] * 1e3),
                2,
            ),
        )

    if show_plots or save_plots:
        visualize_plant(
            hopp_config,
            greenheart_config,
            turbine_config,
            wind_cost_results,
            hopp_results,
            platform_results,
            desal_results,
            h2_storage_results,
            electrolyzer_physics_results,
            design_scenario,
            colors,
            plant_design_number,
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir,
        )
    savepaths = [
        output_dir + "data/",
        output_dir + "data/lcoe/",
        output_dir + "data/lcoh/",
    ]
    for sp in savepaths:
        if not os.path.exists(sp):
            os.makedirs(sp)

    pf_lcoh.get_cost_breakdown().to_csv(
        savepaths[2]
        + "cost_breakdown_lcoh_design%i_incentive%i_%sstorage.csv"
        % (
            plant_design_number,
            incentive_option,
            greenheart_config["h2_storage"]["type"],
        )
    )
    pf_lcoe.get_cost_breakdown().to_csv(
        savepaths[1]
        + "cost_breakdown_lcoe_design%i_incentive%i_%sstorage.csv"
        % (
            plant_design_number,
            incentive_option,
            greenheart_config["h2_storage"]["type"],
        )
    )

    # create dataframe for saving all the stuff
    greenheart_config["design_scenario"] = design_scenario
    greenheart_config["plant_design_number"] = plant_design_number
    greenheart_config["incentive_options"] = incentive_option

    # save power usage data
    if len(solver_results) > 0:
        hours = len(hopp_results["combined_hybrid_power_production_hopp"])
        annual_energy_breakdown = {
            "electricity_generation_kwh": sum(
                hopp_results["combined_hybrid_power_production_hopp"]
            ),
            "electrolyzer_kwh": sum(
                electrolyzer_physics_results["power_to_electrolyzer_kw"]
            ),
            "renewable_kwh": solver_results[0] * hours,
            "grid_power_kwh": solver_results[1] * hours,
            "desal_kwh": solver_results[2] * hours,
            "h2_transport_compressor_power_kwh": solver_results[3] * hours,
            "h2_storage_power_kwh": solver_results[4] * hours,
        }

    ######################### save detailed ORBIT cost information
    if wind_cost_results.orbit_project:
        _, orbit_capex_breakdown, wind_capex_multiplier = adjust_orbit_costs(
            orbit_project=wind_cost_results.orbit_project,
            greenheart_config=greenheart_config,
        )

        # orbit_capex_breakdown["Onshore Substation"] = orbit_project.phases["ElectricalDesign"].onshore_cost
        # discount ORBIT cost information
        for key in orbit_capex_breakdown:
            orbit_capex_breakdown[key] = -npf.fv(
                greenheart_config["finance_parameters"]["costing_general_inflation"],
                greenheart_config["project_parameters"]["cost_year"]
                - greenheart_config["finance_parameters"]["discount_years"]["wind"],
                0.0,
                orbit_capex_breakdown[key],
            )

        # save ORBIT cost information
        ob_df = pd.DataFrame(orbit_capex_breakdown, index=[0]).transpose()
        savedir = output_dir + "data/orbit_costs/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        ob_df.to_csv(
            savedir
            + "orbit_cost_breakdown_lcoh_design%i_incentive%i_%sstorage.csv"
            % (
                plant_design_number,
                incentive_option,
                greenheart_config["h2_storage"]["type"],
            )
        )
        ###############################

        ###################### Save export system breakdown from ORBIT ###################

        _, orbit_capex_breakdown, wind_capex_multiplier = adjust_orbit_costs(
            orbit_project=wind_cost_results.orbit_project,
            greenheart_config=greenheart_config,
        )

        onshore_substation_costs = (
            wind_cost_results.orbit_project.phases["ElectricalDesign"].onshore_cost
            * wind_capex_multiplier
        )

        orbit_capex_breakdown["Export System Installation"] -= onshore_substation_costs

        orbit_capex_breakdown[
            "Onshore Substation and Installation"
        ] = onshore_substation_costs

        # discount ORBIT cost information
        for key in orbit_capex_breakdown:
            orbit_capex_breakdown[key] = -npf.fv(
                greenheart_config["finance_parameters"]["costing_general_inflation"],
                greenheart_config["project_parameters"]["cost_year"]
                - greenheart_config["finance_parameters"]["discount_years"]["wind"],
                0.0,
                orbit_capex_breakdown[key],
            )

        # save ORBIT cost information
        ob_df = pd.DataFrame(orbit_capex_breakdown, index=[0]).transpose()
        savedir = output_dir + "data/orbit_costs/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        ob_df.to_csv(
            savedir
            + "orbit_cost_breakdown_with_onshore_substation_lcoh_design%i_incentive%i_%sstorage.csv"
            % (
                plant_design_number,
                incentive_option,
                greenheart_config["h2_storage"]["type"],
            )
        )

    ##################################################################################
    if (
        hasattr(hopp_results["hybrid_plant"], "dispatch_builder")
        and hopp_results["hybrid_plant"].battery
    ):
        savedir = output_dir + "figures/production/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plot_tools.plot_generation_profile(
            hopp_results["hybrid_plant"],
            start_day=0,
            n_days=10,
            plot_filename=os.path.abspath(savedir + "generation_profile.pdf"),
            font_size=14,
            power_scale=1 / 1000,
            solar_color="r",
            wind_color="b",
            wave_color="g",
            discharge_color="b",
            charge_color="r",
            gen_color="g",
            price_color="r",
            show_price=False,
        )
    else:
        print(
            "generation profile not plotted because HoppInterface does not have a 'dispatch_builder'"
        )

    # save production information
    hourly_energy_breakdown = save_energy_flows(hopp_results["hybrid_plant"], electrolyzer_physics_results, solver_results, hours)
 
    # save hydrogen information
    key = "Hydrogen Hourly Production [kg/hr]"
    np.savetxt(output_dir+"h2_usage", electrolyzer_physics_results["H2_Results"][key], header="# "+key)

    return annual_energy_breakdown, hourly_energy_breakdown
