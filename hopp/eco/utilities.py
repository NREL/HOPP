import os
import os.path
import yaml

import numpy as np
import numpy_financial as npf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

import ORBIT as orbit

from hopp.resource import WindResource


# Function to load inputs
def get_inputs(
    filename_orbit_config,
    filename_turbine_config,
    wind_resource_file,
    filename_floris_config=None,
    verbose=False,
    show_plots=False,
    save_plots=False,
):
    ################ load plant inputs from yaml
    plant_config = orbit.load_config(filename_orbit_config)

    # print plant inputs if desired
    if verbose:
        print("\nPlant configuration:")
        for key in plant_config.keys():
            print(key, ": ", plant_config[key])

    ############### load turbine inputs from yaml

    # load general inputs
    with open(filename_turbine_config, "r") as stream:
        turbine_config = yaml.safe_load(stream)

    # load floris inputs
    if plant_config["wind"]["performance_model"] == "floris":  # TODO replace elements of the file
        assert (
            filename_floris_config is not None
        ), "floris input file must be specified."  # TODO: proper assertion
        with open(filename_floris_config, "r") as f:
            floris_config = yaml.load(f, yaml.FullLoader)
    else:
        floris_config = None

    # print turbine inputs if desired
    if verbose:
        print("\nTurbine configuration:")
        for key in turbine_config.keys():
            print(key, ": ", turbine_config[key])

    ############## provide custom layout for ORBIT and FLORIS if desired
    
    if plant_config["plant"]["layout"] == "custom": #Rob haymond contact person when custom is specified instead of grid
        # generate ORBIT config from floris layout
        for (i, x) in enumerate(floris_config["farm"]["layout_x"]):
            floris_config["farm"]["layout_x"][i] = x + 400
        
        layout_config, layout_data_location = convert_layout_from_floris_for_orbit(floris_config["farm"]["layout_x"], floris_config["farm"]["layout_y"], save_config=True)
        
        # update plant_config with custom layout
        # plant_config = orbit.core.library.extract_library_data(plant_config, additional_keys=layout_config)
        plant_config["array_system_design"]["location_data"] = layout_data_location

    ############## load wind resource
    if wind_resource_file=={}:
        wind_resource = WindResource(
            lat=plant_config["project_location"]["lat"],
            lon=plant_config["project_location"]["lon"],
            year=plant_config["wind_resource_year"],
            wind_turbine_hub_ht=turbine_config["hub_height"],
        )
    else:
        wind_resource = WindResource(
            lat=plant_config["project_location"]["lat"],
            lon=plant_config["project_location"]["lon"],
            year=plant_config["wind_resource_year"],
            wind_turbine_hub_ht=turbine_config["hub_height"],
            filepath=wind_resource_file
            )
        
    
    # adjust mean wind speed if desired
    wind_data = wind_resource._data['data']
    wind_speed = [W[2] for W in wind_data]
    if plant_config["site"]["mean_windspeed"]:
        if np.average(wind_speed) != plant_config["site"]["mean_windspeed"]:
            wind_speed += plant_config["site"]["mean_windspeed"] - np.average(wind_speed)
            for i in np.arange(0, len(wind_speed)):
                # make sure we don't have negative wind speeds after correction
                wind_resource._data['data'][i][2] = np.maximum(wind_speed[i], 0)
    else:
        plant_config["site"]["mean_windspeed"] = np.average(wind_speed)

    if show_plots or save_plots:
        # plot wind resource if desired
        print("\nPlotting Wind Resource")
        wind_speed = [W[2] for W in wind_resource._data["data"]]
        plt.figure(figsize=(9, 6))
        plt.plot(wind_speed)
        plt.title(
            "Wind Speed (m/s) for selected location \n {} \n Average Wind Speed (m/s) {}".format(
                "Gulf of Mexico", np.round(np.average(wind_speed), decimals=3)
            )
        )

        if show_plots:
            plt.show()
        if save_plots:
            savedir = "figures/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            plt.savefig("average_wind_speed.png", bbox_inches="tight")
    print("\n")

    ############## return all inputs

    return plant_config, turbine_config, wind_resource, floris_config

def convert_layout_from_floris_for_orbit(turbine_x, turbine_y, save_config=False):
    
    turbine_x_km = (np.array(turbine_x)*1E-3).tolist()
    turbine_y_km = (np.array(turbine_y)*1E-3).tolist()

    # initialize dict with data for turbines
    turbine_dict = {
                'id': list(range(0,len(turbine_x))),
                'substation_id': ['OSS']*len(turbine_x),
                'name': list(range(0,len(turbine_x))),
                'longitude': turbine_x_km,
                'latitude': turbine_y_km,
                'string': [0]*len(turbine_x), # can be left empty
                'order': [0]*len(turbine_x), # can be left empty
                'cable_length': [0]*len(turbine_x),
                'bury_speed': [0]*len(turbine_x)
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
                'id': 'OSS',
                'substation_id': 'OSS',
                'name': 'OSS',
                'longitude': np.min(turbine_x_km)-200*1e-3,
                'latitude': np.average(turbine_y_km),
                'string': "" , # can be left empty
                'order': "", # can be left empty
                'cable_length': "",
                'bury_speed': ""
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
        df.to_csv(save_location+file_name+".csv", index=False)

    return turbine_dict, file_name

def visualize_plant(
    plant_config,
    orbit_project,
    platform_results,
    desal_results,
    h2_storage_results,
    electrolyzer_physics_results,
    design_scenario,
    colors,
    plant_design_number,
    show_plots=False,
    save_plots=False,
):
    plt.rcParams.update({"font.size": 7})

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

    # Views
    # offshore plant, onshore plant, offshore platform, offshore turbine

    # get plant location

    # get shore location

    # get cable/pipe locations
    cable_array_points = (
        orbit_project.phases["ArraySystemDesign"].coordinates * 1e3
    )  # ORBIT gives coordinates in km, convert to m
    pipe_array_points = (
        orbit_project.phases["ArraySystemDesign"].coordinates * 1e3
    )  # ORBIT gives coordinates in km, convert to m

    # get turbine rotor diameter
    rotor_diameter = orbit_project.config["turbine"]["rotor_diameter"]  # in m
    rotor_radius = rotor_diameter / 2.0

    # get turbine tower base diameter
    tower_base_diameter = orbit_project.config["turbine"]["tower"]["section_diameters"][
        0
    ]  # in m
    tower_base_radius = tower_base_diameter / 2.0

    # get turbine locations
    turbine_x = (
        orbit_project.phases["ArraySystemDesign"].turbines_x.flatten() * 1e3
    )  # ORBIT gives coordinates in km, convert to m
    turbine_x = turbine_x[~np.isnan(turbine_x)]
    turbine_y = (
        orbit_project.phases["ArraySystemDesign"].turbines_y.flatten() * 1e3
    )  # ORBIT gives coordinates in km, convert to m
    turbine_y = turbine_y[~np.isnan(turbine_y)]

    # get offshore substation location and dimensions
    substation_x = (
        orbit_project.phases["ArraySystemDesign"].oss_x * 1e3
    )  # ORBIT gives coordinates in km, convert to m (treated as center)
    substation_y = (
        orbit_project.phases["ArraySystemDesign"].oss_y * 1e3
    )  # ORBIT gives coordinates in km, convert to m (treated as center)
    substation_side_length = 20  # [m] just based on a large substation (https://www.windpowerengineering.com/making-modern-offshore-substation/) since the dimensions are not available in ORBIT

    # set onshore substation dimensions
    onshore_substation_x_side_length = 127.25 # [m] based on 1 acre area https://www.power-technology.com/features/making-space-for-power-how-much-land-must-renewables-use/
    onshore_substation_y_side_length = 31.8 # [m] based on 1 acre area https://www.power-technology.com/features/making-space-for-power-how-much-land-must-renewables-use/

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

    if plant_config["h2_storage"]["type"] == "pressure_vessel":
        h2_storage_area = h2_storage_results["tank_footprint_m2"]
        h2_storage_side = np.sqrt(h2_storage_area)

    electrolyzer_area = electrolyzer_physics_results["equipment_footprint_m2"]
    if design_scenario["electrolyzer_location"] == "turbine":
        electrolyzer_area /= orbit_project.config["plant"]["num_turbines"]
    electrolyzer_side = np.sqrt(electrolyzer_area)

    # compressor side # not sized
    compressor_area = 25
    compressor_side = np.sqrt(compressor_area)

    # get pipe points
    pipe_x = np.array([substation_x - 1000, substation_x])
    pipe_y = np.array([substation_y, substation_y])

    # get cable points
    cable_x = pipe_x
    cable_y = pipe_y

    # set onshore origin
    onshorex = 50
    onshorey = 50

    # plot the stuff
    ## create figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    # onshore plant | offshore plant
    # platform/substation | turbine

    ## add turbines
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
            (x, y),
            radius=rotor_radius,
            color=turbine_rotor_color,
            fill=False,
            label=rlabel,
            zorder=10,
        )
        ax[0, 1].add_patch(turbine_patch)
        # turbine_patch01_tower = patches.Circle((x, y), radius=tower_base_radius, color=turbine_tower_color, fill=False, label=tlabel, zorder=10)
        # ax[0, 1].add_patch(turbine_patch01_tower)
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
    ax[1, 1].add_patch(turbine_patch11_tower)

    # add pipe array
    if (
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
    if design_scenario["h2_storage_location"] != "turbine":
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
    if design_scenario["h2_storage_location"] != "turbine":
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

    ## add equipment platform
    if (
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

    ## add hvdc cable
    if design_scenario["transportation"] == "hvdc":
        ax[0, 0].plot([onshorex+onshore_substation_x_side_length, 1000], [48, 48], "--", color=cable_color, label="HVDC Cable")
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
    if design_scenario["transportation"] == "hvdc":
        onshore_substation_patch00 = patches.Rectangle(
            (
                onshorex + 0.2*onshore_substation_y_side_length,
                onshorey - onshore_substation_y_side_length*1.2,
            ),
            onshore_substation_x_side_length,
            onshore_substation_y_side_length,
            fill=True,
            color=substation_color,
            label="Substation*",
            zorder=11,
        )
        ax[0, 0].add_patch(onshore_substation_patch00)

    ## add transport pipeline
    if design_scenario["transportation"] == "pipeline" or (
        design_scenario["transportation"] == "hvdc"
        and design_scenario["h2_storage_location"] == "platform"
    ):
        linetype = "-."
        label = "Transport Pipeline"
        linewidth = 1.0
        ax[0, 0].plot(
            [onshorex, 1000],
            [onshorey + 2, onshorey + 2],
            linetype,
            color=pipe_color,
            label=label,
            linewidth=linewidth,
            zorder=0,
        )
        ax[0, 1].plot(
            [-5000, substation_x],
            [substation_y + 100, substation_y + 100],
            linetype,
            linewidth=linewidth,
            color=pipe_color,
            label=label,
            zorder=0,
        )
        ax[1, 0].plot(
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
            and design_scenario["h2_storage_location"] == "platform"
        ):
            h2cx = onshorex - compressor_side
            h2cy = onshorey - compressor_side + 2
            h2cax = ax[0, 0]
        else:
            h2cx = substation_x - substation_side_length
            h2cy = substation_y
            h2cax = ax[1, 0]
        # compressor_patch01 = patches.Rectangle((substation_x-substation_side_length, substation_y), compressor_side, compressor_side, color=compressor_color, fill=None, label="Transport Compressor*", hatch="+++", zorder=20)
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
        # ax[0, 1].add_patch(compressor_patch01)
        h2cax.add_patch(compressor_patch10)

    ## add plant components
    ehatch = "///"
    dhatch = "xxxx"
    if design_scenario["electrolyzer_location"] == "onshore" and (
        plant_config["h2_storage"]["type"] != "none"
    ):
        electrolyzer_patch = patches.Rectangle(
            (onshorex - h2_storage_side, onshorey + 4),
            electrolyzer_side,
            electrolyzer_side,
            color=electrolyzer_color,
            fill=None,
            label="Electrolyzer",
            zorder=20,
            hatch=ehatch,
        )
        ax[0, 0].add_patch(electrolyzer_patch)
    elif (design_scenario["electrolyzer_location"] == "platform") and (
        plant_config["h2_storage"]["type"] != "none"
    ):
        dx = equipment_platform_x - equipment_platform_side_length / 2
        dy = equipment_platform_y - equipment_platform_side_length / 2
        e_side_y = equipment_platform_side_length
        e_side_x = electrolyzer_area / e_side_y
        d_side_y = equipment_platform_side_length
        d_side_x = desal_equipment_area / d_side_y
        ex = dx + d_side_x
        ey = dy

        electrolyzer_patch = patches.Rectangle(
            (ex, ey),
            e_side_x,
            e_side_y,
            color=electrolyzer_color,
            fill=None,
            zorder=20,
            label="Electrolyzer",
            hatch=ehatch,
        )
        ax[1, 0].add_patch(electrolyzer_patch)
        desal_patch = patches.Rectangle(
            (dx, dy),
            d_side_x,
            d_side_y,
            color=desal_color,
            zorder=21,
            fill=None,
            label="Desalinator",
            hatch=dhatch,
        )
        ax[1, 0].add_patch(desal_patch)
    elif (design_scenario["electrolyzer_location"] == "turbine") and (
        plant_config["h2_storage"]["type"] != "none"
    ):
        electrolyzer_patch11 = patches.Rectangle(
            (turbine_x[0], turbine_y[0] + tower_base_radius),
            electrolyzer_side,
            electrolyzer_side,
            color=electrolyzer_color,
            fill=None,
            zorder=20,
            label="Electrolyzer",
            hatch=ehatch,
        )
        ax[1, 1].add_patch(electrolyzer_patch11)
        desal_patch11 = patches.Rectangle(
            (turbine_x[0] - desal_equipment_side, turbine_y[0] + tower_base_radius),
            desal_equipment_side,
            desal_equipment_side,
            color=desal_color,
            zorder=21,
            fill=None,
            label="Desalinator",
            hatch=dhatch,
        )
        ax[1, 1].add_patch(desal_patch11)
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
                hatch=ehatch,
            )
            desal_patch01 = patches.Rectangle(
                (x - desal_equipment_side, y + tower_base_radius),
                desal_equipment_side,
                desal_equipment_side,
                color=desal_color,
                zorder=21,
                fill=None,
                label=dlabel,
                hatch=dhatch,
            )
            ax[0, 1].add_patch(electrolyzer_patch01)
            ax[0, 1].add_patch(desal_patch01)
            i += 1

    h2_storage_hatch = "\\\\\\"
    if design_scenario["h2_storage_location"] == "onshore" and (
        plant_config["h2_storage"]["type"] != "none"
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
        ax[0, 0].add_patch(h2_storage_patch)
    elif design_scenario["h2_storage_location"] == "platform" and (
        plant_config["h2_storage"]["type"] != "none"
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
        ax[1, 0].add_patch(h2_storage_patch)
    elif design_scenario["h2_storage_location"] == "turbine":
    
        if plant_config["h2_storage"]["type"] == "turbine":
            h2_storage_patch = patches.Circle(
                (turbine_x[0], turbine_y[0]),
                radius=tower_base_diameter / 2,
                color=h2_storage_color,
                fill=None,
                label="H$_2$ Storage",
                hatch=h2_storage_hatch,
            )
            ax[1, 1].add_patch(h2_storage_patch)
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
                ax[0, 1].add_patch(h2_storage_patch)
        elif plant_config["h2_storage"]["type"] == "pressure_vessel":
            h2_storage_side = np.sqrt(h2_storage_area/plant_config["plant"]["num_turbines"])
            h2_storage_patch = patches.Rectangle(
                (turbine_x[0] - h2_storage_side - desal_equipment_side, turbine_y[0] + tower_base_radius),
                width=h2_storage_side, height=h2_storage_side,
                color=h2_storage_color,
                fill=None,
                label="H$_2$ Storage",
                hatch=h2_storage_hatch,
            )
            ax[1, 1].add_patch(h2_storage_patch)
            i = 0
            for x, y in zip(turbine_x, turbine_y):
                if i == 0:
                    slable = "H$_2$ Storage"
                else:
                    slable = None
                h2_storage_patch = patches.Rectangle(
                    (turbine_x[i] - h2_storage_side - desal_equipment_side, turbine_y[i] + tower_base_radius),
                    width=h2_storage_side, height=h2_storage_side,
                    color=h2_storage_color,
                    fill=None,
                    label=slable,
                    hatch=h2_storage_hatch,
                )
                ax[0, 1].add_patch(h2_storage_patch)
                i += 1
    ax[0, 0].set(xlim=[0, 400], ylim=[0, 300])
    ax[0, 0].set(aspect="equal")

    allpoints = cable_array_points.flatten()
    allpoints = allpoints[~np.isnan(allpoints)]
    roundto = -3
    ax[0, 1].set(
        xlim=[
            round(np.min(allpoints - 2000), ndigits=roundto),
            round(np.max(allpoints + 2000), ndigits=roundto),
        ],
        ylim=[
            round(np.min(turbine_y - 1000), ndigits=roundto),
            round(np.max(turbine_y + 4000), ndigits=roundto),
        ],
    )
    ax[0, 1].set(aspect="equal")
    ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(1000))

    roundto = -2
    ax[1, 0].set(
        xlim=[
            round(substation_x - 400, ndigits=roundto),
            round(substation_x + 100, ndigits=roundto),
        ],
        ylim=[
            round(substation_y - 200, ndigits=roundto),
            round(substation_y + 200, ndigits=roundto),
        ],
    )
    ax[1, 0].set(aspect="equal")

    tower_buffer0 = 20
    tower_buffer1 = 30
    roundto = -1
    ax[1, 1].set(
        xlim=[
            round(
                turbine_x[0] - tower_base_radius - tower_buffer0 - 50, ndigits=roundto
            ),
            round(turbine_x[0] + tower_base_radius + tower_buffer1, ndigits=roundto),
        ],
        ylim=[
            round(turbine_y[0] - tower_base_radius - tower_buffer0, ndigits=roundto),
            round(turbine_y[0] + tower_base_radius + tower_buffer1, ndigits=roundto),
        ],
    )
    ax[1, 1].set(aspect="equal")
    ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax[0,1].legend(frameon=False)
    # ax[0,1].axis('off')

    labels = [
        "(a) Onshore plant",
        "(b) Offshore plant",
        "(c) Equipment platform and substation",
        "(d) NW-most wind turbine",
    ]
    for axi, label in zip(ax.flatten(), labels):
        axi.legend(frameon=False, ncol=2, loc="best")
        axi.set(xlabel="Easting (m)", ylabel="Northing (m)")
        axi.set_title(label, loc="left")

    ## save the plot
    plt.tight_layout()
    if save_plots:
        savepath = "figures/layout/"
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(
            savepath + "plant_layout_%i.png" % (plant_design_number), transparent=True
        )
    if show_plots:
        plt.show()
    return 0


# set up function to post-process HOPP results
def post_process_simulation(
    lcoe,
    lcoh,
    pf_lcoh,
    pf_lcoe,
    hopp_results,
    electrolyzer_physics_results,
    plant_config,
    h2_storage_results,
    capex_breakdown,
    opex_breakdown,
    orbit_project,
    platform_results,
    desal_results,
    design_scenario,
    plant_design_number,
    incentive_option,
    solver_results=[],
    show_plots=False,
    save_plots=False,
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
    print("LCOE: ", round(lcoe * 1e3, 2), "$/MWh")
    print("LCOH: ", round(lcoh, 2), "$/kg")
    print(
        "capacity factor: ",
        round(
            np.sum(hopp_results["combined_power_production_hopp"])
            * 1e-3
            / (plant_config["plant"]["capacity"] * 365 * 24),
            2,
        ),
    )
    print(
        "electrolyzer capacity factor: ",
        round(
            np.sum(electrolyzer_physics_results["energy_to_electrolyzer_kw"])
            * 1e-3
            / (plant_config["electrolyzer"]["rating"] * 365 * 24),
            2,
        ),
    )
    print(
        "Electorlyzer CAPEX installed $/kW: ",
        round(
            capex_breakdown["electrolyzer"]
            / (plant_config["electrolyzer"]["rating"] * 1e3),
            2,
        ),
    )

    if show_plots or save_plots:
        visualize_plant(
            plant_config,
            orbit_project,
            platform_results,
            desal_results,
            h2_storage_results,
            electrolyzer_physics_results,
            design_scenario,
            colors,
            plant_design_number,
            show_plots=show_plots,
            save_plots=save_plots,
        )
    savepaths = ["data", "data/lcoe", "data/lcoh"]
    for sp in savepaths:
        if not os.path.exists(sp):
            os.mkdir(sp)
    pf_lcoh.get_cost_breakdown().to_csv(
        "data/lcoh/cost_breakdown_lcoh_design%i_incentive%i_%sstorage.csv"
        % (plant_design_number, incentive_option, plant_config["h2_storage"]["type"])
    )
    pf_lcoe.get_cost_breakdown().to_csv(
        "data/lcoe/cost_breakdown_lcoe_design%i_incentive%i_%sstorage.csv"
        % (plant_design_number, incentive_option, plant_config["h2_storage"]["type"])
    )

    # create dataframe for saving all the stuff
    plant_config["design_scenario"] = design_scenario
    plant_config["plant_design_number"] = plant_design_number
    plant_config["incentive_options"] = incentive_option

    # save power usage data
    if len(solver_results) > 0:
        hours = len(hopp_results["combined_power_production_hopp"])
        annual_energy_breakdown = {
            "wind_kwh": sum(hopp_results["combined_power_production_hopp"]),
            "electrolyzer_kwh": sum(
                electrolyzer_physics_results["energy_to_electrolyzer_kw"]
            ),
            "renewable_kwh": solver_results[0] * hours,
            "grid_power_kwh": solver_results[1] * hours,
            "desal_kwh": solver_results[2] * hours,
            "h2_transport_compressor_power_kwh": solver_results[3] * hours,
            "h2_storage_power_kwh": solver_results[4] * hours,
        }

    ######################### save detailed ORBIT cost information
    orbit_capex_breakdown = orbit_project.capex_breakdown
    # orbit_capex_breakdown["Onshore Substation"] = orbit_project.phases["ElectricalDesign"].onshore_cost
    # discount ORBIT cost information
    for key in orbit_capex_breakdown:
        orbit_capex_breakdown[key] = -npf.fv(
            plant_config["finance_parameters"]["general_inflation"],
            plant_config["cost_year"]-plant_config["finance_parameters"]["discount_years"]["wind"],
            0.0,
            orbit_capex_breakdown[key],
        )

    # save ORBIT cost information
    ob_df = pd.DataFrame(orbit_capex_breakdown, index=[0]).transpose()
    savedir = "data/orbit_costs/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ob_df.to_csv(savedir+"orbit_cost_breakdown_lcoh_design%i_incentive%i_%sstorage.csv"
        % (plant_design_number, incentive_option, plant_config["h2_storage"]["type"]))
    ###############################

    ###################### Save export system breakdown from ORBIT ###################
    onshore_substation_costs = orbit_project.phases["ElectricalDesign"].onshore_cost

    orbit_capex_breakdown = orbit_project.capex_breakdown

    orbit_capex_breakdown["Export System Installation"] -= onshore_substation_costs

    orbit_capex_breakdown["Onshore Substation and Installation"] = onshore_substation_costs

    # discount ORBIT cost information
    for key in orbit_capex_breakdown:
        orbit_capex_breakdown[key] = -npf.fv(
            plant_config["finance_parameters"]["general_inflation"],
            plant_config["cost_year"]-plant_config["finance_parameters"]["discount_years"]["wind"],
            0.0,
            orbit_capex_breakdown[key],
        )

    # save ORBIT cost information
    ob_df = pd.DataFrame(orbit_capex_breakdown, index=[0]).transpose()
    savedir = "data/orbit_costs/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ob_df.to_csv(savedir+"orbit_cost_breakdown_with_onshore_substation_lcoh_design%i_incentive%i_%sstorage.csv"
        % (plant_design_number, incentive_option, plant_config["h2_storage"]["type"]))

    ##################################################################################

    return annual_energy_breakdown
