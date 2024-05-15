import copy
import numpy as np
import matplotlib.pyplot as plt
import os


from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site
from hopp.simulation.hopp_interface import HoppInterface
from hopp.simulation.technologies.layout.wind_layout_tools import create_grid


# Function to set up the HOPP model
def setup_hopp(
    hopp_config,
    greenheart_config,
    orbit_config,
    turbine_config,
    floris_config,
    design_scenario,
    wind_cost_results=None,
    show_plots=False,
    save_plots=False,
    output_dir="./output/"
):
    
    if "battery" in hopp_config["technologies"].keys() and \
        ("desired_schedule" not in hopp_config["site"].keys() or hopp_config["site"]["desired_schedule"] == []):
        hopp_config["site"]["desired_schedule"] = [greenheart_config["electrolyzer"]["rating"]]*8760
    hopp_site = SiteInfo(**hopp_config["site"])

    # adjust mean wind speed if desired
    wind_data = hopp_site.wind_resource._data["data"]
    wind_speed = [W[2] for W in wind_data]
    if greenheart_config["site"]["mean_windspeed"]:
        if np.average(wind_speed) != greenheart_config["site"]["mean_windspeed"]:
            wind_speed += greenheart_config["site"]["mean_windspeed"] - np.average(
                wind_speed
            )
            for i in np.arange(0, len(wind_speed)):
                # make sure we don't have negative wind speeds after correction
                hopp_site.wind_resource._data["data"][i][2] = np.maximum(
                    wind_speed[i], 0
                )
    else:
        greenheart_config["site"]["mean_windspeed"] = np.average(wind_speed)

    ################ set up HOPP technology inputs
    
    if hopp_config["site"]["wind"]:
        if hopp_config["technologies"]["wind"]["model_name"] == "floris":
            if design_scenario["wind_location"] == "offshore":
                floris_config["farm"]["layout_x"] = (
                    wind_cost_results.orbit_project.phases[
                        "ArraySystemDesign"
                    ].turbines_x.flatten()
                    * 1e3
                )  # ORBIT gives coordinates in km
                # ORBIT produces nans and must be removed for FLORIS layout
                floris_config["farm"]["layout_x"] = floris_config["farm"]["layout_x"][
                    ~np.isnan(floris_config["farm"]["layout_x"])
                ]
                floris_config["farm"]["layout_y"] = (
                    wind_cost_results.orbit_project.phases[
                        "ArraySystemDesign"
                    ].turbines_y.flatten()
                    * 1e3
                )  # ORBIT gives coordinates in km
                # ORBIT produces nans and must be removed for FLORIS layout
                floris_config["farm"]["layout_y"] = floris_config["farm"]["layout_y"][
                    ~np.isnan(floris_config["farm"]["layout_y"])
                ]
                # remove things from turbine_config file that can't be used in FLORIS and set the turbine info in the floris config file

            if design_scenario["wind_location"] == "onshore":
                grid_position = create_grid(
                    site_shape=hopp_site.polygon,
                    center=hopp_site.polygon.centroid,
                    grid_angle=greenheart_config["site"]["wind_layout"]["grid_angle"],
                    intrarow_spacing=(
                        greenheart_config["site"]["wind_layout"]["row_spacing"]
                        * turbine_config["rotor_diameter"]
                    ),
                    interrow_spacing=(
                        greenheart_config["site"]["wind_layout"]["turbine_spacing"]
                        * turbine_config["rotor_diameter"]
                    ),
                    row_phase_offset=greenheart_config["site"]["wind_layout"][
                        "row_phase_offset"
                    ],
                    max_sites=hopp_config["technologies"]["wind"]["num_turbines"],
                )
                # Extracting xy coordinates
                xy_coordinates = [(point.x, point.y) for point in grid_position]
                floris_config["farm"]["layout_x"] = [point.x for point in grid_position]
                floris_config["farm"]["layout_y"] = [point.y for point in grid_position]

                if (
                    len(floris_config["farm"]["layout_x"])
                    < hopp_config["technologies"]["wind"]["num_turbines"]
                ):
                    raise Exception(
                        "size of site is too small, not all turbines were placed."
                    )

            floris_config["farm"]["turbine_type"] = [
                {
                    x: turbine_config[x]
                    for x in turbine_config
                    if x
                    not in {
                        "turbine_rating",
                        "rated_windspeed",
                        "tower",
                        "nacelle",
                        "blade",
                    }
                }
            ]

            hopp_config["technologies"]["wind"]["turbine_rating_kw"] = turbine_config["turbine_rating"] * 1000
            hopp_config["technologies"]["wind"]["floris_config"] = floris_config

        elif hopp_config["technologies"]["wind"]["model_name"] == "sam":
            hopp_config["technologies"]["wind"]["turbine_rating_kw"] = turbine_config["turbine_rating"] * 1000,  # convert from MW to kW
            hopp_config["technologies"]["wind"]["hub_height"] = turbine_config["hub_height"]
            hopp_config["technologies"]["wind"]["rotor_diameter"] = turbine_config["rotor_diameter"]

        else:
            raise (
                ValueError(
                    "Wind model '%s' not available. Please choose one of ['floris', 'sam']"
                )
                % (hopp_config["technologies"]["wind"]["model_name"])
            )

    # setup hopp interface
    hopp_config_internal = copy.deepcopy(hopp_config)

    if "wave" in hopp_config_internal["technologies"].keys():
        wave_cost_dict = hopp_config_internal["technologies"]["wave"].pop("cost_inputs")

    if "battery" in hopp_config_internal["technologies"].keys():
        hopp_config_internal["site"].update({"desired_schedule": hopp_site.desired_schedule})
        
    hi = HoppInterface(hopp_config_internal)
    hi.system.site = hopp_site

    if "wave" in hi.system.technologies.keys():
        hi.system.wave.create_mhk_cost_calculator(wave_cost_dict)
        
    if show_plots or save_plots:
        # plot wind resource if desired
        print("\nPlotting Wind Resource")
        wind_speed = [W[2] for W in hopp_site.wind_resource._data["data"]]
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
            savedir = output_dir + "figures/"
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(savedir + "average_wind_speed.png", bbox_inches="tight")
        print("\n")

    ################ return all the inputs for hopp
    return hi


# Function to run hopp from provided inputs from setup_hopp()
def run_hopp(hi, project_lifetime, verbose=False):

    hi.simulate(project_life=project_lifetime)

    # store results for later use
    hopp_results = {
        "hopp_interface": hi,
        "hybrid_plant": hi.system,
        "combined_hybrid_power_production_hopp": \
            hi.system.grid._system_model.Outputs.system_pre_interconnect_kwac[0:8760],
        "combined_hybrid_curtailment_hopp": hi.system.grid.generation_curtailed,
        "energy_shortfall_hopp": hi.system.grid.missed_load,
        "annual_energies": hi.system.annual_energies,
        "hybrid_npv": hi.system.net_present_values.hybrid,
        "npvs": hi.system.net_present_values,
        "lcoe": hi.system.lcoe_real,
        "lcoe_nom": hi.system.lcoe_nom,
    }
    if verbose:
        print("\nHOPP Results")
        print("Hybrid Annual Energy: ", hopp_results["annual_energies"])
        print("Capacity factors: ", hi.system.capacity_factors)
        print("Real LCOE from HOPP: ", hi.system.lcoe_real)

    return hopp_results
