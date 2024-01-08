import copy 

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site

from greenheart.to_organize.H2_Analysis.hopp_for_h2_floris import (
    hopp_for_h2_floris as hopp_for_h2,
)
from hopp.simulation.hopp_interface import HoppInterface

# Funtion to set up the HOPP model
def setup_hopp(
    hopp_config, 
    eco_config,
    orbit_config,
    turbine_config,
    wind_resource,
    orbit_project,
    floris_config,
    show_plots=False,
    save_plots=False,
):
    # set desired schedule based on electrolyzer capacity
    desired_schedule = [eco_config["electrolyzer"]["rating"]] * 8760

    # generate HOPP SiteInfo class instance
    hopp_site = SiteInfo(
        hub_height=turbine_config["hub_height"],
        desired_schedule=desired_schedule,
        **{k: hopp_config["site"][k] for k in hopp_config["site"].keys() - ["hub_height", "desired_schedule", "follow_desired_schedule"]}
    )

    # replace wind data with previously downloaded and adjusted wind data
    hopp_site.wind_resource = wind_resource

    # update floris_config file with correct input from other files

    ################ set up HOPP technology inputs
    hopp_technologies = {}
    if hopp_config["site"]["wind"]:
        if hopp_config["technologies"]["wind"]["model_name"] == "floris":
            floris_config["farm"]["layout_x"] = (
                orbit_project.phases["ArraySystemDesign"].turbines_x.flatten() * 1e3
            )  # ORBIT gives coordinates in km
            floris_config["farm"]["layout_y"] = (
                orbit_project.phases["ArraySystemDesign"].turbines_y.flatten() * 1e3
            )  # ORBIT gives coordinates in km

            # remove things from turbine_config file that can't be used in FLORIS and set the turbine info in the floris config file
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

            hopp_technologies["wind"] = {
                "turbine_rating_kw": turbine_config["turbine_rating"] * 1000,
                "floris_config": floris_config,  # if not specified, use default SAM models
            }

        elif hopp_config["technologies"]["wind"]["model_name"] == "sam":
            hopp_technologies["wind"] = {
                    "turbine_rating_kw": turbine_config["turbine_rating"] * 1000,  # convert from MW to kW
                    "hub_height": turbine_config["hub_height"],
                    "rotor_diameter": turbine_config["rotor_diameter"],
                }
        else:
            raise(ValueError("Wind model '%s' not available. Please choose one of ['floris', 'sam']") % (hopp_config["technologies"]["wind"]["model_name"]))

        hopp_technologies["wind"].update({"num_turbines": orbit_config["plant"]["num_turbines"]})

        for key in hopp_technologies["wind"]:
            if key in hopp_config["technologies"]["wind"]:
                hopp_config["technologies"]["wind"][key] = hopp_technologies["wind"][key]
            else:
                hopp_config["technologies"]["wind"].update(hopp_technologies["wind"][key])

    hopp_technologies = hopp_config["technologies"]
    print("HOPP TECH KEYS: ", hopp_config["technologies"]["grid"].keys())
    ################ set up scenario dict input for hopp_for_h2()
    hopp_scenario = dict()
    hopp_scenario["Wind ITC"] = eco_config["policy_parameters"]["option1"]["electricity_itc"]
    hopp_scenario["Wind PTC"] = eco_config["policy_parameters"]["option1"]["electricity_ptc"]
    hopp_scenario["H2 PTC"] = eco_config["policy_parameters"]["option1"]["h2_ptc"]
    hopp_scenario["Useful Life"] = orbit_config["project_parameters"][
        "project_lifetime"
    ]
    hopp_scenario["Debt Equity"] = eco_config["finance_parameters"][
        "debt_equity_split"
    ]
    hopp_scenario["Discount Rate"] = eco_config["finance_parameters"]["discount_rate"]
    hopp_scenario["Tower Height"] = turbine_config["hub_height"]
    hopp_scenario["Powercurve File"] = turbine_config["turbine_type"] + ".csv"

    ############### prepare other HOPP for H2 inputs

    # get/set specific wind inputs
    wind_size_mw = (
        orbit_config["plant"]["num_turbines"] * turbine_config["turbine_rating"]
    )
    wind_om_cost_kw = orbit_config["project_parameters"]["opex_rate"]

    wind_cost_kw = (orbit_project.total_capex) / (
        wind_size_mw * 1e3
    )  # should be full plant installation and equipment costs etc minus the export costs

    custom_powercurve = False  # flag to use powercurve file provided in hopp_scenario?

    # get/set specific solar inputs
    solar_size_mw = 0.0
    solar_cost_kw = 0.0

    # get/set specific storage inputs
    storage_size_mw = 0.0
    storage_size_mwh = 0.0
    storage_hours = 0.0

    storage_cost_kw = 0.0
    storage_cost_kwh = 0.0

    # get/set specific electrolyzer inputs
    electrolyzer_size_mw = eco_config["electrolyzer"]["rating"]

    # get/set specific load and source inputs
    kw_continuous = electrolyzer_size_mw * 1e3
    load = [kw_continuous for x in range(0, 8760)]
    grid_connected_hopp = eco_config["project_parameters"]["grid_connection"]


    # add these specific inputs to a dictionary for transfer
    # TODO may need to add this for solar somehow
    # if "solar_om_cost_kw" in plant_config["project_parameters"]:
    #     solar_om_cost_kw = plant_config["project_parameters"]["solar_om_cost_kw"]
    # else:
    solar_om_cost_kw = 0.0

    hopp_h2_args = {
        "wind_size_mw": wind_size_mw,
        "wind_om_cost_kw": wind_om_cost_kw,
        "wind_cost_kw": wind_cost_kw,
        "custom_powercurve": custom_powercurve,
        "solar_size_mw": solar_size_mw,
        "solar_cost_kw": solar_cost_kw,
        "storage_size_mw": storage_size_mw,
        "storage_size_mwh": storage_size_mwh,
        "storage_hours": storage_hours,
        "storage_cost_kw": storage_cost_kw,
        "storage_cost_kwh": storage_cost_kwh,
        "electrolyzer_size": electrolyzer_size_mw,
        "kw_continuous": kw_continuous,
        "load": load,
        "grid_connected_hopp": grid_connected_hopp,
        "turbine_parent_path": "../../input/turbines/",
        "ppa_price": eco_config["project_parameters"]["ppa_price"],
        "solar_om_cost_kw": solar_om_cost_kw 
    }

    ################ return all the inputs for hopp
    return hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args


# Function to run hopp from provided inputs from setup_hopp()
def run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, wave_cost_dict={}, verbose=False):

    hopp_technologies_copy = copy.deepcopy(hopp_technologies)
    if wave_cost_dict == {} and 'wave' in hopp_technologies.keys():
        wave_cost_dict = hopp_technologies_copy["wave"].pop("cost_inputs")
        # wave_cost_dict = hopp_technologies["wave"]["cost_inputs"]
    # run hopp for H2
    (
        hybrid_plant,
        combined_hybrid_power_production_hopp,
        combined_pv_wind_curtailment_hopp,
        energy_shortfall_hopp,
        annual_energies,
        wind_plus_solar_npv,
        npvs,
        lcoe,
        lcoe_nom,
    ) = hopp_for_h2(
        hopp_site,
        hopp_scenario,
        hopp_technologies_copy,
        hopp_h2_args["wind_size_mw"],
        hopp_h2_args["solar_size_mw"],
        hopp_h2_args["storage_size_mw"],
        hopp_h2_args["storage_size_mwh"],
        hopp_h2_args["storage_hours"],
        hopp_h2_args["wind_cost_kw"],
        hopp_h2_args["solar_cost_kw"],
        hopp_h2_args["storage_cost_kw"],
        hopp_h2_args["storage_cost_kwh"],
        hopp_h2_args["kw_continuous"],
        hopp_h2_args["load"],
        hopp_h2_args["custom_powercurve"],
        hopp_h2_args["electrolyzer_size"],
        solar_om_cost_kw=hopp_h2_args["solar_om_cost_kw"],
        grid_connected_hopp=hopp_h2_args["grid_connected_hopp"],
        wind_om_cost_kw=hopp_h2_args["wind_om_cost_kw"],
        turbine_parent_path=hopp_h2_args["turbine_parent_path"],
        ppa_price=hopp_h2_args["ppa_price"],
        wave_cost_dict=wave_cost_dict
    )
    
    # store results for later use
    hopp_results = {
        "hybrid_plant": hybrid_plant,
        "combined_hybrid_power_production_hopp": combined_hybrid_power_production_hopp,
        "combined_pv_wind_curtailment_hopp": combined_pv_wind_curtailment_hopp,
        "energy_shortfall_hopp": energy_shortfall_hopp,
        "annual_energies": annual_energies,
        "wind_plus_solar_npv": wind_plus_solar_npv,
        "npvs": npvs,
        "lcoe": lcoe,
        "lcoe_nom": lcoe_nom,
    }
    if verbose:
        print("\nHOPP Results")
        print("Annual Energies: ", annual_energies)
        print(
            "combined power production: ", sum(combined_hybrid_power_production_hopp)
        )
        print("other ", hybrid_plant.wind.system_capacity_kw)
        print("Theoretical capacity: ", hopp_h2_args["wind_size_mw"] * 1e3 * 365 * 24)
        print(
            "Capacity factor: ",
            sum(combined_hybrid_power_production_hopp)
            / (hopp_h2_args["wind_size_mw"] * 1e3 * 365 * 24),
        )
        print("LCOE from HOPP: ", lcoe)

    return hopp_results
