from hopp.sites import SiteInfo
from hopp.sites import flatirons_site as sample_site

from examples.H2_Analysis.hopp_for_h2_floris import (
    hopp_for_h2_floris as hopp_for_h2,
)

# Funtion to set up the HOPP model
def setup_hopp(
    plant_config,
    turbine_config,
    wind_resource,
    orbit_project,
    floris_config,
    show_plots=False,
    save_plots=False,
):
    ################ set up HOPP site data structure
    # get inputs in correct format to generate HOPP site instance
    hopp_site_input_data = sample_site
    hopp_site_input_data["lat"] = plant_config["project_location"]["lat"]
    hopp_site_input_data["lon"] = plant_config["project_location"]["lon"]
    hopp_site_input_data["year"] = plant_config["wind_resource_year"]
    hopp_site_input_data["no_solar"] = not plant_config["project_parameters"]["solar"]

    # set desired schedule based on electrolyzer capacity
    desired_schedule = [plant_config["electrolyzer"]["rating"]] * 8760

    # generate HOPP SiteInfo class instance
    hopp_site = SiteInfo(
        hopp_site_input_data,
        hub_height=turbine_config["hub_height"],
        desired_schedule=desired_schedule,
    )

    # replace wind data with previously downloaded and adjusted wind data
    hopp_site.wind_resource = wind_resource

    # update floris_config file with correct input from other files

    ################ set up HOPP technology inputs
    hopp_technologies = {}
    if plant_config["wind"]["flag"]:
        if plant_config["wind"]["performance_model"] == "floris":
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
                "num_turbines": plant_config["plant"]["num_turbines"],
                "turbine_rating_kw": turbine_config["turbine_rating"] * 1000,
                "model_name": "floris",
                "timestep": [0, 8759],
                "floris_config": floris_config,  # if not specified, use default SAM models
                "skip_financial": True,
            }
        elif plant_config["wind"]["performance_model"] == "sam":
            hopp_technologies["wind"] = {
                    "num_turbines": plant_config["plant"]["num_turbines"],
                    "turbine_rating_kw": turbine_config["turbine_rating"]
                    * 1000,  # convert from MW to kW
                    "hub_height": turbine_config["hub_height"],
                    "rotor_diameter": turbine_config["rotor_diameter"],
                    "skip_financial": True,
                }
        else:
            raise(ValueError("Wind model '%s' not implemented. Please choose one of ['floris', 'sam']") % (plant_config["wind"]["performance_model"]))
    
    if plant_config["pv"]["flag"]:
        hopp_technologies["pv"] = {"system_capacity_kw": plant_config["pv"]["system_capacity_kw"]}
    if plant_config["battery"]["flag"]:
        hopp_technologies["battery"] = {
            "system_capacity_kwh": plant_config["battery"]["system_capacity_kwh"],
            "system_capacity_kw": plant_config["battery"]["system_capacity_kw"],
        }

    ################ set up scenario dict input for hopp_for_h2()
    hopp_scenario = dict()
    hopp_scenario["Wind ITC"] = plant_config["policy_parameters"]["option1"]["wind_itc"]
    hopp_scenario["Wind PTC"] = plant_config["policy_parameters"]["option1"]["wind_ptc"]
    hopp_scenario["H2 PTC"] = plant_config["policy_parameters"]["option1"]["h2_ptc"]
    hopp_scenario["Useful Life"] = plant_config["project_parameters"][
        "project_lifetime"
    ]
    hopp_scenario["Debt Equity"] = plant_config["finance_parameters"][
        "debt_equity_split"
    ]
    hopp_scenario["Discount Rate"] = plant_config["finance_parameters"]["discount_rate"]
    hopp_scenario["Tower Height"] = turbine_config["hub_height"]
    hopp_scenario["Powercurve File"] = turbine_config["turbine_type"] + ".csv"

    ############### prepare other HOPP for H2 inputs

    # get/set specific wind inputs
    wind_size_mw = (
        plant_config["plant"]["num_turbines"] * turbine_config["turbine_rating"]
    )
    wind_om_cost_kw = plant_config["project_parameters"]["opex_rate"]

    ## extract export cable costs from wind costs
    export_cable_equipment_cost = (
        orbit_project.capex_breakdown["Export System"]
        + orbit_project.capex_breakdown["Offshore Substation"]
    )
    export_cable_installation_cost = (
        orbit_project.capex_breakdown["Export System Installation"]
        + orbit_project.capex_breakdown["Offshore Substation Installation"]
    )
    total_export_cable_system_cost = (
        export_cable_equipment_cost + export_cable_installation_cost
    )
    # wind_cost_kw = (orbit_project.total_capex - total_export_cable_system_cost)/(wind_size_mw*1E3) # should be full plant installation and equipment costs etc minus the export costs
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
    electrolyzer_size_mw = plant_config["electrolyzer"]["rating"]

    # get/set specific load and source inputs
    kw_continuous = electrolyzer_size_mw * 1e3
    load = [kw_continuous for x in range(0, 8760)]
    grid_connected_hopp = plant_config["project_parameters"]["grid_connection"]

    # add these specific inputs to a dictionary for transfer
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
        "ppa_price": plant_config["project_parameters"]["ppa_price"],
    }

    ################ return all the inputs for hopp
    return hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args


# Function to run hopp from provided inputs from setup_hopp()
def run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=False):
    # run hopp for H2
    (
        hybrid_plant,
        combined_pv_wind_power_production_hopp,
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
        hopp_technologies,
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
        grid_connected_hopp=hopp_h2_args["grid_connected_hopp"],
        wind_om_cost_kw=hopp_h2_args["wind_om_cost_kw"],
        turbine_parent_path=hopp_h2_args["turbine_parent_path"],
        ppa_price=hopp_h2_args["ppa_price"],
    )

    # store results for later use
    hopp_results = {
        "hybrid_plant": hybrid_plant,
        "combined_pv_wind_power_production_hopp": combined_pv_wind_power_production_hopp,
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
            "combined power production: ", sum(combined_pv_wind_power_production_hopp)
        )
        print("other ", hybrid_plant.wind.system_capacity_kw)
        print("Theoretical capacity: ", hopp_h2_args["wind_size_mw"] * 1e3 * 365 * 24)
        print(
            "Capacity factor: ",
            sum(combined_pv_wind_power_production_hopp)
            / (hopp_h2_args["wind_size_mw"] * 1e3 * 365 * 24),
        )
        print("LCOE from HOPP: ", lcoe)

    return hopp_results
