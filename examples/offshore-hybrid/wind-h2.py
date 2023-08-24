# general imports
from distutils.command.config import config
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy_financial as npf
import sys
from scipy import optimize
from pprint import pprint

# yaml imports
import yaml
from yamlinclude import YamlIncludeConstructor
from pathlib import Path

PATH = Path(__file__).parent
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/floris/')
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/turbines/')

# visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

# packages needed for setting NREL API key
from hopp.utilities.keys import set_developer_nrel_gov_key, get_developer_nrel_gov_key

# ORBIT imports
from ORBIT import ProjectManager, load_config
from ORBIT.core.library import initialize_library
initialize_library(os.path.join(os.getcwd(), "./input/"))

# system financial model
import ProFAST

# HOPP imports
from hopp.simulation.technologies.resource.wind_resource import WindResource
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site

from hopp.simulation.technologies.hydrogen.h2_storage.pressure_vessel.compressed_gas_storage_model_20221021.Compressed_all import PressureVessel
from hopp.simulation.technologies.hydrogen.desal.desal_model import RO_desal
from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import Underground_Pipe_Storage
from hopp.simulation.technologies.hydrogen.h2_transport.h2_export_pipe import run_pipe_analysis
from hopp.simulation.technologies.hydrogen.h2_transport.h2_pipe_array import run_pipe_array_const_diam
from hopp.simulation.technologies.hydrogen.electrolysis.PEM_costs_Singlitico_model import PEMCostsSingliticoModel
from hopp.simulation.technologies.hydrogen.electrolysis.pem_mass_and_footprint import mass as run_electrolyzer_mass
from hopp.simulation.technologies.hydrogen.electrolysis.pem_mass_and_footprint import footprint as run_electrolyzer_footprint
from hopp.simulation.technologies.hydrogen.h2_storage.on_turbine.on_turbine_hydrogen_storage import PressurizedTower
from hopp.simulation.technologies.offshore.fixed_platform import install_platform, calc_platform_opex, calc_substructure_mass_and_cost
from hopp.simulation.technologies.hydrogen.h2_transport.h2_compression import Compressor

# OSW specific HOPP imports
# Jared: take all these models with a grain of salt and check them before trusting final result because they are built specifically for the OSW project and may not be sufficiently general
import hopp.tools.hopp_tools as hopp_tools
from hopp.to_organize.H2_Analysis.hopp_for_h2_floris import hopp_for_h2_floris as hopp_for_h2; use_floris = True
from hopp.simulation.technologies.hydrogen.electrolysis.H2_cost_model import basic_H2_cost_model


################ Set API key
global NREL_API_KEY
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env or with an env var

# Function to load inputs
def get_inputs(turbine_model="osw_18MW", verbose=False, show_plots=False, save_plots=False):

    ################ load plant inputs from yaml
    plant_config = load_config("./input/plant/orbit-config-"+turbine_model+".yaml")

    # print plant inputs if desired
    if verbose:
        print("\nPlant configuration:")
        for key in plant_config.keys():
            print(key, ": ", plant_config[key])

    ############### load turbine inputs from yaml

    # load general inputs
    if turbine_model == "osw_18MW":
        turbine_yaml_path = "./input/turbines/"+turbine_model+".yaml"
        with open(turbine_yaml_path, 'r') as stream:
            turbine_config = yaml.safe_load(stream)
    else:
        raise(ValueError("Turbine type not integrated"))

    # load floris inputs
    if use_floris: #TODO replace elements of the file
        floris_dir = "./input/floris/"
        floris_file = floris_dir + "floris_input_iea_18MW_osw" + ".yaml"
        with open(floris_file, 'r') as f:
            floris_config = yaml.load(f, yaml.FullLoader)
    else:
        floris_config = None

    # print turbine inputs if desired
    if verbose:
        print("\nTurbine configuration:")
        for key in turbine_config.keys():
            print(key, ": ", turbine_config[key])

    ############## load wind resource
    wind_resource = WindResource(lat=plant_config["project_location"]["lat"],
        lon=plant_config["project_location"]["lon"], year=plant_config["wind_resource_year"],
        wind_turbine_hub_ht=turbine_config["hub_height"])

    if show_plots or save_plots:
        # plot wind resource if desired
        print("\nPlotting Wind Resource")
        wind_speed = [W[2] for W in wind_resource._data['data']]
        plt.figure(figsize=(9,6))
        plt.plot(wind_speed)
        plt.title('Wind Speed (m/s) for selected location \n {} \n Average Wind Speed (m/s) {}'.format("Gulf of Mexico",np.round(np.average(wind_speed), decimals=3)))
        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig('figures/average_wind_speed.png',bbox_inches='tight')
    print("\n")

    ############## return all inputs

    return plant_config, turbine_config, wind_resource, floris_config

# Function to run orbit from provided inputs - this is just for wind costs
def run_orbit(plant_config, verbose=False, weather=None):

    # set up ORBIT
    project = ProjectManager(plant_config, weather=weather)

    # run ORBIT
    project.run(availability=plant_config["installation_availability"])

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
        print("Export cable installation CAPEX: %.2f M USD" %(project.phases["ExportCableInstallation"].installation_capex*1E-6))
        print("\n")

    return project

# Funtion to set up the HOPP model
def setup_hopp(plant_config, turbine_config, wind_resource, orbit_project, floris_config, show_plots=False, save_plots=False):

    ################ set up HOPP site data structure
    # get inputs in correct format to generate HOPP site instance
    hopp_site_input_data = sample_site
    hopp_site_input_data["lat"] = plant_config["project_location"]["lat"]
    hopp_site_input_data["lon"] = plant_config["project_location"]["lon"]
    hopp_site_input_data["year"] = plant_config["wind_resource_year"]
    solar = plant_config["project_parameters"]["solar"]

    # set desired schedule based on electrolyzer capacity
    desired_schedule = [plant_config["electrolyzer"]["rating"]]*8760
    desired_schedule = []

    # generate HOPP SiteInfo class instance
    hopp_site = SiteInfo(hopp_site_input_data, hub_height=turbine_config["hub_height"], desired_schedule=desired_schedule, solar=solar)

    # replace wind data with previously downloaded and adjusted wind data
    hopp_site.wind_resource = wind_resource

    # update floris_config file with correct input from other files

    ################ set up HOPP technology inputs
    if use_floris:

        floris_config["farm"]["layout_x"] = orbit_project.phases["ArraySystemDesign"].turbines_x.flatten()*1E3 # ORBIT gives coordinates in km
        floris_config["farm"]["layout_y"] = orbit_project.phases["ArraySystemDesign"].turbines_y.flatten()*1E3 # ORBIT gives coordinates in km

        # remove things from turbine_config file that can't be used in FLORIS and set the turbine info in the floris config file
        floris_config["farm"]["turbine_type"] = [{x: turbine_config[x] for x in turbine_config if x not in {'turbine_rating', 'rated_windspeed', 'tower', 'nacelle', 'blade'}}]

        hopp_technologies = {'wind': {
                                'num_turbines': plant_config["plant"]["num_turbines"],
                                'turbine_rating_kw': turbine_config["turbine_rating"]*1000,
                                'model_name': 'floris',
                                'timestep': [0,8759],
                                'floris_config': floris_config, # if not specified, use default SAM models
                                'skip_financial': True},
                            }
    else:
        hopp_technologies = {'wind':
                                {'num_turbines': plant_config["plant"]["num_turbines"],
                                'turbine_rating_kw': turbine_config["turbine_rating"]*1000, # convert from MW to kW
                                'hub_height': turbine_config["hub_height"],
                                'rotor_diameter': turbine_config["rotor_diameter"],
                                'skip_financial': True
                                }
                            }

    ################ set up scenario dict input for hopp_for_h2()
    hopp_scenario = dict()
    hopp_scenario['Wind ITC'] = plant_config["policy_parameters"]["option1"]["wind_itc"]
    hopp_scenario['Wind PTC'] = plant_config["policy_parameters"]["option1"]["wind_ptc"]
    hopp_scenario['H2 PTC'] = plant_config["policy_parameters"]["option1"]["h2_ptc"]
    hopp_scenario['Useful Life'] = plant_config["project_parameters"]["project_lifetime"]
    hopp_scenario['Debt Equity'] = plant_config["finance_parameters"]["debt_equity_split"]
    hopp_scenario['Discount Rate'] = plant_config["finance_parameters"]["discount_rate"]
    hopp_scenario['Tower Height'] = turbine_config["hub_height"]
    hopp_scenario['Powercurve File'] = turbine_config["turbine_type"]+".csv"

    ############### prepare other HOPP for H2 inputs

    # get/set specific wind inputs
    wind_size_mw = plant_config["plant"]["num_turbines"]*turbine_config["turbine_rating"]
    wind_om_cost_kw = plant_config["project_parameters"]["opex_rate"]

    ## extract export cable costs from wind costs
    export_cable_equipment_cost = orbit_project.capex_breakdown["Export System"] + orbit_project.capex_breakdown["Offshore Substation"]
    export_cable_installation_cost = orbit_project.capex_breakdown["Export System Installation"] + orbit_project.capex_breakdown["Offshore Substation Installation"]
    total_export_cable_system_cost = export_cable_equipment_cost + export_cable_installation_cost
    # wind_cost_kw = (orbit_project.total_capex - total_export_cable_system_cost)/(wind_size_mw*1E3) # should be full plant installation and equipment costs etc minus the export costs
    wind_cost_kw = (orbit_project.total_capex)/(wind_size_mw*1E3) # should be full plant installation and equipment costs etc minus the export costs

    custom_powercurve = False # flag to use powercurve file provided in hopp_scenario?

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
    kw_continuous = electrolyzer_size_mw*1E3
    load = [kw_continuous for x in range(0, 8760)]
    grid_connected_hopp = plant_config["project_parameters"]["grid_connection"]

    # add these specific inputs to a dictionary for transfer
    hopp_h2_args = {"wind_size_mw": wind_size_mw,
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
                    "ppa_price": plant_config["project_parameters"]["ppa_price"]}

    ################ return all the inputs for hopp
    return hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args

# Function to run hopp from provided inputs from setup_hopp()
def run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=False):
    # run hopp for H2
    hybrid_plant, \
    combined_pv_wind_power_production_hopp, \
    combined_pv_wind_curtailment_hopp, \
    energy_shortfall_hopp, \
    annual_energies, \
    wind_plus_solar_npv, \
    npvs, \
    lcoe, \
    lcoe_nom \
    = hopp_for_h2(hopp_site, hopp_scenario, hopp_technologies,
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
                    ppa_price=hopp_h2_args["ppa_price"])

    # store results for later use
    hopp_results = {"hybrid_plant": hybrid_plant,
                    "combined_pv_wind_power_production_hopp": combined_pv_wind_power_production_hopp,
                    "combined_pv_wind_curtailment_hopp": combined_pv_wind_curtailment_hopp,
                    "energy_shortfall_hopp": energy_shortfall_hopp,
                    "annual_energies": annual_energies,
                    "wind_plus_solar_npv": wind_plus_solar_npv,
                    "npvs": npvs,
                    "lcoe": lcoe,
                    "lcoe_nom": lcoe_nom}
    if verbose:
        print("\nHOPP Results")
        print("Annual Energies: ", annual_energies)
        print("combined power production: ", sum(combined_pv_wind_power_production_hopp))
        print("other ", hybrid_plant.wind.system_capacity_kw)
        print("Theoretical capacity: ", hopp_h2_args["wind_size_mw"]*1E3*365*24)
        print("Capacity factor: ", sum(combined_pv_wind_power_production_hopp)/(hopp_h2_args["wind_size_mw"]*1E3*365*24))
        print("LCOE from HOPP: ", lcoe)

    return hopp_results

def run_electrolyzer_physics(hopp_results, hopp_scenario, hopp_h2_args, plant_config, wind_resource, design_scenario, show_plots=False, save_plots=False,  verbose=False):
    # parse inputs to provide to hopp_tools call
    hybrid_plant = hopp_results["hybrid_plant"]

    if plant_config["project_parameters"]["grid_connection"]:
        # print(np.ones(365*24)*(hopp_h2_args["electrolyzer_size"]*1E3))
        energy_to_electrolyzer_kw = np.ones(365*24)*(hopp_h2_args["electrolyzer_size"]*1E3)
    else:
        energy_to_electrolyzer_kw = hopp_results["combined_pv_wind_power_production_hopp"]

    scenario = hopp_scenario
    wind_size_mw = hopp_h2_args["wind_size_mw"]
    solar_size_mw = hopp_h2_args["solar_size_mw"]
    electrolyzer_size_mw = hopp_h2_args["electrolyzer_size"]
    kw_continuous = hopp_h2_args["kw_continuous"]
    electrolyzer_capex_kw = plant_config["electrolyzer"]["electrolyzer_capex"]
    lcoe = hopp_results["lcoe"]

    # run electrolyzer model
    H2_Results, \
    _, \
    electrical_generation_timeseries \
        = hopp_tools.run_H2_PEM_sim(hybrid_plant,
                energy_to_electrolyzer_kw,
                scenario,
                wind_size_mw,
                solar_size_mw,
                electrolyzer_size_mw,
                kw_continuous,
                electrolyzer_capex_kw,
                lcoe)

    # calculate utilization rate
    energy_capacity = hopp_h2_args["electrolyzer_size"]*365*24 # MWh
    energy_available = sum(energy_to_electrolyzer_kw)*1E-3 # MWh
    capacity_factor_electrolyzer = energy_available/energy_capacity

    # calculate mass and foorprint of system
    mass_kg = run_electrolyzer_mass(electrolyzer_size_mw)
    footprint_m2 = run_electrolyzer_footprint(electrolyzer_size_mw)

    # store results for return
    electrolyzer_physics_results = {"H2_Results": H2_Results,
                            "electrical_generation_timeseries": electrical_generation_timeseries,
                            "capacity_factor": capacity_factor_electrolyzer,
                            "equipment_mass_kg": mass_kg,
                            "equipment_footprint_m2": footprint_m2,
                            "energy_to_electrolyzer_kw": energy_to_electrolyzer_kw}

    if verbose:
        print("\nElectrolyzer Physics:") #61837444.34555772 145297297.29729727
        print("H2 Produced Annually: ", H2_Results["hydrogen_annual_output"])
        print("Max H2 hourly (tonnes): ", max(H2_Results["hydrogen_hourly_production"])*1E-3)
        print("Max H2 daily (tonnes): ", max(np.convolve(H2_Results["hydrogen_hourly_production"], np.ones(24), mode='valid'))*1E-3)
        prodrate = 1.0/50.0 # kg/kWh
        roughest = electrical_generation_timeseries*prodrate
        print("Energy to electrolyzer (kWh): ", sum(energy_to_electrolyzer_kw))
        print("Energy per kg (kWh/kg): ", energy_available*1E3/H2_Results["hydrogen_annual_output"])
        print("Max hourly based on est kg/kWh (kg): ", max(roughest))
        print("Max daily rough est (tonnes): ", max(np.convolve(roughest, np.ones(24), mode='valid'))*1E-3)
        print("Capacity Factor Electrolyzer: ", H2_Results["cap_factor"])

    if save_plots or show_plots:

        N = 24*7*4
        fig, ax = plt.subplots(3,2, sharex=True, sharey="row")

        wind_speed = [W[2] for W in wind_resource._data['data']]

        # plt.title("4-week running average")
        pad = 5
        ax[0,0].annotate("Hourly", xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
        ax[0,1].annotate("4-week running average", xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

        ax[0,0].plot(wind_speed)
        ave_x = range(4*7*24-1, len(H2_Results["hydrogen_hourly_production"])+1)
        print(len(ave_x))
        ax[0,1].plot(ave_x, np.convolve(wind_speed, np.ones(N)/(N), mode='valid'))
        ax[0,0].set(ylabel="Wind\n(m/s)", ylim=[0,30], xlim=[0,len(wind_speed)])

        ax[1,0].plot(electrical_generation_timeseries*1E-3)
        ax[1,0].axhline(y = 400, color = 'r', linestyle = '--', label="Nameplate Capacity")
        ax[1,1].plot(ave_x[:-1], np.convolve(electrical_generation_timeseries*1E-3, np.ones(N)/(N), mode='valid'))
        ax[1,1].axhline(y = 400, color = 'r', linestyle = '--', label="Nameplate Capacity")
        ax[1,0].set(ylabel="Power\n(MW)", ylim=[0,500], xlim=[0,len(wind_speed)])
        # ax[1].legend(frameon=False, loc="best")

        ax[2,0].plot(H2_Results["hydrogen_hourly_production"])
        ax[2,1].plot(ave_x[:-1], np.convolve(H2_Results["hydrogen_hourly_production"], np.ones(N)/(N), mode='valid'))
        ax[2,0].set(xlabel="Hour", ylabel="Hydrogen\n(kg/hr)", ylim=[0,6000], xlim=[0,len(H2_Results["hydrogen_hourly_production"])])
        ax[2,1].set(xlabel="Hour", ylim=[0,6000], xlim=[4*7*24-1,len(H2_Results["hydrogen_hourly_production"]+4*7*24+2)])

        plt.tight_layout()
        if save_plots:
            plt.savefig("figures/production/production_overview_%i.png" %(design_scenario["id"]), transparent=True)
        if show_plots:
            plt.show()

    return electrolyzer_physics_results

def run_electrolyzer_cost(electrolyzer_physics_results, hopp_scenario, plant_config, design_scenario, verbose=False):

    # unpack inputs
    H2_Results = electrolyzer_physics_results["H2_Results"]
    electrolyzer_size_mw = plant_config["electrolyzer"]["rating"]
    useful_life = plant_config["project_parameters"]["project_lifetime"]
    atb_year = plant_config["atb_year"]
    electrical_generation_timeseries = electrolyzer_physics_results["electrical_generation_timeseries"]
    nturbines = plant_config["plant"]["num_turbines"]

    # run hydrogen production cost model - from hopp examples
    if design_scenario["h2_location"] == "onshore":
        offshore = 0
    else:
        offshore = 1

    if design_scenario["h2_location"] == "turbine":
        per_turb_electrolyzer_size_mw = electrolyzer_size_mw/nturbines
        per_turb_h2_annual_output = H2_Results['hydrogen_annual_output']/nturbines
        per_turb_electrical_generation_timeseries = electrical_generation_timeseries/nturbines

        cf_h2_annuals, per_turb_electrolyzer_total_capital_cost, per_turb_electrolyzer_OM_cost, per_turb_electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(plant_config["electrolyzer"]["electrolyzer_capex"], plant_config["electrolyzer"]["time_between_replacement"], per_turb_electrolyzer_size_mw, useful_life, atb_year,
            per_turb_electrical_generation_timeseries, per_turb_h2_annual_output, hopp_scenario['H2 PTC'], hopp_scenario['Wind ITC'], include_refurb_in_opex=False, offshore=offshore)

        electrolyzer_total_capital_cost = per_turb_electrolyzer_total_capital_cost*nturbines
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost*nturbines

    else:
        cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(plant_config["electrolyzer"]["electrolyzer_capex"], plant_config["electrolyzer"]["time_between_replacement"], electrolyzer_size_mw, useful_life, atb_year,
            electrical_generation_timeseries, H2_Results['hydrogen_annual_output'], hopp_scenario['H2 PTC'], hopp_scenario['Wind ITC'], include_refurb_in_opex=False, offshore=offshore)

    # package outputs for return
    electrolyzer_cost_results = {"electrolyzer_total_capital_cost": electrolyzer_total_capital_cost,
                                "electrolyzer_OM_cost_annual": electrolyzer_OM_cost}

    # print some results if desired
    if verbose:
        print("\nHydrogen Cost Results:")
        print("Electrolyzer Total CAPEX $/kW: ", electrolyzer_total_capital_cost/(electrolyzer_size_mw*1E3))
        print("Electrolyzer O&M $/kW: ", electrolyzer_OM_cost/(electrolyzer_size_mw*1E3))
        print("Electrolyzer O&M $/kg: ", electrolyzer_OM_cost/H2_Results['hydrogen_annual_output'])

    return electrolyzer_cost_results

def run_desal(plant_config, electrolyzer_physics_results, design_scenario, verbose=False):

    if verbose:
        print("\n")
        print("Desal Results")

    if design_scenario["h2_location"] == "onshore":
        desal_results = {"feed_water_flowrat_m3perhr": 0,
                        "desal_capex_usd": 0,
                        "desal_opex_usd_per_year": 0,
                        "power_for_desal_kw": 0,
                        "fresh_water_capacity_m3_per_hour": 0,
                        "equipment_mass_kg": 0,
                        "equipment_footprint_m2": 0
                        }
    else:
        freshwater_kg_per_hr = (electrolyzer_physics_results["H2_Results"]['water_annual_usage']/(365*24)) # convert from kg/yr to kg/hr

        if design_scenario["h2_location"] == "platform":

            desal_capacity_m3_per_hour, feedwater_m3_per_hr, desal_power, desal_capex, desal_opex, \
                desal_mass_kg, desal_size_m2 \
                = RO_desal(freshwater_kg_per_hr, salinity="Seawater")

            # package outputs
            desal_results = {"fresh_water_flowrate_m3perhr": desal_capacity_m3_per_hour,
                            "feed_water_flowrat_m3perhr": feedwater_m3_per_hr,
                            "desal_capex_usd": desal_capex,
                            "desal_opex_usd_per_year": desal_opex,
                            "power_for_desal_kw": desal_power,
                            "equipment_mass_kg": desal_mass_kg,
                            "equipment_footprint_m2": desal_size_m2
                            }

            if verbose:
                print("Fresh water needed (m^3/hr): ", desal_capacity_m3_per_hour)

        elif design_scenario["h2_location"] == "turbine":

            nturbines = plant_config["plant"]["num_turbines"]

            # size for per-turbine desal #TODO consider using individual power generation time series from each turbine
            in_turb_freshwater_kg_per_hr = freshwater_kg_per_hr/nturbines

            per_turb_desal_capacity_m3_per_hour, per_turb_feedwater_m3_per_hr, per_turb_desal_power, per_turb_desal_capex, per_turb_desal_opex, \
                per_turb_desal_mass_kg, per_turb_desal_size_m2 \
                = RO_desal(in_turb_freshwater_kg_per_hr, salinity="Seawater")

            fresh_water_flowrate = nturbines*per_turb_desal_capacity_m3_per_hour
            feed_water_flowrate = nturbines*per_turb_feedwater_m3_per_hr
            desal_capex = nturbines*per_turb_desal_capex
            desal_opex = nturbines*per_turb_desal_opex
            power_for_desal = nturbines*per_turb_desal_power

            # package outputs
            desal_results = {"fresh_water_flowrate_m3perhr": fresh_water_flowrate,
                            "feed_water_flowrat_m3perhr": feed_water_flowrate,
                            "desal_capex_usd": desal_capex,
                            "desal_opex_usd_per_year": desal_opex,
                            "power_for_desal_kw": power_for_desal,
                            "per_turb_equipment_mass_kg": per_turb_desal_mass_kg,
                            "per_turb_equipment_footprint_m2": per_turb_desal_size_m2
                            }

        if verbose:
            print("Fresh water needed (m^3/hr): ", desal_results["fresh_water_flowrate_m3perhr"])
            print("Requested fresh water (m^3/hr):", freshwater_kg_per_hr/997)


    if verbose:

        for key in desal_results.keys():
            print("Average", key, " ", np.average(desal_results[key]))
        print("\n")

    return desal_results

def run_h2_pipe_array(plant_config, orbit_project, electrolyzer_physics_results, design_scenario, verbose):

    if design_scenario["h2_location"] == "turbine" and design_scenario["h2_storage"] != "turbine":
        # get pipe lengths from ORBIT using cable lengths (horizontal only)
        pipe_lengths = orbit_project.phases["ArraySystemDesign"].sections_distance

        turbine_h2_flowrate = max(electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"])*((1./60.)**2)/plant_config["plant"]["num_turbines"]
        m_dot = np.ones_like(pipe_lengths)*turbine_h2_flowrate  # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
        p_inlet = 31            # Inlet pressure [bar] - assumed outlet pressure from electrolyzer model
        p_outlet = 10           # Outlet pressure [bar] - about 20 bar drop
        depth = plant_config["site"]["depth"] # depth of pipe [m]

        capex, opex = run_pipe_array_const_diam(pipe_lengths, depth, p_inlet, p_outlet, m_dot)

        h2_pipe_array_results = {"capex": capex, "opex": opex}
    else:
        h2_pipe_array_results = {"capex": 0.0, "opex": 0.0}

    return h2_pipe_array_results

def run_h2_transport_compressor(plant_config, electrolyzer_physics_results, design_scenario, verbose=False):

    if design_scenario["transportation"] == "pipeline" or (design_scenario["h2_storage"] != "onshore" and design_scenario["h2_location"] == "onshore"):

        ########## compressor model from Jamie Kee based on HDSAM
        flow_rate_kg_per_hr = max(electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"]) # kg/hr
        number_of_compressors = 2 # a third will be added as backup in the code
        p_inlet = 20 # bar
        p_outlet = plant_config["h2_transport_compressor"]["outlet_pressure"] # bar
        flow_rate_kg_d = flow_rate_kg_per_hr*24.0

        compressor = Compressor(p_outlet,flow_rate_kg_d, p_inlet=p_inlet, n_compressors=number_of_compressors)
        compressor.compressor_power()
        system_power_kw = compressor.compressor_system_power()
        total_capex,total_OM = compressor.compressor_costs() #2016$ , 2016$/y

        print(f'CAPEX: {round(total_capex,2)} $')
        print(f'Annual operating expense: {round(total_OM,2)} $/yr')

        h2_transport_compressor_results = {"compressor_power": system_power_kw,
                                "compressor_capex": total_capex,
                                "compressor_opex": total_OM}

    else:
        compressor = None
        h2_transport_compressor_results = {"compressor_power": 0.0,
                                "compressor_capex": 0.0,
                                "compressor_opex": 0.0}
        flow_rate_kg_per_hr = 0.0

    if verbose:
        print("\nCompressor Results:")
        print("Total H2 Flowrate (kg/hr): ", flow_rate_kg_per_hr)
        print("Compressor_power (kW): ", h2_transport_compressor_results['compressor_power'])
        print("Compressor capex [USD]: ", h2_transport_compressor_results['compressor_capex'])
        print("Compressor opex [USD/yr]: ", h2_transport_compressor_results['compressor_opex']) # annual

    return compressor, h2_transport_compressor_results

def run_h2_transport_pipe(plant_config, electrolyzer_physics_results, design_scenario, verbose=False):

    # prepare inputs
    export_pipe_length = plant_config["site"]["distance_to_landfall"]                  # Length [km]
    mass_flow_rate = max(electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"])*((1.0/60.0)**2)  # from [kg/hr] to mass flow rate in [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = plant_config["h2_transport_compressor"]["outlet_pressure"] # Inlet pressure [bar]
    p_outlet = plant_config["h2_transport_pipe"]["outlet_pressure"]           # Outlet pressure [bar]
    depth = plant_config["site"]["depth"]   # depth of pipe [m]

    # run model
    if (design_scenario["transportation"] == "pipeline") or (design_scenario["h2_storage"] != "onshore" and design_scenario["h2_location"] == "onshore"):
        h2_transport_pipe_results = run_pipe_analysis(export_pipe_length, mass_flow_rate, p_inlet, p_outlet, depth)
    else:
        h2_transport_pipe_results = pd.DataFrame.from_dict({"index": 0,
                                    "Grade": ["none"],
                                    "Outer diameter (mm)": [0*141.3],
                                    "Inner Diameter (mm)": [0*134.5],
                                    "Schedule": ["none"],
                                    "Thickness (mm)": [0*3.4],
                                    "volume [m3]": [0*30.969133941093407],
                                    "weight [kg]": [0*242798.01009817232],
                                    "mat cost [$]": [0*534155.6222159792],
                                    "labor cost [$]": [0*2974375.749734022],
                                    "misc cost [$]": [0*970181.8962542458],
                                    "ROW cost [$]": [0*954576.9166912301],
                                    "total capital cost [$]": [0*5433290.0184895478],
                                    'annual operating cost [$]': [0.0]})
    if verbose:
        print("\nH2 Transport Pipe Results")
        for col in h2_transport_pipe_results.columns:
            if col == "index": continue
            print(col, h2_transport_pipe_results[col][0])
        print("\n")


    return h2_transport_pipe_results

def run_h2_storage(plant_config, turbine_config, electrolyzer_physics_results, design_scenario, verbose=False):

    nturbines = plant_config["plant"]["num_turbines"]

    if design_scenario["h2_storage"] == "platform":
        if plant_config["h2_storage"]["type"] != "pressure_vessel" and plant_config["h2_storage"]["type"] != "none":
            raise ValueError("Only pressure vessel storage can be used on the off shore platform")

    # initialize output dictionary
    h2_storage_results = dict()

    storage_hours = plant_config["h2_storage"]["days"]*24
    storage_max_fill_rate = np.max(electrolyzer_physics_results["H2_Results"]["hydrogen_hourly_production"])

    ##################### get storage capacity from turbine storage model
    if plant_config["h2_storage"]["capacity_from_max_on_turbine_storage"] == True:
        turbine = {
                    'tower_length': turbine_config["tower"]["length"],
                    'section_diameters': turbine_config["tower"]["section_diameters"],
                    'section_heights': turbine_config["tower"]["section_heights"]
            }

        h2_storage = PressurizedTower(plant_config["atb_year"], turbine)
        h2_storage.run()

        h2_storage_capacity_single_turbine = h2_storage.get_capacity_H2() # kg

        h2_capacity = nturbines*h2_storage_capacity_single_turbine # in kg
    ###################################
    else:
        h2_capacity = round(storage_hours*storage_max_fill_rate)

    if plant_config["h2_storage"]["type"] == "none":
        h2_storage_results["h2_capacity"] = 0.0
    else:
        h2_storage_results["h2_capacity"] = h2_capacity

    # if storage_hours == 0:
    if plant_config["h2_storage"]["type"] == "none" or design_scenario["h2_storage"] == "none":

        h2_storage_results["storage_capex"] = 0.0
        h2_storage_results["storage_opex"] = 0.0
        h2_storage_results["storage_energy"] = 0.0

        h2_storage = None

    elif design_scenario["h2_storage"] == "turbine":

        turbine= {
                'tower_length': turbine_config["tower"]["length"],
                'section_diameters': turbine_config["tower"]["section_diameters"],
                'section_heights': turbine_config["tower"]["section_heights"]
        }

        h2_storage = PressurizedTower(plant_config["atb_year"], turbine)
        h2_storage.run()

        h2_storage_results["storage_capex"] = nturbines*h2_storage.get_capex()
        h2_storage_results["storage_opex"] = nturbines*h2_storage.get_opex()

        if verbose:
            print("On-turbine H2 storage:")
            print("mass empty (single turbine): ", h2_storage.get_mass_empty())
            print("H2 capacity (kg) - single turbine: ", h2_storage.get_capacity_H2())
            print("storage pressure: ", h2_storage.get_pressure_H2())

        h2_storage_results["storage_energy"] = 0.0 # low pressure, so no additional compression needed beyond electolyzer

    elif plant_config["h2_storage"]["type"] == "pipe":

        # for more information, see https://www.nrel.gov/docs/fy14osti/58564.pdf
        # initialize dictionary for pipe storage parameters
        storage_input = dict()

        # pull parameters from plat_config file
        storage_input['H2_storage_kg'] = h2_capacity
        storage_input['compressor_output_pressure'] = plant_config["h2_storage_compressor"]["output_pressure"]

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
        h2_storage_results["storage_energy"] = energy # total in kWh
        h2_storage_results["tank_mass_full_kg"] = h2_storage.get_tank_mass(h2_capacity)[1] + h2_capacity
        h2_storage_results["tank_footprint_m2"] = h2_storage.get_tank_footprint(h2_capacity, upright=True)[1]
        h2_storage_results["tank volume (m^3)"] = h2_storage.compressed_gas_function.Vtank
        h2_storage_results["Number of tanks"] = h2_storage.compressed_gas_function.number_of_tanks
        if verbose:
            print("ENERGY FOR STORAGE: ", energy*1E-3/(365*24), " MW")
            print("Tank volume (M^3): ", h2_storage_results["tank volume (m^3)"])
            print("Single Tank capacity (kg): ", h2_storage.compressed_gas_function.single_tank_h2_capacity_kg)
            print("N Tanks: ", h2_storage_results["Number of tanks"])

    elif plant_config["h2_storage"]["type"] == "salt_cavern":
        #TODO replace this rough estimate with real numbers
        h2_storage = None
        capex = 36.0*h2_capacity # based on Papadias 2021 table 7
        opex = 0.021*capex # based on https://www.pnnl.gov/sites/default/files/media/file/Hydrogen_Methodology.pdf

        h2_storage_results["storage_capex"] = capex
        h2_storage_results["storage_opex"] = opex
        h2_storage_results["storage_energy"] = 0.0
    else:
        raise(ValueError("H2 storage type %s was given, but must be one of ['none', 'pipe', 'pressure_vessel', 'salt_cavern]"))

    if verbose:
        print("\nH2 Storage Results:")
        print('H2 storage capex: ${0:,.0f}'.format(h2_storage_results['storage_capex']))
        print('H2 storage annual opex: ${0:,.0f}/yr'.format(h2_storage_results['storage_opex']))

    return h2_storage, h2_storage_results

def run_equipment_platform(plant_config, design_scenario, electrolyzer_physics_results, h2_storage_results, desal_results, verbose=False):

    topmass = 0.0 # tonnes
    toparea = 0.0 # m^2

    if design_scenario["h2_location"] == "platform" or design_scenario["h2_storage"] == "platform":


        """"equipment_mass_kg": desal_mass_kg,
                        "equipment_footprint_m2": desal_size_m2"""

        if design_scenario["h2_location"] == "platform":
            topmass += electrolyzer_physics_results["equipment_mass_kg"]*1E-3 # from kg to tonnes
            topmass += desal_results["equipment_mass_kg"]*1E-3 # from kg to tonnes
            toparea += electrolyzer_physics_results["equipment_footprint_m2"]
            toparea += desal_results["equipment_footprint_m2"]

        if design_scenario["h2_storage"] == "platform" and plant_config["h2_storage"]["type"] != "none":
            topmass += h2_storage_results["tank_mass_full_kg"]*1E-3 # from kg to tonnes
            toparea += h2_storage_results["tank_footprint_m2"]

        distance = plant_config["site"]["distance_to_landfall"]

        installation_cost = install_platform(topmass, toparea, distance, install_duration=plant_config["platform"]["installation_days"])

        depth = plant_config["site"]["depth"] # depth of pipe [m]

        capex, platform_mass = calc_substructure_mass_and_cost(topmass, toparea, depth)

        opex_rate = plant_config["platform"]["opex_rate"]
        total_opex = calc_platform_opex(capex, opex_rate)

        total_capex = capex + installation_cost

    else:
        platform_mass = 0.0
        total_capex = 0.0
        total_opex = 0.0

    platform_results = {"topmass_kg": topmass,
                        "toparea_m2": toparea,
                        "platform_mass_tonnes": platform_mass,
                        "capex": total_capex,
                        "opex": total_opex}
    if verbose:
        print("\nPlatform Results")
        for key in platform_results.keys():
            print(key, "%.2f" %(platform_results[key]))

    return platform_results

def run_capex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, design_scenario, desal_results, platform_results, verbose=False):

    total_wind_installed_costs_with_export = orbit_project.total_capex

    array_cable_equipment_cost = orbit_project.capex_breakdown["Array System"]
    array_cable_installation_cost = orbit_project.capex_breakdown["Array System Installation"]
    total_array_cable_system_capex = array_cable_equipment_cost + array_cable_installation_cost

    export_cable_equipment_cost = orbit_project.capex_breakdown["Export System"]
    export_cable_installation_cost = orbit_project.capex_breakdown["Export System Installation"]
    total_export_cable_system_capex = export_cable_equipment_cost + export_cable_installation_cost

    substation_equipment_cost = orbit_project.capex_breakdown["Offshore Substation"]
    substation_installation_cost = orbit_project.capex_breakdown["Offshore Substation Installation"]
    total_substation_capex = substation_equipment_cost + substation_installation_cost

    total_electrical_export_system_cost = total_array_cable_system_capex + total_substation_capex + total_export_cable_system_capex

    ## desal capex
    if desal_results != None:
        desal_capex = desal_results["desal_capex_usd"]
    else:
        desal_capex = 0.0

    ## electrolyzer capex
    electrolyzer_total_capital_cost = electrolyzer_cost_results["electrolyzer_total_capital_cost"]

    ## adjust wind cost to remove export
    if design_scenario["h2_location"] == "turbine" and design_scenario["h2_storage"] == "turbine":
        unused_export_system_cost = total_array_cable_system_capex + total_export_cable_system_capex + total_substation_capex
    elif design_scenario["h2_location"] == "turbine" and design_scenario["h2_storage"] == "platform":
        unused_export_system_cost = total_export_cable_system_capex #TODO check assumptions here
    elif design_scenario["h2_location"] == "platform" and design_scenario["h2_storage"] == "platform":
        unused_export_system_cost = total_export_cable_system_capex #TODO check assumptions here
    elif design_scenario["h2_location"] == "platform" and design_scenario["h2_storage"] == "onshore":
        unused_export_system_cost = total_export_cable_system_capex #TODO check assumptions here
    else:
        unused_export_system_cost = 0.0

    total_used_export_system_costs = total_electrical_export_system_cost - unused_export_system_cost

    total_wind_cost_no_export = total_wind_installed_costs_with_export - total_electrical_export_system_cost

    if design_scenario["h2_location"] == "platform" or design_scenario["h2_storage"] == "platform" :
        platform_costs = platform_results["capex"]
    else:
        platform_costs = 0.0

    # h2 transport
    h2_transport_compressor_capex = h2_transport_compressor_results["compressor_capex"]
    h2_transport_pipe_capex = h2_transport_pipe_results["total capital cost [$]"][0]

    ## h2 storage
    if plant_config["h2_storage"]["type"] == "none":
        h2_storage_capex = 0.0
    elif plant_config["h2_storage"]["type"] == "pipe": # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif plant_config["h2_storage"]["type"] == "pressure_vessel": # pressure vessel storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif plant_config["h2_storage"]["type"] == "salt_cavern":  # salt cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]

    # store opex component breakdown
    capex_breakdown = {"wind": total_wind_cost_no_export,
                    #   "cable_array": total_array_cable_system_capex,
                    #   "substation": total_substation_capex,
                      "platform": platform_costs,
                      "electrical_export_system": total_used_export_system_costs,
                      "desal": desal_capex,
                      "electrolyzer": electrolyzer_total_capital_cost,
                      "h2_pipe_array": h2_pipe_array_results["capex"],
                      "h2_transport_compressor": h2_transport_compressor_capex,
                      "h2_transport_pipeline": h2_transport_pipe_capex,
                      "h2_storage": h2_storage_capex}

    # discount capex to appropriate year for unified costing
    for key in capex_breakdown.keys():

        if key == "h2_storage":
            if design_scenario["h2_storage"] == "turbine":
                cost_year = plant_config["finance_parameters"]["discount_years"][key][design_scenario["h2_storage"]]
            else:
                cost_year = plant_config["finance_parameters"]["discount_years"][key][plant_config["h2_storage"]["type"]]
        else:
            cost_year = plant_config["finance_parameters"]["discount_years"][key]

        periods = plant_config["cost_year"] - cost_year
        capex_base = capex_breakdown[key]
        capex_breakdown[key] = -npf.fv(plant_config["finance_parameters"]["general_inflation"], periods, 0.0, capex_breakdown[key])

    total_system_installed_cost = sum(capex_breakdown[key] for key in capex_breakdown.keys())

    if verbose:
        print("\nCAPEX Breakdown")
        for key in capex_breakdown.keys():
            print(key, "%.2f" %(capex_breakdown[key]*1E-6), " M")

        print("\nTotal system CAPEX: ", "$%.2f" % (total_system_installed_cost*1E-9), " B")

    return total_system_installed_cost, capex_breakdown

def run_opex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, desal_results, platform_results, verbose=False, total_export_system_cost=0):

    # WIND ONLY Total O&M expenses including fixed, variable, and capacity-based, $/year
    annual_operating_cost_wind = max(orbit_project.monthly_opex.values())*12 #np.average(hopp_results["hybrid_plant"].wind.om_total_expense)

    # H2 OPEX
    platform_operating_costs = platform_results["opex"] #TODO update this

    annual_operating_cost_h2 = electrolyzer_cost_results["electrolyzer_OM_cost_annual"]

    h2_transport_compressor_opex = h2_transport_compressor_results['compressor_opex'] # annual

    h2_transport_pipeline_opex = h2_transport_pipe_results['annual operating cost [$]'][0] # annual

    storage_opex = h2_storage_results["storage_opex"]
    # desal OPEX
    if desal_results != None:
        desal_opex = desal_results["desal_opex_usd_per_year"]
    else:
        desal_opex = 0.0
    annual_operating_cost_desal = desal_opex

    # store opex component breakdown
    opex_breakdown_annual = {"wind_and_electrical": annual_operating_cost_wind,
                      "platform": platform_operating_costs,
                    #   "electrical_export_system": total_export_om_cost,
                      "desal": annual_operating_cost_desal,
                      "electrolyzer": annual_operating_cost_h2,
                      "h2_pipe_array": h2_pipe_array_results["opex"],
                      "h2_transport_compressor": h2_transport_compressor_opex,
                      "h2_transport_pipeline": h2_transport_pipeline_opex,
                      "h2_storage": storage_opex}

    # discount opex to appropriate year for unified costing
    for key in opex_breakdown_annual.keys():

        if key == "h2_storage":
            cost_year = plant_config["finance_parameters"]["discount_years"][key][plant_config["h2_storage"]["type"]]
        else:
            cost_year = plant_config["finance_parameters"]["discount_years"][key]

        periods = plant_config["cost_year"] - cost_year
        opex_breakdown_annual[key] = -npf.fv(plant_config["finance_parameters"]["general_inflation"], periods, 0.0, opex_breakdown_annual[key])

    # Calculate the total annual OPEX of the installed system
    total_annual_operating_costs = sum(opex_breakdown_annual.values())

    if verbose:
        print("\nAnnual OPEX Breakdown")
        for key in opex_breakdown_annual.keys():
            print(key, "%.2f" %(opex_breakdown_annual[key]*1E-6), " M")

        print("\nTotal Annual OPEX: ", "$%.2f" % (total_annual_operating_costs*1E-6), " M")
        print(opex_breakdown_annual)
    return total_annual_operating_costs, opex_breakdown_annual

def run_profast_lcoe(plant_config, orbit_project, capex_breakdown, opex_breakdown, hopp_results, design_scenario, verbose=False, show_plots=False, save_plots=False):
    gen_inflation = plant_config["finance_parameters"]["general_inflation"]

    if design_scenario["h2_storage"] == "onshore" or design_scenario["h2_location"] == "onshore":
        land_cost = 1E6 #TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST('blank')
    pf.set_params('commodity', {"name":'electricity',"unit":"kWh","initial price":100,"escalation": gen_inflation})
    pf.set_params('capacity', hopp_results["annual_energies"]["wind"]/365.0) #kWh/day
    pf.set_params('maintenance',{"value":0,"escalation": gen_inflation})
    pf.set_params('analysis start year', plant_config["atb_year"]+1)
    pf.set_params('operating life', plant_config["project_parameters"]["project_lifetime"])
    pf.set_params('installation months', (orbit_project.installation_time/(365*24))*(12.0/1.0))
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets', land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_config["project_parameters"]["project_lifetime"])
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',1)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax', plant_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent', plant_config["finance_parameters"]["property_tax"] + plant_config["finance_parameters"]["property_insurance"])
    pf.set_params('admin expense percent', plant_config["finance_parameters"]["administrative_expense_percent_of_sales"])
    pf.set_params('total income tax rate', plant_config["finance_parameters"]["total_income_tax_rate"])
    pf.set_params('capital gains tax rate', plant_config["finance_parameters"]["capital_gains_tax_rate"])
    pf.set_params('sell undepreciated cap', True)
    pf.set_params('tax losses monetized', True)
    pf.set_params('operating incentives taxable', True)
    pf.set_params('general inflation rate', gen_inflation)
    pf.set_params('leverage after tax nominal discount rate', plant_config["finance_parameters"]["discount_rate"])
    pf.set_params('debt equity ratio of initial financing', (plant_config["finance_parameters"]["debt_equity_split"]/(100-plant_config["finance_parameters"]["debt_equity_split"])))
    pf.set_params('debt type', plant_config["finance_parameters"]["debt_type"])
    pf.set_params('loan period if used', plant_config["finance_parameters"]["loan_period"])
    pf.set_params('debt interest rate', plant_config["finance_parameters"]["debt_interest_rate"])
    pf.set_params('cash onhand percent', plant_config["finance_parameters"]["cash_onhand_months"])

    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="Wind System",cost=capex_breakdown["wind"], depr_type=plant_config["finance_parameters"]["depreciation_method"], depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])

    if not (design_scenario["h2_location"] == "turbine" and design_scenario["h2_storage"] == "turbine"):
        pf.add_capital_item(name="Electrical Export system",cost=capex_breakdown["electrical_export_system"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])

    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Wind and Electrical Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["wind_and_electrical"],escalation=gen_inflation)

    sol = pf.solve_price()

    lcoe = sol['price']

    if verbose:
        print("\nProFAST LCOE: ", "%.2f" % (lcoe*1E3), "$/MWh")

    if show_plots or save_plots:
        pf.plot_costs_yearly(per_kg=False, scale='M', remove_zeros=True, remove_depreciation=False, fileout="figures/wind_only/annual_cash_flow_wind_only_%i.png" %(design_scenario["id"]), show_plot=show_plots)
        pf.plot_costs_yearly2(per_kg=False, scale='M', remove_zeros=True, remove_depreciation=False, fileout="figures/wind_only/annual_cash_flow_wind_only_%i.html" %(design_scenario["id"]), show_plot=show_plots)
        pf.plot_capital_expenses(fileout="figures/wind_only/capital_expense_only_%i.png" %(design_scenario["id"]), show_plot=show_plots)
        pf.plot_cashflow(fileout="figures/wind_only/cash_flow_wind_only_%i.png" %(design_scenario["id"]), show_plot=show_plots)
        pf.plot_costs(fileout="figures/wind_only/cost_breakdown_%i.png" %(design_scenario["id"]), show_plot=show_plots)

    return lcoe, pf

def run_profast_grid_only(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown, hopp_results, design_scenario, verbose=False, show_plots=False, save_plots=False):

    gen_inflation = plant_config["finance_parameters"]["general_inflation"]

    if design_scenario["h2_storage"] == "onshore" or design_scenario["h2_location"] == "onshore":
        land_cost = 1E6 #TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST('blank')
    pf.set_params('commodity', {"name":'Hydrogen',"unit":"kg","initial price":100,"escalation": gen_inflation})
    pf.set_params('capacity', electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output']/365.0) #kg/day
    pf.set_params('maintenance',{"value":0,"escalation": gen_inflation})
    pf.set_params('analysis start year', plant_config["atb_year"]+1)
    pf.set_params('operating life', plant_config["project_parameters"]["project_lifetime"])
    # pf.set_params('installation months', (orbit_project.installation_time/(365*24))*(12.0/1.0))
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets', land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_config["project_parameters"]["project_lifetime"])
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',1)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax', plant_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent', plant_config["finance_parameters"]["property_tax"] + plant_config["finance_parameters"]["property_insurance"])
    pf.set_params('admin expense percent', plant_config["finance_parameters"]["administrative_expense_percent_of_sales"])
    pf.set_params('total income tax rate', plant_config["finance_parameters"]["total_income_tax_rate"])
    pf.set_params('capital gains tax rate', plant_config["finance_parameters"]["capital_gains_tax_rate"])
    pf.set_params('sell undepreciated cap', True)
    pf.set_params('tax losses monetized', True)
    pf.set_params('operating incentives taxable', True)
    pf.set_params('general inflation rate', gen_inflation)
    pf.set_params('leverage after tax nominal discount rate', plant_config["finance_parameters"]["discount_rate"])
    pf.set_params('debt equity ratio of initial financing', (plant_config["finance_parameters"]["debt_equity_split"]/(100-plant_config["finance_parameters"]["debt_equity_split"])))
    pf.set_params('debt type', plant_config["finance_parameters"]["debt_type"])
    pf.set_params('loan period if used', plant_config["finance_parameters"]["loan_period"])
    pf.set_params('debt interest rate', plant_config["finance_parameters"]["debt_interest_rate"])
    pf.set_params('cash onhand percent', plant_config["finance_parameters"]["cash_onhand_months"])

    #----------------------------------- Add capital items to ProFAST ----------------
    # pf.add_capital_item(name="Wind System",cost=capex_breakdown["wind"], depr_type=plant_config["finance_parameters"]["depreciation_method"], depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])
    pf.add_capital_item(name="Electrical Export system",cost=capex_breakdown["electrical_export_system"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])

    electrolyzer_refurbishment_schedule = np.zeros(plant_config["project_parameters"]["project_lifetime"])
    refurb_period = round(plant_config["electrolyzer"]["time_between_replacement"]/(24*365))
    electrolyzer_refurbishment_schedule[refurb_period:plant_config["project_parameters"]["project_lifetime"]:refurb_period] = plant_config["electrolyzer"]["replacement_cost_percent"]
    # print(electrolyzer_refurbishment_schedule)
    pf.add_capital_item(name="Electrolysis System",cost=capex_breakdown["electrolyzer"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=list(electrolyzer_refurbishment_schedule))

    pf.add_capital_item(name="Hydrogen Storage System",cost=capex_breakdown["h2_storage"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=[0])
    # pf.add_capital_item(name ="Desalination system",cost=capex_breakdown["desal"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])

    #-------------------------------------- Add fixed costs--------------------------------
    # pf.add_fixed_cost(name="Wind Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["wind"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Electrical Export Fixed O&M Cost", usage=1.0,unit='$/year',cost=opex_breakdown["electrical_export_system"],escalation=gen_inflation)
    # pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["desal"],escalation=gen_inflation)
    pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost", usage=1.0, unit='$/year', cost=opex_breakdown["electrolyzer"],escalation=gen_inflation)
    pf.add_fixed_cost(name="Hydrogen Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_breakdown["h2_storage"],escalation=gen_inflation)

    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Water',usage=electrolyzer_physics_results["H2_Results"]['water_annual_usage']/electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output'],unit='kg-water', cost='US Average', escalation=gen_inflation)

    energy_purchase = plant_config["electrolyzer"]["rating"]*1E3

    pf.add_fixed_cost(name='Electricity from grid', usage=1.0, unit='$/year', cost=energy_purchase*plant_config["project_parameters"]["ppa_price"], escalation=gen_inflation)

    sol = pf.solve_price()

    lcoh = sol['price']
    if verbose:
        print("\nLCOH grid only: ", "%.2f" % (lcoh), "$/kg")

    return lcoh, pf

def run_profast_full_plant_model(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown, hopp_results, incentive_option, design_scenario, verbose=False, show_plots=False, save_plots=False):

    gen_inflation = plant_config["finance_parameters"]["general_inflation"]

    if design_scenario["h2_storage"] == "onshore" or design_scenario["h2_location"] == "onshore":
        land_cost = 1E6 #TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST('blank')
    pf.set_params('commodity', {"name":'Hydrogen',"unit":"kg","initial price":100,"escalation": gen_inflation})
    pf.set_params('capacity', electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output']/365.0) #kg/day
    pf.set_params('maintenance',{"value":0,"escalation": gen_inflation})
    pf.set_params('analysis start year', plant_config["atb_year"]+1)
    pf.set_params('operating life', plant_config["project_parameters"]["project_lifetime"])
    pf.set_params('installation months', (orbit_project.installation_time/(365*24))*(12.0/1.0)) # convert from hours to months
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets', land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_config["project_parameters"]["project_lifetime"])
    pf.set_params('demand rampup', 0)
    pf.set_params('long term utilization', 1) #TODO should use utilization
    pf.set_params('credit card fees', 0)
    pf.set_params('sales tax', plant_config["finance_parameters"]["sales_tax_rate"])
    pf.set_params('license and permit',{'value':00, 'escalation':gen_inflation})
    pf.set_params('rent',{'value':0, 'escalation':gen_inflation})
    # TODO how to handle property tax and insurance for fully offshore?
    pf.set_params('property tax and insurance percent', plant_config["finance_parameters"]["property_tax"] + plant_config["finance_parameters"]["property_insurance"])
    pf.set_params('admin expense percent', plant_config["finance_parameters"]["administrative_expense_percent_of_sales"])
    pf.set_params('total income tax rate', plant_config["finance_parameters"]["total_income_tax_rate"])
    pf.set_params('capital gains tax rate', plant_config["finance_parameters"]["capital_gains_tax_rate"])
    pf.set_params('sell undepreciated cap', True)
    pf.set_params('tax losses monetized', True)
    pf.set_params('operating incentives taxable', True) # TODO check with Matt and tell Kaitlin
    pf.set_params('general inflation rate', gen_inflation)
    pf.set_params('leverage after tax nominal discount rate', plant_config["finance_parameters"]["discount_rate"])
    pf.set_params('debt equity ratio of initial financing', (plant_config["finance_parameters"]["debt_equity_split"]/(100-plant_config["finance_parameters"]["debt_equity_split"]))) #TODO this may not be put in right
    pf.set_params('debt type', plant_config["finance_parameters"]["debt_type"])
    pf.set_params('loan period if used', plant_config["finance_parameters"]["loan_period"])
    pf.set_params('debt interest rate', plant_config["finance_parameters"]["debt_interest_rate"])
    pf.set_params('cash onhand percent', plant_config["finance_parameters"]["cash_onhand_months"])

    #----------------------------------- Add capital and fixed items to ProFAST ----------------
    pf.add_capital_item(name="Wind System",cost=capex_breakdown["wind"], depr_type=plant_config["finance_parameters"]["depreciation_method"], depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])
    pf.add_fixed_cost(name="Wind and Electrical Export Fixed O&M Cost",usage=1.0, unit='$/year',cost=opex_breakdown["wind_and_electrical"],escalation=gen_inflation)

    if not (design_scenario["h2_location"] == "turbine" and design_scenario["h2_storage"] == "turbine"):
        pf.add_capital_item(name="Electrical Export system",cost=capex_breakdown["electrical_export_system"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period"],refurb=[0])
        # TODO assess if this makes sense (electrical export O&M included in wind O&M)

    electrolyzer_refurbishment_schedule = np.zeros(plant_config["project_parameters"]["project_lifetime"])
    refurb_period = round(plant_config["electrolyzer"]["time_between_replacement"]/(24*365))
    electrolyzer_refurbishment_schedule[refurb_period:plant_config["project_parameters"]["project_lifetime"]:refurb_period] = plant_config["electrolyzer"]["replacement_cost_percent"]

    pf.add_capital_item(name="Electrolysis System",cost=capex_breakdown["electrolyzer"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=list(electrolyzer_refurbishment_schedule))
    pf.add_fixed_cost(name="Electrolysis System Fixed O&M Cost", usage=1.0, unit='$/year', cost=opex_breakdown["electrolyzer"],escalation=gen_inflation)

    if design_scenario["h2_location"] == "turbine":
        pf.add_capital_item(name="H2 Pipe Array System",cost=capex_breakdown["h2_pipe_array"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=[0])
        pf.add_fixed_cost(name="H2 Pipe Array Fixed O&M Cost", usage=1.0, unit='$/year', cost=opex_breakdown["h2_pipe_array"], escalation=gen_inflation)

    if (design_scenario["h2_storage"] == "onshore" and design_scenario["h2_location"] != "onshore") or (design_scenario["h2_storage"] != "onshore" and design_scenario["h2_location"] == "onshore"):
        pf.add_capital_item(name="H2 Transport Compressor System",cost=capex_breakdown["h2_transport_compressor"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=[0])
        pf.add_capital_item(name="H2 Transport Pipeline System",cost=capex_breakdown["h2_transport_pipeline"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=[0])

        pf.add_fixed_cost(name="H2 Transport Compression Fixed O&M Cost", usage=1.0, unit='$/year', cost=opex_breakdown["h2_transport_compressor"], escalation=gen_inflation)
        pf.add_fixed_cost(name="H2 Transport Pipeline Fixed O&M Cost", usage=1.0, unit='$/year', cost=opex_breakdown["h2_transport_pipeline"], escalation=gen_inflation)

    if plant_config["h2_storage"]["type"] != "none":
        pf.add_capital_item(name="Hydrogen Storage System",cost=capex_breakdown["h2_storage"], depr_type=plant_config["finance_parameters"]["depreciation_method"],depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"],refurb=[0])
        pf.add_fixed_cost(name="Hydrogen Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_breakdown["h2_storage"],escalation=gen_inflation)

    #---------------------- Add feedstocks, note the various cost options-------------------
    if design_scenario["h2_location"] == "onshore":
        pf.add_feedstock(name='Water',usage=electrolyzer_physics_results["H2_Results"]['water_annual_usage']/electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output'],unit='kg-water', cost='US Average', escalation=gen_inflation)
    else:
        pf.add_capital_item(name="Desal System", cost=capex_breakdown["desal"], depr_type=plant_config["finance_parameters"]["depreciation_method"], depr_period=plant_config["finance_parameters"]["depreciation_period_electrolyzer"], refurb=[0])
        pf.add_fixed_cost(name="Desal Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_breakdown["desal"],escalation=gen_inflation)

    if plant_config["project_parameters"]["grid_connection"]:
        annual_energy_shortfall = np.sum(hopp_results["energy_shortfall_hopp"])
        energy_purchase = annual_energy_shortfall

        pf.add_fixed_cost(name='Electricity from grid', usage=1.0, unit='$/year', cost=energy_purchase*plant_config["project_parameters"]["ppa_price"], escalation=gen_inflation)


    #------------------------------------- add incentives -----------------------------------
    """ Note: units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = plant_config["policy_parameters"]["option%s" %(incentive_option)]

    # add wind_itc (% of wind capex)
    wind_itc_value_percent_wind_capex = incentive_dict["wind_itc"]
    wind_itc_value_dollars = wind_itc_value_percent_wind_capex*(capex_breakdown["wind"] + capex_breakdown["electrical_export_system"])
    pf.set_params('one time cap inct',
        {"value":wind_itc_value_dollars,
        "depr type":plant_config["finance_parameters"]["depreciation_method"],
        "depr period":plant_config["finance_parameters"]["depreciation_period"],
        "depreciable":True})

    # add wind_ptc ($/kW)
    # adjust from 1992 dollars to start year
    wind_ptc_in_dollars_per_kw = -npf.fv(gen_inflation, plant_config["atb_year"]+round((orbit_project.installation_time/(365*24)))-1992, 0,  incentive_dict["wind_ptc"]) # given in 1992 dollars but adjust for inflation
    kw_per_kg_h2 = sum(hopp_results["combined_pv_wind_power_production_hopp"])/electrolyzer_physics_results["H2_Results"]['hydrogen_annual_output']
    wind_ptc_in_dollars_per_kg_h2 = wind_ptc_in_dollars_per_kw*kw_per_kg_h2
    pf.add_incentive(name='Wind PTC', value=wind_ptc_in_dollars_per_kg_h2, decay=-gen_inflation, sunset_years=10, tax_credit=True) #TODO check decay

    # add h2_ptc ($/kg)
    h2_ptc_inflation_adjusted = -npf.fv(gen_inflation, plant_config["atb_year"]+round((orbit_project.installation_time/(365*24)))-2022, 0,  incentive_dict["h2_ptc"])
    pf.add_incentive(name='H2 PTC', value=h2_ptc_inflation_adjusted, decay=-gen_inflation, sunset_years=10, tax_credit=True) #TODO check decay

    #------------------------------------ solve and post-process -----------------------------

    sol = pf.solve_price()

    df = pf.cash_flow_out_table

    lcoh = sol['price']

    if verbose:
        print("\nProFAST LCOH: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST NPV: ", "%.2f" % (sol['NPV']))
        print("ProFAST IRR: ", "%.5f" % (max(sol['irr'])))
        print("ProFAST LCO: ", "%.2f" % (sol['lco']), "$/kg")
        print("ProFAST Profit Index: ", "%.2f" % (sol['profit index']))
        print("ProFAST payback period: ", sol["investor payback period"])

        MIRR = npf.mirr(df["Investor cash flow"], plant_config["finance_parameters"]["debt_interest_rate"], plant_config["finance_parameters"]["discount_rate"]) # TODO probably ignore MIRR
        NPV = npf.npv(plant_config["finance_parameters"]["general_inflation"], df["Investor cash flow"])
        ROI = np.sum(df["Investor cash flow"])/abs(np.sum(df["Investor cash flow"][df["Investor cash flow"]<0])) # ROI is not a good way of thinking about the value of the project

        #TODO project level IRR - capex and operating cash flow

        #note: hurdle rate typically 20% IRR before investing in it due to typically optimistic assumptions
        #note: negative retained earnings (keeping debt, paying down equity) - to get around it, do another line for retained earnings and watch dividends paid by the rpoject (net income/equity should stay positive this way)

        print("Investor NPV: ", np.round(NPV*1E-6, 2), "M USD")
        print("Investor MIRR: ", np.round(MIRR, 5), "")
        print("Investor ROI: ", np.round(ROI, 5), "")

    if save_plots or show_plots:


        if not os.path.exists("figures"):
            os.mkdir("figures")
            os.mkdir("figures/lcoh_breakdown")
            os.mkdir("figures/capex")
            os.mkdir("figures/annual_cash_flow")
        savepaths = ["figures/capex/", "figures/annual_cash_flow/", "figures/lcoh_breakdown/", "data/"]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.mkdir(savepath)

        pf.plot_capital_expenses(fileout="figures/capex/capital_expense_%i.pdf" %(design_scenario["id"]), show_plot=show_plots)
        pf.plot_cashflow(fileout="figures/annual_cash_flow/cash_flow_%i.png" %(design_scenario["id"]), show_plot=show_plots)

        pf.cash_flow_out_table.to_csv("data/cash_flow_%i.csv" %(design_scenario["id"]))

        pf.plot_costs("figures/lcoh_breakdown/lcoh_%i" %(design_scenario["id"]), show_plot=show_plots)

    return lcoh, pf

def visualize_plant(plant_config, orbit_project, platform_results, desal_results, h2_storage_results, electrolyzer_physics_results, design_scenario, colors, plant_design_number, show_plots=False, save_plots=False):

    plt.rcParams.update({'font.size': 7})

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
    cable_array_points = orbit_project.phases["ArraySystemDesign"].coordinates*1E3 # ORBIT gives coordinates in km, convert to m
    pipe_array_points = orbit_project.phases["ArraySystemDesign"].coordinates*1E3 # ORBIT gives coordinates in km, convert to m

    # get turbine rotor diameter
    rotor_diameter = orbit_project.config["turbine"]["rotor_diameter"] # in m
    rotor_radius = rotor_diameter/2.0

    # get turbine tower base diameter
    tower_base_diameter = orbit_project.config["turbine"]["tower"]["section_diameters"][0] # in m
    tower_base_radius = tower_base_diameter/2.0

    # get turbine locations
    turbine_x = orbit_project.phases["ArraySystemDesign"].turbines_x.flatten()*1E3 # ORBIT gives coordinates in km, convert to m
    turbine_x = turbine_x[~np.isnan(turbine_x)]
    turbine_y = orbit_project.phases["ArraySystemDesign"].turbines_y.flatten()*1E3 # ORBIT gives coordinates in km, convert to m
    turbine_y = turbine_y[~np.isnan(turbine_y)]

    # get substation location and dimensions
    substation_x = orbit_project.phases["ArraySystemDesign"].oss_x*1E3 # ORBIT gives coordinates in km, convert to m (treated as center)
    substation_y = orbit_project.phases["ArraySystemDesign"].oss_y*1E3 # ORBIT gives coordinates in km, convert to m (treated as center)
    substation_side_length = 20 # [m] just based on a large substation (https://www.windpowerengineering.com/making-modern-offshore-substation/) since the dimensions are not available in ORBIT

    # get equipment platform location and dimensions
    equipment_platform_area = platform_results["toparea_m2"]
    equipment_platform_side_length = np.sqrt(equipment_platform_area)
    equipment_platform_x = substation_x - substation_side_length - equipment_platform_side_length/2 # [m] (treated as center)
    equipment_platform_y = substation_y # [m] (treated as center)

    # get platform equipment dimensions
    if design_scenario["h2_location"] == "turbine":
        desal_equipment_area = desal_results["per_turb_equipment_footprint_m2"] #equipment_footprint_m2
    elif design_scenario["h2_location"] == "platform":
        desal_equipment_area = desal_results["equipment_footprint_m2"]
    else:
        desal_equipment_area = 0

    desal_equipment_side = np.sqrt(desal_equipment_area)

    if (design_scenario["h2_storage"] != "turbine") and (plant_config["h2_storage"]["type"] != "none"):
        h2_storage_area = h2_storage_results["tank_footprint_m2"]
        h2_storage_side = np.sqrt(h2_storage_area)

    electrolyzer_area = electrolyzer_physics_results["equipment_footprint_m2"]
    if design_scenario["h2_location"] == "turbine":
        electrolyzer_area /= orbit_project.config["plant"]["num_turbines"]
    electrolyzer_side = np.sqrt(electrolyzer_area)

    # compressor side # not sized
    compressor_area = 25
    compressor_side = np.sqrt(compressor_area)

    # get pipe points
    pipe_x = np.array([substation_x-1000, substation_x])
    pipe_y = np.array([substation_y, substation_y])

    # get cable points
    cable_x = pipe_x
    cable_y = pipe_y

    # set onshor origin
    onshorex = 50
    onshorey = 50

    # plot the stuff
    ## create figure
    fig, ax = plt.subplots(2,2, figsize=(12,6))

    # onshore plant | offshore plant
    # platform/substation | turbine

    ## add turbines
    i = 0
    for (x, y) in zip(turbine_x, turbine_y):
        if i == 0:
            rlabel = "Wind Turbine Rotor"
            tlabel = "Wind Turbine Tower"
            i += 1
        else:
            rlabel = None
            tlabel = None
        turbine_patch = patches.Circle((x, y), radius=rotor_radius, color=turbine_rotor_color, fill=False, label=rlabel, zorder=10)
        ax[0, 1].add_patch(turbine_patch)
        # turbine_patch01_tower = patches.Circle((x, y), radius=tower_base_radius, color=turbine_tower_color, fill=False, label=tlabel, zorder=10)
        # ax[0, 1].add_patch(turbine_patch01_tower)
    # turbine_patch11_rotor = patches.Circle((turbine_x[0], turbine_y[0]), radius=rotor_radius, color=turbine_rotor_color, fill=False, label=None, zorder=10)
    tlabel = "Wind Turbine Tower"
    turbine_patch11_tower = patches.Circle((turbine_x[0], turbine_y[0]), radius=tower_base_radius, color=turbine_tower_color, fill=False, label=tlabel, zorder=10)
    # ax[1, 1].add_patch(turbine_patch11_rotor)
    ax[1, 1].add_patch(turbine_patch11_tower)

    # add pipe array
    if design_scenario["h2_storage"] != "turbine" and design_scenario["h2_location"] == "turbine":
        i = 0
        for point_string in pipe_array_points:
            if i == 0:
                label = "Array Pipes"
                i += 1
            else:
                label = None
            ax[0, 1].plot(point_string[:,0], point_string[:,1]-substation_side_length/2, ":", color=pipe_color, zorder=0, linewidth=1, label=label)
            ax[1, 0].plot(point_string[:,0], point_string[:,1]-substation_side_length/2, ":", color=pipe_color, zorder=0, linewidth=1, label=label)
            ax[1, 1].plot(point_string[:,0], point_string[:,1]-substation_side_length/2, ":", color=pipe_color, zorder=0, linewidth=1, label=label)

    ## add cables
    if design_scenario["h2_storage"] != "turbine":
        i = 0
        for point_string in cable_array_points:
            if i == 0:
                label = "Array Cables"
                i += 1
            else:
                label = None
            ax[0, 1].plot(point_string[:,0], point_string[:,1]+substation_side_length/2, "-", color=cable_color, zorder=0, linewidth=1, label=label)
            ax[1, 0].plot(point_string[:,0], point_string[:,1]+substation_side_length/2, "-", color=cable_color, zorder=0, linewidth=1, label=label)
            ax[1, 1].plot(point_string[:,0], point_string[:,1]+substation_side_length/2, "-", color=cable_color, zorder=0, linewidth=1, label=label)

    ## add substation
    if design_scenario["h2_storage"] != "turbine":
        substation_patch01 = patches.Rectangle((substation_x-substation_side_length, substation_y-substation_side_length/2), substation_side_length, substation_side_length,
                                            fill=True, color=substation_color, label="Substation*", zorder=11)
        substation_patch10 = patches.Rectangle((substation_x-substation_side_length, substation_y-substation_side_length/2), substation_side_length, substation_side_length,
                                            fill=True, color=substation_color, label="Substation*", zorder=11)
        ax[0, 1].add_patch(substation_patch01)
        ax[1, 0].add_patch(substation_patch10)

    ## add equipment platform
    if design_scenario["h2_storage"] == "platform" or design_scenario["h2_location"] == "platform": # or design_scenario["transportation"] == "pipeline":
        equipment_platform_patch01 = patches.Rectangle((equipment_platform_x - equipment_platform_side_length/2, equipment_platform_y - equipment_platform_side_length/2),
                                                    equipment_platform_side_length, equipment_platform_side_length, color=equipment_platform_color, fill=True, label="Equipment Platform", zorder=1)
        equipment_platform_patch10 = patches.Rectangle((equipment_platform_x - equipment_platform_side_length/2, equipment_platform_y - equipment_platform_side_length/2),
                                                    equipment_platform_side_length, equipment_platform_side_length, color=equipment_platform_color, fill=True, label="Equipment Platform", zorder=1)
        ax[0, 1].add_patch(equipment_platform_patch01)
        ax[1, 0].add_patch(equipment_platform_patch10)

    ## add hvdc cable
    if design_scenario["transportation"] == "hvdc":
        ax[0, 0].plot([50, 1000], [48, 48], "--", color=cable_color, label="HVDC Cable")
        ax[0, 1].plot([-5000, substation_x], [substation_y-100, substation_y-100], "--", color=cable_color, label="HVDC Cable", zorder=0)
        ax[1, 0].plot([-5000, substation_x], [substation_y-2, substation_y-2], "--", color=cable_color, label="HVDC Cable", zorder=0)
    ## add transport pipeline
    if design_scenario["transportation"] == "pipeline" or (design_scenario["transportation"] == "hvdc" and design_scenario["h2_storage"] == "platform"):
        linetype = "-."
        label = "Transport Pipeline"
        linewidth = 1.0
        ax[0, 0].plot([onshorex, 1000], [onshorey+2, onshorey+2], linetype, color=pipe_color, label=label, linewidth=linewidth, zorder=0)
        ax[0, 1].plot([-5000, substation_x], [substation_y+100, substation_y+100], linetype, linewidth=linewidth, color=pipe_color, label=label, zorder=0)
        ax[1, 0].plot([-5000, substation_x], [substation_y+2, substation_y+2], linetype, linewidth=linewidth, color=pipe_color, label=label, zorder=0)

        if design_scenario["transportation"] == "hvdc" and design_scenario["h2_storage"] == "platform":
            h2cx = onshorex - compressor_side
            h2cy = onshorey - compressor_side + 2
            h2cax = ax[0, 0]
        else:
            h2cx = substation_x-substation_side_length
            h2cy = substation_y
            h2cax = ax[1, 0]
        # compressor_patch01 = patches.Rectangle((substation_x-substation_side_length, substation_y), compressor_side, compressor_side, color=compressor_color, fill=None, label="Transport Compressor*", hatch="+++", zorder=20)
        compressor_patch10 = patches.Rectangle((h2cx, h2cy), compressor_side, compressor_side, color=compressor_color, fill=None, label="Transport Compressor*", hatch="+++", zorder=20)
        # ax[0, 1].add_patch(compressor_patch01)
        h2cax.add_patch(compressor_patch10)

    ## add plant components
    ehatch = "///"
    dhatch = "xxxx"
    if design_scenario["h2_location"] == "onshore" and (plant_config["h2_storage"]["type"] != "none"):
        electrolyzer_patch = patches.Rectangle((onshorex-h2_storage_side, onshorey+4), electrolyzer_side, electrolyzer_side, color=electrolyzer_color, fill=None, label="Electrolyzer", zorder=20, hatch=ehatch)
        ax[0, 0].add_patch(electrolyzer_patch)
    elif (design_scenario["h2_location"] == "platform") and (plant_config["h2_storage"]["type"] != "none"):

        dx = equipment_platform_x - equipment_platform_side_length/2
        dy = equipment_platform_y - equipment_platform_side_length/2
        e_side_y = equipment_platform_side_length
        e_side_x = electrolyzer_area/e_side_y
        d_side_y = equipment_platform_side_length
        d_side_x = desal_equipment_area/d_side_y
        ex = dx + d_side_x
        ey = dy

        electrolyzer_patch = patches.Rectangle((ex, ey), e_side_x, e_side_y, color=electrolyzer_color, fill=None,
                                                zorder=20, label="Electrolyzer", hatch=ehatch)
        ax[1, 0].add_patch(electrolyzer_patch)
        desal_patch = patches.Rectangle((dx, dy), d_side_x, d_side_y, color=desal_color,
                                        zorder=21, fill=None, label="Desalinator", hatch=dhatch)
        ax[1, 0].add_patch(desal_patch)
    elif (design_scenario["h2_location"] == "turbine") and (plant_config["h2_storage"]["type"] != "none"):
        electrolyzer_patch11 = patches.Rectangle((turbine_x[0], turbine_y[0]+tower_base_radius), electrolyzer_side, electrolyzer_side, color=electrolyzer_color, fill=None,
                                                zorder=20, label="Electrolyzer", hatch=ehatch)
        ax[1, 1].add_patch(electrolyzer_patch11)
        desal_patch11 = patches.Rectangle((turbine_x[0]-desal_equipment_side, turbine_y[0]+tower_base_radius), desal_equipment_side, desal_equipment_side, color=desal_color,
                                        zorder=21, fill=None, label="Desalinator", hatch=dhatch)
        ax[1, 1].add_patch(desal_patch11)
        i = 0
        for (x, y) in zip(turbine_x, turbine_y):
            if i == 0:
                elable = "Electrolyzer"
                dlabel = "Desalinator"
            else:
                elable = None
                dlabel = None
            electrolyzer_patch01 = patches.Rectangle((x, y+tower_base_radius), electrolyzer_side, electrolyzer_side, color=electrolyzer_color, fill=None,
                                                zorder=20, label=None, hatch=ehatch)
            desal_patch01 = patches.Rectangle((x-desal_equipment_side, y+tower_base_radius), desal_equipment_side, desal_equipment_side, color=desal_color,
                                        zorder=21, fill=None, label=None, hatch=dhatch)
            ax[0,1].add_patch(electrolyzer_patch01)
            ax[0,1].add_patch(desal_patch01)

    h2_storage_hatch = "\\\\\\"
    if design_scenario["h2_storage"] == "onshore" and (plant_config["h2_storage"]["type"] != "none"):
        h2_storage_patch = patches.Rectangle((onshorex-h2_storage_side, onshorey-h2_storage_side-2), h2_storage_side, h2_storage_side, color=h2_storage_color, fill=None, label="H$_2$ Storage", hatch=h2_storage_hatch)
        ax[0, 0].add_patch(h2_storage_patch)
    elif  design_scenario["h2_storage"] == "platform" and (plant_config["h2_storage"]["type"] != "none"):
        s_side_y = equipment_platform_side_length
        s_side_x = h2_storage_area/s_side_y
        sx = equipment_platform_x - equipment_platform_side_length/2
        sy = equipment_platform_y - equipment_platform_side_length/2
        if design_scenario["h2_location"] == "platform":
            sx += equipment_platform_side_length - s_side_x

        h2_storage_patch = patches.Rectangle((sx, sy), s_side_x, s_side_y, color=h2_storage_color, fill=None, label="H$_2$ Storage", hatch=h2_storage_hatch)
        ax[1, 0].add_patch(h2_storage_patch)
    elif design_scenario["h2_storage"] == "turbine" and (plant_config["h2_storage"]["type"] != "none"):
        h2_storage_patch = patches.Circle((turbine_x[0], turbine_y[0]), radius=tower_base_diameter/2, color=h2_storage_color, fill=None, label="H$_2$ Storage", hatch=h2_storage_hatch)
        ax[1, 1].add_patch(h2_storage_patch)
        i = 0
        for (x, y) in zip(turbine_x, turbine_y):
            if i == 0:
                slable = "H$_2$ Storage"
            else:
                slable = None
            h2_storage_patch = patches.Circle((x, y), radius=tower_base_diameter/2, color=h2_storage_color, fill=None, label=None, hatch=h2_storage_hatch)
            ax[0, 1].add_patch(h2_storage_patch)

    ax[0, 0].set(xlim=[0, 250], ylim=[0, 200])
    ax[0, 0].set(aspect="equal")

    allpoints = cable_array_points.flatten()
    allpoints = allpoints[~np.isnan(allpoints)]
    roundto = -3
    ax[0,1].set(xlim=[round(np.min(allpoints-2000), ndigits=roundto), round(np.max(allpoints+2000), ndigits=roundto)],
            ylim=[round(np.min(turbine_y-1000), ndigits=roundto), round(np.max(turbine_y+4000), ndigits=roundto)])
    ax[0,1].set(aspect="equal")
    ax[0,1].xaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax[0,1].yaxis.set_major_locator(ticker.MultipleLocator(1000))

    roundto = -2
    ax[1,0].set(xlim=[round(substation_x-400, ndigits=roundto), round(substation_x+100, ndigits=roundto)],
            ylim=[round(substation_y-200, ndigits=roundto), round(substation_y+200, ndigits=roundto)])
    ax[1,0].set(aspect="equal")

    tower_buffer0 = 20
    tower_buffer1 = 30
    roundto = -1
    ax[1,1].set(xlim=[round(turbine_x[0]-tower_base_radius-tower_buffer0-50, ndigits=roundto), round(turbine_x[0]+tower_base_radius+tower_buffer1, ndigits=roundto)],
            ylim=[round(turbine_y[0]-tower_base_radius-tower_buffer0, ndigits=roundto), round(turbine_y[0]+tower_base_radius+tower_buffer1, ndigits=roundto)])
    ax[1,1].set(aspect="equal")
    ax[1,1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[1,1].yaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax[0,1].legend(frameon=False)
    # ax[0,1].axis('off')

    labels = ["(a) Onshore plant", "(b) Offshore plant", "(c) Equipment platform and substation", "(d) NW-most wind turbine"]
    for (axi, label) in zip(ax.flatten(), labels):
        axi.legend(frameon=False, ncol=2, loc="best")
        axi.set(xlabel="Easting (m)", ylabel="Northing (m)")
        axi.set_title(label, loc="left")

    ## save the plot
    plt.tight_layout()
    if save_plots:
        savepath = "figures/layout/"
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+"plant_layout_%i.png" %(plant_design_number), transparent=True)
    if show_plots:
        plt.show()
    return 0

# set up function to post-process HOPP results
def post_process_simulation(lcoe, lcoh, pf_lcoh, pf_lcoe, hopp_results, electrolyzer_physics_results, plant_config, h2_storage_results, capex_breakdown, opex_breakdown, orbit_project, platform_results, desal_results, design_scenario, plant_design_number, incentive_option, solver_results=[], show_plots=False, save_plots=False):#, lcoe, lcoh, lcoh_with_grid, lcoh_grid_only):

    # colors (official NREL color palette https://brand.nrel.gov/content/index/guid/color_palette?parent=61)
    colors = ["#0079C2", "#00A4E4", "#F7A11A", "#FFC423", "#5D9732", "#8CC63F", "#5E6A71", "#D1D5D8", "#933C06", "#D9531E"]
    # load saved results

    # post process results
    print("LCOE: ", round(lcoe*1E3, 2), "$/MWh")
    print("LCOH: ", round(lcoh, 2), "$/kg")
    print("wind capacity factor: ", round(np.sum(hopp_results["combined_pv_wind_power_production_hopp"])*1E-3/(plant_config["plant"]["capacity"]*365*24), 2))
    print("electrolyzer capacity factor: ", round(np.sum(electrolyzer_physics_results["energy_to_electrolyzer_kw"])*1E-3/(plant_config["electrolyzer"]["rating"]*365*24), 2))
    print("Electorlyzer CAPEX installed $/kW: ", round(capex_breakdown["electrolyzer"]/(plant_config["electrolyzer"]["rating"]*1E3), 2))

    if show_plots or save_plots:
        visualize_plant(plant_config, orbit_project, platform_results, desal_results, h2_storage_results, electrolyzer_physics_results, design_scenario, colors, plant_design_number, show_plots=show_plots, save_plots=save_plots)

    if not os.path.exists("data"):
        os.mkdir("data")
        os.mkdir("data/lcoe")
        os.mkdir("data/lcoh")
    pf_lcoh.get_cost_breakdown().to_csv("data/lcoh/cost_breakdown_lcoh_design%i_incentive%i_%sstorage.csv" %(plant_design_number, incentive_option, plant_config["h2_storage"]["type"]))
    pf_lcoe.get_cost_breakdown().to_csv("data/lcoe/cost_breakdown_lcoe_design%i_incentive%i_%sstorage.csv" %(plant_design_number, incentive_option, plant_config["h2_storage"]["type"]))

    # create dataframe for saving all the stuff
    plant_config["design_scenario"] = design_scenario
    plant_config["plant_design_number"] = plant_design_number
    plant_config["incentive_options"] = incentive_option

    # save power usage data
    if len(solver_results) > 0:
        hours = len(hopp_results["combined_pv_wind_power_production_hopp"])
        annual_energy_breakdown = {"wind_kwh": sum(hopp_results["combined_pv_wind_power_production_hopp"]),
                                "electrolyzer_kwh": sum(electrolyzer_physics_results["energy_to_electrolyzer_kw"]),
                                "desal_kwh": solver_results[1]*hours,
                                "h2_transport_compressor_power_kwh": solver_results[2]*hours,
                                "h2_storage_power_kwh": solver_results[3]*hours}

    return annual_energy_breakdown

# set up function to run base line case
def run_simulation(electrolyzer_rating=None, plant_size=None, verbose=False, show_plots=False, save_plots=False, use_profast=True, storage_type=None, incentive_option=1, plant_design_scenario=1, output_level=1):

    # load inputs as needed
    plant_config, turbine_config, wind_resource, floris_config = get_inputs(verbose=verbose, show_plots=show_plots, save_plots=save_plots)

    if electrolyzer_rating != None:
        plant_config["electrolyzer"]["rating"] = electrolyzer_rating

    if storage_type != None:
        plant_config["h2_storage"]["type"] = storage_type

    if plant_size != None:
        plant_config["plant"]["capacity"] = plant_size
        plant_config["plant"]["num_turbines"] = int(plant_size/turbine_config["turbine_rating"])
        print(plant_config["plant"]["num_turbines"])

    design_scenario = plant_config["plant_design"]["scenario%s" %(plant_design_scenario)]
    design_scenario["id"] = plant_design_scenario

    # run orbit for wind plant construction and other costs

    ## TODO get correct weather (wind, wave) inputs for ORBIT input (possibly via ERA5)
    orbit_project = run_orbit(plant_config, weather=None, verbose=verbose)

    # setup HOPP model
    hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args = setup_hopp(plant_config, turbine_config, wind_resource, orbit_project, floris_config, show_plots=show_plots, save_plots=save_plots)

    # run HOPP model
    hopp_results = run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=verbose)

    # this portion of the system is inside a function so we can use a solver to determine the correct energy availability for h2 production
    def energy_internals(hopp_results=hopp_results, hopp_site=hopp_site, hopp_technologies=hopp_technologies, hopp_scenario=hopp_scenario, hopp_h2_args=hopp_h2_args, orbit_project=orbit_project, design_scenario=design_scenario, plant_config=plant_config, turbine_config=turbine_config, wind_resource=wind_resource, floris_config=floris_config, electrolyzer_rating=electrolyzer_rating, plant_size=plant_size, verbose=verbose, show_plots=show_plots, save_plots=save_plots, use_profast=use_profast, storage_type=storage_type, incentive_option=incentive_option, plant_design_scenario=plant_design_scenario, output_level=output_level, solver=True, power_for_peripherals_kw_in=0.0, breakdown=False):

        hopp_results_internal = dict(hopp_results)

        # set energy input profile
        ### subtract peripheral power from supply to get what is left for electrolyzer
        remaining_power_profile_in = np.zeros_like(hopp_results["combined_pv_wind_power_production_hopp"])
        for i in range(len(hopp_results["combined_pv_wind_power_production_hopp"])):
            r = hopp_results["combined_pv_wind_power_production_hopp"][i] - power_for_peripherals_kw_in
            if r > 0:
                # print(r)
                remaining_power_profile_in[i] = r

        hopp_results_internal["combined_pv_wind_power_production_hopp"] = tuple(remaining_power_profile_in)

        # run electrolyzer physics model
        electrolyzer_physics_results = run_electrolyzer_physics(hopp_results_internal, hopp_scenario, hopp_h2_args, plant_config, wind_resource, design_scenario, show_plots=show_plots, save_plots=save_plots, verbose=verbose)

        # run electrolyzer cost model
        electrolyzer_cost_results = run_electrolyzer_cost(electrolyzer_physics_results, hopp_scenario, plant_config, design_scenario, verbose=verbose)

        desal_results = run_desal(plant_config, electrolyzer_physics_results, design_scenario, verbose)

        # run array system model
        h2_pipe_array_results = run_h2_pipe_array(plant_config, orbit_project, electrolyzer_physics_results, design_scenario, verbose)

        # compressor #TODO size correctly
        h2_transport_compressor, h2_transport_compressor_results = run_h2_transport_compressor(plant_config, electrolyzer_physics_results, design_scenario, verbose=verbose)

        # transport pipeline
        h2_transport_pipe_results = run_h2_transport_pipe(plant_config, electrolyzer_physics_results, design_scenario, verbose=verbose)

        # pressure vessel storage
        pipe_storage, h2_storage_results = run_h2_storage(plant_config, turbine_config, electrolyzer_physics_results, design_scenario, verbose=verbose)

        total_energy_available = np.sum(hopp_results["combined_pv_wind_power_production_hopp"])

        ### get all energy non-electrolyzer usage in kw
        desal_power_kw = desal_results["power_for_desal_kw"]

        h2_transport_compressor_power_kw = h2_transport_compressor_results["compressor_power"] # kW

        h2_storage_energy_kwh = h2_storage_results["storage_energy"]
        h2_storage_power_kw = h2_storage_energy_kwh*(1.0/(365*24))

        total_accessory_power_kw = desal_power_kw + h2_transport_compressor_power_kw + h2_storage_power_kw

        ### subtract peripheral power from supply to get what is left for electrolyzer
        remaining_power_profile = np.zeros_like(hopp_results["combined_pv_wind_power_production_hopp"])
        for i in range(len(hopp_results["combined_pv_wind_power_production_hopp"])):
            r = hopp_results["combined_pv_wind_power_production_hopp"][i] - total_accessory_power_kw
            if r > 0:
                remaining_power_profile[i] = r

        if verbose and not solver:
            print("\nEnergy/Power Results:")
            print("Supply (MWh): ", total_energy_available)
            print("Desal (kW): ", desal_power_kw)
            print("Transport compressor (kW): ", h2_transport_compressor_power_kw)
            print("Storage compression, refrigeration, etc (kW): ", h2_storage_power_kw)

        if (show_plots or save_plots) and not solver:
            fig, ax = plt.subplots(1)
            plt.plot(np.asarray(hopp_results["combined_pv_wind_power_production_hopp"])*1E-6, label="Total Energy Available")
            plt.plot(remaining_power_profile*1E-6, label="Energy Available for Electrolysis")
            plt.xlabel("Hour")
            plt.ylabel("Power (GW)")
            plt.tight_layout()
            if save_plots:
                savepath = "figures/power_series/"
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                plt.savefig(savepath+"power_%i.png" %(design_scenario["id"]), transparent=True)
            if show_plots:
                plt.show()
        if solver:
            if breakdown:
                return total_accessory_power_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw
            else:
                return total_accessory_power_kw
        else:
            return electrolyzer_physics_results, electrolyzer_cost_results, desal_results, h2_pipe_array_results, h2_transport_compressor, h2_transport_compressor_results, h2_transport_pipe_results, pipe_storage, h2_storage_results, total_accessory_power_kw

    # define function to provide to the brent solver
    def energy_residual_function(power_for_peripherals_kw_in):

        # get results for current design
        # print("power peri in: ", power_for_peripherals_kw_in)
        power_for_peripherals_kw_out = energy_internals(power_for_peripherals_kw_in=power_for_peripherals_kw_in, solver=True, verbose=False)

        # collect residual
        power_residual = power_for_peripherals_kw_out - power_for_peripherals_kw_in
        # print("\nresidual: ", power_residual)

        return power_residual

    def simple_solver(initial_guess=0.0):

        # get results for current design
        total_accessory_power_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw = energy_internals(power_for_peripherals_kw_in=initial_guess, solver=True, verbose=False, breakdown=True)

        return total_accessory_power_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw

    #################### solving for energy needed for non-electrolyzer components ####################################
    # this approach either exactly over over-estimates the energy needed for non-electrolyzer components
    solver_results = simple_solver(0)
    solver_result = solver_results[0]

    # this is a check on the simple solver
    # print("\nsolver result: ", solver_result)
    # residual = energy_residual_function(solver_result)
    # print("\nresidual: ", residual)

    # this approach exactly sizes the energy needed for the non-electrolyzer components (according to the current models anyway)
    # solver_result = optimize.brentq(energy_residual_function, -10, 20000, rtol=1E-5)
    # OptimizeResult = optimize.root(energy_residual_function, 11E3, tol=1)
    # solver_result = OptimizeResult.x
    # print(solver_result)
    ##################################################################################################################

    # get results for final design
    electrolyzer_physics_results, electrolyzer_cost_results, desal_results, h2_pipe_array_results, h2_transport_compressor, h2_transport_compressor_results, h2_transport_pipe_results, pipe_storage, h2_storage_results, total_accessory_power \
        = energy_internals(solver=False, power_for_peripherals_kw_in=solver_result)

    ## end solver loop here
    platform_results = run_equipment_platform(plant_config, design_scenario, electrolyzer_physics_results, h2_storage_results, desal_results, verbose=verbose)

    ################# OSW intermediate calculations" aka final financial calculations
    # does LCOE even make sense if we are only selling the H2? I think in this case LCOE should not be used, rather LCOH should be used. Or, we could use LCOE based on the electricity actually used for h2
    # I think LCOE is just being used to estimate the cost of the electricity used, but in this case we should just use the cost of the electricity generating plant since we are not selling to the grid. We
    # could build in a grid connection later such that we use LCOE for any purchased electricity and sell any excess electricity after H2 production
    # actually, I think this is what OSW is doing for LCOH

    # TODO double check full-system CAPEX
    capex, capex_breakdown = run_capex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, design_scenario, desal_results, platform_results, verbose=verbose)

    # TODO double check full-system OPEX
    opex_annual, opex_breakdown_annual = run_opex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, desal_results, platform_results, verbose=verbose, total_export_system_cost=capex_breakdown["electrical_export_system"])

    print("wind capacity factor: ", np.sum(hopp_results["combined_pv_wind_power_production_hopp"])*1E-3/(plant_config["plant"]["capacity"]*365*24))

    if use_profast:
        lcoe, pf_lcoe = run_profast_lcoe(plant_config, orbit_project, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
        lcoh_grid_only, pf_grid_only = run_profast_grid_only(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
        lcoh, pf_lcoh = run_profast_full_plant_model(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, incentive_option, design_scenario, verbose=verbose, show_plots=show_plots, save_plots=save_plots)

    ################# end OSW intermediate calculations
    power_breakdown = post_process_simulation(lcoe, lcoh, pf_lcoh, pf_lcoe, hopp_results, electrolyzer_physics_results, plant_config, h2_storage_results, capex_breakdown, opex_breakdown_annual, orbit_project, platform_results, desal_results, design_scenario, plant_design_scenario, incentive_option, solver_results=solver_results, show_plots=show_plots, save_plots=save_plots)#, lcoe, lcoh, lcoh_with_grid, lcoh_grid_only)

    # return
    if output_level == 0:
        return 0
    elif output_level == 1:
        return lcoh
    elif output_level == 2:
        return lcoh, lcoe, capex_breakdown, opex_breakdown_annual, pf_lcoh, electrolyzer_physics_results
    elif output_level == 3:
        return lcoh, lcoe, capex_breakdown, opex_breakdown_annual, pf_lcoh, electrolyzer_physics_results, pf_lcoe, power_breakdown

# run the stuff
if __name__ == "__main__":

    run_simulation(verbose=False, show_plots=False, save_plots=False,  use_profast=True, incentive_option=1, plant_design_scenario=1)
