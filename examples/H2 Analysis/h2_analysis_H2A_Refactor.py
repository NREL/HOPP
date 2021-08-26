import sys
import os
from dotenv import load_dotenv
from math import sin, pi
# from hybrid.reopt import REopt
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so
import pandas as pd
import pickle
import json
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from plot_reopt_results import plot_reopt_results
from run_reopt import run_reopt
from hopp_for_h2 import hopp_for_h2
from run_h2a import run_h2a
from simple_dispatch import SimpleDispatch
import numpy as np
from lcoe.lcoe import lcoe as lcoe_calc

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Site Name'] = list()
    save_outputs_dict['Scenario Choice'] = list()
    save_outputs_dict['Site Lat'] = list()
    save_outputs_dict['Site Lon'] = list()
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['kW continuous load'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['PTC'] = list()
    save_outputs_dict['ITC'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['Storage Enabled'] = list()
    save_outputs_dict['Wind Cost kW'] = list()
    save_outputs_dict['Solar Cost kW'] = list()
    save_outputs_dict['Storage Cost kW'] = list()
    save_outputs_dict['Storage Cost kWh'] = list()
    save_outputs_dict['Storage Hours'] = list()
    save_outputs_dict['Wind MW built'] = list()
    save_outputs_dict['Solar MW built'] = list()
    save_outputs_dict['Storage MW built'] = list()
    save_outputs_dict['Storage MWh built'] = list()
    save_outputs_dict['Electrolyzer Size'] = list()
    save_outputs_dict['Electrolyzer Total System Size'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['Built Interconnection Size'] = list()
    save_outputs_dict['REOpt Interconnection Size'] = list()
    save_outputs_dict['Total Installed Cost $(HOPP)'] = list()
    save_outputs_dict['Total Yearly Electrical Output'] = list()
    save_outputs_dict['LCOE'] = list()
    save_outputs_dict['Total Annual H2 production (kg)'] = list()
    save_outputs_dict['Levelized Cost H2/kg (new method - no operational costs)'] = list()
    save_outputs_dict['Levelized Cost H2/kg (new method - with operational costs)'] = list()
    save_outputs_dict['Levelized cost of H2 (excl. electricity) (H2A)'] = list()
    save_outputs_dict['Levelized H2 Elec Feedstock Cost/kg (HOPP)'] = list()
    save_outputs_dict['Total H2 cost/kg'] = list()
    save_outputs_dict['H2 Elec Feedstock Cost/kg (HOPP) Net Cap Cost Method'] = list()
    save_outputs_dict['H2A scaled total install cost'] = list()
    save_outputs_dict['H2A scaled total install cost per kw'] = list()
    save_outputs_dict['REOpt Energy Shortfall'] = list()
    save_outputs_dict['REOpt Curtailment'] = list()
    save_outputs_dict['HOPP Total Generation'] = list()
    save_outputs_dict['HOPP Energy Shortfall'] = list()
    save_outputs_dict['HOPP Curtailment'] = list()
    save_outputs_dict['Battery Generation'] = list()
    save_outputs_dict['Electricity to Grid'] = list()

    return save_outputs_dict


#TODO:
# - *Functionize output dataframe setup
# - *Streamline inputs in code versus from scenario csv file
# - *Functionize REopt result plotting
# - - *Add a warning to the REopt plots when REopt has not been run
# - Functionize the REopt run and results return (Add new inputs inc: forced min wind size, forced solar size, battery on or off, on grid or off grid)
# - - *Fix naming of pre-computes to be specific or as general as needed
# - - *Condense all outputs (e.g. wind_size, solar_size etc to REoptResultsDF)
# - *Functionize HOPP run
# - Functionize H2A run
# - - Add forced electrolyzer size and price.
# - - Add kg h2 output (as an input) and kw continuous load input
# - Functionize Output writing


# Step 1: Establish output structure and special inputs
# save_all_runs = pd.DataFrame()
save_outputs_dict = establish_save_output_dict()
year = 2013
sample_site['year'] = year
useful_life = 30
critical_load_factor_list = [0.9]
run_reopt_flag = False

custom_powercurve = True
storage_used = False
battery_can_grid_charge = False
grid_connected_hopp = True
kw_continuous = 5000
electrolyzer_sizes = [50, 100, 150, 200]
interconnection_size_mw = 100
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

# Step 2: Load scenarios from .csv and enumerate
scenarios_df = pd.read_csv('H2 Baseline Future Scenarios Test Refactor.csv')
for electrolyzer_size in electrolyzer_sizes:
    for critical_load_factor in critical_load_factor_list:
        for i, scenario in scenarios_df.iterrows():
            # TODO: Make scenario_choice, lookup all other values from dataframe from csv.

            # TODO:
            # -Pass through rotor diameter to pySAM
            # -Add wind, solar, storage installed costs
            # -Fix "H2 H2 xxx" text

            print(scenario)
            scenario_choice = scenario['Scenario Name']
            site_name = scenario['Site Name']
            sample_site['lat'] = scenario['Lat']
            sample_site['lon'] = scenario['Long']
            lat = scenario['Lat']
            lon = scenario['Long']
            atb_year = scenario['ATB Year']
            ptc_avail = scenario['PTC Available']
            itc_avail = scenario['ITC Available']
            forced_sizes = scenario['Force Plant Size']
            if forced_sizes:
                forced_wind_size = scenario['Wind Size MW']
                forced_solar_size = scenario['Solar Size MW']
            tower_height = scenario['Tower Height']
            rotor_diameter = scenario['Rotor Diameter']
            turbine_rating = scenario['Turbine Rating']
            wind_cost_kw = scenario['Wind Cost KW']
            custom_powercurve_path = scenario['Powercurve File']
            solar_cost_kw = scenario['Solar Cost KW']
            storage_cost_kw = scenario['Storage Cost kW']
            storage_cost_kwh = scenario['Storage Cost kWh']
            debt_equity_split = scenario['Debt Equity']

            #Todo: Add useful life to .csv scenario input instead
            scenario['Useful Life'] = useful_life

            site = SiteInfo(sample_site, hub_height=tower_height)

            # Step 3: Set up REopt run
            # ------------------------- #
            wind_size_mw, solar_size_mw, storage_size_mw,\
            storage_size_mwh, storage_hours, reopt_results, REoptResultsDF = run_reopt(site, scenario, load,
                                                         interconnection_size_mw*1000,
                                                         critical_load_factor, useful_life,
              battery_can_grid_charge, storage_used, run_reopt_flag)

            # Step 4: Set up HOPP run
            # ------------------------- #
            if forced_sizes:
                solar_size_mw = forced_solar_size
                wind_size_mw = forced_wind_size

            technologies = {'solar': solar_size_mw,  # mw system capacity
                            'wind': wind_size_mw,  # mw system capacity
                            'grid': interconnection_size_mw,
                            'collection_system': True}

            hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
            energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
                hopp_for_h2(site, scenario, technologies,
                            wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                kw_continuous, load,
                custom_powercurve,
                rotor_diameter,
                interconnection_size_mw, grid_connected_hopp=True)

            wind_installed_cost = hybrid_plant.wind.financial_model.SystemCosts.total_installed_cost
            solar_installed_cost = hybrid_plant.solar.financial_model.SystemCosts.total_installed_cost
            hybrid_installed_cost = hybrid_plant.grid.financial_model.SystemCosts.total_installed_cost

            # Step 5: Run Simple Dispatch Model
            # ------------------------- #
            bat_model = SimpleDispatch(combined_pv_wind_curtailment_hopp, energy_shortfall_hopp, len(energy_shortfall_hopp),
                                       storage_size_mw * 1000)

            battery_used, excess_energy, battery_SOC = bat_model.run()
            combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + excess_energy

            # Step 6: Run the Python H2A model
            # ------------------------- #
            #TODO: Refactor H2A model call
            # Should take as input (electrolyzer size, cost, electrical timeseries, total system electrical usage (kwh/kg),
            # Should give as ouptut (h2 costs by net cap cost, levelized, total_unit_cost of hydrogen etc)   )

            electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp

            H2_Results, H2A_Results = run_h2a(electrical_generation_timeseries, kw_continuous, electrolyzer_size,
                                  hybrid_plant, reopt_results, scenario,
            combined_pv_wind_curtailment_hopp, lcoe, total_system_electrical_usage=55.5)

            # Step 6.5: Intermediate financial calculation
            #TODO:
            # - Get Hybrid installed cost (wind, solar, storage)
            # - Get total amount of H2 produced
            # - Direct H2/kg
            # - Levelized H2/kg
            # operating_cost = 25000000  # $million/year
            # capital_cost = 500000000  # $million
            # discount_rate = 0.07  # %
            # lifetime = 20
            # annual_output = 2000000000  # kWh
            # NB annual output in kWh (but unit doesn't matter)
            # lcoe(annual_output, capital_cost, operating_cost, discount_rate, lifetime)


            total_elec_production = np.sum(electrical_generation_timeseries) #REMOVE
            total_hopp_installed_cost = hybrid_plant.grid.financial_model.SystemCosts.total_installed_cost
            total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
            total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
            annual_operating_cost_hopp = (wind_size_mw * 1000 * 42) + (solar_size_mw * 1000 * 13)
            annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
            total_annual_operating_costs = annual_operating_cost_hopp + annual_operating_cost_h2
            h_lcoe_no_op_cost = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                               0, 0.07, useful_life)

            h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                               total_annual_operating_costs, 0.07, useful_life)

            # Step 7: Print  results
            # ------------------------- #
            #TODO: Tidy up these print statements
            print("Future Scenario: {}".format(scenario['Scenario Name']))
            print("Wind Cost per KW: {}".format(scenario['Wind Cost KW']))
            print("PV Cost per KW: {}".format(scenario['Solar Cost KW']))
            print("Storage Cost per KW: {}".format(scenario['Storage Cost kW']))
            print("Storage Cost per KWh: {}".format(scenario['Storage Cost kWh']))
            print("Wind Size built: {}".format(wind_size_mw))
            print("PV Size built: {}".format(solar_size_mw))
            print("Storage Size built: {}".format(storage_size_mw))
            print("Storage Size built: {}".format(storage_size_mwh))
            print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
            print("Total Yearly Electrical Output: {}".format(total_elec_production))
            print("Total Yearly Hydrogen Production: {}".format(H2_Results['hydrogen_annual_output']))
            print("Levelized Cost H2/kg (new method - no operational costs)".format(h_lcoe_no_op_cost))
            print("Capacity Factor of Electrolyzer: {}".format(H2_Results['cap_factor']))

            print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
                H2_Results['feedstock_cost_h2_levelized_hopp']))
            print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(H2A_Results['Total Hydrogen Cost ($/kgH2)']))
            print("Total unit cost of H2 ($/kg) : {}".format(H2_Results['total_unit_cost_of_hydrogen']))
            print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(
                H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp']))

            # Step 8: Plot REopt results
            plot_reopt_results(REoptResultsDF, site_name, atb_year, critical_load_factor,
                               useful_life, tower_height,
                               wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, lcoe,
                               H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp'],
                               H2_Results['feedstock_cost_h2_levelized_hopp'],
                               hybrid_installed_cost, H2A_Results['Total Hydrogen Cost ($/kgH2)'],
                               H2_Results['total_unit_cost_of_hydrogen'],
                               output_dir='results/',
                               monthly_separation=False, reopt_was_run=run_reopt_flag)

            # Step 9: Plot Hydrogen Production profile


            # Step 9: Save outputs
            # ------------------------- #
            #TODO: Place in function
            save_outputs_dict['Site Name'].append(site_name)
            save_outputs_dict['Scenario Choice'].append(scenario_choice)
            save_outputs_dict['Site Lat'].append(lat)
            save_outputs_dict['Site Lon'].append(lon)
            save_outputs_dict['ATB Year'].append(atb_year)
            save_outputs_dict['Resource Year'].append(year)
            save_outputs_dict['Critical Load Factor'].append(critical_load_factor)
            save_outputs_dict['kW continuous load'].append(kw_continuous)
            save_outputs_dict['Useful Life'].append(useful_life)
            save_outputs_dict['PTC'].append(ptc_avail)
            save_outputs_dict['ITC'].append(itc_avail)
            save_outputs_dict['Hub Height (m)'].append(tower_height)
            save_outputs_dict['Storage Enabled'].append(storage_used)
            save_outputs_dict['Wind Cost kW'].append(wind_cost_kw)
            save_outputs_dict['Solar Cost kW'].append(solar_cost_kw)
            save_outputs_dict['Storage Cost kW'].append(storage_cost_kw)
            save_outputs_dict['Storage Cost kWh'].append(storage_cost_kwh)
            save_outputs_dict['Storage Hours'].append(storage_hours)
            save_outputs_dict['Wind MW built'].append(wind_size_mw)
            save_outputs_dict['Solar MW built'].append(solar_size_mw)
            save_outputs_dict['Storage MW built'].append(storage_size_mw)
            save_outputs_dict['Storage MWh built'].append(storage_size_mwh)
            save_outputs_dict['Battery Can Grid Charge'].append(battery_can_grid_charge)
            save_outputs_dict['Built Interconnection Size'].append(hybrid_plant.interconnect_kw)
            save_outputs_dict['REOpt Interconnection Size'].append(interconnection_size_mw*1000)
            save_outputs_dict['Total Installed Cost $(HOPP)'].append(total_hopp_installed_cost)
            save_outputs_dict['Total Yearly Electrical Output'].append(total_elec_production)
            save_outputs_dict['LCOE'].append(lcoe)
            save_outputs_dict['Total Annual H2 production (kg)'].append(H2_Results['hydrogen_annual_output'])
            save_outputs_dict['Levelized Cost H2/kg (new method - no operational costs)'].append(h_lcoe_no_op_cost)
            save_outputs_dict['Levelized Cost H2/kg (new method - with operational costs)'].append(h_lcoe)
            save_outputs_dict['Levelized H2 Elec Feedstock Cost/kg (HOPP)'].append(H2_Results['feedstock_cost_h2_levelized_hopp'])
            save_outputs_dict['Levelized cost of H2 (excl. electricity) (H2A)'].append(H2A_Results['Total Hydrogen Cost ($/kgH2)'])
            save_outputs_dict['Total H2 cost/kg'].append(H2_Results['total_unit_cost_of_hydrogen'])
            save_outputs_dict['H2 Elec Feedstock Cost/kg (HOPP) Net Cap Cost Method'].append(H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp'])
            save_outputs_dict['REOpt Energy Shortfall'].append(np.sum(REoptResultsDF['energy_shortfall']))
            save_outputs_dict['REOpt Curtailment'].append(np.sum(REoptResultsDF['combined_pv_wind_curtailment']))
            save_outputs_dict['Grid Connected HOPP'].append(grid_connected_hopp)
            save_outputs_dict['HOPP Total Generation'].append(np.sum(hybrid_plant.grid.generation_profile_from_system[0:8759]))
            save_outputs_dict['HOPP Energy Shortfall'].append(np.sum(energy_shortfall_hopp))
            save_outputs_dict['HOPP Curtailment'].append(np.sum(combined_pv_wind_curtailment_hopp))
            save_outputs_dict['Battery Generation'].append(np.sum(battery_used))
            save_outputs_dict['Electricity to Grid'].append(np.sum(excess_energy))
            save_outputs_dict['Electrolyzer Size'].append(H2A_Results['electrolyzer_size'])
            save_outputs_dict['Electrolyzer Total System Size'].append(H2A_Results['total_plant_size'])
            save_outputs_dict['H2A scaled total install cost'].append(H2A_Results['scaled_total_installed_cost'])
            save_outputs_dict['H2A scaled total install cost per kw'].append(H2A_Results['scaled_total_installed_cost_kw'])


    # save_all_runs = save_all_runs.append(save_outputs_dict, sort=False)

# Create dataframe from outputs and save
save_outputs_dict_df = pd.DataFrame(save_outputs_dict)
save_outputs_dict_df.to_csv("results/H2_Analysis_{}.csv".format(site_name))
