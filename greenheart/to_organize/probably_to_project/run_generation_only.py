import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site
from hopp.utilities.keys import set_developer_nrel_gov_key
from greenheart.to_organize.H2_Analysis import simple_dispatch
from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from greenheart.to_organize import hopp_tools
from greenheart.to_organize import inputs_py
import copy 
from greenheart.to_organize import plot_results

"""
Perform a LCOE analysis for a few locations across the U.S. to demonstrate analysis across locations  using HOPP
A few notes:
# 1. physical interaction effects ignored currently 
# 2. shared BOS to be re-implemented
"""

#Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

#Step 1: User Inputs for scenario
resource_year = 2013
atb_years = [
            2020,
            2030,
            2050
            ]
policy = {
    'option 1': {'Wind ITC': 0, 'Wind PTC': 0},
    'option 2': {'Wind ITC': 26, 'Wind PTC': 0},
    'option 3': {'Wind ITC': 6, 'Wind PTC': 0},
    'option 4': {'Wind ITC': 30, 'Wind PTC': 0},
    'option 5': {'Wind ITC': 50, 'Wind PTC': 0},
}

sample_site['year'] = resource_year
useful_life = 30
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = True    #A flag that is applicable when using PySam WindPower (not FLORIS)
storage_used = True
battery_can_grid_charge = False
grid_connected_hopp = False
floris = True

# Technology sizing
interconnection_size_mw = 1000
wind_size_mw = 1000
solar_size_mw = 500
storage_size_mw = 100
storage_size_mwh = 400 # 100 MW, 4 hr

scenario_choice = 'Example HOPP buildout'

site_selection = [
                'Site 1',
                'Site 2',
                'Site 3',
                'Site 4'
                ]
scenario = dict()
kw_continuous = (wind_size_mw + solar_size_mw) * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

#Site lat and lon will be set by data loaded from Orbit runs

# Financial inputs
discount_rate = 0.07
debt_equity_split = 60

# These inputs are not used in this analysis (no solar or storage)
solar_cost_kw = 640  # TODO: take from input file for different years (fine if just looking at current year)
storage_cost_kw = 1500
storage_cost_kwh = 380

# Flags (TODO: update documentation)
forced_sizes = True # no REopt

# Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
# sell_price = 0.01
# buy_price = 0.01
sell_price = False
buy_price = False

# Set paths for results, floris and orbit
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/hybrids/results/'
floris_dir = parent_path + '/floris_input_files/'

print('Parent path = ', parent_path)

# Site specific turbine information
path = ('examples/hybrids/location_based_costs.xlsx')
xl = pd.ExcelFile(path)

# outputs
save_outputs_dict = inputs_py.establish_save_output_dict()
save_all_runs = list()

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = True
plot_wind = True
print_results = True


for i in policy:
    # set policy values
    scenario, policy_option = hopp_tools.set_policy_values(scenario, policy, i)
    print('Wind PTC: ', scenario['Wind PTC'])

    for atb_year in atb_years:
        for site_location in site_selection:
            scenario_df = xl.parse()
            scenario_df.set_index(["Parameter"], inplace = True)
            
            site_df = scenario_df[site_location]

            turbine_model = str(site_df['Turbine Rating'])+'MW'

            # set turbine values
            scenario, nTurbs, floris_config = hopp_tools.set_turbine_model(turbine_model, scenario, parent_path,floris_dir, floris)

            scenario['Useful Life'] = useful_life

            # financials
            scenario = hopp_tools.set_financial_info(scenario, debt_equity_split, discount_rate)

            # site info
            site_df, sample_site = hopp_tools.set_site_info(site_df, sample_site)
            site_name = site_df['State']
            site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])

            #Assign scenario cost details (TODO: hard-coded from spreadsheet - not a standardized way)
            if atb_year == 2020:
                total_capex = site_df['2020 CapEx']
                wind_om_cost_kw = site_df['2020 OpEx ($/kw-yr)']
            if atb_year == 2030:
                total_capex = site_df['2030 CapEx']
                wind_om_cost_kw = site_df['2030 OpEx ($/kw-yr)']
            if atb_year == 2050:
                total_capex = site_df['2050 CapEx']
                wind_om_cost_kw = site_df['2050 OpEx ($/kw-yr)']

            capex_multiplier = site_df['CapEx Multiplier']
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier

            #Plot Wind Data to ensure offshore data is sound
            wind_data = site.wind_resource._data['data']
            wind_speed = [W[2] for W in wind_data]
            plot_results.plot_wind_results(wind_data, site_name, site_df['Representative coordinates'], results_dir, plot_wind)

            
            # Run HOPP
            combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe = \
                hopp_tools.run_HOPP(scenario,
                                    site,
                                    sample_site,
                                    forced_sizes,
                                    solar_size_mw,
                                    wind_size_mw,
                                    storage_size_mw,
                                    storage_size_mwh,
                                    wind_cost_kw, 
                                    solar_cost_kw, 
                                    storage_cost_kw, 
                                    storage_cost_kwh,
                                    kw_continuous, 
                                    load,
                                    interconnection_size_mw,
                                    wind_om_cost_kw,
                                    nTurbs,
                                    floris_config,
                                    floris)

            generation_summary_df = pd.DataFrame({'Generation profile (kW)': hybrid_plant.grid.generation_profile[0:8760] })
            generation_summary_df.to_csv(os.path.join(results_dir, 'Generation Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,scenario['Powercurve File'])))

            #Step 4: Plot HOPP Results
            plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
                                    energy_shortfall_hopp,
                                    combined_pv_wind_curtailment_hopp,
                                    load,
                                    results_dir,
                                    site_name,
                                    atb_year,
                                    turbine_model,
                                    hybrid_plant,
                                    plot_power_production)

            #Step 5: Run Simple Dispatch Model
            combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used, excess_energy = \
                hopp_tools.run_battery(energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, combined_pv_wind_power_production_hopp)
            
            plot_results.plot_battery_results(combined_pv_wind_curtailment_hopp, 
                        energy_shortfall_hopp,
                        combined_pv_wind_storage_power_production_hopp,
                        combined_pv_wind_power_production_hopp,
                        battery_SOC,
                        battery_used,
                        results_dir,
                        site_name,atb_year,turbine_model,
                        load,
                        plot_battery)

            

            # grid information (on and off-grid systems)
            cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer = hopp_tools.grid(combined_pv_wind_storage_power_production_hopp,
                                                                                    sell_price,
                                                                                    excess_energy,
                                                                                    buy_price,
                                                                                    kw_continuous,
                                                                                    plot_grid)

            # calculate financials simple (no H2)
            total_elec_production = np.sum(combined_pv_wind_storage_power_production_hopp) 
            cf = total_elec_production / (interconnection_size_mw * 1000 * 8760)
            # plt.plot(energy_to_electrolyzer)
            # plt.show()

            print(scenario)

            if print_results:
                # ------------------------- #
                #TODO: Tidy up these print statements
                print("Future Scenario: {}".format(site_location))
                print("Wind CapEx Cost per KW: {}".format(total_capex))
                print("PV Cost per KW: {}".format(solar_cost_kw))
                print("Storage Cost per KW: {}".format(storage_cost_kw))
                print("Storage Cost per KWh: {}".format(storage_cost_kwh))
                print("Wind Size built: {}".format(wind_size_mw))
                print("PV Size built: {}".format(solar_size_mw))
                print("Storage Size built: {}".format(storage_size_mw))
                print("Storage Size built: {}".format(storage_size_mwh))
                print("Total Yearly Electrical Output: {}".format(total_elec_production))
                print('Combined Capacity Factor: ', cf)
                print("Levelized cost of Electricity (cents/kWh): {}".format(lcoe))
                print('=========================================================')


print('Done')