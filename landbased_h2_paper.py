import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
from tools.resource import *
from tools.resource.resource_loader import site_details_creator
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    # save_outputs_dict['Site Name'] = list()
    # save_outputs_dict['Scenario Choice'] = list()
    # save_outputs_dict['Site Lat'] = list()
    # save_outputs_dict['Site Lon'] = list()
    # save_outputs_dict['ATB Year'] = list()
    # save_outputs_dict['Resource Year'] = list()
    # save_outputs_dict['Turbine Model'] = list()
    # save_outputs_dict['Critical Load Factor'] = list()
    # save_outputs_dict['kW continuous load'] = list()
    # save_outputs_dict['Useful Life'] = list()
    # save_outputs_dict['PTC'] = list()
    # save_outputs_dict['ITC'] = list()
    # save_outputs_dict['Discount Rate'] = list()
    # save_outputs_dict['Debt Equity'] = list()
    # save_outputs_dict['Hub Height (m)'] = list()
    # save_outputs_dict['Storage Enabled'] = list()
    # save_outputs_dict['Wind Cost kW'] = list()
    # save_outputs_dict['Solar Cost kW'] = list()
    # save_outputs_dict['Storage Cost kW'] = list()
    # save_outputs_dict['Storage Cost kWh'] = list()
    # save_outputs_dict['Storage Hours'] = list()
    # save_outputs_dict['Wind MW built'] = list()
    # save_outputs_dict['Solar MW built'] = list()
    # save_outputs_dict['Storage MW built'] = list()
    # save_outputs_dict['Storage MWh built'] = list()
    # save_outputs_dict['Electrolyzer Stack Size'] = list()
    # save_outputs_dict['Electrolyzer Total System Size'] = list()
    # save_outputs_dict['Battery Can Grid Charge'] = list()
    # save_outputs_dict['Grid Connected HOPP'] = list()
    # save_outputs_dict['Built Interconnection Size'] = list()
    # save_outputs_dict['Total Installed Cost $(HOPP)'] = list()
    # save_outputs_dict['Total Yearly Electrical Output'] = list()
    # save_outputs_dict['LCOE'] = list()
    # save_outputs_dict['Total Annual H2 production (kg)'] = list()
    # save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = list()
    # # save_outputs_dict['Levelized Cost/kg H2 (lcoe using installed and operation costs)'] = list()
    # save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = list()
    # save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = list()
    # save_outputs_dict['H2A scaled total install cost'] = list()
    # save_outputs_dict['H2A scaled total install cost per kw'] = list()
    # save_outputs_dict['HOPP Total Generation'] = list()
    # save_outputs_dict['Wind Capacity Factor'] = list()
    # save_outputs_dict['HOPP Energy Shortfall'] = list()
    # save_outputs_dict['HOPP Curtailment'] = list()
    # save_outputs_dict['Battery Generation'] = list()
    # save_outputs_dict['Electricity to Grid'] = list()
    
    return save_outputs_dict

def save_the_things():
    save_outputs_dict['Site Name'].append(site_name)
    save_outputs_dict['Scenario Choice'].append(scenario_choice)
    save_outputs_dict['Site Lat'].append(lat)
    save_outputs_dict['Site Lon'].append(lon)
    save_outputs_dict['ATB Year'].append(atb_year)
    save_outputs_dict['Resource Year'].append(resource_year)
    save_outputs_dict['Turbine Model'].append(turbine_name)
    save_outputs_dict['Critical Load Factor'].append(critical_load_factor)
    save_outputs_dict['kW continuous load'].append(kw_continuous)
    save_outputs_dict['Useful Life'].append(useful_life)
    save_outputs_dict['PTC'].append(ptc_avail)
    save_outputs_dict['ITC'].append(itc_avail)
    save_outputs_dict['Discount Rate'].append(discount_rate)
    save_outputs_dict['Debt Equity'].append(debt_equity_split)
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
    save_outputs_dict['Total Installed Cost $(HOPP)'].append(total_hopp_installed_cost)
    save_outputs_dict['Total Yearly Electrical Output'].append(total_elec_production)
    save_outputs_dict['LCOE'].append(lcoe)
    save_outputs_dict['Total Annual H2 production (kg)'].append(H2_Results['hydrogen_annual_output'])
    save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'].append(gut_check_h2_cost_kg)
    # save_outputs_dict['Levelized Cost/kg H2 (lcoe using installed and operation costs)'].append(h_lcoe)
    save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'].append(LCOH_cf_method)
    save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'].append(LCOH_cf_method_w_operating_costs)
    save_outputs_dict['Grid Connected HOPP'].append(grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'].append(np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'].append(hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'].append(np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'].append(np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'].append(np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'].append(np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size'].append(H2A_Results['electrolyzer_size'])
    save_outputs_dict['Electrolyzer Total System Size'].append(H2A_Results['total_plant_size'])
    save_outputs_dict['H2A scaled total install cost'].append(H2A_Results['scaled_total_installed_cost'])
    save_outputs_dict['H2A scaled total install cost per kw'].append(H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict
def save_the_things_alt():
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Scenario Choice'] = (scenario_choice)
    save_outputs_dict['Site Lat'] = (lat)
    save_outputs_dict['Site Lon'] = (lon)
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_name)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['kW continuous load'] = (kw_continuous)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['PTC'] = (ptc_avail)
    save_outputs_dict['ITC'] = (itc_avail)
    save_outputs_dict['Discount Rate'] = (discount_rate)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['Storage Enabled'] = (storage_used)
    save_outputs_dict['Wind Cost kW'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost kW'] = (solar_cost_kw)
    save_outputs_dict['Storage Cost kW'] = (storage_cost_kw)
    save_outputs_dict['Storage Cost kWh'] = (storage_cost_kwh)
    save_outputs_dict['Storage Hours'] = (storage_hours)
    save_outputs_dict['Wind MW built'] = (wind_size_mw)
    save_outputs_dict['Solar MW built'] = (solar_size_mw)
    save_outputs_dict['Storage MW built'] = (storage_size_mw)
    save_outputs_dict['Storage MWh built'] = (storage_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    save_outputs_dict['Total Yearly Electrical Output'] = (total_elec_production)
    save_outputs_dict['LCOE'] = (lcoe)
    save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = (gut_check_h2_cost_kg)
    # save_outputs_dict['Levelized Cost/kg H2 (lcoe using installed and operation costs)'].append(h_lcoe)
    save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method)
    save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size'] = (H2A_Results['electrolyzer_size'])
    save_outputs_dict['Electrolyzer Total System Size'] = (H2A_Results['total_plant_size'])
    save_outputs_dict['H2A scaled total install cost'] = (H2A_Results['scaled_total_installed_cost'])
    save_outputs_dict['H2A scaled total install cost per kw'] = (H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict



load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env


#Establish Site Details
resource_year = 2013
atb_years = [2022,2025,2030,2035]
ptc_options = ['yes', 'no']
N_lat = 50 #50 # number of data points
N_lon = 95 #95
desired_lats = np.linspace(23.833504, 49.3556, N_lat)
desired_lons = np.linspace(-129.22923, -65.7146, N_lon)
load_resource_from_file = False
resource_dir = Path(__file__).parent.parent.parent / "resource_files"
sitelist_name = 'filtered_site_details_{}_lats_{}_lons_{}_resourceyear'.format(N_lat, N_lon, resource_year)

if load_resource_from_file:
    # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
    site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, resource_year)  # Return contains
    site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))
    site_details = filter_sites(site_details, location='usa only')
else:
    # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
    # - to force API resource download)
    if os.path.exists(sitelist_name):
        site_details = pd.read_csv(sitelist_name)
    else:
        site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, resource_dir)
        # Filter to locations in USA
        site_details = filter_sites(site_details, location='usa only')
        site_details.to_csv(sitelist_name)
print("Site Details Created")

sample_site['year'] = resource_year
useful_life = 25
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = True
storage_used = True
battery_can_grid_charge = True
grid_connected_hopp = False
interconnection_size_mw = 100
electrolyzer_size = 100

# which plots to show
plot_power_production = False
plot_battery = False
plot_grid = False
plot_h2 = False
turbine_name = '2020ATB_7MW'
h2_model ='Simple'  
# h2_model = 'H2A'

scenario = dict()
kw_continuous = electrolyzer_size * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

scenario_choice = 'Onshore Wind-H2 Analysis'
# site_selection = 'Site 1'
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results/'

itc_avail = 'no'
discount_rate = 0.089
forced_sizes = True
force_electrolyzer_cost = True
forced_wind_size = 100
forced_solar_size = 0
forced_storage_size_mw = 0
forced_storage_size_mwh = 0

sell_price = False
buy_price = False

# Define Turbine Characteristics based on user selected turbine.
if turbine_name == '2020ATB_7MW':
    #This is the onshore option
    custom_powercurve_path = '2020ATB_NREL_Reference_7MW_200.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_12MW_214.html
    tower_height = 175
    rotor_diameter = 200
    turbine_rating_mw = 7
    wind_cost_kw = 1300
if turbine_name == '2020ATB_12MW':
    custom_powercurve_path = '2020ATB_NREL_Reference_12MW_214.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_12MW_214.html
    tower_height = 136
    rotor_diameter = 214
    turbine_rating_mw = 12
    wind_cost_kw = 1300
elif turbine_name == '2020ATB_15MW':
    custom_powercurve_path = '2020ATB_NREL_Reference_15MW_240.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_15MW_240.html
    tower_height = 150
    rotor_diameter = 240
    turbine_rating_mw = 15
    wind_cost_kw =  1300
elif turbine_name == '2020ATB_18MW':
    custom_powercurve_path = '2020ATB_NREL_Reference_18MW_263.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_18MW_263.html
    tower_height = 156
    rotor_diameter = 263
    turbine_rating_mw = 18
    wind_cost_kw = 1300
    
print("Powercurve Path: ", custom_powercurve_path)
# site_details = site_details.set_index('site_nums')
save_outputs_dict = establish_save_output_dict()
save_all_runs = list()

for ptc_avail in ptc_options:
    for atb_year in atb_years:
        for i, site_deet in enumerate(site_details.iterrows()):
            if i == 0: continue
            else:
                site_deet = site_deet[1]
                print(site_deet)
                lat = site_deet['Lat']
                lon = site_deet['Lon']
                sample_site['lat'] = lat
                sample_site['lon'] = lon
                sample_site['no_solar'] = True
                # sample_site['no_wind'] = False
                site = SiteInfo(sample_site, hub_height=tower_height)
                site_name = (str(lat)+","+str(lon))

            if atb_year == 2022:
                forced_electrolyzer_cost = 400
                wind_cost_kw = 1310
                solar_cost_kw = 9999
                storage_cost_kw = 219
                storage_cost_kwh = 286
                debt_equity_split = 60
                wind_om_cost_kw = 44
            elif atb_year == 2025:
                forced_electrolyzer_cost = 300
                wind_cost_kw = 1081
                solar_cost_kw = 9999
                storage_cost_kw = 162
                storage_cost_kwh = 211
                debt_equity_split = 60
                wind_om_cost_kw = 44
            elif atb_year == 2030:
                forced_electrolyzer_cost = 150
                wind_cost_kw = 704
                solar_cost_kw = 9999
                storage_cost_kw = 110
                storage_cost_kwh = 143
                debt_equity_split = 60
                wind_om_cost_kw = 44
            elif atb_year == 2035:
                forced_electrolyzer_cost = 100
                wind_cost_kw = 660
                solar_cost_kw = 9999
                storage_cost_kw = 103
                storage_cost_kwh = 134
                debt_equity_split = 60
                wind_om_cost_kw = 44

            scenario['Useful Life'] = useful_life
            scenario['Debt Equity'] = debt_equity_split
            scenario['PTC Available'] = ptc_avail
            scenario['ITC Available'] = itc_avail
            scenario['Discount Rate'] = discount_rate
            scenario['Tower Height'] = tower_height
            scenario['Powercurve File'] = custom_powercurve_path

            if forced_sizes:
                solar_size_mw = forced_solar_size
                wind_size_mw = forced_wind_size
                storage_size_mw = forced_storage_size_mw
                storage_size_mwh = forced_storage_size_mwh
                storage_hours = 0

            if storage_size_mw > 0:
                technologies = {
                                'wind':
                                    {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                        'turbine_rating_kw': turbine_rating_mw*1000,
                                        'hub_height': tower_height,
                                        'rotor_diameter': rotor_diameter},
                                'battery': {
                                    'system_capacity_kwh': storage_size_mwh * 1000,
                                    'system_capacity_kw': storage_size_mw * 1000
                                    }
                                }
            else:
                        technologies = {
                                'wind':
                                    {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                        'turbine_rating_kw': turbine_rating_mw*1000,
                                        'hub_height': tower_height,
                                        'rotor_diameter': rotor_diameter}
                                }

            hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
            energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
                hopp_for_h2(site, scenario, technologies,
                            wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                kw_continuous, load,
                custom_powercurve,
                electrolyzer_size, grid_connected_hopp=True, wind_om_cost_kw=wind_om_cost_kw)

            wind_installed_cost = hybrid_plant.wind.total_installed_cost
            if solar_size_mw > 0:
                solar_installed_cost = hybrid_plant.pv.total_installed_cost
            else:
                solar_installed_cost = 0
            hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

            print("HOPP run complete")

            bat_model = SimpleDispatch()
            bat_model.Nt = len(energy_shortfall_hopp)
            bat_model.curtailment = combined_pv_wind_curtailment_hopp
            bat_model.shortfall = energy_shortfall_hopp

            battery_used, excess_energy, battery_SOC = bat_model.run()
            combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used

            if sell_price:
                profit_from_selling_to_grid = np.sum(excess_energy)*sell_price
            else:
                profit_from_selling_to_grid = 0.0

            # buy_price = False # if you want to force no buy from grid
            if buy_price:
                cost_to_buy_from_grid = buy_price

                for i in range(len(combined_pv_wind_storage_power_production_hopp)):
                    if combined_pv_wind_storage_power_production_hopp[i] < kw_continuous:
                        cost_to_buy_from_grid += (kw_continuous-combined_pv_wind_storage_power_production_hopp[i])*buy_price
                        combined_pv_wind_storage_power_production_hopp[i] = kw_continuous
            else:
                cost_to_buy_from_grid = 0.0

            energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]

            electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
            electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

            adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
            #NB: adjusted_installed_cost does NOT include the electrolyzer cost
            useful_life = scenario['Useful Life']
            net_capital_costs = 0

            # system_rating = electrolyzer_size
            system_rating = wind_size_mw + solar_size_mw
            H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size,
                            kw_continuous,forced_electrolyzer_cost,lcoe,adjusted_installed_cost,useful_life,
                            net_capital_costs)


            H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']

            print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
            print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
            print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))

            from examples.H2_Analysis.desal_model import RO_desal

            water_usage_electrolyzer = H2_Results['water_hourly_usage']
            m3_water_per_kg_h2 = 0.01
            desal_system_size_m3_hr = electrolyzer_size * (1000/55.5) * m3_water_per_kg_h2
            est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate
            # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
            Power = electrical_generation_timeseries
            fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals = RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=30)
            print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size, desal_system_size_m3_hr))
            print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
            print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
            print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
            # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
            print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

            total_elec_production = np.sum(electrical_generation_timeseries)
            total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
            # total_hopp_installed_cost_pipeline = hybrid_plant_pipeline.grid._financial_model.SystemCosts.total_installed_cost
            total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
            print(H2A_Results['scaled_total_installed_cost_kw'])
            total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
            # total_system_installed_cost_pipeline = total_hopp_installed_cost_pipeline + total_electrolyzer_cost
            annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
            annual_operating_cost_desal = desal_opex
            total_annual_operating_costs =  annual_operating_cost_h2 + annual_operating_cost_desal + cost_to_buy_from_grid - profit_from_selling_to_grid


            h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                                total_annual_operating_costs, discount_rate, useful_life)

            # Cashflow Financial Calculation
            discount_rate = scenario['Discount Rate']
            cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
            if solar_size_mw > 0:
                cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
            else:
                cf_solar_annuals = np.zeros(useful_life)
            if h2_model == 'H2A':
                cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.  
            elif h2_model == 'Simple':
                electrolyzer_capex = forced_electrolyzer_cost*electrolyzer_size*1000
                electrolyzer_opex_without_replacements = electrolyzer_capex * 0.05
                electrolyzer_variable_costs = [H2_Results['hydrogen_annual_output']*0.024]*useful_life
                cf_h2_annuals = - np.add(simple_cash_annuals(useful_life, useful_life, electrolyzer_capex, electrolyzer_opex_without_replacements, 0.03),electrolyzer_variable_costs)
            print("CF H2 Annuals",cf_h2_annuals)

            cf_operational_annuals = [-total_annual_operating_costs for i in range(useful_life)]

            cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals[:len(cf_wind_annuals)]],['Wind', 'Solar', 'H2'])

            cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))

            #NPVs of wind, solar, H2
            npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
            npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
            npv_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
            print("NPV H2 Costs using {} model: {}".format(h2_model,npv_h2_costs))
            npv_operating_costs = npf.npv(discount_rate, cf_operational_annuals)
            npv_desal_costs = -desal_capex
            print("Desal CAPEX: ",desal_capex)

            npv_total_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs
            npv_total_costs_w_operating_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs+npv_operating_costs

            LCOH_cf_method_wind = -npv_wind_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            LCOH_cf_method_solar = -npv_solar_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            LCOH_cf_method_h2_costs = -npv_h2_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            LCOH_cf_method_desal_costs = -npv_desal_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            LCOH_cf_method_operating_costs = -npv_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)

            LCOH_cf_method = -npv_total_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            LCOH_cf_method_w_operating_costs = -npv_total_costs_w_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
            financial_summary_df = pd.DataFrame([scenario['Useful Life'], wind_cost_kw, solar_cost_kw, forced_electrolyzer_cost,
                                                    scenario['Debt Equity'], atb_year, ptc_avail, itc_avail,
                                                    discount_rate, npv_wind_costs, npv_solar_costs, npv_h2_costs, LCOH_cf_method, LCOH_cf_method_w_operating_costs],
                                                ['Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'Electrolyzer Cost KW', 'Debt Equity',
                                                    'ATB Year', 'PTC available', 'ITC available', 'Discount Rate', 'NPV Wind Expenses', 'NPV Solar Expenses', 'NPV H2 Expenses', 'LCOH cf method HVDC','LCOH cf method HVDC w/operating cost'])
            financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}.csv'.format(site_name,atb_year,ptc_avail)))

            # Gut Check H2 calculation (non-levelized)
            total_installed_and_operational_lifetime_cost = total_system_installed_cost + (30 * total_annual_operating_costs)
            lifetime_h2_production = useful_life * H2_Results['hydrogen_annual_output']
            gut_check_h2_cost_kg = total_installed_and_operational_lifetime_cost / lifetime_h2_production

            print("Gut Check H2 Cost:",gut_check_h2_cost_kg)
            print("HVDC Scenario: LCOH w/o Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method)
            print("HVDC Scenario: LCOH WITH Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_w_operating_costs)

            print_results = False
            print_h2_results = True
            save_outputs_dict = save_the_things_alt()
            save_all_runs.append(save_outputs_dict)
            save_outputs_dict = establish_save_output_dict()

if print_results:
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

if print_h2_results:
    print('Total Lifetime H2(kg) produced: {}'.format(lifetime_h2_production))
    print("Gut-check H2 cost/kg: {}".format(gut_check_h2_cost_kg))
    print("LCOH CF Method (doesn't include grid electricity cost if used)", LCOH_cf_method)
    print("LCOH CF Method (includes operating costs + electricity)", LCOH_cf_method_w_operating_costs)
# ------------------------- #

save_outputs = True
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    save_all_runs_df.to_csv(os.path.join(results_dir, "H2_Analysis_Landbased_All.csv"))



