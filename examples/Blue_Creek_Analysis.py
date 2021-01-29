import os
from dotenv import load_dotenv
from math import sin, pi
from hybrid.reopt import REopt
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so
import pandas as pd
import json

from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from tools.analysis import create_cost_calculator


# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Set up output dataframe
save_all_runs = pd.DataFrame()
save_outputs_loop = dict()
save_outputs_loop['Count'] = list()
save_outputs_loop['Site Lat'] = list()
save_outputs_loop['Site Lon'] = list()
save_outputs_loop['ATB Year'] = list()
save_outputs_loop['Resource Year'] = list()
save_outputs_loop['Site Name'] = list()
save_outputs_loop['PTC'] = list()
save_outputs_loop['ITC'] = list()
save_outputs_loop['Hub Height (m)'] = list()
save_outputs_loop['Useful Life'] = list()
save_outputs_loop['Storage Enabled'] = list()
save_outputs_loop['Wind Cost kW'] = list()
save_outputs_loop['Solar Cost kW'] = list()
save_outputs_loop['Storage Cost kW'] = list()
save_outputs_loop['Site Lon'] = list()
save_outputs_loop['Wind MW built'] = list()
save_outputs_loop['Solar MW built'] = list()
save_outputs_loop['Storage MW built'] = list()
save_outputs_loop['Wind AEP (GWh)'] = list()
save_outputs_loop['Wind Total Installed Cost ($)'] = list()
save_outputs_loop['PPA Price'] = list()
save_outputs_loop['LCOE'] = list()
save_outputs_loop['IRR'] = list()
save_outputs_loop['Turbine Choice'] = list()
save_outputs_loop['Turbine Hub Height'] = list()
save_outputs_loop['Turbine Rotor Diameter'] = list()
save_outputs_loop['Wind Speed Average'] = list()

# Get resource
site_name = 'Blue Creek'
lat = 40.966
lon = -84.598
year = 2013
sample_site['year'] = year
sample_site['lat'] = lat
sample_site['lon'] = lon
tower_height_list = [97, 98]  # [80, 90, 100, 110, 120, 130, 140]
site = SiteInfo(sample_site, hub_height=97)

# Set run parameters
storage_used = False
on_grid = True  # Storage can charge from grid by default
atb_year_list = ['custom_cost']  # 2020, 2025, 2030, 'custom_cost']  # Will determine cost

useful_life_list = [25]  # , 30, 35]
itc_avail_list = [True]  # ], False]
ptc_avail_list = [True]  # , False]
debt_equity_split = 90  # %debt
re_opt_optimized_sizing = False
# Desired sizing below will be used if REopt sizing optimization = False, or a solution is not found
desired_wind_installed_capacity_mw = 304
desired_solar_installed_capacity_mw = 0
desired_interconnect_capacity_mw = 304
# Desired cost/mw below will be used if not running a standard 2020, 2025, 2030 ATB cost scenario
desired_wind_cost_mw = 1974  # $/kw - 1428 Turbine, 544.5 BOS
desired_solar_cost_mw = 9999
# Additional Blue Creek-specific details
desired_wind_om_cost_kw_yr = 18  #hybrid_plant.wind.SystemCosts.om_capacity = 18
land_lease_payment_million_yr = 4.7  #Not sure where to implement this yet, possibly roll into O&M cost for cashflow equivalent
losses_exc_wake = 13.9  # Hard to specify straight loss - set all loss params (hybrid_plant.wind.system_model.Losses.xyz to zero and set turb_generic_loss to 13.9
turbine_choice = 'gamesa_g90_2mw_100hh' # Will use SAM power curve as it's "close enough" (~1-2%) Find where power curve comes from and make switchable
rotor_diameter = 90  # hybrid_plant.wind.system_model.Turbine.wind_turbine_rotor_diameter = 90
# Import powercurve
powercurve_filename = ['powercurve_{}'.format(turbine_choice)]
powercurve_file = open('powercurve_gamesa_g90_2mw_100hh')
powercurve_data = json.load(powercurve_file)
powercurve_file.close()


# Set up HOPP models
solar_model = SolarPlant(site, 20000)
wind_model = WindPlant(site, 20000)
fin_model = so.default("GenericSystemSingleOwner")
filepath = os.path.dirname(os.path.abspath(__file__))
fileout = os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json")
count = 0

# Set up REopt run
MTC02_yr = 10640  # Metric tons of CO2 per year
MTC02_yr_to_kw_continuous_conversion = 0.2779135  # Metric tons CO2-to kWe conversion (avg. US carbon intensity)
kw_continuous = MTC02_yr * MTC02_yr_to_kw_continuous_conversion
load = [kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

for useful_life in useful_life_list:
    for atb_year in atb_year_list:
        if atb_year == 'custom_cost':
            print("Custom ATB year")
            wind_cost_kw = desired_wind_cost_mw
            pv_cost_kw = desired_solar_cost_mw
            storage_cost_kw = 9999
        elif atb_year == 2020:
            print("I shoudln't be in here")
            wind_cost_kw = 1629  # (2020 = 1629, 2025 = 1301, 2030 = 940) # NREL 2020 ATB
            pv_cost_kw = 1340  # (2020 = 1340, 2025 = 1014, 2030 = 688)
            storage_cost_kw = 1455  # (2025 = 779, 2030 = 567)
        elif atb_year == 2025:
            wind_cost_kw = 1301  # (2020 = 1629, 2025 = 1301, 2030 = 940) # NREL 2020 ATB
            pv_cost_kw = 1014  # (2020 = 1340, 2025 = 1014, 2030 = 688)
            storage_cost_kw = 779  # (2025 = 779, 2030 = 567)
        elif atb_year == 2030:
            wind_cost_kw = 940  # (2020 = 1629, 2025 = 1301, 2030 = 940) # NREL 2020 ATB
            pv_cost_kw = 688  # (2020 = 1340, 2025 = 1014, 2030 = 688)
            storage_cost_kw = 567  # (2025 = 779, 2030 = 567)

        for ptc_avail in ptc_avail_list:
            for itc_avail in itc_avail_list:
                for tower_height in tower_height_list:
                    site = SiteInfo(sample_site, hub_height=tower_height)
                    count = count + 1

                    if re_opt_optimized_sizing:
                        reopt = REopt(lat=lat,
                                      lon=lon,
                                      load_profile=load,
                                      urdb_label=urdb_label,
                                      solar_model=solar_model,
                                      wind_model=wind_model,
                                      fin_model=fin_model,
                                      interconnection_limit_kw=20000,
                                      fileout=os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json"))

                        reopt.set_rate_path(os.path.join(filepath, 'data'))

                        reopt.post['Scenario']['Site']['Wind']['installed_cost_us_dollars_per_kw'] = wind_cost_kw  # ATB
                        reopt.post['Scenario']['Site']['PV']['installed_cost_us_dollars_per_kw'] = pv_cost_kw
                        reopt.post['Scenario']['Site']['Storage'] = {'min_kw': 0.0, 'max_kw': 100000.0, 'min_kwh': 0.0, 'max_kwh': 400000.0,
                                                                     'internal_efficiency_pct': 0.975, 'inverter_efficiency_pct': 0.96,
                                                                     'rectifier_efficiency_pct': 0.96, 'soc_min_pct': 0.2, 'soc_init_pct': 0.5,
                                                                     'canGridCharge': True, 'installed_cost_us_dollars_per_kw': storage_cost_kw,
                                                                     'installed_cost_us_dollars_per_kwh': 420.0,
                                                                     'replace_cost_us_dollars_per_kw': 410.0,
                                                                     'replace_cost_us_dollars_per_kwh': 200.0, 'inverter_replacement_year': 10,
                                                                     'battery_replacement_year': 10, 'macrs_option_years': 7,
                                                                     'macrs_bonus_pct': 1.0, 'macrs_itc_reduction': 0.5, 'total_itc_pct': 0.0,
                                                                     'total_rebate_us_dollars_per_kw': 0, 'total_rebate_us_dollars_per_kwh': 0}
                        reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_us_dollars_per_kwh'] = 0.05
                        reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_above_site_load_us_dollars_per_kwh'] = 0.0
                        reopt.post['Scenario']['Site']['Financial']['analysis_years'] = useful_life
                        if not storage_used:
                            reopt.post['Scenario']['Site']['Storage']['max_kw'] = 0
                        if ptc_avail:
                            reopt.post['Scenario']['Site']['Wind']['pbi_us_dollars_per_kwh'] = 0.022
                        else:
                            reopt.post['Scenario']['Site']['Wind']['pbi_us_dollars_per_kwh'] = 0.0
                        if itc_avail:
                            reopt.post['Scenario']['Site']['PV']['federal_itc_pct'] = 0.26
                        else:
                            reopt.post['Scenario']['Site']['PV']['federal_itc_pct'] = 0.0

                        reopt.post['Scenario']['Site']['LoadProfile']['doe_reference_name'] = "FlatLoad"
                        reopt.post['Scenario']['Site']['LoadProfile']['annual kwh'] = 8760 * 3000
                        reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = 1
                        reopt.post['Scenario']['Site']['LoadProfile']['outage_start_hour'] = 10
                        reopt.post['Scenario']['Site']['LoadProfile']['outage_end_hour'] = 8750
                        result = reopt.get_reopt_results(force_download=True)

                        solar_size_mw = result['outputs']['Scenario']['Site']['PV']['size_kw'] / 1000
                        wind_size_mw = result['outputs']['Scenario']['Site']['Wind']['size_kw'] / 1000
                        storage_size_mw = result['outputs']['Scenario']['Site']['Storage']['size_kw'] / 1000
                        interconnection_size_mw = reopt.interconnection_limit_kw / 1000
                    else:
                        solar_size_mw = desired_solar_installed_capacity_mw
                        wind_size_mw = desired_wind_installed_capacity_mw
                        storage_size_mw = 0
                        interconnection_size_mw = desired_interconnect_capacity_mw

                    # Set up HOPP technologies array
                    technologies = {'solar': solar_size_mw,  # mw system capacity
                                    'wind': wind_size_mw,  # mw system capacity
                                    'grid': interconnection_size_mw,
                                    'collection_system': True}

                    # Create model
                    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)
                    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                                              bos_cost_source='CostPerMW',
                                                                              wind_installed_cost_mw=wind_cost_kw*1000,
                                                                              solar_installed_cost_mw=pv_cost_kw*1000))

                    if solar_size_mw > 0:
                        hybrid_plant.solar.financial_model.FinancialParameters.analysis_period = useful_life
                        hybrid_plant.solar.financial_model.FinancialParameters.debt_percent = debt_equity_split
                        if itc_avail:
                            hybrid_plant.solar.financial_model.TaxCreditIncentives.itc_fed_percent = 26
                        else:
                            hybrid_plant.solar.financial_model.TaxCreditIncentives.itc_fed_percent = 0

                    if 'wind' in technologies:
                        hybrid_plant.wind.financial_model.FinancialParameters.analysis_period = useful_life
                        hybrid_plant.wind.financial_model.FinancialParameters.debt_percent = debt_equity_split
                        if ptc_avail:
                            ptc_val = 0.022
                        else:
                            ptc_val = 0.0

                        interim_list = list(
                            hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount)
                        interim_list[0] = ptc_val
                        hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)

                        # Add other project-specific parameters
                        hybrid_plant.wind.system_model.Turbine.wind_turbine_hub_ht = tower_height
                        hybrid_plant.wind.system_model.Turbine.wind_turbine_rotor_diameter = rotor_diameter
                        hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_windspeeds = \
                            powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
                        hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_powerout = \
                            powercurve_data['turbine_powercurve_specification']['turbine_power_output']

                    hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
                    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
                    hybrid_plant.ppa_price = 0.045
                    ptc_val = 0.022
                    interim_list = list(
                        hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount)
                    interim_list[0] = ptc_val
                    hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
                    # Run HybridSimulation
                    hybrid_plant.simulate(useful_life)

                    # Save the outputs
                    annual_energies = hybrid_plant.annual_energies
                    wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.solar
                    npvs = hybrid_plant.net_present_values
                    lcoe = hybrid_plant.lcoe_real.hybrid
                    irr = hybrid_plant.internal_rate_of_returns


                    wind_installed_cost = hybrid_plant.wind.financial_model.SystemCosts.total_installed_cost
                    solar_installed_cost = hybrid_plant.solar.financial_model.SystemCosts.total_installed_cost
                    hybrid_installed_cost = hybrid_plant.grid.financial_model.SystemCosts.total_installed_cost

                    print("Wind Cost per KW: {}".format(wind_cost_kw))
                    print("PV Cost per KW: {}".format(pv_cost_kw))
                    print("Storage Cost per KW: {}".format(storage_cost_kw))
                    print("Wind Size built: {}".format(wind_size_mw))
                    print("PV Size built: {}".format(solar_size_mw))
                    print("Storage Size built: {}".format(storage_size_mw))
                    print("Levelized cost of Electricity: {}".format(lcoe))

                    save_outputs_loop['Count'].append(count)
                    save_outputs_loop['Site Lat'].append(lat)
                    save_outputs_loop['Site Lon'].append(lon)
                    save_outputs_loop['ATB Year'].append(atb_year)
                    save_outputs_loop['Resource Year'].append(year)
                    save_outputs_loop['Site Name'].append(site_name)
                    save_outputs_loop['PTC'].append(ptc_avail)
                    save_outputs_loop['ITC'].append(itc_avail)
                    save_outputs_loop['Hub Height (m)'].append(tower_height)
                    save_outputs_loop['Useful Life'].append(useful_life)
                    save_outputs_loop['Storage Enabled'].append(storage_used)
                    save_outputs_loop['Wind Cost kW'].append(wind_cost_kw)
                    save_outputs_loop['Solar Cost kW'].append(pv_cost_kw)
                    save_outputs_loop['Storage Cost kW'].append(storage_cost_kw)
                    save_outputs_loop['Wind MW built'].append(wind_size_mw)
                    save_outputs_loop['Solar MW built'].append(solar_size_mw)
                    save_outputs_loop['Storage MW built'].append(storage_size_mw)
                    save_outputs_loop['Wind AEP (GWh)'].append(hybrid_plant.annual_energies.wind)
                    save_outputs_loop['Wind Total Installed Cost ($)'].append(hybrid_plant.wind.financial_model.SystemCosts.total_installed_cost)
                    save_outputs_loop['PPA Price'].append(hybrid_plant.ppa_price)
                    save_outputs_loop['LCOE'].append(lcoe)
                    save_outputs_loop['IRR'].append(irr)
                    save_outputs_loop['Turbine Choice'].append(turbine_choice)
                    save_outputs_loop['Turbine Hub Height'].append(tower_height)
                    save_outputs_loop['Turbine Rotor Diameter'].append(rotor_diameter)
                    save_outputs_loop['Wind Speed Average'].append(hybrid_plant.wind.system_model.Outputs.wind_speed_average)



save_outputs_loop_df = pd.DataFrame(save_outputs_loop)
save_outputs_loop_df.to_csv("As_Built_Analysis_{}.csv".format(site_name))

