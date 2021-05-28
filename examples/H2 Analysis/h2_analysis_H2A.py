import sys
sys.path.append('/Users/jannoni/Desktop/Desktop/Repos/HOPP_H2/HOPP_Private/')

import os
from dotenv import load_dotenv
from math import sin, pi
from hybrid.reopt import REopt
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so
import pandas as pd

from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from tools.analysis import create_cost_calculator

# simple battery dispatch model
from simple_dispatch import SimpleDispatch
from plot_power_to_load import plot_power_to_load
from plot_shortfall_curtailment import plot_shortfall_curtailment
from plot_battery import plot_battery
import numpy as np


# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Set up output dataframe
save_all_runs = pd.DataFrame()
save_outputs_loop = dict()
save_outputs_loop['Scenario Choice'] = list()
save_outputs_loop['Count'] = list()
save_outputs_loop['Site Lat'] = list()
save_outputs_loop['Site Lon'] = list()
save_outputs_loop['ATB Year'] = list()
save_outputs_loop['Resource Year'] = list()
save_outputs_loop['Site Name'] = list()
# save_outputs_loop['MT C02'] = list()
save_outputs_loop['Critical Load Factor'] = list()
save_outputs_loop['kW continuous load'] = list()
save_outputs_loop['PTC'] = list()
save_outputs_loop['ITC'] = list()
save_outputs_loop['Hub Height (m)'] = list()
save_outputs_loop['Useful Life'] = list()
save_outputs_loop['Storage Enabled'] = list()
save_outputs_loop['Wind Cost kW'] = list()
save_outputs_loop['Solar Cost kW'] = list()
save_outputs_loop['Storage Cost kW'] = list()
save_outputs_loop['Storage Cost kWh'] = list()
save_outputs_loop['Storage Hours'] = list()
save_outputs_loop['Site Lon'] = list()
save_outputs_loop['Wind MW built'] = list()
save_outputs_loop['Solar MW built'] = list()
save_outputs_loop['Storage MW built'] = list()
save_outputs_loop['Storage MWh built'] = list()
save_outputs_loop['Battery Can Grid Charge'] = list()
save_outputs_loop['Built Interconnection Size'] = list()
save_outputs_loop['REOpt Interconnection Size'] = list()
save_outputs_loop['LCOE'] = list()
save_outputs_loop['Levelized cost of H2 (excl. electricity) (H2A)'] = list()
save_outputs_loop['Levelized H2 Elec Feedstock Cost/kg (HOPP)'] = list()
save_outputs_loop['Total H2 cost/kg'] = list()
save_outputs_loop['H2 Elec Feedstock Cost/kg (HOPP) Net Cap Cost Method'] = list()
save_outputs_loop['H2 Elec Feedstock Cost/kg (REopt) Net Cap Cost Method'] = list()
save_outputs_loop['REOpt Energy Shortfall'] = list()
save_outputs_loop['REOpt Curtailment'] = list()
save_outputs_loop['Grid Connected HOPP'] = list()
save_outputs_loop['HOPP Total Generation'] = list()
save_outputs_loop['HOPP Energy Shortfall'] = list()
save_outputs_loop['HOPP Curtailment'] = list()
save_outputs_loop['Battery Generation'] = list()
save_outputs_loop['Electricity to Grid'] = list()

# Get resource
site_name = 'IOWA'
lat = 42.952  #flatirons_site['lat']
lon = -94.453  #flatirons_site
# site_name = 'Plainview Bioenergy - Texas'
# lat = 46.1  #flatirons_site['lat']
# lon = -94.33  #flatirons_site
year = 2013
sample_site['year'] = year
sample_site['lat'] = lat
sample_site['lon'] = lon
useful_life = 30
critical_load_factor_list = [0.9]
run_reopt = True
custom_powercurve = True

#Load scenarios from .csv and enumerate
scenarios_df = pd.read_csv('H2 Baseline Future Scenarios.csv')
# scenarios_df = pd.read_csv('H2 Baseline Future Scenarios_Optimals_Future_Iowa.csv')
for critical_load_factor in critical_load_factor_list:
    for i, scenario in scenarios_df.iterrows():
        # TODO: Make scenario_choice, lookup all other values from dataframe from csv.

        # TODO:
        # -Pass through rotor diameter to pySAM
        # -Add wind, solar, storage installed costs
        # -Fix "H2 H2 xxx" text

        print(scenario)
        scenario_choice = scenario['Scenario Name']
        atb_year = scenario['ATB Year']
        ptc_avail = scenario['PTC Available']
        itc_avail = scenario['ITC Available']
        tower_height = scenario['Tower Height']
        rotor_diameter = scenario['Rotor Diameter']
        wind_cost_kw = scenario['Wind Cost KW']
        pv_cost_kw = scenario['Solar Cost KW']
        storage_cost_kw = scenario['Storage Cost kW']
        storage_cost_kwh = scenario['Storage Cost kWh']
        debt_equity_split = scenario['Debt Equity']

        site = SiteInfo(sample_site)

        storage_used = True
        battery_can_grid_charge = True
        grid_connected_hopp = True

        # Set up REopt run
        kg_per_day_desired = 5000
        kw_continuous = 5000 # 50 MW continuous load - equivalent to 909kg H2 per hr at 55 kWh/kg electrical intensity
        load = [kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
        urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

        solar_model = SolarPlant(site, 20000)
        wind_model = WindPlant(site, 20000)
        fin_model = so.default("GenericSystemSingleOwner")
        filepath = os.path.dirname(os.path.abspath(__file__))
        fileout = os.path.join(filepath, "../data", "REoptResultsNoExportAboveLoad.json")
        count = 0

        site = SiteInfo(sample_site, hub_height=tower_height)
        count = count + 1


        reopt = REopt(lat=lat,
                      lon=lon,
                      load_profile=load,
                      urdb_label=urdb_label,
                      solar_model=solar_model,
                      wind_model=wind_model,
                      fin_model=fin_model,
                      interconnection_limit_kw=100000,
                      fileout=os.path.join(filepath, "../data", "REoptResultsNoExportAboveLoad.json"))

        reopt.set_rate_path(os.path.join(filepath, '../data'))

        reopt.post['Scenario']['Site']['Wind']['installed_cost_us_dollars_per_kw'] = wind_cost_kw  # ATB
        reopt.post['Scenario']['Site']['PV']['installed_cost_us_dollars_per_kw'] = pv_cost_kw
        reopt.post['Scenario']['Site']['Storage'] = {'min_kw': 0.0, 'max_kw': 100000000.0, 'min_kwh': 0.0, 'max_kwh': 33300000.0,
                                                     'internal_efficiency_pct': 0.975, 'inverter_efficiency_pct': 0.96,
                                                     'rectifier_efficiency_pct': 0.96, 'soc_min_pct': 0.2, 'soc_init_pct': 0.5,
                                                     'canGridCharge': battery_can_grid_charge, 'installed_cost_us_dollars_per_kw': storage_cost_kw,
                                                     'installed_cost_us_dollars_per_kwh': storage_cost_kwh,
                                                     'replace_cost_us_dollars_per_kw': storage_cost_kw,
                                                     'replace_cost_us_dollars_per_kwh': storage_cost_kwh, 'inverter_replacement_year': 10,
                                                     'battery_replacement_year': 10, 'macrs_option_years': 7,
                                                     'macrs_bonus_pct': 1.0, 'macrs_itc_reduction': 0.5, 'total_itc_pct': 0.0,
                                                     'total_rebate_us_dollars_per_kw': 0, 'total_rebate_us_dollars_per_kwh': 0}
        reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_us_dollars_per_kwh'] = 0.01
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
        reopt.post['Scenario']['Site']['LoadProfile']['annual kwh'] = 8760 * kw_continuous
        critical_load_pct = critical_load_factor
        reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = critical_load_pct
        reopt.post['Scenario']['Site']['LoadProfile']['outage_start_hour'] = 10
        reopt.post['Scenario']['Site']['LoadProfile']['outage_end_hour'] = 8750

        if run_reopt == True:
            result = reopt.get_reopt_results(force_download=True)

            # if scenario['Scenario Name'] == 'Iowa 2020 Moderate':
            if scenario['Scenario Name'] == 'ATB 2020 Moderate':
                import pickle
                pickle.dump(result, open("results/reopt_precomputes/results_ATB_moderate_2020_{}_{}.p".format(site_name, critical_load_factor), "wb"))

        else:
            import pickle
            if critical_load_factor == 0.9:
                result = pickle.load(open("results/reopt_precomputes/results_ATB_moderate_2020_{}_0.9.p".format(site_name), "rb"))
            elif critical_load_factor == 0.5:
                result = pickle.load(open("results/reopt_precomputes/results_ATB_moderate_2020_{}_0.5.p".format(site_name), "rb"))
        if result['outputs']['Scenario']['Site']['PV']['size_kw']:
            solar_size_mw = result['outputs']['Scenario']['Site']['PV']['size_kw'] / 1000

        wind_size_mw = result['outputs']['Scenario']['Site']['Wind']['size_kw'] / 1000
        if result['outputs']['Scenario']['Site']['Storage']['size_kw']:
            storage_size_mw = result['outputs']['Scenario']['Site']['Storage']['size_kw'] / 1000
            storage_size_mwh = result['outputs']['Scenario']['Site']['Storage']['size_kwh'] / 1000
            storage_hours = storage_size_mwh / storage_size_mw
        interconnection_size_mw = reopt.interconnection_limit_kw / 1000
        print('Solar size = ', solar_size_mw)
        print('Wind size = ', wind_size_mw)
        print('Storage size = ', storage_size_mw)
        print('Interconnection size = ', interconnection_size_mw)


        # Create a dataframe of desired REopt results to visualize
        # result['outputs']['Scenario']['Site']['PV']['year_one_power_production_series_kw']
        # result['outputs']['Scenario']['Site']['PV']['year_one_to_battery_series_kw']
        # result['outputs']['Scenario']['Site']['PV']['year_one_to_load_series_kw']
        # result['outputs']['Scenario']['Site']['PV']['year_one_to_grid_series_kw']
        #
        # result['outputs']['Scenario']['Site']['Storage']['year_one_to_load_series_kw']
        # result['outputs']['Scenario']['Site']['Storage']['year_one_to_grid_series_lw']
        # result['outputs']['Scenario']['Site']['Storage']['year_one_soc_series_pct']
        #
        # result['outputs']['Scenario']['Site']['Wind']['year_one_power_production_series_kw']
        # result['outputs']['Scenario']['Site']['Wind']['year_one_to_battery_series_kw']
        # result['outputs']['Scenario']['Site']['Wind']['year_one_to_load_series_kw']
        # result['outputs']['Scenario']['Site']['Wind']['year_one_to_grid_series_kw']
        # result['outputs']['Scenario']['Site']['Wind']['year_one_curtailed_production_series_kw']

        reopt_site_result = result['outputs']['Scenario']['Site']
        generated_date = pd.date_range(start='1/1/2018 00:00:00', end='12/31/2018 23:00:00', periods=8760)
        combined_pv_wind_power_production = [x + y for x, y in zip(reopt_site_result['PV']['year_one_power_production_series_kw']
                                                                   , reopt_site_result['Wind']['year_one_power_production_series_kw'])]
        combined_pv_wind_storage_power_production = [x + y for x, y in zip(combined_pv_wind_power_production,
                                                                           reopt_site_result['Storage']['year_one_to_load_series_kw'])]
        energy_shortfall = [y - x for x, y in zip(combined_pv_wind_storage_power_production, load)]
        energy_shortfall = [x if x > 0 else 0 for x in energy_shortfall]

        combined_pv_wind_curtailment = [x + y for x, y in zip(reopt_site_result['PV']['year_one_curtailed_production_series_kw']
                                                                   , reopt_site_result['Wind']['year_one_curtailed_production_series_kw'])]

        reopt_result_dict = {'Date':
                                 generated_date,
                             'pv_power_production':
                                 reopt_site_result['PV']
                                 ['year_one_power_production_series_kw'],
                             'pv_power_to_grid':
                                 reopt_site_result['PV']
                                 ['year_one_to_grid_series_kw'],
                             'pv_power_to_load':
                                reopt_site_result['PV']['year_one_to_load_series_kw'],
                             'pv_power_to_battery':
                                 reopt_site_result['PV']['year_one_to_battery_series_kw'],
                             'pv_power_curtailed':
                                 reopt_site_result['PV']['year_one_curtailed_production_series_kw'],
                             'wind_power_production':
                                 reopt_site_result['Wind']
                                 ['year_one_power_production_series_kw'],
                             'wind_power_to_grid':
                                 reopt_site_result['Wind']
                                 ['year_one_to_grid_series_kw'],
                             'wind_power_to_load':
                                 reopt_site_result['Wind']['year_one_to_load_series_kw'],
                             'wind_power_to_battery':
                                 reopt_site_result['Wind']['year_one_to_battery_series_kw'],
                             'wind_power_curtailed':
                                 reopt_site_result['Wind']['year_one_curtailed_production_series_kw'],
                             'combined_pv_wind_power_production':
                                 combined_pv_wind_power_production,
                             'combined_pv_wind_storage_power_production':
                                 combined_pv_wind_storage_power_production,
                             'storage_power_to_load':
                                 reopt_site_result['Storage']['year_one_to_load_series_kw'],
                             'storage_power_to_grid':
                                 reopt_site_result['Storage']['year_one_to_grid_series_kw'],
                             'battery_soc_pct':
                                 reopt_site_result['Storage']['year_one_soc_series_pct'],
                             'energy_shortfall':
                                 energy_shortfall
                             }

        REoptResultsDF = pd.DataFrame(reopt_result_dict)
        # Visualize REopt results
        import plotly.express as px
        import matplotlib.pyplot as plt
        import numpy as np

        # Set up HOPP run
        technologies = {'solar': solar_size_mw,  # mw system capacity
                        'wind': wind_size_mw,  # mw system capacity
                        'grid': interconnection_size_mw,
                        'collection_system': True}

        # Create model
        if grid_connected_hopp:
            interconnection_size_mw = reopt.interconnection_limit_kw
        else:
            interconnection_size_mw = kw_continuous/1000

        hybrid_plant = HybridSimulation(technologies, site,
                                        interconnect_kw=interconnection_size_mw*1000,
                                        storage_kw=storage_size_mw * 1000,
                                        storage_kwh=storage_size_mwh * 1000,
                                        storage_hours=storage_hours)
        hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                                  bos_cost_source='CostPerMW',
                                                                  wind_installed_cost_mw=wind_cost_kw*1000,
                                                                  solar_installed_cost_mw=pv_cost_kw*1000,
                                                                  storage_installed_cost_mw=storage_cost_kw*1000,
                                                                  storage_installed_cost_mwh=storage_cost_kwh*1000
                                                                  ))

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
            hybrid_plant.wind.system_model.Turbine.wind_turbine_hub_ht = tower_height

        if custom_powercurve:
            import json
            powercurve_file = open('powercurve_custom_3MW')
            powercurve_data = json.load(powercurve_file)
            powercurve_file.close()
            hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_windspeeds = \
                powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
            hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_powerout = \
                powercurve_data['turbine_powercurve_specification']['turbine_power_output']

        hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
        hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
        hybrid_plant.ppa_price = 0.05
        hybrid_plant.simulate(useful_life)

        # HOPP Specific Energy Metrics
        energy_shortfall_hopp = [y - x for x, y in
                                 zip(hybrid_plant.grid.generation_profile_from_system[0:8759], load)]
        energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
        combined_pv_wind_power_production_hopp = hybrid_plant.grid.system_model.Outputs.system_pre_interconnect_kwac[0:8759]
        combined_pv_wind_curtailment_hopp = [x-y for x, y in zip(hybrid_plant.grid.system_model.Outputs.system_pre_interconnect_kwac[0:8759],
                                                                  hybrid_plant.grid.system_model.Outputs.gen[0:8759])]

        # super simple dispatch battery model with no forecasting TODO: add forecasting
        bat_model = SimpleDispatch(combined_pv_wind_curtailment_hopp, energy_shortfall_hopp, len(energy_shortfall_hopp),
                                   storage_size_mw * 1000)

        battery_used, excess_energy, battery_SOC = bat_model.run()
        combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + excess_energy

        # Save the outputs
        annual_energies = hybrid_plant.annual_energies
        wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.solar
        npvs = hybrid_plant.net_present_values
        lcoe = hybrid_plant.lcoe_real.hybrid

        #-------------------------Dan's Edits-------------------------------------#
        # TODO: Test H2A integration
        # Run the Python H2A model
        import H2AModel
        total_system_electrical_usage = 55.5                                                        # kWh/kgH2
        generation_timeseries = combined_pv_wind_storage_power_production_hopp                      # Hourly Generation
        avg_generation = np.mean(generation_timeseries)                                             # Avg Generation

        # Set Capacity Factor

        if avg_generation < kw_continuous:
            cap_factor = avg_generation/kw_continuous
        else:
            cap_factor = 1

        hydrogen_hourly_production = np.divide(generation_timeseries, total_system_electrical_usage)  # hourly hydrogen production (kg)

        # Get Daily Hydrogen Production - Add Every 24 hours
        i = 0
        daily_H2_production = []
        while i < 8760:
            x = sum(hydrogen_hourly_production[i:i + 25])
            daily_H2_production.append(x)
            i = i + 25

        avg_daily_H2_production = np.mean(daily_H2_production)                           # kgH2/day
        hydrogen_annual_output = sum(hydrogen_hourly_production)                         # kgH2/year
        elec_remainder_after_h2 = combined_pv_wind_curtailment_hopp
        # elec_revenue_after_h2 = np.sum(elec_remainder_after_h2 * 0.05)
        print("Surplus Electrical Revenue")
        # Hydrogen Plant - levelized H2 cost
        results = H2AModel.H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output)
        # results['Capital Related Costs']
        # results['Fixed O&M']
        # results['Variable Costs/Feedstock']
        # results['Total Hydrogen Cost ($/kgH2)']
        # Hybrid Plant - levelized H2 Cost - HOPP
        h2a_costs = results['Total Hydrogen Cost ($/kgH2)']
        feedstock_cost_h2_levelized_hopp = lcoe * 55.5/100  # $/kg
        # Hybrid Plant - levelized H2 Cost - HOPP
        feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = hybrid_plant.grid.financial_model.Outputs.adjusted_installed_cost /\
                                      (hydrogen_annual_output * useful_life)             # $/kgH2
        # Total Hydrogen Cost ($/kgH2)
        total_unit_cost_of_hydrogen = h2a_costs + feedstock_cost_h2_levelized_hopp

        #feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = hybrid_plant.grid.financial_model.Outputs.adjusted_installed_cost /\
        #                                                     ((kw_continuous/55.5)*(8760 * useful_life))
        feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt = result['outputs']['Scenario']['Site']\
        ['Financial']['net_capital_costs'] / ((kw_continuous/55.5)*(8760*useful_life))
        #--------------------------------------------------------------------------------------------------------#
        wind_installed_cost = hybrid_plant.wind.financial_model.SystemCosts.total_installed_cost
        solar_installed_cost = hybrid_plant.solar.financial_model.SystemCosts.total_installed_cost
        hybrid_installed_cost = hybrid_plant.grid.financial_model.SystemCosts.total_installed_cost

        print("Future Scenario: {}".format(scenario_choice))
        print("Wind Cost per KW: {}".format(wind_cost_kw))
        print("PV Cost per KW: {}".format(pv_cost_kw))
        print("Storage Cost per KW: {}".format(storage_cost_kw))
        print("Storage Cost per KWh: {}".format(storage_cost_kwh))
        print("Wind Size built: {}".format(wind_size_mw))
        print("PV Size built: {}".format(solar_size_mw))
        print("Storage Size built: {}".format(storage_size_mw))
        print("Storage Size built: {}".format(storage_size_mwh))
        print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
        print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
            feedstock_cost_h2_levelized_hopp))
        print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(h2a_costs))
        print("Total unit cost of H2 ($/kg) : {}".format(total_unit_cost_of_hydrogen))
        print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp))
        print("kg H2 cost from net cap cost/lifetime h2 production (REopt): {}".format(
            feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt))

        # Plot REopt results
        REoptResultsDF.index = REoptResultsDF.Date
        monthly_separation = False
        if monthly_separation:
            # Group by months
            df_mean = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).mean()
            df_std = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).std()
            df_n = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).count()
            z = 1.96
            df_ci = z * df_std / df_n.applymap(np.sqrt)
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            month_map = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }

            mapped = [f"{month_map[m]}-{h}" for m, h in df_mean.index.values]
            y = range(df_mean.index.values.shape[0])

            xticks_major = [x * 24 for x in range(1, 13)]
            xticks_minor = list(range(0, 24 * 12, 6))
            xlabels_major = [month_map[m / 24].ljust(13) for m in xticks_major]
            xlabels_minor = ["", "06", "12", "18"] + ["06", "12", "18", "24"] * 11
        else:
            # Group by hours
            df_mean = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).mean()
            df_std = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).std()
            df_n = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).count()
            z = 1.96
            df_ci = z * df_std / df_n.applymap(np.sqrt)
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            y = range(df_mean.index.values.shape[0])

            xticks_major = [x for x in range(1, 24)]
            xticks_minor = list(range(0, 24))
            xlabels_major = [hr for hr in xticks_major]
            xlabels_minor = xticks_minor

        #Plot 1 - Energy supplied to load
        titletext = 'PV and Wind Power at {} plant | ATB Year {} \n Critical Load Factor (0-1): {:,.2f} \n' \
                      ' Wind Size (MW): {:,.2f} | Solar Size (MW): {:,.2f} \n Storage Size (MW): {:,.2f} | Storage Size MWh: {:,.2f} \n' \
                      ' HOPP LCOE: {:,.2f}c | H2 Levelized Electrical Cost ($/kg): ${:,.2f} | Hybrid Installed Cost: ${:,.2f} \n' \
                      ' H2A Levelized Plant Costs ($/kg): ${:,.2f} | Total Levelized H2 Cost ($/kg): ${:,.2f} \n'\
                      ' Total Energy Provided (MWh): {:,.2f} Total Energy Shortfall: {:,.2f} Total Energy Curtailed (MWh) {:,.2f}'\
            .format(site_name, atb_year, critical_load_pct, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh,
            lcoe, feedstock_cost_h2_levelized_hopp, hybrid_installed_cost, h2a_costs, total_unit_cost_of_hydrogen,
            np.sum(combined_pv_wind_storage_power_production),
            np.sum(energy_shortfall), np.sum(combined_pv_wind_curtailment))
        save_location = "wind_pv_{}_production_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
                                                                             atb_year, useful_life,
                                                                             critical_load_factor, tower_height)
        ylim = [0, 10000]
        plot_power_to_load(titletext, df_mean, df_ci, y, ylim,
                           colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                           save_location = save_location)


        # Plot 2 - SHORTFALL & CURTAILMENT
        save_location = "wind_pv_{}_shortfall_curtailment_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
            atb_year, useful_life, critical_load_factor, tower_height)
        ylim = [0, 10000]
        plot_shortfall_curtailment(titletext, df_mean, df_ci, y, ylim,
                           colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                           save_location=save_location)

        # Plot 3 - Battery Only
        save_location = "battery_{}_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
                                                                                                    atb_year,
                                                                                                    useful_life,
                                                                                                    critical_load_factor,
                                                                                                    tower_height)
        ylim = [0, 10000]
        plot_battery(titletext, df_mean, df_ci, y, ylim,
                                   colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                                   save_location=save_location)

        # Save outputs
        save_outputs_loop['Scenario Choice'].append(scenario_choice)
        save_outputs_loop['Count'].append(count)
        save_outputs_loop['Site Lat'].append(lat)
        save_outputs_loop['Site Lon'].append(lon)
        save_outputs_loop['ATB Year'].append(atb_year)
        save_outputs_loop['Resource Year'].append(year)
        save_outputs_loop['Site Name'].append(site_name)
        # save_outputs_loop['MT C02'].append(MTC02_yr)
        save_outputs_loop['Critical Load Factor'].append(critical_load_factor)
        save_outputs_loop['kW continuous load'].append(kw_continuous)
        save_outputs_loop['PTC'].append(ptc_avail)
        save_outputs_loop['ITC'].append(itc_avail)
        save_outputs_loop['Hub Height (m)'].append(tower_height)
        save_outputs_loop['Useful Life'].append(useful_life)
        save_outputs_loop['Storage Enabled'].append(storage_used)
        save_outputs_loop['Wind Cost kW'].append(wind_cost_kw)
        save_outputs_loop['Solar Cost kW'].append(pv_cost_kw)
        save_outputs_loop['Storage Cost kW'].append(storage_cost_kw)
        save_outputs_loop['Storage Cost kWh'].append(storage_cost_kwh)
        save_outputs_loop['Storage Hours'].append(storage_hours)
        save_outputs_loop['Wind MW built'].append(wind_size_mw)
        save_outputs_loop['Solar MW built'].append(solar_size_mw)
        save_outputs_loop['Storage MW built'].append(storage_size_mw)
        save_outputs_loop['Storage MWh built'].append(storage_size_mwh)
        save_outputs_loop['Battery Can Grid Charge'].append(battery_can_grid_charge)
        save_outputs_loop['Built Interconnection Size'].append(hybrid_plant.interconnect_kw)
        save_outputs_loop['REOpt Interconnection Size'].append(reopt.interconnection_limit_kw)
        save_outputs_loop['LCOE'].append(lcoe)
        save_outputs_loop['Levelized H2 Elec Feedstock Cost/kg (HOPP)'].append(feedstock_cost_h2_levelized_hopp)
        save_outputs_loop['Levelized cost of H2 (excl. electricity) (H2A)'].append(h2a_costs)
        save_outputs_loop['Total H2 cost/kg'].append(total_unit_cost_of_hydrogen)
        save_outputs_loop['H2 Elec Feedstock Cost/kg (HOPP) Net Cap Cost Method'].append(feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp)
        save_outputs_loop['H2 Elec Feedstock Cost/kg (REopt) Net Cap Cost Method'].append(feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt)
        save_outputs_loop['REOpt Energy Shortfall'].append(np.sum(energy_shortfall))
        save_outputs_loop['REOpt Curtailment'].append(np.sum(combined_pv_wind_curtailment))
        save_outputs_loop['Grid Connected HOPP'].append(grid_connected_hopp)
        save_outputs_loop['HOPP Total Generation'].append(np.sum(hybrid_plant.grid.generation_profile_from_system[0:8759]))
        save_outputs_loop['HOPP Energy Shortfall'].append(np.sum(energy_shortfall_hopp))
        save_outputs_loop['HOPP Curtailment'].append(np.sum(combined_pv_wind_curtailment_hopp))
        save_outputs_loop['Battery Generation'].append(np.sum(battery_used))
        save_outputs_loop['Electricity to Grid'].append(np.sum(excess_energy))

# save_all_runs = save_all_runs.append(save_outputs_loop, sort=False)

save_outputs_loop_df = pd.DataFrame(save_outputs_loop)
save_outputs_loop_df.to_csv("results/H2_Analysis_{}.csv".format(site_name))
