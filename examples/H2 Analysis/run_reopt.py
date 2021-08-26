import numpy as np
import os
from hybrid.reopt import REopt
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so
import pandas as pd
import pickle

def run_reopt(site, scenario, load, interconnection_limit_kw, critical_load_factor, useful_life,
              battery_can_grid_charge,
              storage_used, run_reopt_flag):

    # kw_continuous = forced_system_size  # 5 MW continuous load - equivalent to 909kg H2 per hr at 55 kWh/kg electrical intensity

    urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898
    solar_model = SolarPlant(site, 20000)
    wind_model = WindPlant(site, 20000)
    fin_model = so.default("GenericSystemSingleOwner")
    filepath = os.path.dirname(os.path.abspath(__file__))
    fileout = os.path.join(filepath, "../data", "REoptResultsNoExportAboveLoad.json")
    # site = SiteInfo(sample_site, hub_height=tower_height)
    count = 1
    reopt = REopt(lat=scenario['Lat'],
                  lon=scenario['Long'],
                  load_profile=load,
                  urdb_label=urdb_label,
                  solar_model=solar_model,
                  wind_model=wind_model,
                  fin_model=fin_model,
                  interconnection_limit_kw=interconnection_limit_kw,
                  fileout=os.path.join(filepath, "../data", "REoptResultsNoExportAboveLoad.json"))

    reopt.set_rate_path(os.path.join(filepath, '../data'))

    reopt.post['Scenario']['Site']['Wind']['installed_cost_us_dollars_per_kw'] = scenario['Wind Cost KW']  # ATB
    reopt.post['Scenario']['Site']['PV']['installed_cost_us_dollars_per_kw'] = scenario['Solar Cost KW']
    reopt.post['Scenario']['Site']['Storage'] = {'min_kw': 0.0, 'max_kw': 1000000.0, 'min_kwh': 0.0,
                                                 'max_kwh': 33300000.0,
                                                 'internal_efficiency_pct': 0.975, 'inverter_efficiency_pct': 0.96,
                                                 'rectifier_efficiency_pct': 0.96, 'soc_min_pct': 0.2,
                                                 'soc_init_pct': 0.5,
                                                 'canGridCharge': battery_can_grid_charge,
                                                 'installed_cost_us_dollars_per_kw': scenario['Storage Cost kW'],
                                                 'installed_cost_us_dollars_per_kwh': scenario['Storage Cost kWh'],
                                                 'replace_cost_us_dollars_per_kw': scenario['Storage Cost kW'],
                                                 'replace_cost_us_dollars_per_kwh': scenario['Storage Cost kWh'],
                                                 'inverter_replacement_year': 10,
                                                 'battery_replacement_year': 10, 'macrs_option_years': 7,
                                                 'macrs_bonus_pct': 1.0, 'macrs_itc_reduction': 0.5,
                                                 'total_itc_pct': 0.0,
                                                 'total_rebate_us_dollars_per_kw': 0,
                                                 'total_rebate_us_dollars_per_kwh': 0}
    reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_us_dollars_per_kwh'] = 0.01
    reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_above_site_load_us_dollars_per_kwh'] = 0.0
    reopt.post['Scenario']['Site']['Financial']['analysis_years'] = useful_life
    if not storage_used:
        reopt.post['Scenario']['Site']['Storage']['max_kw'] = 0
    if scenario['PTC Available']:
        reopt.post['Scenario']['Site']['Wind']['pbi_us_dollars_per_kwh'] = 0.022
    else:
        reopt.post['Scenario']['Site']['Wind']['pbi_us_dollars_per_kwh'] = 0.0
    if scenario['ITC Available']:
        reopt.post['Scenario']['Site']['PV']['federal_itc_pct'] = 0.26
    else:
        reopt.post['Scenario']['Site']['PV']['federal_itc_pct'] = 0.0

    reopt.post['Scenario']['Site']['LoadProfile']['doe_reference_name'] = "FlatLoad"
    reopt.post['Scenario']['Site']['LoadProfile']['annual kwh'] = load #8760 * kw_continuous
    reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = critical_load_factor
    reopt.post['Scenario']['Site']['LoadProfile']['outage_start_hour'] = 10
    reopt.post['Scenario']['Site']['LoadProfile']['outage_end_hour'] = 8750

    if run_reopt_flag:
        result = reopt.get_reopt_results(force_download=True)
        pickle.dump(result, open("results/reopt_precomputes/results_{}_{}_{}.p".format(
            scenario['Site Name'], scenario['Scenario Name'], critical_load_factor), "wb"))

    else:
        print("Not running reopt. Loading Dummy data")
        result = pickle.load(
            open("results/reopt_precomputes/results_ATB_moderate_2020_IOWA_0.9.p", "rb"))

    if result['outputs']['Scenario']['Site']['PV']['size_kw']:
        solar_size_mw = result['outputs']['Scenario']['Site']['PV']['size_kw'] / 1000

    wind_size_mw = result['outputs']['Scenario']['Site']['Wind']['size_kw'] / 1000
    if result['outputs']['Scenario']['Site']['Storage']['size_kw']:
        storage_size_mw = result['outputs']['Scenario']['Site']['Storage']['size_kw'] / 1000
        storage_size_mwh = result['outputs']['Scenario']['Site']['Storage']['size_kwh'] / 1000
        storage_hours = storage_size_mwh / storage_size_mw

    reopt_site_result = result['outputs']['Scenario']['Site']
    generated_date = pd.date_range(start='1/1/2018 00:00:00', end='12/31/2018 23:00:00', periods=8760)
    combined_pv_wind_power_production = [x + y for x, y in
                                         zip(reopt_site_result['PV']['year_one_power_production_series_kw']
                                             , reopt_site_result['Wind']['year_one_power_production_series_kw'])]
    combined_pv_wind_storage_power_production = [x + y for x, y in zip(combined_pv_wind_power_production,
                                                                       reopt_site_result['Storage'][
                                                                           'year_one_to_load_series_kw'])]
    energy_shortfall = [y - x for x, y in zip(combined_pv_wind_storage_power_production, load)]
    energy_shortfall = [x if x > 0 else 0 for x in energy_shortfall]

    combined_pv_wind_curtailment = [x + y for x, y in
                                    zip(reopt_site_result['PV']['year_one_curtailed_production_series_kw']
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
                             energy_shortfall,
                         'combined_pv_wind_curtailment':
                             combined_pv_wind_curtailment
                         }

    REoptResultsDF = pd.DataFrame(reopt_result_dict)

    return wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours, result, REoptResultsDF