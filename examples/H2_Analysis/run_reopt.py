import numpy as np
import os
from hybrid.reopt import REopt
from hybrid.pv_source import PVPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so
import pandas as pd
import pickle
# import post_and_poll
from dotenv import load_dotenv
import json

#TODO:
# - Change to developer API.
# - Integrate all new features of post_and_poll with REOpt class (from hybrid/reopt_dev)


def run_reopt(site, scenario, load, interconnection_limit_kw, critical_load_factor, useful_life,
              battery_can_grid_charge,
              storage_used, run_reopt_flag):

    # kw_continuous = forced_system_size  # 5 MW continuous load - equivalent to 909kg H2 per hr at 55 kWh/kg electrical intensity

    urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898
    pv_config = {'system_capacity_kw': 20000}
    solar_model = PVPlant(site, pv_config)
    # ('num_turbines', 'turbine_rating_kw', 'rotor_diameter', 'hub_height', 'layout_mode', 'layout_params')
    wind_config = {'num_turbines': np.floor(scenario['Wind Size MW'] / scenario['Turbine Rating']),
                       'rotor_dimeter': scenario['Rotor Diameter'], 'hub_height': scenario['Tower Height'],
                   'turbine_rating_kw': scenario['Turbine Rating']}

    wind_model = WindPlant(site, wind_config)
    fin_model = so.default("GenericSystemSingleOwner")
    filepath = os.path.dirname(os.path.abspath(__file__))
    fileout = 'reopt_result_test_intergration.json'
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
                  off_grid=True,
                  fileout=fileout)

    reopt.set_rate_path(os.path.join(filepath, '../data'))

    reopt.post['Scenario']['Site']['Wind']['installed_cost_us_dollars_per_kw'] = scenario['Wind Cost KW']  # ATB
    reopt.post['Scenario']['Site']['PV']['installed_cost_us_dollars_per_kw'] = scenario['Solar Cost KW']
    reopt.post['Scenario']['Site']['Storage'] = {'min_kw': 0.0, 'max_kw': 0.99e9, 'min_kwh': 0.0,
                                                 'max_kwh': 0.99e9,
                                                 'internal_efficiency_pct': 0.975, 'inverter_efficiency_pct': 0.96,
                                                 'rectifier_efficiency_pct': 0.96, 'soc_min_pct': 0.2,
                                                 'soc_init_pct': 0.5,
                                                 'canGridCharge': battery_can_grid_charge,
                                                 'installed_cost_us_dollars_per_kw': scenario['Storage Cost KW'],
                                                 'installed_cost_us_dollars_per_kwh': scenario['Storage Cost KWh'],
                                                 'replace_cost_us_dollars_per_kw': scenario['Storage Cost KW'],
                                                 'replace_cost_us_dollars_per_kwh': scenario['Storage Cost KWh'],
                                                 'inverter_replacement_year': 10,
                                                 'battery_replacement_year': 10, 'macrs_option_years': 7,
                                                 'macrs_bonus_pct': 1.0, 'macrs_itc_reduction': 0.5,
                                                 'total_itc_pct': 0.0,
                                                 'total_rebate_us_dollars_per_kw': 0,
                                                 'total_rebate_us_dollars_per_kwh': 0}

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

    # reopt.post['Scenario']['Site']['LoadProfile']['doe_reference_name'] = "FlatLoad"
    # reopt.post['Scenario']['Site']['LoadProfile']['annual_kwh'] = load #8760 * kw_continuous
    reopt.post['Scenario']['Site']['LoadProfile']['loads_kw'] = load
    reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = critical_load_factor

    off_grid = False
    reopt.post['Scenario']['optimality_tolerance_techs'] = 0.05

    if off_grid == True:
        # reopt.post['Scenario']['Site'].pop('Wind')
        # reopt.post['Scenario']['Site']['Wind']['min_kw'] = 10000
        dictforstuff = {"off_grid_flag": True}
        reopt.post['Scenario'].update(dictforstuff)
        reopt.post['Scenario']['optimality_tolerance_techs'] = 0.05
        reopt.post['Scenario']["timeout_seconds"] = 3600
        # reopt.post['Scenario']['Site']['LoadProfile'].pop('annual kwh')
        reopt.post['Scenario']['Site'].pop('ElectricTariff')
        reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = 1.0
        f = open('massproducer_offgrid (1).json')
        data_for_post = json.load(f)
        reopt.post['Scenario']['Site']['Financial'] = data_for_post['Scenario']['Site']['Financial']
    else:
        reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_us_dollars_per_kwh'] = 0.01
        reopt.post['Scenario']['Site']['ElectricTariff']['wholesale_rate_above_site_load_us_dollars_per_kwh'] = 0.01
        reopt.post['Scenario']['Site']['LoadProfile']['outage_start_hour'] = 10
        reopt.post['Scenario']['Site']['LoadProfile']['outage_end_hour'] = 11

    from pathlib import Path
    post_path = 'results/reopt_precomputes/reopt_post'
    post_path_abs = Path(__file__).parent / post_path
    if not os.path.exists(post_path_abs.parent):
        os.mkdir(post_path_abs.parent)
    with open(post_path_abs, 'w') as outfile:
        json.dump(reopt.post, outfile)
    # mass_producer_dict = {
    #     "mass_units": "kg",
    #     "time_units": "hr",
    #     "min_mass_per_time": 10.0,
    #     "max_mass_per_time": 10.0,
    #     "electric_consumed_to_mass_produced_ratio_kwh_per_mass": 71.7,
    #     "thermal_consumed_to_mass_produced_ratio_kwh_per_mass": 0.0,
    #     "feedstock_consumed_to_mass_produced_ratio": 0.0,
    #     "installed_cost_us_dollars_per_mass_per_time": 10.0,
    #     "om_cost_us_dollars_per_mass_per_time": 1.5,
    #     "om_cost_us_dollars_per_mass": 0.0,
    #     "mass_value_us_dollars_per_mass": 5.0,
    #     "feedstock_cost_us_dollars_per_mass": 0.0,
    #     "macrs_option_years": 0,
    #     "macrs_bonus_pct": 0
    # }
    # reopt.post['Scenario']['Site']['MassProducer'] = mass_producer_dict

    if run_reopt_flag:
        #NEW METHOD
        load_dotenv()
        result = reopt.get_reopt_results()

        #BASIC INITIAL TEST FOR NEW METHOD
        # result = post_and_poll.get_api_results(data_for_post, NREL_API_KEY, 'https://offgrid-electrolyzer-reopt-dev-api.its.nrel.gov/v1',
        #                       'reopt_result_test_intergration.json')

        # f = open('massproducer_offgrid (1).json')
        # data_for_post = json.load(f)

        #OLD METHOD
        # result = reopt.get_reopt_results(force_download=True)


        pickle.dump(result, open("results/reopt_precomputes/results_{}_{}_{}.p".format(
            scenario['Site Name'], scenario['Scenario Name'], critical_load_factor), "wb"))

    else:
        print("Not running reopt. Loading Dummy data")
        precompute_path = 'results/reopt_precomputes/'
        precompute_path_abs = Path(__file__).parent / precompute_path
        result = pickle.load(
            open(os.path.join(precompute_path_abs, "results_ATB_moderate_2020_IOWA_0.9.p"), "rb"))

    if result['outputs']['Scenario']['Site']['PV']['size_kw']:
        solar_size_mw = result['outputs']['Scenario']['Site']['PV']['size_kw'] / 1000

    if result['outputs']['Scenario']['Site']['Wind']['size_kw']:
        wind_size_mw = result['outputs']['Scenario']['Site']['Wind']['size_kw'] / 1000

    if result['outputs']['Scenario']['Site']['Storage']['size_kw']:
        storage_size_mw = result['outputs']['Scenario']['Site']['Storage']['size_kw'] / 1000
        storage_size_mwh = result['outputs']['Scenario']['Site']['Storage']['size_kwh'] / 1000
        storage_hours = storage_size_mwh / storage_size_mw

    reopt_site_result = result['outputs']['Scenario']['Site']
    generated_date = pd.date_range(start='1/1/2018 00:00:00', end='12/31/2018 23:00:00', periods=8760)
    if reopt_site_result['Wind']['size_kw'] == 0:

        reopt_site_result['Wind']['year_one_power_production_series_kw'] = np.zeros(8760)
        reopt_site_result['Wind']['year_one_to_grid_series_kw'] = np.zeros(8760)
        reopt_site_result['Wind']['year_one_to_load_series_kw'] = np.zeros(8760)
        reopt_site_result['Wind']['year_one_to_battery_series_kw'] = np.zeros(8760)
        reopt_site_result['Wind']['year_one_curtailed_production_series_kw'] = np.zeros(8760)
        wind_size_mw = 0

    if reopt_site_result['PV']['size_kw'] == 0:
        reopt_site_result['PV']['year_one_power_production_series_kw'] = np.zeros(8760)
        reopt_site_result['PV']['year_one_to_grid_series_kw'] = np.zeros(8760)
        reopt_site_result['PV']['year_one_to_load_series_kw'] = np.zeros(8760)
        reopt_site_result['PV']['year_one_to_battery_series_kw'] = np.zeros(8760)
        reopt_site_result['PV']['year_one_curtailed_production_series_kw'] = np.zeros(8760)
        solar_size_mw = 0

    if reopt_site_result['Storage']['size_kw'] == 0:
        reopt_site_result['Storage']['year_one_soc_series_pct'] = np.zeros(8760)
        reopt_site_result['Storage']['year_one_to_massproducer_series_kw'] = np.zeros(8760)
        storage_size_mw = 0
        storage_size_mwh = 0
        storage_hours = 0

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