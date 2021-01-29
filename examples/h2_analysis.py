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
save_outputs_loop['MT C02'] = list()
save_outputs_loop['kW continuous load'] = list()
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
save_outputs_loop['LCOE'] = list()
save_outputs_loop['H2 Elec Feedstock Cost/kW'] = list()

# Get resource
site_name = 'Plainview Bioenergy - Texas'
lat = 46.1  #flatirons_site['lat']
lon = -94.33  #flatirons_site
year = 2013
sample_site['year'] = year
sample_site['lat'] = lat
sample_site['lon'] = lon
tower_height_list = [140]  # [80, 90, 100, 110, 120, 130, 140]
site = SiteInfo(sample_site)

# Set run parameters
# atb_year = 2020  # 2025, 2030
# ptc_avail = True
# itc_avail = True
storage_used = True
on_grid = True  # Storage can charge from grid by default
atb_year_list = [2020]#, 2025, 2030, 'Custom']  # Will determine cost

useful_life_list = [25]#, 30, 35]
itc_avail_list = [True]#, False]
ptc_avail_list = [True]#, False]
debt_equity_split = 90  # %debt
re_opt_optimized_sizing = True


# Set up REopt run
MTC02_yr = 10640  # Metric tons of CO2 per year
MTC02_yr_to_kw_continuous_conversion = 0.2779135  # Metric tons CO2-to kWe conversion (avg. US carbon intensity)
kw_continuous = MTC02_yr * MTC02_yr_to_kw_continuous_conversion
load = [kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

solar_model = SolarPlant(site, 20000)
wind_model = WindPlant(site, 20000)
fin_model = so.default("GenericSystemSingleOwner")
filepath = os.path.dirname(os.path.abspath(__file__))
fileout = os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json")
count = 0

for useful_life in useful_life_list:
    for atb_year in atb_year_list:
        if atb_year == 2020:
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
        else:
            print("Custom ATB year")
            wind_cost_kw = 999
            pv_cost_kw = 999
            storage_cost_kw = 999

        for ptc_avail in ptc_avail_list:
            for itc_avail in itc_avail_list:
                for tower_height in tower_height_list:
                    site = SiteInfo(sample_site, hub_height=tower_height)
                    count = count + 1
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
                    critical_load_pct = 0.6
                    reopt.post['Scenario']['Site']['LoadProfile']['critical_load_pct'] = critical_load_pct
                    reopt.post['Scenario']['Site']['LoadProfile']['outage_start_hour'] = 10
                    reopt.post['Scenario']['Site']['LoadProfile']['outage_end_hour'] = 8750
                    result = reopt.get_reopt_results(force_download=True)

                    # result['outputs']['Scenario']['Site']['Wind']['year_one_to_load_series_kw']
                    solar_size_mw = result['outputs']['Scenario']['Site']['PV']['size_kw'] / 1000
                    wind_size_mw = result['outputs']['Scenario']['Site']['Wind']['size_kw'] / 1000
                    storage_size_mw = result['outputs']['Scenario']['Site']['Storage']['size_kw'] / 1000
                    storage_size_mwh = result['outputs']['Scenario']['Site']['Storage']['size_kwh'] / 1000
                    interconnection_size_mw = reopt.interconnection_limit_kw / 1000

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
                    energy_shortfall = [x if x>0 else 0 for x in energy_shortfall]

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
                                         'energy_shortfall':
                                             energy_shortfall
                                         }

                    REoptResultsDF = pd.DataFrame(reopt_result_dict)
                    # Visualize REopt results
                    import plotly.express as px
                    import matplotlib.pyplot as plt
                    import numpy as np




                    #Set up HOPP run
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
                        hybrid_plant.wind.system_model.Turbine.wind_turbine_hub_ht = tower_height

                    hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
                    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
                    hybrid_plant.ppa_price = 0.05
                    hybrid_plant.simulate(useful_life)

                    # Save the outputs
                    annual_energies = hybrid_plant.annual_energies
                    wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.solar
                    npvs = hybrid_plant.net_present_values
                    lcoe = hybrid_plant.lcoe_real.hybrid
                    feedstock_cost_h2 = lcoe * 44/100  # $/kg
                    feedstock_cost_h2_via_net_cap_cost_lifetime_h2 = result['outputs']['Scenario']['Site']['Financial']['net_capital_costs']\
                                                                     /((kw_continuous/44)*(8760*useful_life))

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
                    print("Levelized cost of H2 (electricity feedstock): {}".format(feedstock_cost_h2))
                    print("kg H2 cost from net cap cost/lifetime h2 production: {}".format(feedstock_cost_h2_via_net_cap_cost_lifetime_h2))

                    # Plot REopt results
                    REoptResultsDF.index = REoptResultsDF.Date
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

                    fig, ax = plt.subplots(figsize=(15, 7))
                    fig2, ax2 = plt.subplots(figsize=(15, 7))
                    for i in y:
                        if i % 24 == 0:
                            ax.plot(y[i: i + 24], df_mean.pv_power_production[i: i + 24], marker="o", label="$PV Power$",
                                    c=colors[1])
                            ax.fill_between(
                                y[i: i + 24],
                                (df_mean.pv_power_production - df_ci.pv_power_production)[i: i + 24],
                                (df_mean.pv_power_production + df_ci.pv_power_production)[i: i + 24],
                                alpha=0.3, color=colors[1], label="$PV Power$ 95% CI"
                            )

                            ax.plot(y[i: i + 24], df_mean.wind_power_production[i: i + 24], marker="o", label="$Wind Power$",
                                    c=colors[0])
                            ax.fill_between(
                                y[i: i + 24],
                                (df_mean.wind_power_production - df_ci.wind_power_production)[i: i + 24],
                                (df_mean.wind_power_production + df_ci.wind_power_production)[i: i + 24],
                                alpha=0.3, color=colors[0], label="$Wind Power$ 95% CI"
                            )
                            ax.plot(y[i: i + 24], df_mean.combined_pv_wind_power_production[i: i + 24], marker="o", label="$Wind + PV Combined$", c=colors[3])
                            ax.fill_between(
                                y[i: i + 24],
                                (df_mean.combined_pv_wind_power_production - df_ci.combined_pv_wind_power_production)[i: i + 24],
                                (df_mean.combined_pv_wind_power_production + df_ci.combined_pv_wind_power_production)[i: i + 24],
                                alpha=0.3, color=colors[3], label="$Wind + PV Combined$ 95% CI"
                            )

                            ax.plot(y[i: i + 24], df_mean.combined_pv_wind_storage_power_production[i: i + 24], marker="o", label="$Wind + PV + Storage Combined$", c=colors[2])
                            ax.fill_between(
                                y[i: i + 24],
                                (df_mean.combined_pv_wind_storage_power_production - df_ci.combined_pv_wind_storage_power_production)[i: i + 24],
                                (df_mean.combined_pv_wind_storage_power_production + df_ci.combined_pv_wind_storage_power_production)[i: i + 24],
                                alpha=0.3, color=colors[2], label="$Wind + PV + Storage Combined$ 95% CI"
                            )

                            ax.plot(y[i: i + 24], df_mean.storage_power_to_load[i: i + 24], marker="o", label="$Storage Power$", c=colors[4])
                            ax.fill_between(
                                y[i: i + 24],
                                (df_mean.storage_power_to_load - df_ci.storage_power_to_load)[i: i + 24],
                                (df_mean.storage_power_to_load + df_ci.storage_power_to_load)[i: i + 24],
                                alpha=0.3, color=colors[4], label="$Storage Power$ 95% CI"
                            )

                            ax2.plot(y[i: i + 24], df_mean.wind_power_curtailed[i: i + 24], marker="o", label="$Wind Curtailed$", c=colors[0])
                            ax2.fill_between(
                                y[i: i + 24],
                                (df_mean.wind_power_curtailed - df_ci.wind_power_curtailed)[i: i + 24],
                                (df_mean.wind_power_curtailed + df_ci.wind_power_curtailed)[i: i + 24],
                                alpha=0.3, color=colors[0], label="$Wind Curtailed$ 95% CI"
                            )

                            ax2.plot(y[i: i + 24], df_mean.pv_power_curtailed[i: i + 24], marker="o", label="$Solar Curtailed$", c=colors[1])
                            ax2.fill_between(
                                y[i: i + 24],
                                (df_mean.pv_power_curtailed - df_ci.pv_power_curtailed)[i: i + 24],
                                (df_mean.pv_power_curtailed + df_ci.pv_power_curtailed)[i: i + 24],
                                alpha=0.3, color=colors[1], label="$Solar Curtailed$ 95% CI"
                            )
                            ax2.plot(y[i: i + 24], df_mean.energy_shortfall[i: i + 24], marker="o", label="$Energy Shortfall$", c=colors[2])
                            ax2.fill_between(
                                y[i: i + 24],
                                (df_mean.energy_shortfall - df_ci.energy_shortfall)[i: i + 24],
                                (df_mean.energy_shortfall + df_ci.energy_shortfall)[i: i + 24],
                                alpha=0.3, color=colors[2], label="$Energy Shortfall$ 95% CI"
                            )

                    xticks_major = [x * 24 for x in range(1, 13)]
                    xticks_minor = list(range(0, 24 * 12, 6))
                    xlabels_major = [month_map[m / 24].ljust(13) for m in xticks_major]
                    xlabels_minor = ["", "06", "12", "18"] + ["06", "12", "18", "24"] * 11

                    ax.set_ylabel("Power (kW)")
                    ax.set_xlim(0, 24)
                    ax.set_xticks(xticks_major)
                    for t in ax.get_xticklabels():
                        t.set_y(-0.05)
                    ax.set_xticks(xticks_minor, minor=True)
                    ax.set_xticklabels(xlabels_major, ha="right")
                    ax.set_xticklabels(xlabels_minor, minor=True)
                    ax.set_xlabel("Hour of day")
                    # TODO:
                    # Add wind size built
                    # Add solar size built
                    # Add storage size built
                    # Add REopt Wind LCOE, Solar LCOE
                    # Add H2 Cost
                    # Add total energy generated
                    # Add total energy curtailed

                    titletext_1 = [["PV and Wind Power at Plainview TX H2 plant - 1/1/2013 - 12/31/2013",
                                   "Critical Load Factor (0-1): {}, Wind Size (MW): {} Solar Size (MW): {} Storage Size (MW): {} Storage Size MWh: {}"\
                                       .format(critical_load_pct, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh)],[
                                   "REopt LCOE Wind: {}, REopt LCOE Solar: {}, HOPP LCOE: {}, H2 Elec Cost ($/kg): {},"
                                   " Total Energy Provided (MWh):"
                                   " {} Total Energy Curtailed (MWh) {}".format(
                                       result['outputs']['Scenario']['Site']['Wind']['lcoe_us_dollars_per_kwh'],
                                        result['outputs']['Scenario']['Site']['PV']['lcoe_us_dollars_per_kwh'],
                                        lcoe,
                                        feedstock_cost_h2, 1, 1)]]
                    ax.set_title(titletext_1)
                    plt.grid(alpha=0.7)
                    plt.grid(alpha=0.2, which="minor")

                    ax2.set_ylabel("Power (kW)")
                    ax2.set_xlim(0, 24)
                    ax2.set_xticks(xticks_major)
                    for t in ax2.get_xticklabels():
                        t.set_y(-0.05)
                    ax2.set_xticks(xticks_minor, minor=True)
                    ax2.set_xticklabels(xlabels_major, ha="right")
                    ax2.set_xticklabels(xlabels_minor, minor=True)
                    ax2.set_xlabel("Hour of day")
                    ax2.set_title(titletext_1)

                    handles, labels = ax.get_legend_handles_labels()
                    labels_set = ["$PV Power$", "$PV Power$ 95% CI",
                                  "$Wind Power$", "$Wind Power$ 95% CI",
                                  "$Wind + PV Combined$", "$Wind + PV Combined$ 95% CI",
                                  "$Wind + PV + Storage Combined$", "$Wind + PV + Storage Combined$ 95% CI",
                                  "$Storage Power$", "$Storage Power$ 95% CI"]

                    handles2, labels2 = ax2.get_legend_handles_labels()
                    labels_set2 = ["$Wind Curtailed$", "$Wind Curtailed$ 95% CI",
                                   "$Solar Curtailed$", "$Solar Curtailed$ 95% CI",
                                   "$Energy Shortfall$", "$Energy Shortfall$ 95% CI"]

                    ax.grid(alpha=0.7)
                    ax.grid(alpha=0.2, which="minor")

                    ax2.grid(alpha=0.7)
                    ax2.grid(alpha=0.2, which="minor")

                    ix_filter = [labels.index(el) for el in labels_set]
                    handles = [handles[ix] for ix in ix_filter]
                    labels = [labels[ix] for ix in ix_filter]
                    ax.legend(handles, labels, ncol=5, loc="lower left")

                    ix_filter2 = [labels2.index(el) for el in labels_set2]
                    handles2 = [handles2[ix] for ix in ix_filter2]
                    labels2 = [labels2[ix] for ix in ix_filter2]
                    ax2.legend(handles2, labels2, ncol=3, loc="lower left")

                    plt.tight_layout()
                    fig2.tight_layout()
                    plt.savefig("wind_pv_plainview_production.png", dpi=240, bbox_to_inches="tight")
                    fig2.savefig("wind_pv_plainview_shortfall_curtailment.png", dpi=240, bbox_to_inches="tight")
                    plt.show()
                    fig2.show()

                    save_outputs_loop['Count'].append(count)
                    save_outputs_loop['Site Lat'].append(lat)
                    save_outputs_loop['Site Lon'].append(lon)
                    save_outputs_loop['ATB Year'].append(atb_year)
                    save_outputs_loop['Resource Year'].append(year)
                    save_outputs_loop['Site Name'].append(site_name)
                    save_outputs_loop['MT C02'].append(MTC02_yr)
                    save_outputs_loop['kW continuous load'].append(kw_continuous)
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
                    save_outputs_loop['LCOE'].append(lcoe)
                    save_outputs_loop['H2 Elec Feedstock Cost/kW'].append(feedstock_cost_h2)

# save_all_runs = save_all_runs.append(save_outputs_loop, sort=False)

save_outputs_loop_df = pd.DataFrame(save_outputs_loop)
save_outputs_loop_df.to_csv("H2_Analysis_{}.csv".format(site_name))

