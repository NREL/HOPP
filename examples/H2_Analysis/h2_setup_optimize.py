import sys
import os
import hybrid
from dotenv import load_dotenv
from math import sin, pi
import PySAM.Singleowner as so
import pandas as pd
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.run_h2_PEM import run_h2_PEM
import numpy as np
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
from tools.analysis import create_cost_calculator
import json

import warnings
warnings.filterwarnings("ignore")

def setup_power_calcs(scenario,solar_size_mw,storage_size_mwh,storage_size_mw,interconnection_size_mw):
    """
    A function to facilitate plant setup for POWER calculations, assuming one wind turbine.
    
    INPUT VARIABLES
    scenario: dict, the H2 scenario of interest
    solar_size_mw: float, the amount of solar capacity in MW
    storage_size_mwh: float, the amount of battery storate capacity in MWh
    storage_size_mw: float, the amount of battery storate capacity in MW
    interconnection_size_mw: float, the interconnection size in MW

    OUTPUTS
    hybrid_plant: the hybrid plant object from HOPP for power calculations
    """

    # Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

    # Step 1: Establish output structure and special inputs
    year = 2013
    sample_site['year'] = year
    useful_life = 30
    custom_powercurve = True
    electrolyzer_size = 50000

    sample_site['lat'] = scenario['Lat']
    sample_site['lon'] = scenario['Long']
    tower_height = scenario['Tower Height']

    scenario['Useful Life'] = useful_life

    site = SiteInfo(sample_site, hub_height=tower_height)
    
    technologies = {'pv':
                        {'system_capacity_kw': solar_size_mw * 1000},
                    'wind':
                        {'num_turbines': 1,
                            'turbine_rating_kw': scenario['Turbine Rating']*1000,
                            'hub_height': scenario['Tower Height'],
                            'rotor_diameter': scenario['Rotor Diameter']},
                    'battery': {
                        'system_capacity_kwh': storage_size_mwh * 1000,
                        'system_capacity_kw': storage_size_mw * 1000
                        },
                    'grid': {
                        'interconnect_kw': electrolyzer_size
                        }
                    }
    dispatch_options = {'battery_dispatch': 'heuristic'}
    hybrid_plant = HybridSimulation(technologies, site, dispatch_options=dispatch_options)

    hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33   

    if custom_powercurve:
        powercurve_file = open(scenario['Powercurve File'])
        powercurve_data = json.load(powercurve_file)
        powercurve_file.close()
        hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = \
            powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
        hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = \
            powercurve_data['turbine_powercurve_specification']['turbine_power_output']

    hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000

    return hybrid_plant


def setup_cost_calcs(scenario,hybrid_plant,electrolyzer_size_mw,wind_size_mw,solar_size_mw,
                    solar_cost_multiplier=1.0):
    """
    A function to facilitate plant setup for COST calculations. 
    
    INPUT VARIABLES
    scenario: dict, the H2 scenario of interest
    hybrid_plant: the hybrid plant object from setup_power_calcs, so we don't need to reinput everything
    electrolyzer_size_mw: float, the exlectrolyzer capacity in MW
    wind_size_mw: float, the amount of wind capacity in MW
    solar_size_mw: float, the amount of solar capacity in MW
    solar_cost_multiplier: float, if you want to test design sensitivities to solar costs. Multiplies the solar costs

    OUTPUTS
    hybrid_plant: the hybrid plant object from HOPP for cost calculations
    """

    # Step 1: Establish output structure and special inputs
    year = 2013
    sample_site['year'] = year
    useful_life = 30

    sample_site['lat'] = scenario['Lat']
    sample_site['lon'] = scenario['Long']
    wind_cost_kw = scenario['Wind Cost KW']
    solar_cost_kw = scenario['Solar Cost KW']*solar_cost_multiplier
    storage_cost_kw = scenario['Storage Cost KW']
    storage_cost_kwh = scenario['Storage Cost KWh']

    #Todo: Add useful life to .csv scenario input instead
    scenario['Useful Life'] = useful_life

    interconnection_size_mw = electrolyzer_size_mw

    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                              bos_cost_source='CostPerMW',
                                                              wind_installed_cost_mw=wind_cost_kw * 1000,
                                                              solar_installed_cost_mw=solar_cost_kw * 1000,
                                                              storage_installed_cost_mw=storage_cost_kw * 1000,
                                                              storage_installed_cost_mwh=storage_cost_kwh * 1000
                                                              ))
    hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33   
    if solar_size_mw > 0:
        hybrid_plant.pv._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.pv._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['ITC Available']:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 26
        else:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0

    if wind_size_mw > 0:
        hybrid_plant.wind._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['PTC Available']:
            ptc_val = 0.022
        else:
            ptc_val = 0.0

        interim_list = list(
            hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        hybrid_plant.wind._system_model.Turbine.wind_turbine_hub_ht = scenario['Tower Height']

    hybrid_plant.ppa_price = 0.05
    hybrid_plant.wind.system_capacity_kw = wind_size_mw*1000
    return hybrid_plant


def calculate_h_lcoe_continuous(bat_model,electrolyzer_size_mw,wind_size_mw,solar_size_mw,battery_storage_mwh,battery_charge_rate,battery_discharge_rate,
                        scenario,buy_from_grid=False,sell_to_grid=False,solar_cost_multiplier=1.0,interconnection_size_mw=False):
    """
    A function to calculate h_lcoe within an optimization. The wind generation is calculated for a single turbine, then scaled up to the entire capacity.
    This means that no wake losses are captured, but allows the power generation to be continuous with respect to wind capacity. The h_lcoe is not actually 
    continuous at this point, because the costs assume integer values of turbines. This will need to be improved for gradient-based optimization.
    
    INPUT VARIABLES
    bat_model: the battery model to be used. See example scripts.
    electrolyzer_size_mw: float, the exlectrolyzer capacity in MW
    wind_size_mw: float, the amount of wind capacity in MW
    solar_size_mw: float, the amount of solar capacity in MW
    battery_storage_mwh: float, the amount of battery storage in MWh
    battery_charge_rate: float, the battery charge rate in MW
    battery_discharge_rate: float, the battery discharge rate in MW
    scenario: dict, the H2 scenario of interest
    buy_from_grid: Bool, can the plant buy from the grid
    sell_to_grid: Bool, can the plant sell to the grid
    solar_cost_multiplier: float, if you want to test design sensitivities to solar costs. Multiplies the solar costs
    interconnection_size_mw: interconnection size in MW. False if not grid connected


    OUTPUTS
    h_lcoe: float, the levelized cost of hydrogen
    aep: float, the total energy production from the wind and solar
    h2_output: float, the total annual output of hydrogen
    total_system_installed_cost: float, the total system installed costs
    total_annual_operating_costs: float, the total annual operating costs
    electrolyzer_cf: float, the electrolyzer capacity factor
    """

    if interconnection_size_mw:
        interconnection_size = interconnection_size_mw
    else:
        interconnection_size = electrolyzer_size_mw * 1000.0
    wind_size_mw = scenario['Turbine Rating']
    if battery_storage_mwh < 1E-6:
        battery_storage_mwh = 1E-6
    if battery_discharge_rate < 1E-6:
        battery_discharge_rate = 1E-6
        
    hybrid_plant = setup_power_calcs(scenario,solar_size_mw,battery_storage_mwh,battery_discharge_rate,interconnection_size) 

    useful_life = scenario["Useful Life"]

    kw_continuous = electrolyzer_size_mw*1000
    load = [kw_continuous for x in
            range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

    sample_site['lat'] = scenario['Lat']
    sample_site['lon'] = scenario['Long']
    force_electrolyzer_cost = scenario['Force Electrolyzer Cost']
    if force_electrolyzer_cost:
        forced_electrolyzer_cost = scenario['Electrolyzer Cost KW']


    # Step 4: HOPP run
    # ------------------------- #

    hybrid_plant.simulate_power(useful_life)

    # HOPP Specific Energy Metrics
    pv_power_production = hybrid_plant.pv.generation_profile[0:8760]
    wind_power_production_1 = hybrid_plant.wind.generation_profile[0:8760]
    effective_n_turbs = wind_size_mw/scenario['Turbine Rating']
    wind_power_production = np.array(wind_power_production_1)*effective_n_turbs/hybrid_plant.wind.num_turbines


    combined_pv_wind_power_production_hopp = pv_power_production + wind_power_production
    energy_shortfall_hopp = [x - y for x, y in
                             zip(load,combined_pv_wind_power_production_hopp)]
    energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
 
    combined_pv_wind_curtailment_hopp = [x - y for x, y in
                             zip(combined_pv_wind_power_production_hopp,load)]
    combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]

    # Step 5: Run Simple Dispatch Model
    # # ------------------------- #
    bat_model.Nt = len(combined_pv_wind_curtailment_hopp)
    bat_model.curtailment = combined_pv_wind_curtailment_hopp
    bat_model.shortfall = energy_shortfall_hopp

    bat_model.battery_storage = battery_storage_mwh * 1000
    bat_model.charge_rate = battery_charge_rate * 1000
    bat_model.discharge_rate = battery_discharge_rate * 1000

    battery_used, excess_energy, battery_SOC = bat_model.run()
    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used

    if sell_to_grid:
        profit_from_selling_to_grid = np.sum(excess_energy)*sell_to_grid
    else:
        profit_from_selling_to_grid = 0.0

    if buy_from_grid:
        cost_to_buy_from_grid = 0.0
        
        for i in range(len(combined_pv_wind_storage_power_production_hopp)):
            if combined_pv_wind_storage_power_production_hopp[i] < kw_continuous:
                cost_to_buy_from_grid += (kw_continuous-combined_pv_wind_storage_power_production_hopp[i])*buy_from_grid
                combined_pv_wind_storage_power_production_hopp[i] = kw_continuous
    else:
        cost_to_buy_from_grid = 0.0

    energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]
    electrolyzer_cf = np.sum(energy_to_electrolyzer)/(kw_continuous*len(energy_to_electrolyzer))

    # Step 6: Run the Python H2A model
    # ------------------------- #
    hybrid_plant = setup_cost_calcs(scenario,hybrid_plant,electrolyzer_size_mw,wind_size_mw,solar_size_mw,
                        solar_cost_multiplier=solar_cost_multiplier)
    hybrid_plant.simulate_financials(useful_life)
    lcoe = hybrid_plant.lcoe_real.hybrid

    # print("lcoe: ", lcoe)

    electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
    electrical_generation_timeseries[:] = energy_to_electrolyzer[:]
    
    adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
    net_capital_costs = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost

    H2_Results, H2A_Results = run_h2_PEM(electrical_generation_timeseries,electrolyzer_size_mw,
                    kw_continuous,forced_electrolyzer_cost,lcoe,adjusted_installed_cost,useful_life,
                    net_capital_costs)

    # print(H2_Results['hydrogen_annual_output'])
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
    total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
    total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
    annual_operating_cost_hopp = (wind_size_mw * 1000 * 42) + (solar_size_mw * 1000 * 13)
    annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
    total_annual_operating_costs = annual_operating_cost_hopp + annual_operating_cost_h2 + cost_to_buy_from_grid - profit_from_selling_to_grid

    h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                    total_annual_operating_costs, 0.07, useful_life)

    aep = np.sum(combined_pv_wind_power_production_hopp)
    h2_output = H2_Results['hydrogen_annual_output']

    return h_lcoe, aep, h2_output, total_system_installed_cost, total_annual_operating_costs,electrolyzer_cf


if __name__=="__main__":
    bat_model = SimpleDispatch()

    electrolyzer_size_mw = 500
    solar_capacity_mw = 500
    wind_capacity_mw = 500
    battery_storage_mwh = 500
    battery_storage_mw = 0

    N = 20
    h_lcoe = np.zeros(N)
    pv_wind = np.zeros(N)
    hydrogen_annual_output = np.zeros(N)
    total_installed_cost = np.zeros(N)
    total_operating_cost = np.zeros(N)
    CF = np.zeros(N)

    # n_turbines = np.linspace(10,100,N)
    battery_storage_mwh = np.linspace(0.0,1000.0,N)

    scenarios_df = pd.read_csv('single_scenario.csv') 
    for i, s in scenarios_df.iterrows():
        scenario = s

    buy_from_grid = False
    for i in range(N):
        h_lcoe[i], pv_wind[i], hydrogen_annual_output[i], total_installed_cost[i], total_operating_cost[i],CF[i] = \
                        calculate_h_lcoe_continuous(bat_model,electrolyzer_size_mw,wind_capacity_mw,
                        solar_capacity_mw,battery_storage_mwh[i],battery_storage_mw,battery_storage_mw,
                        scenario,buy_from_grid=buy_from_grid,sell_to_grid=False)

    print("h_lcoe: ", h_lcoe)
    print("pv_wind: ", pv_wind)
    print("h2 output: ", hydrogen_annual_output)
    print("total_installed_cost: ", total_installed_cost)
    print("total_operating_cost: ", total_operating_cost)


    metric = battery_storage_mwh
    axis_label=("battery_storage_mwh")

    plt.figure(figsize=(8,6))
    plt.subplot(321)
    plt.plot(metric,h_lcoe,"o")
    plt.xlabel(axis_label)
    plt.title("h_lcoe")

    plt.subplot(322)
    plt.plot(metric,CF,"o")
    plt.xlabel(axis_label)
    plt.title("electrolyzer capacity factor")

    plt.subplot(323)
    plt.plot(metric,hydrogen_annual_output,"o")
    plt.xlabel(axis_label)
    plt.title("hydrogen_annual_output")

    plt.subplot(324)
    plt.plot(metric,total_installed_cost,"o")
    plt.xlabel(axis_label)
    plt.title("total_installed_cost")

    plt.subplot(325)
    plt.plot(metric,total_operating_cost,"o")
    plt.xlabel(axis_label)
    plt.title("total_operating_cost")

    plt.subplot(326)
    plt.plot(metric,pv_wind,"o")
    plt.xlabel(axis_label)
    plt.title("pv_wind")

    plt.tight_layout()
    plt.show()