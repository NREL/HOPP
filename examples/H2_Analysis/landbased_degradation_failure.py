from pathlib import Path
import pathlib
from matplotlib import use
import matplotlib.pyplot as plt
from hybrid.hybrid_simulation import HybridSimulation
from examples.H2_Analysis.degradation import Degradation
from examples.H2_Analysis.failure import Failure
from examples.H2_Analysis.desal_model import RO_desal
import numpy as np
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env
from tools.analysis.bos.cost_calculator import create_cost_calculator

import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Scenario Name'] = list()
    save_outputs_dict['Site Name'] = list()
    #save_outputs_dict['Substructure Technology'] = list()
    save_outputs_dict['Site Lat'] = list()
    save_outputs_dict['Site Lon'] = list()
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Turbine Model'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['System Load (kW)'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['PTC'] = list()
    save_outputs_dict['ITC'] = list()
    save_outputs_dict['Discount Rate'] = list()
    save_outputs_dict['Debt Equity'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['BSS Enabled'] = list()
    save_outputs_dict['Wind Cost kW'] = list()
    save_outputs_dict['Solar Cost kW'] = list()
    save_outputs_dict['BSS Cost kW'] = list()
    save_outputs_dict['BSS Cost kWh'] = list()
    save_outputs_dict['Wind MW built'] = list()
    save_outputs_dict['Solar MW built'] = list()
    save_outputs_dict['BSS MW built'] = list()
    save_outputs_dict['BSS MWh built'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Built Interconnection Size'] = list()
    save_outputs_dict['Wind Generation MW'] = list()
    save_outputs_dict['Degraded Wind Generation MW'] = list()
    save_outputs_dict['Failed Wind Generation MW'] = list()
    save_outputs_dict['Solar Generation MW'] = list()
    save_outputs_dict['Degraded Solar Generation MW'] = list()
    save_outputs_dict['Failed Solar Generation MW'] = list()
    save_outputs_dict['Hybrid Generation MW'] = list()
    save_outputs_dict['Degraded Hybrid Generation MW'] = list()
    save_outputs_dict['Failed Hybrid Generation MW'] = list()
    save_outputs_dict['Battery Used MW'] = list()
    save_outputs_dict['Degraded Battery Used MW'] = list()
    save_outputs_dict['Total Annual H2 Production kg'] = list()
    save_outputs_dict['Degraded Total Annual H2 Production kg'] = list()
    save_outputs_dict['Total Wind Repairs'] = list()
    save_outputs_dict['Total Solar Repairs'] = list()
    save_outputs_dict['Total Battery Repairs'] = list()
    save_outputs_dict['Total Electrolyzer Repairs'] = list()
    save_outputs_dict['HOPP LCOE'] = list()
    save_outputs_dict['HOMP LCOE (Degradation and Failure)'] = list()
    save_outputs_dict['HOPP LCOH'] = list()
    save_outputs_dict['HOMP LCOH (Degradation and Failure)'] = list()
    
    return save_outputs_dict

def save_HOPP_outputs():
    save_outputs_dict['Scenario Name'] = (scenario_choice)
    save_outputs_dict['Site Name'] = (site_name)
    #save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['Site Lat'] = (sample_site['lat'])
    save_outputs_dict['Site Lon'] = (sample_site['lon'])
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (sample_site['year'])
    save_outputs_dict['Turbine Model'] = (turbine_name)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (static_load)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['PTC'] = (ptc_avail)
    save_outputs_dict['ITC'] = (itc_avail)
    save_outputs_dict['Discount Rate'] = (discount_rate)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['BSS Enabled'] = (bss_used)
    save_outputs_dict['Wind Cost kW'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost kW'] = (solar_cost_kw)
    save_outputs_dict['BSS Cost kW'] = (bss_cost_kw)
    save_outputs_dict['BSS Cost kWh'] = (bss_cost_kwh)
    save_outputs_dict['Wind MW built'] = (wind_size_mw)
    save_outputs_dict['Solar MW built'] = (solar_size_mw)
    save_outputs_dict['BSS MW built'] = (bss_size_mw)
    save_outputs_dict['BSS MWh built'] = (bss_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Wind Generation MW'] = (np.sum(hybrid_plant.wind.generation_profile)/1000)
    #save_outputs_dict['Degraded Wind Generation MW'] = (np.sum(hybrid_degradation.wind_degraded_generation)/1000)
    #save_outputs_dict['Failed Wind Generation MW'] = (np.sum(hybrid_failure.wind_failed_generation)/1000)
    save_outputs_dict['Solar Generation MW'] = (np.sum(hybrid_plant.pv.generation_profile)/1000)
    #save_outputs_dict['Degraded Solar Generation MW'] = (np.sum(hybrid_degradation.pv_degraded_generation)/1000)
    #save_outputs_dict['Failed Solar Generation MW'] = (np.sum(hybrid_failure.pv_failed_generation)/1000)
    save_outputs_dict['Hybrid Generation MW'] = (np.sum(np.add(hybrid_plant.pv.generation_profile, hybrid_plant.wind.generation_profile))/1000)
    #save_outputs_dict['Degraded Hybrid Generation MW'] = (np.sum(np.add(hybrid_degradation.pv_degraded_generation, hybrid_degradation.wind_degraded_generation))/1000)
    #save_outputs_dict['Failed Hybrid Generation MW'] = (np.sum(np.add(hybrid_failure.pv_failed_generation, hybrid_failure.wind_failed_generation))/1000)
    save_outputs_dict['Battery Used MW'] = (np.sum(battery_used)/1000)
    #save_outputs_dict['Degraded Battery Used MW'] = (np.sum(battery_deg)/1000)
    save_outputs_dict['Total Annual H2 Production kg'] = (np.sum(hydrogen_hourly_production))
    #save_outputs_dict['Degraded Total Annual H2 Production kg'] = (np.sum(hybrid_degradation.hydrogen_hourly_production))
    # save_outputs_dict['Total Wind Repairs'] = (np.sum(hybrid_failure.wind_repair))
    # save_outputs_dict['Total Solar Repairs'] = (np.sum(hybrid_failure.pv_repair))
    # save_outputs_dict['Total Battery Repairs'] = (np.sum(hybrid_degradation.battery_repair))
    # save_outputs_dict['Total Electrolyzer Repairs'] = (np.sum(hybrid_degradation.electrolyzer_repair))
    save_outputs_dict['HOPP LCOE'] = (0)
    # save_outputs_dict['HOMP LCOE (Degradation and Failure)'] = (0)
    save_outputs_dict['HOPP LCOH'] = (0)
    # save_outputs_dict['HOMP LCOH (Degradation and Failure)'] = (0)
    # save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    # save_outputs_dict['LCOE'] = (lcoe)
    # save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    # save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = (gut_check_h2_cost_kg)
    # save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method)
    # save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs)
    # save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_pipeline)
    # save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs_pipeline)
    # save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    # save_outputs_dict['HOPP Total Electrical Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    # save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = (total_elec_production)
    # save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    # save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    # save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    # save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    # save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    # save_outputs_dict['Electrolyzer Stack Size'] = (H2A_Results['electrolyzer_size'])
    # save_outputs_dict['Electrolyzer Total System Size'] = (H2A_Results['total_plant_size'])
    # save_outputs_dict['H2A scaled total install cost'] = (H2A_Results['scaled_total_installed_cost'])
    # save_outputs_dict['H2A scaled total install cost per kw'] = (H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict

def save_HOMP_outputs():
    save_outputs_dict['Scenario Name'] = (scenario_choice)
    save_outputs_dict['Site Name'] = (site_name)
    #save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['Site Lat'] = (sample_site['lat'])
    save_outputs_dict['Site Lon'] = (sample_site['lon'])
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (sample_site['year'])
    save_outputs_dict['Turbine Model'] = (turbine_name)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (static_load)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['PTC'] = (ptc_avail)
    save_outputs_dict['ITC'] = (itc_avail)
    save_outputs_dict['Discount Rate'] = (discount_rate)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['BSS Enabled'] = (bss_used)
    save_outputs_dict['Wind Cost kW'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost kW'] = (solar_cost_kw)
    save_outputs_dict['BSS Cost kW'] = (bss_cost_kw)
    save_outputs_dict['BSS Cost kWh'] = (bss_cost_kwh)
    save_outputs_dict['Wind MW built'] = (wind_size_mw)
    save_outputs_dict['Solar MW built'] = (solar_size_mw)
    save_outputs_dict['BSS MW built'] = (bss_size_mw)
    save_outputs_dict['BSS MWh built'] = (bss_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Wind Generation MW'] = (np.sum(hybrid_plant.wind.generation_profile)/1000)
    save_outputs_dict['Degraded Wind Generation MW'] = (np.sum(hybrid_degradation.wind_degraded_generation)/1000)
    save_outputs_dict['Failed Wind Generation MW'] = (np.sum(hybrid_failure.wind_failed_generation)/1000)
    save_outputs_dict['Solar Generation MW'] = (np.sum(hybrid_plant.pv.generation_profile)/1000)
    save_outputs_dict['Degraded Solar Generation MW'] = (np.sum(hybrid_degradation.pv_degraded_generation)/1000)
    save_outputs_dict['Failed Solar Generation MW'] = (np.sum(hybrid_failure.pv_failed_generation)/1000)
    save_outputs_dict['Hybrid Generation MW'] = (np.sum(np.add(hybrid_plant.pv.generation_profile, hybrid_plant.wind.generation_profile))/1000)
    save_outputs_dict['Degraded Hybrid Generation MW'] = (np.sum(np.add(hybrid_degradation.pv_degraded_generation, hybrid_degradation.wind_degraded_generation))/1000)
    save_outputs_dict['Failed Hybrid Generation MW'] = (np.sum(np.add(hybrid_failure.pv_failed_generation, hybrid_failure.wind_failed_generation))/1000)
    # save_outputs_dict['Battery Used MW'] = (np.sum(battery_used)/1000)
    save_outputs_dict['Degraded Battery Used MW'] = (np.sum(battery_deg)/1000)
    # save_outputs_dict['Total Annual H2 Production kg'] = (np.sum(hydrogen_hourly_production))
    save_outputs_dict['Degraded Total Annual H2 Production kg'] = (np.sum(hybrid_degradation.hydrogen_hourly_production))
    save_outputs_dict['Total Wind Repairs'] = (np.sum(hybrid_failure.wind_repair))
    save_outputs_dict['Total Solar Repairs'] = (np.sum(hybrid_failure.pv_repair))
    save_outputs_dict['Total Battery Repairs'] = (np.sum(hybrid_degradation.battery_repair))
    save_outputs_dict['Total Electrolyzer Repairs'] = (np.sum(hybrid_degradation.electrolyzer_repair))
    save_outputs_dict['HOPP LCOE'] = (0)
    save_outputs_dict['HOMP LCOE (Degradation and Failure)'] = (0)
    save_outputs_dict['HOPP LCOH'] = (0)
    save_outputs_dict['HOMP LCOH (Degradation and Failure)'] = (0)
    # save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    # save_outputs_dict['LCOE'] = (lcoe)
    # save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    # save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = (gut_check_h2_cost_kg)
    # save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method)
    # save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs)
    # save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_pipeline)
    # save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs_pipeline)
    # save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    # save_outputs_dict['HOPP Total Electrical Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    # save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = (total_elec_production)
    # save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    # save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    # save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    # save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    # save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    # save_outputs_dict['Electrolyzer Stack Size'] = (H2A_Results['electrolyzer_size'])
    # save_outputs_dict['Electrolyzer Total System Size'] = (H2A_Results['total_plant_size'])
    # save_outputs_dict['H2A scaled total install cost'] = (H2A_Results['scaled_total_installed_cost'])
    # save_outputs_dict['H2A scaled total install cost per kw'] = (H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict
    
# examples_dir = Path(__file__).resolve().parents[1]

# Set API key
set_nrel_key_dot_env()

# Establish place to save all outputs from analysis
save_outputs_dict = establish_save_output_dict()
save_all_runs = list()

# Load scenarios from .csv and enumerate
parent_path = os.path.abspath('')
print(parent_path)
scenarios_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/default_HOMP_scenarios.csv'))
scenarios_df

# Runs through each scenario in HOMP scenarios file
for i, scenario in scenarios_df.iterrows():
    scenario_choice = scenario['Scenario Name']
    
    # Set plant location
    site_name = scenario['Site Name']
    location_type = scenario['Location Type']
    sample_site['lat'] = scenario['Lat']
    sample_site['lon'] = scenario['Long']
    sample_site['year'] = scenario['Resource Year']

    # Set financial assumptions and plant life
    atb_year = scenario['ATB Year']
    ptc_avail = scenario['PTC Available']   #TODO: What PTC should be used?
    itc_avail = scenario['ITC Available']   #TODO: What ITC should be used?
    discount_rate = scenario['Discount Rate']
    debt_equity_split = scenario['Debt Equity']
    useful_life = scenario['Useful Life']

    # Set wind, solar, BSS, electrolyzer and interconnection capacities (in MW)
    # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_7MW_200.html
    wind_size_mw = scenario['Wind Size MW']
    solar_size_mw = scenario['Solar Size MW']
    bss_size_mw = scenario['BSS Size MW']
    bss_size_mwh = scenario['BSS Size MWh']
    electrolyzer_size_mw = scenario['Electrolyzer Size MW']
    interconnection_size_mw = scenario['Interconnect Size MW']

    # Set wind, solar, BSS, electrolyzer and interconnection CapEx (in $/KW)
    #Onshore analysis will use ATB cost/kW 
    wind_cost_kw = scenario['Wind Cost KW']
    solar_cost_kw = scenario['Solar Cost KW']
    bss_cost_kw = scenario['BSS Cost KW']
    bss_cost_kwh = scenario['BSS Cost KWh']
    electrolyzer_cost_kw = scenario['Electrolyzer Cost KW']

    # Set wind turbine characteristics
    turbine_name = scenario['Turbine Name']
    turbine_rating_mw = scenario['Turbine Rating MW']
    tower_height = scenario['Tower Height']
    rotor_diameter = scenario['Rotor Diameter']
    custom_powercurve_path = scenario['Powercurve File']

    # Set grid connection
    buy = scenario['Buy From Grid']
    sell = scenario['Sell To Grid']

    if buy and sell:
        buy_price = 0.01
        sell_price = 0.01
    else:
        buy_price = False
        sell_price = False
    
    #If solar size = 0, remove request for solar data
    if solar_size_mw == 0:
        sample_site['no_solar'] = True

    if bss_size_mw > 0:
        bss_used = False
    else:
        bss_used = True

    # Set up site location through SiteInfo Class
    site = SiteInfo(sample_site, hub_height=tower_height)

    # Set plant load to be electrolyzer rated capacity
    load = [electrolyzer_size_mw*1000] * useful_life * 8760
    static_load = electrolyzer_size_mw*1000

    # TODO: Make sure all of these are what I want and correct
    critical_load_factor = 1
    custom_powercurve = True
    battery_can_grid_charge = False
    grid_connected_hopp = False


    # which plots to show
    plot_power_production = True
    plot_battery = True
    plot_grid = True
    plot_h2 = True

    scenario = dict()

    parent_path = os.path.abspath('')
    results_dir = parent_path + '/examples/H2_Analysis/results/'

    # Plot wind data to ensure it's sound
    # wind_data = site.wind_resource._data['data']
    # wind_speed = [x[2] for x in wind_data]
    # plt.plot(wind_speed)
    # plt.title('Wind Speed (m/s) for selected location \n {} \n Average Wind Speed (m/s) {}'.format(site_name,np.average(wind_speed)))
    # plt.savefig(os.path.join(results_dir,'Average Wind Speed_{}'.format(site_name)),bbox_inches='tight')


    if solar_size_mw > 0 and bss_size_mw > 0:
        print("Wind, Solar, and Storage")
        technologies = {'pv':
                            {'system_capacity_kw': solar_size_mw * 1000},
                        'wind':
                            {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'hub_height': tower_height,
                                'rotor_diameter': rotor_diameter},
                        'battery': {
                            'system_capacity_kwh': bss_size_mwh * 1000,
                            'system_capacity_kw': bss_size_mw * 1000
                            }
                        }

    elif solar_size_mw > 0:
        print("Wind and Solar, no storage")
        technologies = {'pv':
                    {'system_capacity_kw': solar_size_mw * 1000},
                'wind':
                    {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                        'turbine_rating_kw': turbine_rating_mw*1000,
                        'hub_height': tower_height,
                        'rotor_diameter': rotor_diameter},
                }
    else:
        print('Wind Only')
        technologies = {
            'wind':
                {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                    'turbine_rating_kw': turbine_rating_mw*1000,
                    'hub_height': tower_height,
                    'rotor_diameter': rotor_diameter},
            }

    # Create model
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                                bos_cost_source='CostPerMW',
                                                                wind_installed_cost_mw=wind_cost_kw * 1000,
                                                                solar_installed_cost_mw=solar_cost_kw * 1000,
                                                                storage_installed_cost_mw=bss_cost_kw * 1000,
                                                                storage_installed_cost_mwh=bss_cost_kwh * 1000
                                                                ))
    # TODO: Add O&M costs for each technology and determine whether if statement is needed when battery off                                                              
    hybrid_plant.set_om_costs_per_kw(pv_om_per_kw=None, wind_om_per_kw=None, hybrid_om_per_kw=None)

    if solar_size_mw > 0:
        hybrid_plant.pv._financial_model.FinancialParameters.analysis_period = useful_life
        hybrid_plant.pv._financial_model.FinancialParameters.debt_percent = debt_equity_split
        # hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
        if itc_avail:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 26
        else:
            hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0

    if 'wind' in technologies:
        hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33
        hybrid_plant.wind.wake_model = 3
        hybrid_plant.wind.value("wake_int_loss", 3)
        hybrid_plant.wind._financial_model.FinancialParameters.analysis_period = useful_life
        # hybrid_plant.wind.om_capacity = 
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = debt_equity_split
        hybrid_plant.wind._financial_model.value("debt_option", 0)
        if ptc_avail == 'yes':
            ptc_val = 0.025
        elif ptc_avail == 'no':
            ptc_val = 0.0

        interim_list = list(
            hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        hybrid_plant.wind._system_model.Turbine.wind_turbine_hub_ht = tower_height

    if custom_powercurve:
        parent_path = os.path.abspath(os.path.dirname(__file__))
        powercurve_file = open(os.path.join(parent_path, custom_powercurve_path))
        powercurve_file_extension = pathlib.Path(os.path.join(parent_path, custom_powercurve_path)).suffix
        if powercurve_file_extension == '.csv':
            curve_data = pd.read_csv(os.path.join(parent_path, custom_powercurve_path))            
            wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
            curve_power = curve_data['Power [kW]']
            hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
            hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power

        else:
            powercurve_data = json.load(powercurve_file)
            powercurve_file.close()
            hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = \
                powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
            hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = \
                powercurve_data['turbine_powercurve_specification']['turbine_power_output']


    hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hybrid_plant.ppa_price = 0.05
    hybrid_plant.simulate(useful_life)

    HOMP = ['yes', 'no']
    for options in HOMP:
        if options == 'yes':
            # Save the outputs
            generation_profile = hybrid_plant.generation_profile

            # Instantiate degradation
            hybrid_degradation = Degradation(technologies, True, electrolyzer_size_mw, useful_life, generation_profile, load)

            # Simulate wind and pv degradation
            hybrid_degradation.simulate_generation_degradation()

            # Assign output from generation degradation to be fed into failure class
            pv_deg = [x for x in hybrid_degradation.pv_degraded_generation]
            wind_deg = [x for x in hybrid_degradation.wind_degraded_generation]

            degraded_generation = dict()
            degraded_generation['pv'] = pv_deg
            degraded_generation['wind'] = wind_deg

            # Instantiate failure 
            # Use degraded generation profiles
            hybrid_failure = Failure(technologies, True, electrolyzer_size_mw, useful_life,degraded_generation,load, False)

            # Add failures to generation technology (pv and wind)
            hybrid_failure.simulate_generation_failure()

            # Feeds generation profile that has degradation and failure into battery simulation
            pv_deg_fail = [x for x in hybrid_failure.pv_failed_generation]
            wind_deg_fail = [x for x in hybrid_failure.wind_failed_generation]
            hybrid_degradation.hybrid_degraded_generation = np.add(pv_deg_fail, wind_deg_fail)

            if bss_size_mw > 0:
                # Simulates battery degradation
                hybrid_degradation.simulate_battery_degradation()

                battery_deg = [x for x in hybrid_degradation.battery_used]

                # Simulates battery failure
                # hybrid_failure.battery_used = battery_deg

                # hybrid_failure.simulate_battery_failure(input_battery_use=True)

                # Set combined_pv_wind_storage_power_production for electrolyzer simulation
                # battery_deg_fail = [x for x in hybrid_failure.battery_used]
            else:
                battery_deg = [0] * useful_life * 8760
                hybrid_degradation.battery_repair = [0]

            hybrid_degradation.combined_pv_wind_storage_power_production = np.add(np.add(pv_deg_fail, wind_deg_fail), battery_deg)

            # Simulate electrolyzer degradation
            hybrid_degradation.simulate_electrolyzer_degradation()

            # Set degraded hydrogen production for electrolyzer failure
            hydrogen_deg = [x for x in hybrid_degradation.hydrogen_hourly_production]

            # hybrid_failure.hydrogen_hourly_production = hydrogen_deg

            # Simulate electrolyzer failure
            # hybrid_failure.simulate_electrolyzer_failure(input_hydrogen_production=True)
            
            water_hourly_usage = hybrid_degradation.water_hourly_usage  #Will need to change if electrolyzer also has failure

            water_usage_electrolyzer = water_hourly_usage
            m3_water_per_kg_h2 = 0.01
            desal_system_size_m3_hr = electrolyzer_size_mw * (1000/55.5) * m3_water_per_kg_h2
            est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate
            # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
            Power = [x for x in hybrid_degradation.combined_pv_wind_storage_power_production]
            fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals = RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=useful_life)
            print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size_mw, desal_system_size_m3_hr))
            print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
            print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
            print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
            # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
            print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

            print_results = False
            print_h2_results = False
            save_outputs_dict = save_HOMP_outputs()
            save_all_runs.append(save_outputs_dict)
            save_outputs_dict = establish_save_output_dict()

        if options == 'no':
            if bss_size_mw > 0:
                hybrid_generation = np.add(hybrid_plant.wind.generation_profile, hybrid_plant.pv.generation_profile)

                energy_shortfall = [x - y for x, y in
                                        zip(load,hybrid_generation)]
                energy_shortfall = [x if x > 0 else 0 for x in energy_shortfall]
                combined_pv_wind_curtailment = [x - y for x, y in
                                    zip(hybrid_generation,load)]
                combined_pv_wind_curtailment = [x if x > 0 else 0 for x in combined_pv_wind_curtailment]

                bat_model = SimpleDispatch()
                bat_model.Nt = len(energy_shortfall)
                bat_model.curtailment = combined_pv_wind_curtailment
                bat_model.shortfall = energy_shortfall
                bat_model.battery_storage = bss_size_mw
                bat_model.charge_rate = bss_size_mwh
                bat_model.discharge_rate = bss_size_mwh
                battery_used, excess_energy, battery_SOC = bat_model.run()
            else:
                battery_used = [0] * useful_life * 8760

            combined_pv_wind_storage_power_production = np.add(hybrid_generation, battery_used)
            
            kw_continuous = electrolyzer_size_mw * 1000
            energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production]
            electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
            electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

            in_dict = dict()
            in_dict['electrolyzer_system_size_MW'] = electrolyzer_size_mw
            in_dict['P_input_external_kW'] = electrical_generation_timeseries
            out_dict = dict()
            hydrogen_hourly_production = []
            water_houly_usage = []

            el = PEM_electrolyzer_LT(in_dict, out_dict)
            el.h2_production_rate()
            el.water_supply()

            hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
            water_hourly_usage = out_dict['water_used_kg_hr']

            water_usage_electrolyzer = water_hourly_usage
            m3_water_per_kg_h2 = 0.01
            desal_system_size_m3_hr = electrolyzer_size_mw * (1000/55.5) * m3_water_per_kg_h2
            est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate
            # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
            Power = [x for x in combined_pv_wind_storage_power_production]
            fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals = RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=useful_life)
            print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size_mw, desal_system_size_m3_hr))
            print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
            print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
            print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
            # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
            print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

            print_results = False
            print_h2_results = False
            save_outputs_dict = save_HOPP_outputs()
            save_all_runs.append(save_outputs_dict)
            save_outputs_dict = establish_save_output_dict()

    wind_installed_cost = hybrid_plant.wind.total_installed_cost
    if solar_size_mw > 0:
        solar_installed_cost = hybrid_plant.pv.total_installed_cost
    else:
        solar_installed_cost = 0
    if bss_size_mw > 0:
        bss_installed_cost = hybrid_plant.battery.total_installed_cost
    else:
        bss_installed_cost = 0

    hybrid_installed_cost = hybrid_plant.grid.total_installed_cost



    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.plot(fresh_water_flowrate[200:300],"--",label="Freshwater flowrate from desal")
    # plt.plot(feed_water_flowrate[200:300],"--",label="Feedwater flowrate to desal")
    # plt.legend()
    # plt.title('Freshwater flowrate (m^3/hr) from desal  (Snapshot)')
    # # plt.show()

    # plt.subplot(1,2,2)
    # plt.plot(operational_flags[200:300],"--",label="Operational Flag")
    # plt.legend()
    # plt.title('Desal Equipment Operational Status (Snapshot) \n 0 = Not enough power to operate \n 1 = Operating at reduced capacity \n 2 = Operating at full capacity')
    # plt.savefig(os.path.join(results_dir,'Desal Flows_{}_{}_{}'.format(site_name,atb_year,ptc_avail)),bbox_inches='tight')
    # plt.show()

plot_degradation = False
plot_failure = False

if plot_degradation:
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.title("Max power generation vs degraded power generation")
    plt.plot(hybrid_degradation.wind_degraded_generation[175200:175344],label="degraded wind")
    plt.plot(hybrid_plant.wind.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()
    
    plt.subplot(312)
    plt.plot(hybrid_degradation.pv_degraded_generation[175200:175344],label="degraded pv")
    plt.plot(hybrid_plant.pv.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()

    plt.subplot(313)
    plt.plot(hybrid_degradation.hybrid_degraded_generation[175200:175344], label="degraded hybrid generation")
    plt.plot(load[175200:175344], label = "load profile")
    plt.ylabel("Power Production (kW)")
    plt.xlabel("Time (hour)")
    plt.legend()
    plt.show()

if plot_failure:
    plt.figure(figsize=(10,6))
    plt.subplot(411)
    plt.title("Max power generation vs failed power generation")
    plt.plot(hybrid_failure.wind_failed_generation[175200:175344],label="failed wind")
    plt.plot(hybrid_plant.wind.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()
    
    plt.subplot(412)
    plt.plot(hybrid_failure.pv_failed_generation[175200:175344],label="failed pv")
    plt.plot(hybrid_plant.pv.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()

    plt.subplot(413)
    plt.plot(hybrid_failure.hybrid_failure_generation[175200:175344], label="failed hybrid generation")
    plt.plot(load[175200:175344], label = "load profile")
    plt.ylabel("Power Production (kW)")
    plt.xlabel("Time (hour)")
    plt.legend()

    plt.plot(figsize=(10,8))
    plt.plot(hybrid_degradation.hydrogen_hourly_production[61300:61440], label ='degraded')
    plt.plot(hybrid_failure.hydrogen_hourly_production[61300:61440],"--", label = 'failed')
    plt.title('Hydrogen production rate [kg/hr]')
    plt.legend()
    plt.show()

save_outputs = True
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    save_all_runs_df.to_csv(os.path.join(results_dir, "HOMP_Analysis.csv"))


print('Done')