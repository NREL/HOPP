from audioop import avg
import os
import sys

from matplotlib import use
sys.path.append('')
import pathlib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
from hybrid.keys import set_nrel_key_dot_env
from tools.analysis.bos.cost_calculator import CostCalculator, create_cost_calculator
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
from hybrid.hybrid_simulation import HybridSimulation
from examples.H2_Analysis.degradation import Degradation
from examples.H2_Analysis.failure import Failure
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
from lcoe.lcoe import lcoe as lcoe_calc
from examples.H2_Analysis.desal_model import RO_desal
import warnings
warnings.filterwarnings("ignore")

"""
Analysis file to run HOMP landbased scenarios.

Sets up hybrid simulation to analyze LCOE and LCOH
if degradation and failure of technologies (wind, pv, bss, electrolyzer)
were included.
"""
def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Scenario Name'] = list()
    save_outputs_dict['Site Name'] = list()
    save_outputs_dict['HOMP on/off'] = list()
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
    save_outputs_dict['Electrolyzer Cost kW'] = list()
    save_outputs_dict['Wind Fixed O&M kW'] = list()
    save_outputs_dict['Solar Fixed O&M kW'] = list()
    save_outputs_dict['BSS Fixed O&M kW'] = list()
    save_outputs_dict['Electrolyzer Fixed O&M $/kg'] = list()
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
    save_outputs_dict['Total H2 Production kg'] = list()
    save_outputs_dict['Degraded Total H2 Production kg'] = list()
    save_outputs_dict['Avg Annual H2 production (kg)'] = list()
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
    save_outputs_dict['HOMP on/off'] = (HOMP_options)
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
    save_outputs_dict['Electrolyzer Cost kW'] = (electrolyzer_cost_kw)
    save_outputs_dict['Wind Fixed O&M kW'] = (wind_fixed_om_kw)
    save_outputs_dict['Solar Fixed O&M kW'] = (solar_fixed_om_kw)
    save_outputs_dict['BSS Fixed O&M kW'] = (bss_fixed_om_kw)
    save_outputs_dict['Electrolyzer Fixed O&M $/kg'] = (electrolyzer_fixed_om_kg)
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
    save_outputs_dict['Total H2 Production kg'] = (np.sum(hydrogen_hourly_production))
    #save_outputs_dict['Degraded Total Annual H2 Production kg'] = (np.sum(hybrid_degradation.hydrogen_hourly_production))
    save_outputs_dict['Avg Annual H2 production (kg)'] = (avg_annual_hydrogen_production)
    # save_outputs_dict['Total Wind Repairs'] = (np.sum(hybrid_failure.wind_repair))
    # save_outputs_dict['Total Solar Repairs'] = (np.sum(hybrid_failure.pv_repair))
    # save_outputs_dict['Total Battery Repairs'] = (np.sum(hybrid_degradation.battery_repair))
    # save_outputs_dict['Total Electrolyzer Repairs'] = (np.sum(hybrid_degradation.electrolyzer_repair))
    save_outputs_dict['HOPP LCOE'] = (LCOE)
    # save_outputs_dict['HOMP LCOE (Degradation and Failure)'] = (0)
    save_outputs_dict['HOPP LCOH'] = (LCOH_cf_method)
    # save_outputs_dict['HOMP LCOH (Degradation and Failure)'] = (0)
    # save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    # save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    # save_outputs_dict['HOPP Total Electrical Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    # save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = (total_elec_production)
    # save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall))
    # save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment))
    # save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    # save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    return save_outputs_dict

def save_HOMP_outputs():
    save_outputs_dict['Scenario Name'] = (scenario_choice)
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['HOMP on/off'] = (HOMP_options)
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
    save_outputs_dict['Degraded Total H2 Production kg'] = (np.sum(hybrid_degradation.hydrogen_hourly_production))
    save_outputs_dict['Total Wind Repairs'] = (np.sum(hybrid_failure.wind_repair))
    save_outputs_dict['Total Solar Repairs'] = (np.sum(hybrid_failure.pv_repair))
    save_outputs_dict['Total Battery Repairs'] = (np.sum(hybrid_degradation.battery_repair))
    save_outputs_dict['Total Electrolyzer Repairs'] = (np.sum(hybrid_degradation.electrolyzer_repair))
    save_outputs_dict['HOPP LCOE'] = (0)
    save_outputs_dict['HOMP LCOE (Degradation and Failure)'] = (LCOE_HOMP)
    save_outputs_dict['HOPP LCOH'] = (0)
    save_outputs_dict['HOMP LCOH (Degradation and Failure)'] = (LCOH_cf_method_HOMP)
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

    # Set wind, solar, BSS, electrolyzer CapEx ($/KW)
    #Onshore analysis will use ATB cost/kW 
    wind_cost_kw = scenario['Wind Cost $/KW']
    solar_cost_kw = scenario['Solar Cost $/KW']
    bss_cost_kw = scenario['BSS Cost $/KW']
    bss_cost_kwh = scenario['BSS Cost $/KWh']
    electrolyzer_cost_kw = scenario['Electrolyzer Cost $/KW']

    # Set wind, solar, BSS, electrolyzer Fixed OpEx
    # TODO: Source for wind, solar and bss
    # https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf
    wind_fixed_om_kw = scenario['Wind Fixed O&M $/kW']
    solar_fixed_om_kw = scenario['Solar Fixed O&M $/kW']
    bss_fixed_om_kw = scenario['BSS Fixed O&M $/kW']
    electrolyzer_fixed_om_kg = scenario['Electrolyzer Fixed O&M $/kg']

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

    # Add costs to model
    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                                bos_cost_source='CostPerMW',
                                                                wind_installed_cost_mw=wind_cost_kw * 1000,
                                                                solar_installed_cost_mw=solar_cost_kw * 1000,
                                                                storage_installed_cost_mw=bss_cost_kw * 1000,
                                                                storage_installed_cost_mwh=bss_cost_kwh * 1000
                                                                ))
    
    hybrid_plant.pv.value('om_capacity', (solar_fixed_om_kw,))   # Capacity-based O&M amount [$/kWcap]
    hybrid_plant.wind.value('om_capacity', (wind_fixed_om_kw,)) # Capacity-based O&M amount [$/kWcap]
    if bss_size_mw > 0:
        hybrid_plant.battery.value('om_batt_capacity_cost', (bss_fixed_om_kw,)) # Capacity-based O&M amount [$/kWcap]


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
    for HOMP_options in HOMP:
        if HOMP_options == 'yes':
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
            total_degraded_hydrogen_production = np.sum(hydrogen_deg)
            avg_annual_degraded_hydrogen_production = total_degraded_hydrogen_production/ (useful_life*8760)
            # hybrid_failure.hydrogen_hourly_production = hydrogen_deg

            # Simulate electrolyzer failure
            # hybrid_failure.simulate_electrolyzer_failure(input_hydrogen_production=True)
            
            water_hourly_usage = hybrid_degradation.water_hourly_usage  #Will need to change if electrolyzer also has failure

            # Set up desalination model and run it
            water_usage_electrolyzer = water_hourly_usage
            m3_water_per_kg_h2 = 0.01
            desal_system_size_m3_hr = electrolyzer_size_mw * (1000/55.5) * m3_water_per_kg_h2
            est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate

            Power = [x for x in hybrid_degradation.combined_pv_wind_storage_power_production]
            fresh_water_flowrate, feed_water_flowrate, operational_flags, \
                desal_capex, desal_opex, desal_annuals = \
                    RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=useful_life)
            
            h2_model = "Simple"

            if h2_model == 'H2A':
                #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
                pass  
            elif h2_model == 'Simple':
                #https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf
                stack_capital_cost = 342   #$/kW
                mechanical_bop_cost = 36  #$/kW
                electrical_bop_cost = 82  #$/kW
                installation_factor = 12/100  #%
                stack_replacment_cost = 15/100  #% of installed capital cost
                plant_lifetime = 40    #years
                fixed_OM = 0.24     #$/kg H2
                
                inflation_2016to2022 = 1 + (23.46/100) # 2016$ to 2022$

                electrolyzer_installed_capex_kw = electrolyzer_cost_kw * installation_factor * inflation_2016to2022
                electrolyzer_total_installed_capex = electrolyzer_installed_capex_kw*electrolyzer_size_mw*1000
                
                electrolyzer_fixed_opex = electrolyzer_fixed_om_kg * avg_annual_degraded_hydrogen_production
                cf_h2_annuals = - simple_cash_annuals(useful_life, useful_life, electrolyzer_total_installed_capex,\
                     electrolyzer_fixed_opex, 0.03)
            #print("CF H2 Annuals",cf_h2_annuals)
            
            # Set replacment costs
            inverter_replace_cost = 0.25 * 250              # $0.25/kW for string inverters ($0.14/kW for central inverters); largest string inverters are 250kW and 350kW https://www.nrel.gov/docs/fy22osti/80694.pdf
            wind_replace_cost = 300000                      # Table from HOMP
            battery_replace_cost = 8000                     # Battery replacement cost can be $5k-$11k Table from HOMP
            electro_replace_cost = 0.15 * electrolyzer_total_installed_capex     # 15% of installed CapEx. Table from HOMP

            # Cashflow Financial Calculation
            cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
            if solar_size_mw > 0:
                cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
                print(cf_solar_annuals)
            else:
                cf_solar_annuals = np.zeros(useful_life)
            if bss_size_mw > 0:
                cf_battery_annuals = hybrid_plant.battery._financial_model.Outputs.cf_annual_costs
            else:
                cf_battery_annuals = np.zeros(useful_life)

            ## Add replacement cashflows
            pv_cf = np.add((hybrid_failure.pv_repair * inverter_replace_cost), cf_solar_annuals)
            wind_cf = np.add((hybrid_failure.wind_repair * wind_replace_cost), cf_wind_annuals)
            if bss_size_mw > 0:
                battery_cf = np.add(np.append((hybrid_degradation.battery_repair * battery_replace_cost),[0]), cf_battery_annuals)
            else:
                battery_cf = np.zeros(useful_life)
            electrolyzer_cf = np.add((hybrid_degradation.electrolyzer_repair * electro_replace_cost), cf_h2_annuals)


            # cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals[:len(cf_wind_annuals)]],['Wind', 'Solar', 'H2'])

            # cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))

            #NPVs of wind, solar, BSS and electrolyzer
            npv_wind_costs = npf.npv(discount_rate, wind_cf)
            npv_solar_costs = npf.npv(discount_rate, pv_cf)
            npv_bss_costs = npf.npv(discount_rate, battery_cf)
            npv_h2_costs = npf.npv(discount_rate, electrolyzer_cf)
            print("NPV H2 Costs using {} model: {}".format(h2_model,npv_h2_costs))
            npv_desal_costs = npf.npv(discount_rate, -desal_annuals)

            # Calculate total NPVs
            total_npv_elec = npv_wind_costs + npv_solar_costs + npv_bss_costs
            total_npv_h2 = total_npv_elec + npv_h2_costs + npv_desal_costs
            print("npv bss: ", npv_bss_costs)
           
            LCOE_HOMP = -total_npv_elec / ((np.sum(hybrid_failure.pv_failed_generation) \
                + np.sum(hybrid_failure.wind_failed_generation))*useful_life * 8760)       # $/kWh
            
            LCOH_cf_method_wind = -npv_wind_costs / total_degraded_hydrogen_production
            LCOH_cf_method_solar = -npv_solar_costs / total_degraded_hydrogen_production
            LCOH_cf_method_h2_costs = -npv_h2_costs / total_degraded_hydrogen_production
            LCOH_cf_method_desal_costs = -npv_desal_costs / total_degraded_hydrogen_production

            LCOH_cf_method_HOMP = -total_npv_h2 / total_degraded_hydrogen_production
            financial_summary_df = pd.DataFrame([useful_life, wind_cost_kw, solar_cost_kw, bss_cost_kw, bss_cost_kwh, electrolyzer_cost_kw,
                                                    debt_equity_split, atb_year, ptc_avail, itc_avail,
                                                    discount_rate, npv_wind_costs, npv_solar_costs, npv_bss_costs, npv_h2_costs,
                                                    npv_desal_costs, LCOE_HOMP, LCOH_cf_method_HOMP],
                                                ['Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'BSS Cost KW', 'BSS Cost KWh',
                                                    'Electrolyzer Cost KW', 'Debt Equity','ATB Year', 'PTC available', 'ITC available',
                                                    'Discount Rate', 'NPV Wind Expenses','NPV Solar Expenses', 'NPV BSS Expenses', 
                                                    'NPV H2 Expenses','NPV Desal Expenses', 'LCOE $/kWh','LCOH cf method'])
            financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}.csv'.format(scenario_choice,HOMP_options)))

            print("LCOE HOMP: ", LCOE_HOMP, "$/kWh")
            print("LCOH HOMP:", LCOH_cf_method_HOMP, "$/kg")
            print_results = False
            print_h2_results = False
            save_outputs_dict = save_HOMP_outputs()
            save_all_runs.append(save_outputs_dict)
            save_outputs_dict = establish_save_output_dict()

        if HOMP_options == 'no':
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

            total_hydrogen_production = np.sum(hydrogen_hourly_production)
            avg_annual_hydrogen_production = total_hydrogen_production/(useful_life)
            print('Avg annual h2 production: ', avg_annual_hydrogen_production)
            # Set up desalination model and run it
            water_usage_electrolyzer = water_hourly_usage
            m3_water_per_kg_h2 = 0.01
            desal_system_size_m3_hr = electrolyzer_size_mw * (1000/55.5) * m3_water_per_kg_h2
            est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate

            Power = [x for x in combined_pv_wind_storage_power_production]
            fresh_water_flowrate, feed_water_flowrate, operational_flags,\
                 desal_capex, desal_opex, desal_annuals = \
                     RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=useful_life)

            h2_model = "Simple"

            if h2_model == 'H2A':
                #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
                pass  
            elif h2_model == 'Simple':
                #https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf
                stack_capital_cost = 342   #$/kW
                mechanical_bop_cost = 36  #$/kW
                electrical_bop_cost = 82  #$/kW
                installation_factor = 12/100  #%
                stack_replacment_cost = 15/100  #% of installed capital cost
                plant_lifetime = 40    #years
                fixed_OM = 0.24     #$/kg H2
                
                inflation_2016to2022 = 1 + (23.46/100) # 2016$ to 2022$

                electrolyzer_installed_capex_kw = electrolyzer_cost_kw * installation_factor * inflation_2016to2022
                electrolyzer_total_installed_capex = electrolyzer_installed_capex_kw*electrolyzer_size_mw*1000
                
                electrolyzer_fixed_opex = electrolyzer_fixed_om_kg * avg_annual_hydrogen_production
                electrolyzer_replacement_costs = [0]*useful_life
                cf_h2_annuals = - np.add(simple_cash_annuals(useful_life, useful_life, electrolyzer_total_installed_capex,\
                     electrolyzer_fixed_opex, 0.03),electrolyzer_replacement_costs)
            #print("CF H2 Annuals",cf_h2_annuals)

            # Cashflow Financial Calculation
            cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
            if solar_size_mw > 0:
                cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
            else:
                cf_solar_annuals = np.zeros(useful_life)
            if bss_size_mw > 0:
                cf_battery_annuals = hybrid_plant.battery._financial_model.Outputs.cf_annual_costs
            else:
                cf_battery_annuals = np.zeros(useful_life)


            # cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals[:len(cf_wind_annuals)]],['Wind', 'Solar', 'H2'])

            # cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))

            #NPVs of wind, solar, BSS and electrolyzer
            npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
            npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
            npv_bss_costs = npf.npv(discount_rate, cf_battery_annuals)
            npv_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
            print("NPV H2 Costs using {} model: {}".format(h2_model,npv_h2_costs))
            npv_desal_costs = npf.npv(discount_rate, -desal_annuals)

            # Calculate total NPVs
            total_npv_elec = npv_wind_costs + npv_solar_costs + npv_bss_costs
            total_npv_h2 = total_npv_elec + npv_h2_costs + npv_desal_costs
            print("npv bss: ", npv_bss_costs)
           
            LCOE = -total_npv_elec / ((np.sum(hybrid_plant.pv.generation_profile) + \
                np.sum(hybrid_plant.wind.generation_profile))*useful_life * 8760)       # $/kWh
            
            LCOH_cf_method_wind = -npv_wind_costs / total_hydrogen_production
            LCOH_cf_method_solar = -npv_solar_costs / total_hydrogen_production
            LCOH_cf_method_h2_costs = -npv_h2_costs / total_hydrogen_production
            LCOH_cf_method_desal_costs = -npv_desal_costs / total_hydrogen_production

            LCOH_cf_method = -total_npv_h2 / total_hydrogen_production
            financial_summary_df = pd.DataFrame([useful_life, wind_cost_kw, solar_cost_kw, bss_cost_kw, bss_cost_kwh, electrolyzer_cost_kw,
                                                    debt_equity_split, atb_year, ptc_avail, itc_avail,
                                                    discount_rate, npv_wind_costs, npv_solar_costs, npv_bss_costs, npv_h2_costs,
                                                    npv_desal_costs, LCOE, LCOH_cf_method],
                                                ['Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'BSS Cost KW', 'BSS Cost KWh',
                                                    'Electrolyzer Cost KW', 'Debt Equity','ATB Year', 'PTC available', 'ITC available',
                                                    'Discount Rate', 'NPV Wind Expenses','NPV Solar Expenses', 'NPV BSS Expenses', 
                                                    'NPV H2 Expenses','NPV Desal Expenses', 'LCOE $/kWh','LCOH cf method'])
            financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}.csv'.format(scenario_choice,HOMP_options)))

            print("LCOE: ", LCOE, "$/kWh")
            print("LCOH:", LCOH_cf_method, "$/kg")



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