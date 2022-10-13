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
    save_outputs_dict['Grid Connected HOPP'].append(grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'].append(np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'].append(hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'].append(np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'].append(np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'].append(np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'].append(np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size'].append(H2A_Results['electrolyzer_size'])
    save_outputs_dict['Electrolyzer Total System Size'].append(H2A_Results['total_plant_size'])
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
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    save_outputs_dict['Percentage Time 50% Load met without storage'] = (perc_time_load_met_no_storage_50)
    save_outputs_dict['Percentage Time 75% Load met without storage'] = (perc_time_load_met_no_storage_75)
    save_outputs_dict['Percentage Time 90% Load met without storage'] = (perc_time_load_met_no_storage_90)
    save_outputs_dict['Percentage Time 95% Load met without storage'] = (perc_time_load_met_no_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met without storage'] = (perc_time_load_met_no_storage_100)
    save_outputs_dict['Percentage Time 50% Load met with storage'] = (perc_time_load_met_with_storage_50)
    save_outputs_dict['Percentage Time 75% Load met with storage'] = (perc_time_load_met_with_storage_75)
    save_outputs_dict['Percentage Time 90% Load met with storage'] = (perc_time_load_met_with_storage_90)
    save_outputs_dict['Percentage Time 95% Load met with storage'] = (perc_time_load_met_with_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met with storage'] = (perc_time_load_met_with_storage_100)
    return save_outputs_dict



load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env


#Establish Site Details
resource_year = 2013
atb_years = [2022]#,2025,2030,2035]
ptc_options = ['yes']#, 'no']
N_lat = 1 #50 # number of data points
N_lon = 1 #95
desired_lats = 43.07 #35.21 #np.linspace(23.833504, 49.3556, N_lat)
desired_lons = -94.23 #101.24 #np.linspace(-129.22923, -65.7146, N_lon)
load_resource_from_file = False
resource_dir = Path(__file__).parent.parent.parent / "resource_files"
print('resource_dir: ', resource_dir)
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
electrolyzer_size = 50

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = False
plot_h2 = False
turbine_name =  'VestasV82_1.65MW_82' #'VestasV47_660kW_47' #
h2_model ='Simple'  
# h2_model = 'H2A'

critical = 'no'

scenario = dict()
kw_continuous = electrolyzer_size * 1000
if critical == 'yes':
    df = pd.read_csv('/Users/cclark2/Desktop/MIRACL/AOP FY22/MultiLab_Report/Loads/Critical_Loads.csv', skiprows=3, usecols = ['Time', 'Total (kWh)'])
    # df = pd.read_csv('/Users/cclark2/Desktop/MIRACL/AOP FY22/MultiLab_Report/Loads/Total_Loads.csv')
    df = df[0:8760]
    load = df['Total (kWh)']  #[kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    load = load.tolist()
else:
    df = pd.read_csv('/Users/cclark2/Desktop/MIRACL/AOP FY22/MultiLab_Report/Loads/Total_Loads.csv', skiprows=0, usecols = ['Total (kWh)'])
    # df = pd.read_csv('/Users/cclark2/Desktop/MIRACL/AOP FY22/MultiLab_Report/Loads/Total_Loads.csv')
    load = df['Total (kWh)']  #[kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    load = load.tolist()
    
scenario_choice = 'Resilience Storage Sizing Analysis'
# site_selection = 'Site 1'
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results/'

itc_avail = 'no'
discount_rate = 0.089
forced_sizes = True
force_electrolyzer_cost = True
forced_wind_size = 25
forced_solar_size = 25
forced_storage_size_mw = 25
forced_storage_size_mwh = 400
storage_size_mwh_options = [i * forced_storage_size_mw for i in range(1,11)]

sell_price = False
buy_price = False

# Define Turbine Characteristics based on user selected turbine.
if turbine_name == 'VestasV47_660kW_47':
    #This is the smaller distributed wind option that most closely matches the Zion 775kW turbines Algona has
    custom_powercurve_path = 'VestasV47_660kW_47.csv' # https://nrel.github.io/turbine-models/VestasV47_660kW_47.html
    tower_height = 60 #65
    rotor_diameter = 47
    turbine_rating_mw = 0.66
    wind_cost_kw = 1310 #ATB 2022 Conservative/Moderate/Advanced CAPEX for Wind Class II
if turbine_name == 'DOE_GE_1.5MW_77':
    #this is the larger distributed wind option that Algona could install since they have plans to add more turbines
    custom_powercurve_path = 'DOE_GE_1.5MW_77.csv' # https://nrel.github.io/turbine-models/DOE_GE_1.5MW_77.html
    tower_height = 80
    rotor_diameter = 77
    turbine_rating_mw = 1.5
    wind_cost_kw = 1310 #ATB 2022 Conservative/Moderate/Advanced CAPEX for Wind Class II
if turbine_name == '2018COE_Market_Average_2.4MW_116':
    #this is the larger distributed wind option that Algona could install since they have plans to add more turbines
    custom_powercurve_path = '2018COE_Market_Average_2.4MW_116.csv' # https://nrel.github.io/turbine-models/2018COE_Market_Average_2.4MW_116.html
    tower_height = 88
    rotor_diameter = 116
    turbine_rating_mw = 2.4
    wind_cost_kw = 1310 #ATB 2022 Conservative/Moderate/Advanced CAPEX for Wind Class II
if turbine_name == 'VestasV82_1.65MW_82':
    #this is the larger distributed wind option that Algona could install since they have plans to add more turbines
    custom_powercurve_path = 'VestasV82_1.65MW_82.csv' # https://nrel.github.io/turbine-models/VestasV82_1.65MW_82.html
    tower_height = 80
    rotor_diameter = 82
    turbine_rating_mw = 1.65
    wind_cost_kw = 1310 #ATB 2022 Conservative/Moderate/Advanced CAPEX for Wind Class II
    
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

for forced_storage_size_mwh in storage_size_mwh_options:
    for ptc_avail in ptc_options:
        for atb_year in atb_years:
            for i, site_deet in enumerate(site_details.iterrows()):
                # if i == 0: continue
                # else:
                site_deet = site_deet[1]
                print(site_deet)
                lat = site_deet['Lat']
                lon = site_deet['Lon']
                sample_site['lat'] = lat
                sample_site['lon'] = lon
                sample_site['no_solar'] = False
                # sample_site['no_wind'] = False
                site = SiteInfo(sample_site, hub_height=tower_height)
                site_name = (str(lat)+","+str(lon))

                if atb_year == 2022:
                    forced_electrolyzer_cost = 400
                    wind_cost_kw = 1462 #1310
                    solar_cost_kw = 1105
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
                    storage_hours = storage_size_mwh / storage_size_mw

                if storage_size_mw > 0:
                    technologies = {
                                    'wind':
                                        {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                            'turbine_rating_kw': turbine_rating_mw*1000,
                                            'hub_height': tower_height,
                                            'rotor_diameter': rotor_diameter},
                                    'pv':
                                        {'system_capacity_kw': solar_size_mw * 1000}, 
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
                                            'rotor_diameter': rotor_diameter},
                                    'pv':
                                        {'system_capacity_kw': solar_size_mw * 1000}
                                    }

                hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
                energy_shortfall_hopp, gen, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
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
                bat_model.battery_storage = storage_size_mwh * 1000
                bat_model.charge_rate = storage_size_mw * 1000
                bat_model.discharge_rate = storage_size_mw * 1000
                battery_used, excess_energy, battery_SOC = bat_model.run()

                #Calculate Metrics after deploying battery
                wind_power_production_hopp = gen.wind
                pv_power_production_hopp = gen.pv
                combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used
                energy_shortfall_post_battery_hopp = [x - y for x, y in
                                zip(load,combined_pv_wind_storage_power_production_hopp)]
                energy_shortfall_post_battery_hopp = [x if x > 0 else 0 for x in energy_shortfall_post_battery_hopp]
                combined_pv_wind_curtailment_post_battery_hopp = [x - y for x, y in
                                zip(combined_pv_wind_storage_power_production_hopp,load)]
                combined_pv_wind_curtailment_post_battery_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_post_battery_hopp]

                #Check Percentage of time load is met
                import check_load_met
                #Generate 50, 75, 90, 95% load profiles
                load_percentages = [0.5,0.75,0.9,0.95,1]
                for load_perc in load_percentages:
                    load_at_perc = [i * load_perc for i in load]
                    globals()[f"perc_time_load_met_no_storage_{int(100*load_perc)}"],globals()[f"perc_time_load_met_with_storage_{int(100*load_perc)}"],\
                    globals()[f"energy_shortfall_no_storage_{int(100*load_perc)}"],globals()[f"energy_shortfall_with_storage_{int(100*load_perc)}"] = \
                        check_load_met.check_load_met(load_at_perc,combined_pv_wind_power_production_hopp,combined_pv_wind_storage_power_production_hopp) 
                #Generate text for plots
                battery_plot_text = '{}MW, {}MWh battery'.format(storage_size_mw,storage_size_mwh)
                battery_plot_results_text = 'Shortfall Before: {0:,.0f}\nShortfall After: {1:,.0f}'.format(
                    np.sum(energy_shortfall_hopp),np.sum(energy_shortfall_post_battery_hopp))
                percentage_time_load_met_results_text = 'Without Storage, this configuration meets:\n50% load {}% of the time \n75% load {}% of the time\n90% load {}% of the time\n95% load {}% of the time and\n100% load {}% of the time'.format(perc_time_load_met_no_storage_50, perc_time_load_met_no_storage_75,perc_time_load_met_no_storage_90,perc_time_load_met_no_storage_95,perc_time_load_met_no_storage_100)
                percentage_time_load_met_with_storage_results_text = 'With {}MW and {}MWh Storage, this configuration meets:\n50% load {}% of the time \n75% load {}% of the time\n90% load {}% of the time\n95% load {}% of the time and\n100% load {}% of the time'.format(storage_size_mw, storage_size_mwh, perc_time_load_met_with_storage_50, perc_time_load_met_with_storage_75,perc_time_load_met_with_storage_90,perc_time_load_met_with_storage_95,perc_time_load_met_with_storage_100)
                print('==============================================')
                print(battery_plot_results_text)
                print(percentage_time_load_met_results_text)
                print(percentage_time_load_met_with_storage_results_text)
                print('==============================================')
                if plot_battery:
                    plt.figure(figsize=(9,6))
                    plt.subplot(311)
                    plt.plot(combined_pv_wind_curtailment_hopp[892:899],label="curtailment") #outage from 892-898
                    plt.plot(energy_shortfall_hopp[892:899],label="shortfall")
                    # plt.plot(combined_pv_wind_curtailment_hopp[200:300],label="curtailment")
                    # plt.plot(energy_shortfall_hopp[200:300],label="shortfall")
                    plt.title(battery_plot_text + '\n' + battery_plot_results_text + '\n' + 'Energy Curtailment and Shortfall')
                    plt.legend()

                    plt.subplot(312)
                    plt.plot(combined_pv_wind_storage_power_production_hopp[892:899],label="wind+pv+storage")
                    plt.plot(combined_pv_wind_power_production_hopp[892:899],"--",label="wind+pv")
                    plt.plot(wind_power_production_hopp[892:899],"--",label="wind")
                    plt.plot(pv_power_production_hopp[892:899],"--",label="pv")
                    plt.plot(load[892:899],"--",label="load")
                    # plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],label="wind+pv+storage")
                    # plt.plot(combined_pv_wind_power_production_hopp[200:300],"--",label="wind+pv")
                    # plt.plot(load[200:300],"--",label="electrolyzer rating")
                    plt.legend()
                    plt.title("Hybrid Plant Power Flows with and without Storage")
                    plt.tight_layout()

                    plt.subplot(313)
                    plt.plot(battery_SOC[892:899],label="State of Charge")
                    plt.plot(battery_used[892:899],"--",label="Battery Used")
                    # plt.plot(battery_SOC[200:300],label="State of Charge")
                    # plt.plot(battery_used[200:300],"--",label="Battery Used")
                    plt.title('Battery State')
                    plt.legend()
                    plt.savefig(os.path.join(results_dir,'Feb Resilience Battery Sizing_{}MW_{}MWh.jpg'.format(storage_size_mw,storage_size_mwh)),bbox_inches='tight')
                    # plt.show()
                    
                    df_outage = pd.DataFrame([combined_pv_wind_curtailment_hopp[892:899], 
                                              energy_shortfall_hopp[892:899], 
                                              combined_pv_wind_storage_power_production_hopp[892:899], 
                                              combined_pv_wind_power_production_hopp[892:899], load[892:899],
                                              battery_SOC[892:899], battery_used[892:899],],
                                              ['Curtailment', 'Shortfall', 'Wind + PV + Storage', 
                                              'Wind + PV Generation', 'Load', 'State of Charge', 'Battery Used'])
                    # df_outage.to_csv(os.path.join(results_dir, "February_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))
                    df_outage.to_csv(os.path.join(results_dir, "Feb_Algona_{}_{}.csv".format(scenario_choice, forced_storage_size_mwh)))


                    # plt.figure(figsize=(9,6))
                    # plt.subplot(311)
                    # plt.plot(combined_pv_wind_curtailment_hopp[3206:3254],label="curtailment") #outage from 892-898
                    # plt.plot(energy_shortfall_hopp[3206:3254],label="shortfall")
                    # plt.title(battery_plot_text + '\n' + battery_plot_results_text + '\n' + 'Energy Curtailment and Shortfall')
                    # plt.legend()

                    # plt.subplot(312)
                    # plt.plot(combined_pv_wind_storage_power_production_hopp[3206:3254],label="wind+pv+storage")
                    # plt.plot(combined_pv_wind_power_production_hopp[3206:3254],"--",label="wind+pv")
                    # plt.plot(wind_power_production_hopp[3206:3254],"--",label="wind")
                    # plt.plot(pv_power_production_hopp[3206:3254],"--",label="pv")
                    # plt.plot(load[3206:3254],"--",label="load")
                    # plt.legend()
                    # plt.title("Hybrid Plant Power Flows with and without Storage")
                    # plt.tight_layout()


                    # plt.subplot(313)
                    # plt.plot(battery_SOC[3206:3254],label="State of Charge")
                    # plt.plot(battery_used[3206:3254],"--",label="Battery Used")
                    # plt.title('Battery State')
                    # plt.legend()
                    # plt.savefig(os.path.join(results_dir,'May Resilience Battery Sizing_{}MW_{}MWh.jpg'.format(storage_size_mw,storage_size_mwh)),bbox_inches='tight')
                    # # plt.show()
                    
                    # df_outage = pd.DataFrame([combined_pv_wind_curtailment_hopp[3206:3254], 
                    #                           energy_shortfall_hopp[3206:3254], 
                    #                           combined_pv_wind_storage_power_production_hopp[3206:3254], 
                    #                           combined_pv_wind_power_production_hopp[3206:3254], load[3206:3254],
                    #                           battery_SOC[3206:3254], battery_used[3206:3254],],
                    #                           ['Curtailment', 'Shortfall', 'Wind + PV + Storage', 
                    #                           'Wind + PV Generation', 'Load', 'State of Charge', 'Battery Used'])
                    # df_outage.to_csv(os.path.join(results_dir, "May_Algona_{}_{}.csv".format(scenario_choice, forced_storage_size_mwh)))


                if plot_grid:
                    plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],label="before buy from grid")
                    plt.suptitle("Power Signal Before Purchasing From Grid")
                
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
                # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
                Power = electrical_generation_timeseries

                total_elec_production = np.sum(electrical_generation_timeseries)
                total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost

                total_system_installed_cost = total_hopp_installed_cost 

                total_annual_operating_costs = cost_to_buy_from_grid - profit_from_selling_to_grid

                # Cashflow Financial Calculation
                discount_rate = scenario['Discount Rate']
                cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
                if solar_size_mw > 0:
                    cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
                else:
                    cf_solar_annuals = np.zeros(useful_life)

                cf_operational_annuals = [-total_annual_operating_costs for i in range(useful_life)]

                cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals],['Wind', 'Solar'])

                cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))

                #NPVs of wind, solar, H2
                npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
                npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
        
                npv_operating_costs = npf.npv(discount_rate, cf_operational_annuals)


                npv_total_costs = npv_wind_costs+npv_solar_costs
                npv_total_costs_w_operating_costs = npv_wind_costs+npv_solar_costs+npv_operating_costs

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
    # ------------------------- #

save_outputs = True
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    save_all_runs_df.to_csv(os.path.join(results_dir, "Feb_ResilienceAnalysisMIRACL_Algona_VestasV82.csv"))



