import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import time

import subprocess

import run_pyfast_for_steel as steel_economics
sys.path.append('../PyFAST/')
import src.PyFAST as PyFAST

warnings.filterwarnings("ignore")

"""
Perform a LCOH analysis for an offshore wind + Hydrogen PEM system

Missing Functionality:
1. Figure out H2A Costs or ammortize cost/kw electrolyzer figure and add opex

~1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~

2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
3. Cost Scaling Based on Plant Size (Shields et. Al report)
4. Integration Required:
* Pressure Vessel Model~
* HVDC Model 
* Pipeline Model

5. Model Development Required:
- Floating Electrolyzer Platform
"""

def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Site Name'] = list()
    save_outputs_dict['Substructure Technology'] = list()
    save_outputs_dict['Site Lat'] = list()
    save_outputs_dict['Site Lon'] = list()
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Turbine Model'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['System Load (kW)'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['Wind PTC'] = list()
    save_outputs_dict['H2 PTC'] = list()
    save_outputs_dict['Wind ITC'] = list()
    save_outputs_dict['Discount Rate'] = list()
    save_outputs_dict['Debt Equity'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['Storage Enabled'] = list()
    save_outputs_dict['Wind Cost kW'] = list()
    save_outputs_dict['Solar Cost kW'] = list()
    save_outputs_dict['Storage Cost kW'] = list()
    save_outputs_dict['Storage Cost kWh'] = list()
    save_outputs_dict['Storage Hours'] = list()
    save_outputs_dict['Wind MW built'] = list()
    save_outputs_dict['Solar MW built'] = list()
    save_outputs_dict['Storage MW built'] = list()
    save_outputs_dict['Storage MWh built'] = list()
    save_outputs_dict['Electrolyzer Stack Size'] = list()
    save_outputs_dict['Electrolyzer Total System Size'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['Built Interconnection Size'] = list()
    save_outputs_dict['Total Installed Cost $(HOPP)'] = list()
    save_outputs_dict['LCOE'] = list()
    save_outputs_dict['Total Annual H2 production (kg)'] = list()
    save_outputs_dict['Electrolyzer capacity factor (-)'] = list()
    save_outputs_dict['Total LCOH ($/kg)'] = list()
    save_outputs_dict['LCOH without transport ($/kg)'] = list()
    save_outputs_dict['Electrolzyer capex LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Electrolyzer FOM LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Electrolyzer VOM LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Ren. capex LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Ren. FOM LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Storage and comp. LCOH contribution ($/kg)'] = list()
    save_outputs_dict['LCFS LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Energy chare LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Fixed demand charge LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Timed demand charge LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Meter cost LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Tax LCOH contribution ($/kg)'] = list()
    save_outputs_dict['Transport LCOH contribution ($/kg)'] = list()  
    #save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = list()
    #save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = list()
    #save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = list()
    #save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = list()
    #save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['HOPP Total Electrical Generation'] = list()
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = list()
    save_outputs_dict['Wind Capacity Factor'] = list()
    save_outputs_dict['HOPP Energy Shortfall'] = list()
    save_outputs_dict['HOPP Curtailment'] = list()
    save_outputs_dict['Battery Generation'] = list()
    save_outputs_dict['Electricity to Grid'] = list()
    save_outputs_dict['Electrolyzer Stack Size'] = list()
    save_outputs_dict['Break even price of steel ($/tonne)'] = list()
    #save_outputs_dict['Electrolyzer Total System Size'] = list()
    #save_outputs_dict['H2A scaled total install cost'] = list()
    #save_outputs_dict['H2A scaled total install cost per kw'] = list()
    
    return save_outputs_dict

def save_the_things():
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['Site Lat'] = (lat)
    save_outputs_dict['Site Lon'] = (lon)
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_model)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (load)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['Wind PTC'] = (scenario['Wind PTC'])
    save_outputs_dict['H2 PTC'] = (scenario['H2 PTC'])
    save_outputs_dict['Wind ITC'] = (scenario['Wind ITC'])
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
    save_outputs_dict['LCOE'] = (lcoe)
    #save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    save_outputs_dict['Total Annual H2 production (kg)'] = hydrogen_annual_production
    save_outputs_dict['Electrolyzer capacity factor (-)'] = RODeO_results_summary_dict['input capacity factor']
    save_outputs_dict['Total LCOH ($/kg)'] = levelized_cost_hydrogen_with_transport
    save_outputs_dict['LCOH without transport ($/kg)'] = RODeO_results_summary_dict['Product NPV cost (US$/kg)']
    save_outputs_dict['Electrolzyer capex LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Input CAPEX (US$/kg)']
    save_outputs_dict['Electrolyzer FOM LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Input FOM (US$/kg)']
    save_outputs_dict['Electrolyzer VOM LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Input VOM (US$/kg)']
    save_outputs_dict['Ren. capex LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Renewable capital cost (US$/kg)']
    save_outputs_dict['Ren. FOM LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Renewable FOM (US$/kg)']
    save_outputs_dict['Storage and comp. LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Storage & compression cost (US$/kg)']
    save_outputs_dict['LCFS LCOH contribution ($/kg)'] = RODeO_results_summary_dict['LCFS_FCEV (US$/kg)']
    save_outputs_dict['Energy chare LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Energy charge (US$/kg)']
    save_outputs_dict['Fixed demand charge LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Fixed demand charge (US$/kg)']
    save_outputs_dict['Timed demand charge LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Timed demand charge (US$/kg)']
    save_outputs_dict['Meter cost LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Meters cost (US$/kg)']
    save_outputs_dict['Tax LCOH contribution ($/kg)'] = RODeO_results_summary_dict['Taxes (US$/kg)']
    save_outputs_dict['Transport LCOH contribution ($/kg)'] = LCOHT_cf_method_pipelineonly    
    #save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = (gut_check_h2_cost_kg)
    #save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method)
    #save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs)
    #save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_pipeline)
    #save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs_pipeline)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Electrical Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = (total_elec_production)
    save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size (MW)'] = RODeO_results_summary_dict['input capacity (MW)']*system_rating_mw # Edit if no longer necessary to scale system to 1 MW in RODeO
    save_outputs_dict['Break even price of steel ($/tonne)'] = steel_breakeven_price
    #save_outputs_dict['Electrolyzer Stack Size'] = (H2A_Results['electrolyzer_size'])
    #save_outputs_dict['Electrolyzer Total System Size'] = (H2A_Results['total_plant_size'])
    #save_outputs_dict['H2A scaled total install cost'] = (H2A_Results['scaled_total_installed_cost'])
    #save_outputs_dict['H2A scaled total install cost per kw'] = (H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict

#Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

#Step 1: User Inputs for scenario
# save_all_runs = pd.DataFrame()

resource_year = 2013
atb_years = [
            #2022,
            2025,
            #2030,
            #2035
            ]
policy = {
    'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
    # 'option 2': {'Wind ITC': 26, 'Wind PTC': 0, "H2 PTC": 0},
    # 'option 3': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 0},
    # 'option 4': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 0.6},
    # 'option 5': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 3},
}

sample_site['year'] = resource_year
useful_life = 30
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = True
storage_used = True
battery_can_grid_charge = True
grid_connected_hopp = False
interconnection_size_mw = 1000
electrolyzer_size = 1000

grid_connected_rodeo = False

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = True
plot_h2 = True
plot_desal = True
plot_wind = True
plot_hvdcpipe = True
plot_hvdcpipe_lcoh = True
turbine_name = ['2020ATB_15MW']
#turbine_name = ['2020ATB_12MW','2020ATB_15MW','2020ATB_18MW']
h2_model ='Simple'  
# h2_model = 'H2A'

scenario = dict()
kw_continuous = electrolyzer_size * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

scenario_choice = 'Offshore Wind-H2 Analysis'
site_selection = [
                'Site 1',
                #'Site 2',
                #'Site 3',
                #'Site 4'
                ]
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results_for_Kaitlin/'

hydrogen_storage_durations = [100,500,600,620,640,700,1000]
#hydrogen_storage_durations = [10,50,100,500,1000]
optimize_storage_duration = 0

#Site lat and lon will be set by data loaded from Orbit runs

discount_rate = 0.07
forced_sizes = True
force_electrolyzer_cost = True
forced_wind_size = 1000
forced_solar_size = 0
forced_storage_size_mw = 0
forced_storage_size_mwh = 0


solar_cost_kw = 9999
storage_cost_kw = 250
storage_cost_kwh = 240
debt_equity_split = 60

# Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
# sell_price = 0.01
# buy_price = 0.01
sell_price = False
buy_price = False

# Read in gams exe and license location
# Create a .txt file in notepad with the locations of the gams .exe file, the .gms RODeO
# version that you want to use, and the location of the gams license file. The text
# should look something like this: 
# "C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt
# Do not push this file to the remote repository because it will be different for every user
# and for every machine, depending on what version of gams they are using and where it is installed
with open('gams_exe_license_locations.txt') as f:
    gams_locations_rodeo_version = f.readlines()
f.close()

save_outputs_dict = establish_save_output_dict()
save_all_runs = list()

for i in policy:
    for atb_year in atb_years:
        for site_location in site_selection:
            for turbine_model in turbine_name:
                for h2_storage_duration in hydrogen_storage_durations:
                
                    i = 'option 1'
                    atb_year = 2022
                    site_location = 'Site 1'
                    turbine_model = '2020ATB_15MW'
                    h2_storage_duration = 500
                    
                    # Set policy values
                    scenario['Wind ITC'] = policy[i]['Wind ITC']
                    scenario['Wind PTC'] = policy[i]['Wind PTC']
                    scenario['H2 PTC'] = policy[i]['H2 PTC']
                    
                    print(scenario['Wind PTC'])
                    # Define Turbine Characteristics based on user selected turbine.
                    if turbine_model == '2020ATB_12MW':
                        custom_powercurve_path = '2020ATB_NREL_Reference_12MW_214.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_12MW_214.html
                        tower_height = 136
                        rotor_diameter = 214
                        turbine_rating_mw = 12
                        wind_cost_kw = 1300
                        # Future Cost Reduction Estimates - ATB 2022: Class 4 Fixed, Class 11 Float
                        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_12MW.csv'))
                        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_12MW.csv'))
    
                    elif turbine_model == '2020ATB_15MW':
                        custom_powercurve_path = '2020ATB_NREL_Reference_15MW_240.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_15MW_240.html
                        tower_height = 150
                        rotor_diameter = 240
                        turbine_rating_mw = 15
                        wind_cost_kw =  1300
                        # Future Cost Reduction Estimates
                        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_15MW.csv'))
                        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_15MW.csv'))
    
                    elif turbine_model == '2020ATB_18MW':
                        custom_powercurve_path = '2020ATB_NREL_Reference_18MW_263.csv' # https://nrel.github.io/turbine-models/2020ATB_NREL_Reference_18MW_263.html
                        tower_height = 156
                        rotor_diameter = 263
                        turbine_rating_mw = 18
                        wind_cost_kw = 1300
                        # Future Cost Reduction Estimates
                        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_18MW.csv'))
                        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_18MW.csv'))
    
                        
                    scenario['Useful Life'] = useful_life
                    scenario['Debt Equity'] = debt_equity_split
                    scenario['Discount Rate'] = discount_rate
                    scenario['Tower Height'] = tower_height
                    scenario['Powercurve File'] = custom_powercurve_path
    
                    print("Powercurve Path: ", custom_powercurve_path)
    
                    
    
                    #Step 2: Extract Scenario Information from ORBIT Runs
                    # Load Excel file of scenarios
                    # OSW sites and cost file including turbines 8/16/2022 
                    path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
                    xl = pd.ExcelFile(path)
    
                    turbinesheet = turbine_model[-4:]
                    #print(xl.sheet_names)
                    scenario_df = xl.parse(turbinesheet)
                    scenario_df.set_index(["Parameter"], inplace = True)
    
                    site_df = scenario_df[site_location]
                    # print(site_df)
                    scenario_df
    
                    #Assign Orbit results to scenario cost details
                    total_capex = site_df['Total CapEx']
                    wind_cost_kw = total_capex
                    
                    site_name = site_df['Representative region']
                    fixed_or_floating_wind = site_df['Substructure technology']
                    latlon = site_df['Representative coordinates']
                    lat, lon = (latlon.split(','))
                    lat = float(lat)
                    lon = float(lon)
                    sample_site['lat'] = lat
                    sample_site['lon'] = lon
                    sample_site['no_solar'] = True
                    # sample_site['no_wind'] = False
                    site = SiteInfo(sample_site, hub_height=tower_height)
                    wind_om_cost_kw = site_df['OpEx, $/kW-yr']
                    wind_net_cf = site_df['Assumed NCF']
                    #Plot Wind Data to ensure offshore data is sound
                    wind_data = site.wind_resource._data['data']
                    # print(wind_data)
    
                    # TODO: Plot and print wind speeds to confirm offshore wind data is sound
                    if plot_wind:
                        wind_speed = [W[2] for W in wind_data]
                        plt.figure(figsize=(9,6))
                        plt.plot(wind_speed)
                        plt.title('Wind Speed (m/s) for selected location \n {} \n lat, lon: {} \n Average Wind Speed (m/s) {}'.format(site_name,latlon,np.average(wind_speed)))
                        plt.savefig(os.path.join(results_dir,'Average Wind Speed_{}'.format(site_name)),bbox_inches='tight')
    
                    #Plot Wind Cost Contributions
                    # Plot a nested pie chart of results
                    group_names=['BOS', 'Soft', 'Turbine']
                    group_size=[site_df['BOS'],site_df['Soft'],site_df['Turbine CapEx']]
                    subgroup_names=['Array System','Export System','Substructure','Mooring System',
                                'Offshore Substation','Scour Protection','Array System Installation',
                                'Export System Installation', 'Offshore Substation Installation',
                                'Scour Protection Installation', 'Substructure Installation','Turbine Installation',
                                'Mooring System Installation',
                                'construction_insurance_capex', 'decomissioning_costs',
                                'construction_financing', 'procurement_contingency_costs',
                                'install_contingency_costs', 'project_completion_capex',
                                    'Turbine CapEx']
    
                    subgroup_vals = site_df[subgroup_names]
                    subgroup_size=[x for x in subgroup_vals]
    
                    bos_names = ['BOS: '+name for name in subgroup_names[:13]]
                    soft_names = ['Soft Cost: ' +name for name in subgroup_names[13:-1]]
                    turbine_names = ['Turbine: '+name for name in subgroup_names[19:]]
                    all_names = (bos_names + soft_names + turbine_names)
                    subgroup_names_legs=[x for x in all_names]
    
                    # Create colors
                    a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
    
                    # First Ring (Inside)
                    fig, ax = plt.subplots(figsize=(12,12))
                    ax.axis('equal')
                    mypie, _ = ax.pie(group_size, radius=1, labels=group_names, labeldistance=.6, colors= 
                    [a(0.6), b(0.6), c(0.6)] )
                    plt.setp( mypie, width=0.3, edgecolor='white')
    
                    # Second Ring (Outside)
                    mypie2, _ = ax.pie(subgroup_size, radius=1.0+0.4, 
                    labels=subgroup_names, labeldistance=1, rotatelabels=True, colors=[a(0.1), a(0.2), 
                    a(0.3), a(0.4), a(0.5), a(0.6), a(0.7), a(0.8), a(0.9), a(1), a(1.1), a(1.2),
                                                                    a(1.3),b(.16),b(.32),b(.48),b(.64),b(.8),b(.9),c(.4)])
                    plt.setp( mypie2, width=0.4, edgecolor='white')
                    plt.margins(0,0)
    
                    plt.legend(loc=(0.9, 0.1))
                    handles, labels = ax.get_legend_handles_labels()
    
                    ax.legend(handles[3:], subgroup_names_legs, loc=(0.4, 0.3))
    
                    plt.legend(subgroup_names_legs,loc='best')
                    # plt.title('ORBIT Cost Contributions for {}'.format(site_name))
                    print('ORBIT Cost Contributions for {}'.format(site_name))
                    plt.savefig(os.path.join(results_dir,'BOS Cost Figure {}_{}.jpg'.format(site_name,turbine_name)),bbox_inches='tight')
                    # plt.show()
    
                    #Display Future Cost Reduction Estimates per turbine
                    # Fixed Wind Cost Reductions
                    if fixed_or_floating_wind == 'Fixed - Monopile':
                        capex_reduction = fixed_cost_reductions_df[str(atb_year)][0]
                        opex_reduction = fixed_cost_reductions_df[str(atb_year)][1]
                        net_cf_increase = fixed_cost_reductions_df[str(atb_year)][2]
                    # Floating Wind Cost Reductions
                    elif fixed_or_floating_wind == 'Floating - semisubmersible':
                        capex_reduction = floating_cost_reductions_df[str(atb_year)][0]
                        opex_reduction = floating_cost_reductions_df[str(atb_year)][1]
                        net_cf_increase = floating_cost_reductions_df[str(atb_year)][2]
    
                    print("For {} wind in {}, capex reduction is estimated to be: {}, opex reduction is: {}, and net capacity factor increase is: {}.".format(fixed_or_floating_wind, str(atb_year), capex_reduction, opex_reduction, net_cf_increase))
    
                    new_wind_cost_kw = wind_cost_kw * (100-float(capex_reduction[:-1]))/100
                    new_wind_om_cost_kw = wind_om_cost_kw * (100-float(opex_reduction[:-1]))/100
                    new_wind_net_cf = wind_net_cf * (100+float(net_cf_increase[:-1]))/100
                    
                    print("Wind Cost in baseline year was {}, reduced to {} in {}".format(wind_cost_kw, new_wind_cost_kw, atb_year))
                    print("Operation and Maintain Cost, reduced from {} to {}".format(wind_om_cost_kw, new_wind_om_cost_kw))
                    print("Net Capacity Factor increased from {} to {}".format(wind_net_cf, new_wind_net_cf))
                    wind_cost_kw = new_wind_cost_kw
                    wind_om_cost_kw = new_wind_om_cost_kw
                    wind_net_cf = new_wind_net_cf
    
                    #Step 3: Run HOPP
                    if forced_sizes:
                        solar_size_mw = forced_solar_size
                        wind_size_mw = forced_wind_size
                        storage_size_mw = forced_storage_size_mw
                        storage_size_mwh = forced_storage_size_mwh
                        storage_hours = 0
    
                    if storage_size_mw > 0:
                        technologies = {#'pv':
                                        #   {'system_capacity_kw': solar_size_mw * 1000},
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
                                technologies = {#'pv':
                                            #{'system_capacity_kw': solar_size_mw * 1000},
                                        'wind':
                                            {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                                'turbine_rating_kw': turbine_rating_mw*1000,
                                                'hub_height': tower_height,
                                                'rotor_diameter': rotor_diameter},
                        #                 'battery': {
                        #                     'system_capacity_kwh': storage_size_mwh * 1000,
                        #                     'system_capacity_kw': storage_size_mw * 1000
                        #                     }
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
                    print(hybrid_plant.om_capacity_expenses)
    
                    #Step 4: Plot HOPP Results
                    if plot_power_production:
                        plt.figure(figsize=(4,4))
                        plt.title("HOPP power production")
                        plt.plot(combined_pv_wind_power_production_hopp[200:300],label="wind + pv")
                        plt.plot(energy_shortfall_hopp[200:300],label="shortfall")
                        plt.plot(combined_pv_wind_curtailment_hopp[200:300],label="curtailment")
                        plt.plot(load[200:300],label="electrolyzer rating")
                        plt.xlabel("Time (hour)")
                        plt.ylabel("Power Production (kW)")
                        # plt.ylim(0,250000)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(results_dir,'HOPP Power Production_{}_{}_{}'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                        # plt.show()
    
                    print("Turbine Power Output (to identify powercurve impact): {0:,.0f} kW".format(hybrid_plant.wind.annual_energy_kw))
                    print("Wind Plant CF: {}".format(hybrid_plant.wind.capacity_factor))
                    print("LCOE: {}".format(hybrid_plant.lcoe_real.hybrid))
    
                    #Step 5: Run Simple Dispatch Model
                    bat_model = SimpleDispatch()
                    bat_model.Nt = len(energy_shortfall_hopp)
                    bat_model.curtailment = combined_pv_wind_curtailment_hopp
                    bat_model.shortfall = energy_shortfall_hopp
                    # print(combined_pv_wind_curtailment_hopp)
                    # print(energy_shortfall_hopp)
    
                    # bat_model.battery_storage = 100 * 1000
                    # bat_model.charge_rate = 100 * 1000
                    # bat_model.discharge_rate = 100 * 1000
    
                    battery_used, excess_energy, battery_SOC = bat_model.run()
                    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used
    
                    if plot_battery:
                        plt.figure(figsize=(9,6))
                        plt.subplot(311)
                        plt.plot(combined_pv_wind_curtailment_hopp[200:300],label="curtailment")
                        plt.plot(energy_shortfall_hopp[200:300],label="shortfall")
                        plt.title('Energy Curtailment and Shortfall')
                        plt.legend()
                        
    
                        plt.subplot(312)
                        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],label="wind+pv+storage")
                        plt.plot(combined_pv_wind_power_production_hopp[200:300],"--",label="wind+pv")
                        plt.plot(load[200:300],"--",label="electrolyzer rating")
                        plt.legend()
                        plt.title("Hybrid Plant Power Flows with and without storage")
                        plt.tight_layout()
                        
                        plt.subplot(313)
                        plt.plot(battery_SOC[200:300],label="state of charge")
                        plt.plot(battery_used[200:300],"--",label="battery used")
                        plt.title('Battery State')
                        plt.legend()
                        plt.savefig(os.path.join(results_dir,'HOPP Full Power Flows_{}_{}_{}'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                        # plt.show()
    
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
    
                    #Plot Dispatch Results
    
                    if plot_grid:
                        plt.figure(figsize=(9,6))
                        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],"--",label="after buy from grid")
                        plt.plot(energy_to_electrolyzer[200:300],"--",label="energy to electrolyzer")
                        plt.legend()
                        plt.title('Power available after purchasing from grid (if enabled)')
                        # plt.show()
                        
                        
                    # Step 6: Run RODeO
                    
                    # Renewable generation profile
                    system_rating_mw = wind_size_mw + solar_size_mw
                    # Renewable output profile needs to be same length as number of time periods in RODeO.
                    # Ideally it would be 8760 but if for some reason a couple hours less, this is a simple fix
                    while len(energy_to_electrolyzer)<8760:
                        energy_to_electrolyzer.append(energy_to_electrolyzer[-1])
                        
                    electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
                    electrical_generation_timeseries[:] = energy_to_electrolyzer[:]
                    # Put electrolyzer input into MW
                    electrical_generation_timeseries = electrical_generation_timeseries/1000
                    # Normalize renewable profile to 1. In my experience (Evan) this helps RODeO run more smoothly but might not be necessary. Should experiment with it.
                    electrical_generation_timeseries = electrical_generation_timeseries/system_rating_mw
                    # Get renewable generation profile into a format that works for RODeO
                    electrical_generation_timeseries_df = pd.DataFrame(electrical_generation_timeseries).reset_index().rename(columns = {'index':'Interval',0:1})
                    electrical_generation_timeseries_df['Interval'] = electrical_generation_timeseries_df['Interval']+1
                    electrical_generation_timeseries_df = electrical_generation_timeseries_df.set_index('Interval')
                    
                    # # Make normalized demand curve for supply-driven cases (mostly to check economics for those cases)
                    # normalized_demand = list(energy_to_electrolyzer/max(energy_to_electrolyzer))
                    # normalized_demand_rounded = [round(item,6) for item in normalized_demand]
                    
                    # normalized_demand_df = pd.DataFrame(normalized_demand_rounded).reset_index().rename(columns = {'index':'Interval',0:1})
                    # normalized_demand_df['Interval'] = normalized_demand_df['Interval']+1
                    # normalized_demand_df = normalized_demand_df.set_index('Interval')
                    
                    # Fill in renewable profile for RODeO with zeros for years 2-20 (because for some reason it neesd this)
                    extra_zeroes = np.zeros_like(energy_to_electrolyzer)
                    for j in range(19):
                        #j=0
                        extra_zeroes_df = pd.DataFrame(extra_zeroes,columns = [j+2]).reset_index().rename(columns = {'index':'Interval',0:j+2})
                        extra_zeroes_df['Interval'] = extra_zeroes_df['Interval']+1
                        extra_zeroes_df = extra_zeroes_df.set_index('Interval')
                        electrical_generation_timeseries_df = electrical_generation_timeseries_df.join(extra_zeroes_df)
                        # normalized_demand_df = normalized_demand_df.join(extra_zeroes_df)
    
                    # Write the renewable generation profile to a .csv file in the RODeO repository, assuming RODeO is installed in the same folder as HOPP
                    ren_profile_name = 'ren_profile_'+str(atb_year) + '_'+site_location.replace(' ','_') + '_'+ turbine_model
                    electrical_generation_timeseries_df.to_csv("examples/H2_Analysis/RODeO_files/Data_files/TXT_files/Ren_profile/" + ren_profile_name + '.csv',sep = ',')
                    

                    # dem_profile_name = 'demand_profile_' + str(atb_year)+ '_' + site_location.replace(' ','_') + '_'+turbine_model
                    # normalized_demand_df.to_csv("examples/H2_Analysis/RODeO_files/Data_Files/TXT_files/Product_consumption/" + dem_profile_name + '.csv',sep = ',')
                    
                    
                    #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
                    if atb_year == 2022:
                        electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
                        time_between_replacement = 40000    #[hrs] 
                    elif atb_year == 2025:
                        electrolyzer_capex_kw = 300
                        time_between_replacement = 80000    #[hrs]
                    elif atb_year == 2030:
                        electrolyzer_capex_kw = 150
                        time_between_replacement = 80000    #[hrs]
                    elif atb_year == 2035:
                        electrolyzer_capex_kw = 100
                        time_between_replacement = 80000    #[hrs]
                                
                    # Storage costs as a function of location
                    if site_location == 'Site 1':
                        h2_storage_cost_USDperkg =25
                        balancing_area = 'p65'
                        hybrid_fixed_om_cost_kw = 103
                    elif site_location == 'Site 2':
                        h2_storage_cost_USDperkg = 540
                        balancing_area ='p124'
                        hybrid_fixed_om_cost_kw = 83
                    elif site_location == 'Site 3':
                        h2_storage_cost_USDperkg = 54
                        balancing_area = 'p128'
                        hybrid_fixed_om_cost_kw = 103
                    elif site_location == 'Site 4':
                        h2_storage_cost_USDperkg = 54
                        balancing_area = 'p9'
                        hybrid_fixed_om_cost_kw = 83
    
                    # Format renewable system cost for RODeO
                    hybrid_installed_cost_perMW = hybrid_installed_cost/system_rating_mw  
                    
                    #Capital costs provide by Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
                    stack_capital_cost = 342   #[$/kW]
                    mechanical_bop_cost = 36  #[$/kW] for a compressor
                    electrical_bop_cost = 82  #[$/kW] for a rectifier

                    # Installed capital cost
                    stack_installation_factor = 12/100  #[%] for stack cost 
                    elec_installation_factor = 12/100   #[%] and electrical BOP 
                    #mechanical BOP install cost = 0%

                    # Indirect capital cost as a percentage of installed capital cost
                    site_prep = 2/100   #[%]
                    engineering_design = 10/100 #[%]
                    project_contingency = 15/100 #[%]
                    permitting = 15/100     #[%]
                    land = 250000   #[$]

                    stack_replacment_cost = 15/100  #[% of installed capital cost]
                    plant_lifetime = 40    #[years]
                    fixed_OM = 0.24     #[$/kg H2]

                    program_record = False

                    # Chose to use numbers provided by GPRA pathways
                    if program_record:
                        total_direct_electrolyzer_cost_kw = (stack_capital_cost*(1+stack_installation_factor)) \
                            + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))
                    else:
                        total_direct_electrolyzer_cost_kw = (electrolyzer_capex_kw * (1+stack_installation_factor)) \
                            + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))

                    # Assign CapEx for electrolyzer from capacity based installed CapEx
                    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size*1000

                    # Add indirect capital costs
                    electrolyzer_total_capital_cost = electrolyzer_total_installed_capex+((site_prep+engineering_design+project_contingency+permitting)\
                        *electrolyzer_total_installed_capex) + land
                        
                    electrolyzer_capex_kw = electrolyzer_total_capital_cost/1000/1000

                    # O&M costs
                    # https://www.sciencedirect.com/science/article/pii/S2542435121003068
                    fixed_OM = 12.8 #[$/kWh-y]
                    property_tax_insurance = 1.5/100    #[% of Cap/y]
                    variable_OM = 1.30  #[$/MWh]

                    elec_cf = sum(energy_to_electrolyzer)/(electrolyzer_size*1000*8760)

                    # Amortized refurbishment expense [$/MWh]
                    amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacment_cost)\
                            *max(((useful_life*8760*elec_cf)/time_between_replacement-1),0)/useful_life/8760/elec_cf*1000

                    total_variable_OM = variable_OM+amortized_refurbish_cost

                    # # Total O&M costs [% of installed cap/year]
                    # total_OM_costs = ((fixed_OM+(property_tax_insurance*total_direct_electrolyzer_cost_kw))/total_direct_electrolyzer_cost_kw\
                    #     +((variable_OM+amortized_refurbish_cost)/1000*8760*(H2_Results['cap_factor']/total_direct_electrolyzer_cost_kw)))
                    
                    # Define electrolyzer capex, fixed opex, and energy consumption (if not pulling from external data)
                    electrolyzer_capex_USD_per_MW = electrolyzer_capex_kw*1000#1542000 # Eventually get from input loop
                    electrolyzer_fixed_opex_USD_per_MW_year = fixed_OM*1000
                    electrolyzer_energy_kWh_per_kg = 55.5 # Eventually get from input loop
                    
                    # Define dealination conversion factors
                    desal_energy_conversion_factor_kWh_per_m3_water = 4 # kWh per m3-H2O
                    m3_water_per_kg_h2 = 0.01 # m3-H2O per kg-H2
                    
                    # Calculate desalination energy requirement per kg of produced hydrogen
                    desal_energy_kWh_per_kg_H2 = m3_water_per_kg_h2*desal_energy_conversion_factor_kWh_per_m3_water
                    
                    # Calculate desal capex and opex per MW of electrolysis power
                    desal_capex_USD_per_MW_of_electrolysis = 32894*(997/3600*1000/electrolyzer_energy_kWh_per_kg*m3_water_per_kg_h2)
                    desal_opex_USD_per_MW_of_EC_per_year = 4841*(997/3600*1000/electrolyzer_energy_kWh_per_kg*m3_water_per_kg_h2)
                    
                    # Incorporate desal cost and efficiency into electrolyzer capex, opex, and energy consumption
                    electrolysis_desal_total_capex_per_MW = electrolyzer_capex_USD_per_MW + desal_capex_USD_per_MW_of_electrolysis
                    electrolysis_desal_total_opex_per_MW_per_year = electrolyzer_fixed_opex_USD_per_MW_year + desal_opex_USD_per_MW_of_EC_per_year
                    electrolysis_desal_total_energy_consumption = electrolyzer_energy_kWh_per_kg + desal_energy_kWh_per_kg_H2
                    
                    # Convert electrolysis energy consumption into LHV efficiency
                    hydrogen_LHV = 120000 #kJ/kg
                    eta_LHV = hydrogen_LHV/3600/electrolysis_desal_total_energy_consumption
                    
                    # Grid connection switfch
                    if grid_connected_rodeo == True:
                        grid_string = 'gridconnected'
                        grid_imports = 1
                    else:
                        grid_string = 'offgrid'
                        grid_imports = 0
                        
                    # Financial parameters
                    inflation_rate = 2.5/100
                    equity_percentage = 40/100
                    bonus_depreciation = 0/100
                    
                    # Set hydrogen break even price guess value
                    # Could in the future replace with H2OPP or H2A estimates 
                    lcoh_guessvalue =50
                    
                    # Set up batch file
                    dir0 = "..\\RODeO\\"
                    dir1 = 'examples\\H2_Analysis\\RODeO_files\\Data_files\\TXT_files\\'
                    dirout = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\'
                    
                   # txt1 = '"C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt'
                    txt1 = gams_locations_rodeo_version[0]
                    scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','_') +'_'+turbine_model+'_'+str(h2_storage_duration)+'_hrstor_'+grid_string
                    
                    scenario_inst = ' --file_name_instance='+scenario_name
                    #scenario_name = ' --file_name_instance='+Scenario1
                    # demand_prof = ' --product_consumed_inst=' + dem_profile_name
                    demand_prof = ' --product_consumed_inst=Product_consumption_flat_hourly_ones'
                    load_prof = ' --load_prof_instance=Additional_load_none_hourly'
                    ren_prof = ' --ren_prof_instance=Ren_profile\\'+ren_profile_name
                    ren_cap = ' --Renewable_MW_instance=1'#+str(system_rating_mw)#'1'
                    energy_price = ' --energy_purchase_price_inst=Elec_prices\\Elec_purch_price_WS_MWh_MC95by35_'+str(balancing_area)+'_'+str(atb_year)
                    #energy_price = ' --energy_purchase_price_inst=Netload_'+str(i1)+' --energy_sale_price_inst=Netload_'+str(i1)
                    #max_input_entry = ' --Max_input_prof_inst=Max_input_cap_'+str(i1)
                    capacity_values = ' --input_cap_instance=1'#+str(system_rating_mw)#+str(storage_power_increment)#+' --output_cap_instance='+str(storage_power_increment)
                    efficiency = ' --input_efficiency_inst='+str(round(eta_LHV,4))#'0.611'#+str(round(math.sqrt(RTE[i1-1]),6))#+' --output_efficiency_inst='+str(round(math.sqrt(RTE[i1-1]),6))

                    wacc_instance = ' --wacc_instance=0.07'                    
                    equity_perc_inst = ' --perc_equity_instance=' + str(round(equity_percentage,4))
                    ror_inst = ' --ror_instance=0.489'
                    roe_inst = ' --roe_instance=0.104'
                    debt_interest_inst = ' --debt_interest_instance=0.0481'
                    cftr_inst = ' --cftr_instance=0.27'
                    inflation_inst = ' --inflation_inst=' + str(round(inflation_rate,3))
                    bonus_dep_frac_inst = ' --bonus_deprec_instance=' + str(round(bonus_depreciation,1))
                    
                    storage_init_inst = ' --storage_init_instance=0.5'
                    storage_final_inst = ' --storage_final_instance=0.5'
                    max_storage_dur_inst= ' --max_stor_disch_inst=1000'
                    
                    storage_cap = ' --storage_cap_instance='+str(h2_storage_duration)#'1000'#+str(stor_dur[i1-1])
                    storage_opt = ' --opt_storage_cap ='+str(optimize_storage_duration)
                    out_dir = ' --outdir='+dirout
                    in_dir = ' --indir='+dir1
                    #out_dir = ' --outdir=C:\\Users\\ereznic2\\Documents\\Projects\\SCS_CRADA\\RODeO\\Projects\\SCS\\Output_GSA_test'
                    #in_dir = ' --indir=C:\\Users\\ereznic2\\Documents\\Projects\\SCS_CRADA\\RODeO\\Projects\\SCS\\Data_files\\TXT_files'
                    product_price_inst = ' --Product_price_instance='+str(lcoh_guessvalue)
                    device_ren_inst = ' --devices_ren_instance=1'
                    input_cap_inst = ' --input_cap_instance=1'#+str(system_rating_mw)#1'
                    allow_import_inst = ' --allow_import_instance='+str(grid_imports)
                    input_LSL_inst = ' --input_LSL_instance=0'
                    ren_capcost = ' --renew_cap_cost_inst='+str(round(hybrid_installed_cost_perMW))#'1230000'
                    input_capcost= ' --input_cap_cost_inst='+str(round(electrolysis_desal_total_capex_per_MW))#'1542000'
                    prodstor_capcost = ' --ProdStor_cap_cost_inst='+str(round(h2_storage_cost_USDperkg))#'26'
                    ren_fom = ' --renew_FOM_cost_inst='+str(1000*hybrid_fixed_om_cost_kw)
                    input_fom = ' --input_FOM_cost_inst='+str(round(electrolysis_desal_total_opex_per_MW_per_year))#'34926.3'
                    ren_vom = ' --renew_VOM_cost_inst=0'
                    input_vom = ' --input_VOM_cost_inst='+str(round(total_variable_OM,2))
                    
                    # Create batch file
                    batch_string = txt1+scenario_inst+demand_prof+ren_prof+load_prof+energy_price+capacity_values+efficiency+storage_cap+storage_opt+ren_cap+out_dir+in_dir\
                                 + product_price_inst+device_ren_inst+allow_import_inst+input_LSL_inst+ren_capcost+input_capcost+prodstor_capcost+ren_fom+input_fom+ren_vom+input_vom\
                                 + wacc_instance+equity_perc_inst+ror_inst+roe_inst+debt_interest_inst+cftr_inst+inflation_inst+bonus_dep_frac_inst\
                                 + storage_init_inst+storage_final_inst  +max_storage_dur_inst  

                    #subprocess.run(batch_string,capture_output = True)
                                      
                                 
                    with open(os.path.join(dir0, 'Output_batch.bat'), 'w') as OPATH:
                        #OPATH.writelines([batch_string,'\n','pause']) # Remove '\n' and 'pause' if not trouble shooting
                        OPATH.writelines([batch_string]) # Remove '\n' and 'pause' if not trouble shooting
                    
                    
                    summary_file_path = dirout + '\\Storage_dispatch_summary_'+scenario_name + '.csv'
                    inputs_file_path = dirout + '\\Storage_dispatch_inputs_'+scenario_name + '.csv'
                    results_file_path = dirout + '\\Storage_dispatch_results_'+scenario_name + '.csv'
                    
                    # Delete currently existing scenario so that RODeO can replace it
                    if os.path.exists(summary_file_path):
                        os.remove(summary_file_path)
                        
                    if os.path.exists(inputs_file_path):
                        os.remove(inputs_file_path)
                        
                    if os.path.exists(results_file_path):
                        os.remove(results_file_path)
                    
                    # Run batch file
                    os.startfile(r'..\\RODeO\\Output_batch.bat')
                    
                    start_time = time.time()
                    
                    # Make sure GAMS has finished and printed results before continuing
                    while os.path.exists(summary_file_path)==False:
                        time_delta = time.time() - start_time
                        print('Waiting for RODeO... Elapsed time: ' + str(round(time_delta))+' s')
                        time.sleep(20)
                        
                    # Make sure the inputs file has been written too
                    while os.path.exists(inputs_file_path)==False:
                        time_delta = time.time() - start_time
                        print('Waiting for RODeO... Elapsed time: ' + str(round(time_delta))+' s')
                        time.sleep(5)
                    
                    # Make sure the results file has been written too
                    while os.path.exists(results_file_path)==False:
                        time_delta = time.time() - start_time
                        print('Waiting for RODeO... Elapsed time: ' + str(round(time_delta))+' s')
                        time.sleep(5)
                    
                    # Is this really the best way to do this? Probably not, but until we figure out
                    # how to use the Python API for GAMS, this is the quickest and easiest way to do it
                    end_time = time.time()
                    print('RoDeO finished! Total elapsed time: ' + str(round(end_time-start_time))+' s')
                    
                    # Get RODeO results summary (high level outputs such as LCOH, capacity factor, cost breakdown, etc.)
                    RODeO_results_summary = pd.read_csv(dirout+'\\Storage_dispatch_summary_'+scenario_name + '.csv',header = 1,sep=',')
                    RODeO_results_summary = RODeO_results_summary.rename(columns = {'Elapsed Time (minutes):':'Parameter',RODeO_results_summary.columns[1]:'Value'}).set_index('Parameter')
                    # Put results into a dictionary
                    RODeO_results_summary_T = RODeO_results_summary.T
                    RODeO_results_summary_dict = RODeO_results_summary_T.iloc[0].to_dict()
        
                    # Examples for reading out RODeO summary results of interest
                    levelized_cost_of_hydrogen_RODeO = RODeO_results_summary_dict['Product NPV cost (US$/kg)']
                    electrolyzer_capacity_factor = RODeO_results_summary_dict['input capacity factor']
                    electrolyzer_renewable_curtailment_MWh = RODeO_results_summary_dict['Curtailment (MWh)']
                    electyrolyzer_renewable_curtailment_percent = 100*RODeO_results_summary_dict['Curtailment (MWh)']/RODeO_results_summary_dict['Renewable Electricity Input (MWh)']
                    storage_duration_hr = RODeO_results_summary_dict['storage capacity (MWh)']/RODeO_results_summary_dict['input efficiency (%)']
                    
                    # Get RODeO operational results (e.g., electrolyzer and storage hourly operation)
                    hydrogen_hourly_inputs_RODeO = pd.read_csv(dirout+'\\Storage_dispatch_inputs_'+scenario_name + '.csv',index_col = None,header = 29)
                    hydrogen_hourly_results_RODeO = pd.read_csv(dirout+'\\Storage_dispatch_results_'+scenario_name + '.csv',index_col = None,header = 26)
                    hydrogen_hourly_results_RODeO['Storage Level (%)'] = 100*hydrogen_hourly_results_RODeO['Storage Level (MW-h)']/(RODeO_results_summary_dict['storage capacity (MWh)'])
                    hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]'] = hydrogen_hourly_results_RODeO['Input Power (MW)']*1000/54.55*system_rating_mw
                    hydrogen_hourly_results_RODeO['Water consumption [kg/hr]'] = hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]']*10 #15.5 might be a better number for centralized electrolysis
                    
                    hydrogen_annual_production = sum(hydrogen_hourly_results_RODeO['Product Sold (units of product)'])*system_rating_mw
                    water_consumption_hourly_array = hydrogen_hourly_results_RODeO['Water consumption [kg/hr]'].to_numpy()
                    
                    # Plot electrolyzer operation
                    plt.figure()
                    plt.plot(hydrogen_hourly_results_RODeO['Interval'], hydrogen_hourly_results_RODeO['Input Power (MW)'])
                    plt.xlabel('Hours',fontsize = 16)
                    plt.ylabel('Electrolyzer Input Power (MW)',fontname = 'Arial', fontsize = 16)
                    plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
                    plt.yticks(fontname = 'Arial',fontsize = 16)
                    #plt.axis([0,8760,0,0.6])
                    plt.tick_params(direction = 'in',width = 1)
                    plt.tight_layout()
                    
                    # Plot storage level
                    plt.figure()
                    plt.plot(hydrogen_hourly_results_RODeO['Interval'], hydrogen_hourly_results_RODeO['Storage Level (%)'])
                    plt.xlabel('Hours',fontsize = 16)
                    plt.ylabel('Storage Level (%)',fontname = 'Arial', fontsize = 16)
                    plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
                    plt.yticks(fontname = 'Arial',fontsize = 16)
                    #plt.axis([0,8760,0,0.6])
                    plt.tick_params(direction = 'in',width = 1)
                    plt.tight_layout()
                    
                    # Plot renewables curtailment
                    plt.figure()
                    plt.plot(hydrogen_hourly_results_RODeO['Interval'], hydrogen_hourly_results_RODeO['Curtailment (MW)'])
                    plt.xlabel('Hours',fontsize = 16)
                    plt.ylabel('Curtailment (MW)',fontname = 'Arial', fontsize = 16)
                    plt.xticks(fontname = 'Arial',fontsize = 16,rotation = 45)
                    plt.yticks(fontname = 'Arial',fontsize = 16)
                    #plt.axis([0,8760,0,0.6])
                    plt.tick_params(direction = 'in',width = 1)
                    plt.tight_layout()
                    
                    # #Step 6: Run the H2_PEM model
                    # #TODO: Refactor H2A model call
                    # # Should take as input (electrolyzer size, cost, electrical timeseries, total system electrical usage (kwh/kg),
                    # # Should give as ouptut (h2 costs by net cap cost, levelized, total_unit_cost of hydrogen etc)   )
    
                    # # electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp
                    # electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
                    # electrical_generation_timeseries[:] = energy_to_electrolyzer[:]
    
                    # adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
                    # #NB: adjusted_installed_cost does NOT include the electrolyzer cost
                    # useful_life = scenario['Useful Life']
                    # net_capital_costs = 0
    
                    # # system_rating = electrolyzer_size
                    # system_rating = wind_size_mw + solar_size_mw
                    # H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size,
                    #                 kw_continuous,electrolyzer_capex_kw,lcoe,adjusted_installed_cost,useful_life,
                    #                 net_capital_costs)
    
    
                    # H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']
                    # H2_Results['cap_factor'] = H2_Results['cap_factor']
                    
                    # print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
                    # print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
                    # print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))
    
                    # if plot_h2:
                    #     hydrogen_hourly_production = H2_Results['hydrogen_hourly_production']
                    #     plt.figure(figsize=(8,8))
                    #     plt.subplot(411)
                    #     plt.plot(electrical_generation_timeseries[200:300])
                    #     plt.ylim(0,max(electrical_generation_timeseries[200:300])*1.2)
                    #     plt.plot(load[200:300],label="electrolyzer rating")
                    #     plt.legend()
                    #     plt.title("Energy to electrolyzer (kW)")
    
                    #     plt.subplot(412)
                    #     plt.plot(hydrogen_hourly_production[200:300])
                    #     plt.ylim(0,max(hydrogen_hourly_production[200:300])*1.2)
                    #     plt.title("Hydrogen production rate (kg/hr)")
    
                        
                    #     plt.subplot(413)
                    #     plt.plot(H2_Results['electrolyzer_total_efficiency'][200:300])
                    #     plt.ylim(0,1)
                    #     plt.title("Electrolyzer Total Efficiency (%)")
                        
                        
                    #     plt.subplot(414)
                    #     plt.plot(H2_Results['water_hourly_usage'][200:300],"--",label="Hourly Water Usage")
                    #     plt.legend()
                    #     plt.title('Hourly Water Usage (kg/hr) \n' 'Total Annual Water Usage: {0:,.0f}kg'.format(H2_Results['water_annual_usage']))
                    #     plt.tight_layout()
                    #     plt.xlabel('Time (hours)')
                    #     plt.savefig(os.path.join(results_dir,'Electrolyzer Flows_{}_{}_{}'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                    #     # plt.show()
    
                    ##Pipeline Model
                    print("Distance to port: ",site_df['Approx. distance to port'])
                    dist_to_port_value = site_df['Approx. distance to port']
                    #pipe_flow_rate = max(H2_Results['hydrogen_hourly_production'])
                    pipe_flow_rate = max(hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]'])
                    dist_to_port = ""
                    for m in dist_to_port_value:
                        if m.isdigit():
                            dist_to_port = dist_to_port + m
                    dist_to_port = int(dist_to_port)
                    print("Water depth: ",site_df['Approx. water depth'])
                    site_depth_value = site_df['Approx. water depth']
                    site_depth = ""
                    for m in site_depth_value:
                        if m.isdigit():
                            site_depth = site_depth + m
                    site_depth = int(site_depth)
    
                    #from examples.H2_Analysis.pipeline_model import Pipeline
                    from examples.H2_Analysis.pipelineASME import PipelineASME
                    in_dict = dict()
                    #in_dict['pipeline_model'] = 'nrwl'
                    #in_dict['pipeline_model'] = 'nexant'
                    #in_dict['pipe_diam_in'] = 24.0
                    in_dict['pipe_diam_in'] = np.linspace(12.0, 48.0, 20)
                    in_dict['pipe_thic_in'] = np.linspace(0.1, 2.0, 50)
                    #in_dict['offshore_bool'] = True 
                    in_dict['flow_rate_kg_hr'] = pipe_flow_rate
                    in_dict['plant_life'] = 30
                    in_dict['useful_life'] = useful_life
                    in_dict['dist_to_h2_load_km'] = int(dist_to_port)
                    in_dict['site_depth_m'] = int(site_depth)
                    in_dict['steel_cost_ton'] = 900.0 # $ US/ton searching for seamless FBE X52 carbon steel > $500-$1000 per ton
                    in_dict['pressure_bar'] = 100#$storage_input['compressor_output_pressure']
                    
                    out_dict = dict()
    
                    print("Pipeline flow rate: ", pipe_flow_rate, "kg/hr")
                    #pipeline_model = Pipeline(in_dict, out_dict)
                    #capex_pipeline, opex_pipeline, pipeline_annuals = pipeline_model.pipeline_cost()
                    #pipeline_cost_kw = capex_pipeline / (wind_size_mw*1000)
                    #print("Pipeline CAPEX: ${0:,.0f}".format(capex_pipeline))
                    #print("Pipeline Cost/kW: ${0:,.0f}/kW".format(pipeline_cost_kw))
                    pipeline_model = PipelineASME(in_dict, out_dict)
                    pipeline_model.pipelineDesign()
                    pipeline_model.pipelineCost()
                    capex_pipeline = np.min(out_dict['pipeline_capex'])
                    opex_pipeline = np.min(out_dict['pipeline_opex'])
                    capex_substation = out_dict['substation_capex']
    
                    total_h2export_system_cost = capex_pipeline + capex_substation
                    #print('Pipeline Model:', in_dict['pipeline_model'])
                    #print('Pipeline length (miles):', out_dict['len_pipeline_miles'])
                    #print('Pipeline CapEx Cost ($USD):', out_dict['pipeline_capex'])
                    #print('Pipeline OpEx Cost ($USD):', out_dict['pipeline_opex'])
                    print("Pipeline Length (km):", out_dict['total_pipeline_length_km'])
                    print("Pipeline Design Pressure (bar):",in_dict['pressure_bar'])
                    print("Pipeline Diameter: {} in, Thickness {} in".format(out_dict['design_diam_in'][0],out_dict['design_thic_in'][0]))
                    print("Pipeline CapEx ($US): ", capex_pipeline)
                    print("Pipeline Opex ($US/year)", opex_pipeline)
                    print("Substation CapEx ($US): ", capex_substation)
                    print("Total H2-Export CapEx:", total_h2export_system_cost)
     
                    #Pipeline vs HVDC cost
                    #Get Equivalent cost of HVDC export system from Orbit runs and remove it
                    export_system_cost_kw = site_df['Export System'] + site_df['Offshore Substation']
                    export_system_installation_cost_kw = site_df['Export System Installation'] + site_df['Offshore Substation Installation']
                    total_export_system_cost_kw = export_system_cost_kw + export_system_installation_cost_kw
                    export_system_cost = export_system_cost_kw * wind_size_mw * 1000
                    export_system_installation_cost = export_system_installation_cost_kw * wind_size_mw * 1000
                    total_export_system_cost = export_system_cost + export_system_installation_cost
                    print("Total HVDC Export System Cost is ${0:,.0f} vs ${1:,.0f} for H2 Pipeline".format(total_export_system_cost, total_h2export_system_cost))
                    
                    # create data
                    if plot_hvdcpipe:
                        barx = ['HVDC', 'Pipeline']
                        #cost_comparison_hvdc_pipeline = [capex_pipeline,total_export_system_cost]
                        cost_comparison_hvdc_pipeline = [total_export_system_cost, total_h2export_system_cost]
                        plt.figure(figsize=(9,6))
                        plt.bar(barx, cost_comparison_hvdc_pipeline)
    
                        plt.ylabel("$USD")
                        plt.legend(["Total CAPEX"])
                        #plt.title("H2 Pipeline vs HVDC cost\n {}\n Model:{}".format(site_name,in_dict['pipeline_model']))
                        plt.title("H2 Pipeline vs HVDC cost\n {}\n Model: ASME Pipeline".format(site_name))
                        plt.savefig(os.path.join(results_dir,'Pipeline Vs HVDC Cost_{}_{}_{}'.format(site_name,atb_year,dist_to_port_value)))
                        #plt.show()
    
                    #*DANGER: Need to make sure this step doesnt have knock-on effects*
                    # Replace export system cost with pipeline cost
                    #new_wind_cost_kw = wind_cost_kw - total_export_system_cost_kw + pipeline_cost_kw
                    new_wind_cost_kw = wind_cost_kw - total_export_system_cost_kw + total_h2export_system_cost/(wind_size_mw*1000)
                    print("Wind Cost was ${0:,.0f}/kW and is now ${1:.0f}/kW".format(wind_cost_kw, new_wind_cost_kw))
    
                    # Run HOPP again to provide wind capital costs in pipeline scenario and without export
                    hybrid_plant_pipeline, combined_pv_wind_power_production_hopp_pipeline, combined_pv_wind_curtailment_hopp_pipeline,\
                    energy_shortfall_hopp_pipeline, annual_energies_pipeline, wind_plus_solar_npv_pipeline, npvs_pipeline, lcoe_pipeline =  \
                        hopp_for_h2(site, scenario, technologies,
                                    wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                        new_wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                        kw_continuous, load,
                        custom_powercurve,
                        electrolyzer_size, grid_connected_hopp=True, wind_om_cost_kw = wind_om_cost_kw)
                        
                    new_wind_cost_kw_noexport = wind_cost_kw - total_export_system_cost_kw
                        
                    hybrid_plant_noexport, combined_pv_wind_power_production_hopp_noexport, combined_pv_wind_curtailment_hopp_noexport,\
                    energy_shortfall_hopp_noexport, annual_energies_noexport, wind_plus_solar_npv_noexport, npvs_noexport, lcoe_noexport =  \
                        hopp_for_h2(site, scenario, technologies,
                                    wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                        new_wind_cost_kw_noexport, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                        kw_continuous, load,
                        custom_powercurve,
                        electrolyzer_size, grid_connected_hopp=True, wind_om_cost_kw = wind_om_cost_kw)
    
    
                    print("HOPP run for pipeline scenario")
                    
                    
                    # Cash flow analysis for the pipeline
                    discount_rate = scenario['Discount Rate']
                    
                    # Installed cost of pipeline only (find a better way in the future)
                    total_hopp_installed_cost_pipeline = hybrid_plant_pipeline.grid._financial_model.SystemCosts.total_installed_cost
                    total_hopp_installed_cost_noexport = hybrid_plant_noexport.grid._financial_model.SystemCosts.total_installed_cost
                    total_hopp_installed_cost_pipeline_only = total_hopp_installed_cost_pipeline - total_hopp_installed_cost_noexport
                    
                    # Cash flows of pipeline only (again, find a better way)
                    cf_wind_annuals_pipeline = hybrid_plant_pipeline.wind._financial_model.Outputs.cf_annual_costs
                    cf_wind_annuals_noexport = hybrid_plant_noexport.wind._financial_model.Outputs.cf_annual_costs
                    cf_pipelineonly = [a-b for a,b in zip(list(cf_wind_annuals_pipeline),list(cf_wind_annuals_noexport))]
                    
                    # Calculate NPV of pipeline only
                    npv_costs_pipelineonly = npf.npv(discount_rate, cf_pipelineonly)
                    
                    # Calculate levelized cost of hydrogen transport via pipeline
                    #LCOH_cf_method_pipelineonly = -npv_costs_pipelineonly / (H2_Results['hydrogen_annual_output'] * useful_life)
                    LCOHT_cf_method_pipelineonly = -npv_costs_pipelineonly / (hydrogen_annual_production * useful_life)
                    
                    # Calculate total levelized cost of hydrogen including transport
                    levelized_cost_hydrogen_with_transport = levelized_cost_of_hydrogen_RODeO + LCOHT_cf_method_pipelineonly
                    
                    
                    # Steel production break-even price analysis
                    
                    hydrogen_consumption_for_steel = 0.06596              # metric tonnes of hydrogen/metric tonne of steel productio
                    # Could be good to make this more conservative, but it is probably fine if demand profile is flat
                    max_steel_production_capacity_mtpy = hydrogen_annual_production/1000/hydrogen_consumption_for_steel
                    
                    # Could connect these to other things in the model
                    steel_capacity_factor = 0.9
                    steel_plant_life = 30
                    
                    # Should connect these to something (AEO, Cambium, etc.)
                    natural_gas_cost = 4                        # $/MMBTU
                    electricity_cost = 48.92                    # $/MWh
                    
                    steel_economics_from_pyfast,steel_economics_summary = steel_economics.run_pyfast_for_steel(max_steel_production_capacity_mtpy,\
                                                                       steel_capacity_factor,steel_plant_life,levelized_cost_hydrogen_with_transport,\
                                                                       electricity_cost,natural_gas_cost)
    
                    steel_breakeven_price = steel_economics_from_pyfast.get('price')
                        
                    # Step 6.5: Intermediate financial calculation
    
                    total_elec_production = np.sum(electrical_generation_timeseries)*system_rating_mw # Remove if we nolonger need to scale RODeO to 1 MW
                    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
                    # total_hopp_installed_cost_pipeline = hybrid_plant_pipeline.grid._financial_model.SystemCosts.total_installed_cost
                    # total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
                    # print(H2A_Results['scaled_total_installed_cost_kw'])
                    # total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
                    # total_system_installed_cost_pipeline = total_hopp_installed_cost_pipeline + total_electrolyzer_cost
                    # annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
                    # annual_operating_cost_desal = 0#desal_opex
                    # total_annual_operating_costs =  annual_operating_cost_h2 + annual_operating_cost_desal + cost_to_buy_from_grid - profit_from_selling_to_grid
    
                    # h_lcoe_no_op_cost = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                    #                    0, 0.07, useful_life)
    
                    # h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                    #                    total_annual_operating_costs, discount_rate, useful_life)
    
                    # Cashflow Financial Calculation
                    discount_rate = scenario['Discount Rate']
                    cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
                    cf_wind_annuals_pipeline = hybrid_plant_pipeline.wind._financial_model.Outputs.cf_annual_costs
                    if solar_size_mw > 0:
                        cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
                    else:
                        cf_solar_annuals = np.zeros(30)
    
    
                    # if h2_model == 'H2A':
                    #     #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
                    #     pass  
                    # elif h2_model == 'Simple':
                    #     # Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
                    #     # https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf
    
                    #     # Capital costs provide by Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
                    #     stack_capital_cost = 342   #[$/kW]
                    #     mechanical_bop_cost = 36  #[$/kW] for a compressor
                    #     electrical_bop_cost = 82  #[$/kW] for a rectifier
    
                    #     # Installed capital cost
                    #     stack_installation_factor = 12/100  #[%] for stack cost 
                    #     elec_installation_factor = 12/100   #[%] and electrical BOP 
                    #     #mechanical BOP install cost = 0%
    
                    #     # Indirect capital cost as a percentage of installed capital cost
                    #     site_prep = 2/100   #[%]
                    #     engineering_design = 10/100 #[%]
                    #     project_contingency = 15/100 #[%]
                    #     permitting = 15/100     #[%]
                    #     land = 250000   #[$]
    
                    #     stack_replacment_cost = 15/100  #[% of installed capital cost]
                    #     plant_lifetime = 40    #[years]
                    #     fixed_OM = 0.24     #[$/kg H2]
    
                    #     program_record = False
    
                    #     # Chose to use numbers provided by GPRA pathways
                    #     if program_record:
                    #         total_direct_electrolyzer_cost_kw = (stack_capital_cost*(1+stack_installation_factor)) \
                    #             + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))
                    #     else:
                    #         total_direct_electrolyzer_cost_kw = (electrolyzer_capex_kw * (1+stack_installation_factor)) \
                    #             + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))
    
                    #     # Assign CapEx for electrolyzer from capacity based installed CapEx
                    #     electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw* electrolyzer_size *1000
    
                    #     # Add indirect capital costs
                    #     electrolyzer_total_capital_cost = ((site_prep+engineering_design+project_contingency+permitting)\
                    #         *electrolyzer_total_installed_capex) + land
    
                    #     # O&M costs
                    #     # https://www.sciencedirect.com/science/article/pii/S2542435121003068
                    #     fixed_OM = 12.8 #[$/kWh-y]
                    #     property_tax_insurance = 1.5/100    #[% of Cap/y]
                    #     variable_OM = 1.30  #[$/MWh]
    
                    #     # Amortized refurbishment expense [$/MWh]
                    #     amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacment_cost)\
                    #             *max(((useful_life*8760*H2_Results['cap_factor'])/time_between_replacement-1),0)/useful_life/8760/H2_Results['cap_factor']*1000
    
                    #     # Total O&M costs [% of installed cap/year]
                    #     total_OM_costs = ((fixed_OM+(property_tax_insurance*total_direct_electrolyzer_cost_kw))/total_direct_electrolyzer_cost_kw\
                    #         +((variable_OM+amortized_refurbish_cost)/1000*8760*(H2_Results['cap_factor']/total_direct_electrolyzer_cost_kw)))
    
                    #     capacity_based_OM = True
                    #     if capacity_based_OM:
                    #         electrolyzer_OM_cost = electrolyzer_total_installed_capex * total_OM_costs     #Capacity based
                    #     else:   
                    #         electrolyzer_OM_cost = fixed_OM  * H2_Results['hydrogen_annual_output'] #Production based - likely not very accurate
    
                    #     cf_h2_annuals = - simple_cash_annuals(useful_life, useful_life, electrolyzer_total_capital_cost,\
                    #         electrolyzer_OM_cost, 0.03)
                        
                    # Include Hydrogen PTC from the Inflation Reduction Act (range $0.60 - $3/kg-H2)
                    #h2_tax_credit = [H2_Results['hydrogen_annual_output']*scenario['H2 PTC']] * useful_life
                    #cf_h2_annuals = np.add(cf_h2_annuals,h2_tax_credit)
                    #
                    #cf_operational_annuals = [-total_annual_operating_costs for i in range(30)]
                    #
                    #cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals],['Wind', 'Solar', 'H2'])
                    #
                    #cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}_{}MW.csv".format(site_name, scenario_choice, atb_year, discount_rate,turbine_rating_mw)))
    # #
                    #NPVs of wind, solar, H2
                    npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
                    
                    npv_wind_costs_pipeline = npf.npv(discount_rate, cf_wind_annuals_pipeline)
                    npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
                    #npv_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
                    npv_h2_costs = RODeO_results_summary_dict['NPV of actual operating profit ($)']*system_rating_mw#npf.npv(discount_rate, cf_h2_annuals)
                    print("NPV H2 Costs using {} model: {}".format(h2_model,npv_h2_costs))
                    #npv_operating_costs = npf.npv(discount_rate, cf_operational_annuals)
                    #npv_desal_costs = -desal_capex
                    #print("Desal CAPEX: ",desal_capex)
    
                    npv_total_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs
                    npv_total_costs_pipeline = npv_wind_costs_pipeline + npv_solar_costs + npv_h2_costs
                    #npv_total_costs_w_operating_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs+npv_operating_costs
                    #npv_total_costs_w_operating_costs_pipeline = npv_wind_costs_pipeline+npv_solar_costs+npv_h2_costs+npv_operating_costs
    
    #                 LCOH_cf_method_wind = -npv_wind_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_wind_pipeline = -npv_wind_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_solar = -npv_solar_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_h2_costs = -npv_h2_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_desal_costs = -npv_desal_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_operating_costs = -npv_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    
    #                 LCOH_cf_method = -npv_total_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_pipeline = -npv_total_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_w_operating_costs = -npv_total_costs_w_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    #                 LCOH_cf_method_w_operating_costs_pipeline = -npv_total_costs_w_operating_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
                    # financial_summary_df = pd.DataFrame([scenario['Useful Life'], wind_cost_kw, solar_cost_kw, electrolyzer_capex_kw,
                    #                                         scenario['Debt Equity'], atb_year, scenario['Wind PTC'], scenario['H2 PTC'],scenario['Wind ITC'],
                    #                                         discount_rate, npv_wind_costs, npv_solar_costs, npv_h2_costs, LCOH_cf_method, LCOH_cf_method_pipeline, LCOH_cf_method_w_operating_costs, LCOH_cf_method_w_operating_costs_pipeline],
                    #                                     ['Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'Electrolyzer Cost KW', 'Debt Equity',
                    #                                         'ATB Year', 'Wind PTC', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses', 'NPV Solar Expenses', 'NPV H2 Expenses', 'LCOH cf method HVDC','LCOH cf method Pipeline','LCOH cf method HVDC w/operating cost','LCOH cf method Pipeline w/operating cost'])
                    financial_summary_df = pd.DataFrame([scenario['Useful Life'], wind_cost_kw, solar_cost_kw, electrolyzer_capex_USD_per_MW/1000,
                                                            scenario['Debt Equity'], atb_year, scenario['Wind PTC'], scenario['H2 PTC'],scenario['Wind ITC'],
                                                            discount_rate, npv_wind_costs, npv_solar_costs, npv_h2_costs, levelized_cost_hydrogen_with_transport],
                                                        ['Useful Life (years)', 'Wind Cost ($/kW)', 'Solar Cost ($/kW)', 'Electrolyzer Cost ($/kW)', 'Debt Equity',
                                                            'ATB Year', 'Wind PTC', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses', 'NPV Solar Expenses', 'NPV H2 Expenses', 'Total LCOH ($/kg)'])
                    financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model)))
    
                    # # Gut Check H2 calculation (non-levelized)
                    # total_installed_and_operational_lifetime_cost = total_system_installed_cost + (30 * total_annual_operating_costs)
                    # lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
                    # gut_check_h2_cost_kg = total_installed_and_operational_lifetime_cost / lifetime_h2_production
    
                    # print("Gut Check H2 Cost:",gut_check_h2_cost_kg)
                    # print("HVDC Scenario: LCOH w/o Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method)
                    # print("HVDC Scenario: LCOH WITH Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_w_operating_costs)
    
                    # print("Pipeline Scenario: LCOH w/o Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_pipeline)
                    # print("Pipeline Scenario: LCOH WITH Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_w_operating_costs_pipeline)
    
                    # Step 7: Plot Results
                    
                    # create data
                    #x = ['HVDC', 'Pipeline']
                    
                    # # plot bars in stack manner
                    # if plot_hvdcpipe_lcoh:
                    #     plt.figure(figsize=(9,6))
                    #     plt.bar(barx, [LCOH_cf_method_wind,LCOH_cf_method_wind_pipeline], color='blue')
                    #     plt.bar(barx, LCOH_cf_method_solar, bottom=[LCOH_cf_method_wind,LCOH_cf_method_wind_pipeline], color='orange')
                    #     plt.bar(barx, LCOH_cf_method_h2_costs, bottom =[(LCOH_cf_method_wind + LCOH_cf_method_solar), (LCOH_cf_method_wind_pipeline + LCOH_cf_method_solar)], color='g')
                    #     plt.bar(barx, LCOH_cf_method_operating_costs, bottom=[(LCOH_cf_method_wind + LCOH_cf_method_solar + LCOH_cf_method_h2_costs),(LCOH_cf_method_wind_pipeline + LCOH_cf_method_solar + LCOH_cf_method_h2_costs)], color='y')
                    #     plt.bar(barx, LCOH_cf_method_desal_costs, bottom=(LCOH_cf_method_wind + LCOH_cf_method_solar + LCOH_cf_method_h2_costs + LCOH_cf_method_operating_costs), color='k')
    
                    #     plt.ylabel("LCOH")
                    #     plt.legend(["Wind", "Solar", "H2", "Operating Costs", "Desal"])
                    #     plt.title("Levelized Cost of hydrogen - Cost Contributors\n {}\n {}\n {} ptc".format(site_name,atb_year,turbine_model))
                    #     plt.savefig(os.path.join(results_dir,'LCOH Barchart_{}_{}_{}.jpg'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                    #     # plt.show()
    
                    print_results = True
                    print_h2_results = True
                    save_outputs_dict = save_the_things()
                    save_all_runs.append(save_outputs_dict)
                    save_outputs_dict = establish_save_output_dict()
    
                    if print_results:
                        # ------------------------- #
                        #TODO: Tidy up these print statements
                        #print("Future Scenario: {}".format(scenario['Scenario Name']))
                        print("Wind Cost per KW: {}".format(wind_cost_kw))
                        print("PV Cost per KW: {}".format(solar_cost_kw))
                        #print("Storage Cost per KW: {}".format(scenario['Storage Cost kW']))
                        #print("Storage Cost per KWh: {}".format(scenario['Storage Cost kWh']))
                        print("Wind Size built: {}".format(wind_size_mw))
                        print("PV Size built: {}".format(solar_size_mw))
                        print("Storage Size built: {}".format(storage_size_mw))
                        print("Storage Size built: {}".format(storage_size_mwh))
                        print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
                        print("Total Yearly Electrical Output: {}".format(total_elec_production))
                        #print("Total Yearly Hydrogen Production: {}".format(H2_Results['hydrogen_annual_output']))
                        print("Total Yearly Hydrogen Production (kg): {}".format(hydrogen_annual_production))
                        #print("Levelized Cost H2/kg (new method - no operational costs)".format(h_lcoe_no_op_cost))
                        print("Levelized Cost H2/kg".format(levelized_cost_hydrogen_with_transport))
                        #print("Capacity Factor of Electrolyzer: {}".format(H2_Results['cap_factor']))
                        print("Capacity Factor of Electrolyzer: {}".format(RODeO_results_summary_dict['input capacity factor']))
    
                    # if print_h2_results:
                    #     print('Total Lifetime H2(kg) produced: {}'.format(lifetime_h2_production))
                    #     print("Gut-check H2 cost/kg: {}".format(gut_check_h2_cost_kg))
                    # #     print("h_lcoe: ", h_lcoe)
                    #     print("LCOH CF Method (doesn't include grid electricity cost if used)", LCOH_cf_method)
                    #     print("LCOH CF Method (includes operating costs + electricity)", LCOH_cf_method_w_operating_costs)
                    #     # print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
                    #     #     H2_Results['feedstock_cost_h2_levelized_hopp']))
                    #     # print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(H2A_Results['Total Hydrogen Cost ($/kgH2)']))
                    #     # print("Total unit cost of H2 ($/kg) : {}".format(H2_Results['total_unit_cost_of_hydrogen']))
                    #     # print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(
                    #     #     H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp']))
    
                        #Step 9: Summarize Results
                        # print('For a {}MW Offshore Wind Plant with {}MW electrolyzer located at {} \n (average wind speed {}m/s) in {}, with a Wind CAPEX cost of {},\n and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} /kg '.
                        #             format(forced_wind_size,electrolyzer_size,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_capex_kw,LCOH_cf_method_w_operating_costs))
    
                        # print("LCOH CF Method (doesn't include grid electricity cost if used)", LCOH_cf_method)
                        # print("LCOH CF Method (includes operating costs + electricity)", LCOH_cf_method_w_operating_costs)

save_outputs = True
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    save_all_runs_df.to_csv(os.path.join(results_dir, "HOPP_RODeO_Analysis_OSW_All.csv"))


print('Done')

