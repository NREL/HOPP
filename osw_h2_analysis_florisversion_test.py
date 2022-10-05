import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2_floris import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
import yaml
import re
from yamlinclude import YamlIncludeConstructor 
cwd = os.getcwd()
print(cwd)

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='/your/conf/dir')


warnings.filterwarnings("ignore")

"""
Perform a LCOH analysis for an offshore wind + Hydrogen PEM system

1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~
2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
3. Cost Scaling Based on Plant Size (Shields et. Al report)
4. Future Model Development Required:
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
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Policy Option'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Turbine Model'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['System Load (kW)'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['Wind PTC ($/kW)'] = list()
    save_outputs_dict['H2 PTC ($/kg)'] = list()
    save_outputs_dict['ITC (%)'] = list()
    save_outputs_dict['Discount Rate (%)'] = list()
    save_outputs_dict['Debt Equity'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['Storage Enabled'] = list()
    save_outputs_dict['Wind Cost ($/kW)'] = list()
    save_outputs_dict['Solar Cost ($/kW)'] = list()
    save_outputs_dict['Storage Cost ($/kW)'] = list()
    save_outputs_dict['Storage Cost ($/kWh)'] = list()
    save_outputs_dict['Electrolyzer Cost ($/kW)'] = list()
    save_outputs_dict['Storage Hours'] = list()
    save_outputs_dict['Wind built (MW)'] = list()
    save_outputs_dict['Solar built (MW)'] = list()
    save_outputs_dict['Storage built (MW)'] = list()
    save_outputs_dict['Storage built (MWh)'] = list()
    save_outputs_dict['Electrolyzer built (MW)'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['Built Interconnection Size (MW)'] = list()
    save_outputs_dict['Wind Total Installed Cost ($)'] = list()
    save_outputs_dict['Desal Total Installed Cost ($)'] = list()
    save_outputs_dict['HVDC Total Installed Cost ($)'] = list()
    save_outputs_dict['Pipeline Total Installed Cost ($)'] = list()
    save_outputs_dict['Electrolyzer Total Installed Capital Cost ($)'] = list()
    save_outputs_dict['Total Plant Capital Cost HVDC ($)'] = list()
    save_outputs_dict['Total Plant Capital Cost Pipeline ($)'] = list()
    save_outputs_dict['Total Plant Operating Cost HVDC ($)'] = list()
    save_outputs_dict['Total Plant Operating Cost Pipeline ($)'] = list()
    save_outputs_dict['Real LCOE ($/MWh)'] = list()
    save_outputs_dict['Nominal LCOE ($/MWh)'] = list()
    save_outputs_dict['Total Annual H2 production (kg)'] = list()
    save_outputs_dict['H2 total tax credit ($)'] = list()
    save_outputs_dict['Total ITC HVDC scenario ($)'] = list()
    save_outputs_dict['Total ITC Pipeline scenario ($)'] = list()
    save_outputs_dict['TLCC Wind ($)'] = list()
    save_outputs_dict['TLCC H2 ($)'] = list()
    save_outputs_dict['TLCC Desal ($)'] = list()
    save_outputs_dict['TLCC HVDC ($)'] = list()
    save_outputs_dict['TLCC Pipeline ($)'] = list()
    save_outputs_dict['LCOH Wind contribution ($/kg)'] = list()
    save_outputs_dict['LCOH H2 contribution ($/kg)'] = list()
    save_outputs_dict['LCOH Desal contribution ($/kg)'] = list()
    save_outputs_dict['LCOH HVDC contribution ($/kg)'] = list()
    save_outputs_dict['LCOH Pipeline contribution ($/kg)'] = list()    
    save_outputs_dict['H_LCOE HVDC scenario no operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE HVDC scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE Pipeline scenario no operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE Pipeline scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 HVDC scenario (CF Method - using annual cashflows per technology) ($/kg)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 Pipeline scenario (CF Method - using annual cashflows per technology) ($/kg)'] = list()
    save_outputs_dict['Gut-Check Cost/kg H2 Pipeline scenario (non-levelized, includes elec if used) ($/kg)'] = list()
    save_outputs_dict['Gut-Check Cost/kg H2 HVDC scenario (non-levelized, includes elec if used) ($/kg)'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['HOPP Total Electrical Generation (kWh)'] = list()
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer (kWh)'] = list()
    save_outputs_dict['Wind Capacity Factor (%)'] = list()
    save_outputs_dict['Electrolyzer Capacity Factor (%)'] = list()
    save_outputs_dict['HOPP Annual Energy Shortfall (kWh)'] = list()
    save_outputs_dict['HOPP Annual Energy Curtailment (kWh)'] = list()
    save_outputs_dict['Battery Generation (kWh)'] = list()
    save_outputs_dict['Electricity to Grid (kWh)'] = list()
    
    return save_outputs_dict

# Dictionary that saves all of the results from the analysis 
# FYI: there are more results calculated in analysis than saved to a file
def save_the_things():
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Policy Option'] = (policy_option)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_model)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (kw_continuous)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['Wind PTC ($/kW)'] = (scenario['Wind PTC'])
    save_outputs_dict['H2 PTC ($/kg)'] = (scenario['H2 PTC'])
    save_outputs_dict['ITC (%)'] = (scenario['Wind ITC'])
    save_outputs_dict['Discount Rate (%)'] = (discount_rate*100)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['Storage Enabled'] = (storage_used)
    save_outputs_dict['Wind Cost ($/kW)'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost ($/kW)'] = (solar_cost_kw)
    save_outputs_dict['Storage Cost ($/kW)'] = (storage_cost_kw)
    save_outputs_dict['Storage Cost ($/kWh)'] = (storage_cost_kwh)
    save_outputs_dict['Electrolyzer Cost ($/kW)'] = (electrolyzer_capex_kw)
    save_outputs_dict['Storage Hours'] = (storage_hours)
    save_outputs_dict['Wind built (MW)'] = (wind_size_mw)
    save_outputs_dict['Solar built (MW)'] = (solar_size_mw)
    save_outputs_dict['Storage built (MW)'] = (storage_size_mw)
    save_outputs_dict['Storage built (MWh)'] = (storage_size_mwh)
    save_outputs_dict['Electrolyzer built (MW)'] = (electrolyzer_size_mw)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['Built Interconnection Size (MW)'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Wind Total Installed Cost ($)'] = (total_hopp_installed_cost)
    save_outputs_dict['Desal Total Installed Cost ($)'] = (total_desal_cost)
    save_outputs_dict['HVDC Total Installed Cost ($)'] = (total_export_system_cost)
    save_outputs_dict['Pipeline Total Installed Cost ($)'] = (total_h2export_system_cost)
    save_outputs_dict['Electrolyzer Total Installed Capital Cost ($)'] = (electrolyzer_total_capital_cost)
    save_outputs_dict['Total Plant Capital Cost HVDC ($)'] = (total_system_installed_cost_hvdc)
    save_outputs_dict['Total Plant Capital Cost Pipeline ($)'] = (total_system_installed_cost_pipeline)
    save_outputs_dict['Total Plant Operating Cost HVDC ($)'] = (total_annual_operating_costs_hvdc)
    save_outputs_dict['Total Plant Operating Cost Pipeline ($)'] = (total_annual_operating_costs_pipeline)
    save_outputs_dict['Real LCOE ($/MWh)'] = (lcoe*10)
    save_outputs_dict['Nominal LCOE ($/MWh)'] = (lcoe_nom*10)
    save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    save_outputs_dict['H2 total tax credit ($)'] = (np.sum(h2_tax_credit))
    save_outputs_dict['Total ITC HVDC scenario ($)'] = (total_itc_hvdc)
    save_outputs_dict['Total ITC Pipeline scenario ($)'] = (total_itc_pipeline)
    save_outputs_dict['TLCC Wind ($)'] = (tlcc_wind_costs)
    save_outputs_dict['TLCC H2 ($)'] = (tlcc_h2_costs)
    save_outputs_dict['TLCC Desal ($)'] = (tlcc_desal_costs)
    save_outputs_dict['TLCC HVDC ($)'] = (tlcc_hvdc_costs)
    save_outputs_dict['TLCC Pipeline ($)'] = (tlcc_pipeline_costs)
    save_outputs_dict['LCOH Wind contribution ($/kg)'] = (LCOH_cf_method_wind)
    save_outputs_dict['LCOH H2 contribution ($/kg)'] = (LCOH_cf_method_h2_costs)
    save_outputs_dict['LCOH Desal contribution ($/kg)'] = (LCOH_cf_method_desal_costs)
    save_outputs_dict['LCOH HVDC contribution ($/kg)'] = (LCOH_cf_method_pipeline)
    save_outputs_dict['LCOH Pipeline contribution ($/kg)'] =  (LCOH_cf_method_hvdc) 
    save_outputs_dict['H_LCOE HVDC scenario no operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_no_op_cost_hvdc)
    save_outputs_dict['H_LCOE HVDC scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_hvdc)
    save_outputs_dict['H_LCOE Pipeline scenario no operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_no_op_cost_pipeline)
    save_outputs_dict['H_LCOE Pipeline scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_pipeline)
    save_outputs_dict['Levelized Cost/kg H2 HVDC scenario (CF Method - using annual cashflows per technology) ($/kg)'] = (LCOH_cf_method_total_hvdc)
    save_outputs_dict['Levelized Cost/kg H2 Pipeline scenario (CF Method - using annual cashflows per technology) ($/kg)'] = (LCOH_cf_method_total_pipeline)
    save_outputs_dict['Gut-Check Cost/kg H2 Pipeline scenario (non-levelized, includes elec if used) ($/kg)'] = (gut_check_h2_cost_kg_pipeline)
    save_outputs_dict['Gut-Check Cost/kg H2 HVDC scenario (non-levelized, includes elec if used) ($/kg)'] = (gut_check_h2_cost_kg_hvdc)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Electrical Generation (kWh)'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer (kWh)'] = (total_elec_production)
    save_outputs_dict['Wind Capacity Factor (%)'] = (hybrid_plant.wind.capacity_factor)
    save_outputs_dict['Electrolyzer Capacity Factor (%)'] = (H2_Results['cap_factor']*100)
    save_outputs_dict['HOPP Annual Energy Shortfall (kWh)'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Annual Energy Curtailment (kWh)'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation (kWh)'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid (kWh)'] = (np.sum(excess_energy))

    return save_outputs_dict

#Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

#Step 1: User Inputs for scenario
atb_years = [
            2022,
            2025,
            2030,
            2035
            ]
policy = {
    'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
    'option 2': {'Wind ITC': 26, 'Wind PTC': 0, "H2 PTC": 0},
    'option 3': {'Wind ITC': 6, 'Wind PTC': 0, "H2 PTC": 0.6},
    'option 4': {'Wind ITC': 30, 'Wind PTC': 0, "H2 PTC": 3},
    'option 5': {'Wind ITC': 50, 'Wind PTC': 0, "H2 PTC": 3},
}

resource_year = 2013
sample_site['year'] = resource_year
useful_life = 30
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = False   #A flag that is applicable when using PySam WindPower (not FLORIS)
storage_used = False
battery_can_grid_charge = False
grid_connected_hopp = False

# Technology sizing
interconnection_size_mw = 1000
electrolyzer_size_mw = 1000
forced_wind_size = 1000     #name forced_wind_size is a legacy that hasn't been removed yet
forced_solar_size = 0
forced_storage_size_mw = 0
forced_storage_size_mwh = 0

turbine_name = [
                '12MW',
                '15MW',
                '18MW'
                ]
scenario_choice = 'Offshore Wind-H2 Analysis'
site_selection = [
                'Site 1',
                'Site 2',
                'Site 3',
                'Site 4'
                ]

scenario = dict()
kw_continuous = electrolyzer_size_mw * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant



parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results/'

floris_dir = parent_path + '/floris_input_files/'


#Site lat and lon will be set by data loaded from Orbit runs

# Finanical inputs
discount_rate = 0.07
debt_equity_split = 60

# Wind costs input from ORBIT analysis
h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
# h2_model = 'H2A'

# These inputs are not used in this analysis (no solar or storage)
solar_cost_kw = 9999
storage_cost_kw = 250
storage_cost_kwh = 240

# Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
# sell_price = 0.01
# buy_price = 0.01
sell_price = False
buy_price = False

forced_sizes = True
force_electrolyzer_cost = True


save_outputs_dict = establish_save_output_dict()
save_all_runs = list()

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = True
plot_h2 = True
plot_desal = True
plot_wind = True
plot_hvdcpipe = True
plot_hvdcpipe_lcoh = True

for option in policy:
    # Set policy values
    policy_option = option.__str__()
    scenario = policy[policy_option]
    print('Policy option: ',option,'ITC: ',scenario['Wind ITC'],'Wind PTC: ',scenario['Wind PTC'], 'H2 PTC: ', scenario['H2 PTC'])
    for atb_year in atb_years:
        for site_location in site_selection:
            site_number = site_location.split(' ')[1]
            for turbine_model in turbine_name:
                # Define Turbine Characteristics based on user selected turbine.
                ########## TEMPERARY ###########
                site_number = 'base'
                site_number = 'singleT'
                site_number = 'osw'
                ################################

                turbine_file = floris_dir + 'floris_input' + turbine_model + '_' + site_number + '.yaml'
                with open(turbine_file, 'r') as f:
                    floris_config = yaml.load(f, yaml.FullLoader)
                    # floris_config = yaml.load(f, yaml.SafeLoader)
                nTurbs = len(floris_config['farm']['layout_x'])
                # turbine_type = floris_config['farm']['turbine_type'][0]
                turbine_type = floris_config['farm']['turbine_type'][0]['turbine_type']
                # print(floris_config['farm']['turbine_type'][0]['turbine_type'])
                
                turbine_rating_mw = float(re.findall('[0-9]+', turbine_type)[0])
                wind_cost_kw = 1300

                # Scaled from reference 15MW turbine: https://github.com/IEAWindTask37/IEA-15-240-RWT
                if turbine_model == '12MW':
                    custom_powercurve_path = '2022atb_osw_12MW.csv' 
                    tower_height = 136
                    rotor_diameter = 215
                    wind_cost_kw = 1300
                    # Future Cost Reduction Estimates - ATB 2022: Class 4 Fixed, Class 11 Float
                    floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_12MW.csv'))
                    fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_12MW.csv'))

                elif turbine_model == '15MW':
                    custom_powercurve_path = '2022atb_osw_15MW.csv'
                    tower_height = 150
                    rotor_diameter = 240
                    wind_cost_kw =  1300
                    # Future Cost Reduction Estimates
                    floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_15MW.csv'))
                    fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_15MW.csv'))

                elif turbine_model == '18MW':
                    custom_powercurve_path = '2022atb_osw_18MW.csv' 
                    tower_height = 161
                    rotor_diameter = 263
                    wind_cost_kw = 1300
                    # Future Cost Reduction Estimates
                    floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_18MW.csv'))
                    fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_18MW.csv'))
                    
                scenario['Useful Life'] = useful_life
                scenario['Debt Equity'] = debt_equity_split
                scenario['Discount Rate'] = discount_rate
                scenario['Tower Height'] = tower_height

                # use this if running PySam WindPower
                # scenario['Powercurve File'] = custom_powercurve_path
                # print("Powercurve Path: ", custom_powercurve_path)

                # #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
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
                wind_cost_kw = site_df['Total CapEx']
                wind_om_cost_kw = site_df['OpEx, $/kW-yr']
                wind_net_cf = site_df['Assumed NCF']        #net capacity factor

                # HVDC Export System
                export_system_cost_kw = site_df['Export System'] + site_df['Offshore Substation']
                export_system_installation_cost_kw = site_df['Export System Installation'] + site_df['Offshore Substation Installation']
                total_export_system_cost_kw = export_system_cost_kw + export_system_installation_cost_kw
                
                wind_cost_kw = wind_cost_kw - total_export_system_cost_kw # Wind System Cost per KW ($US/kW) with no HVDC export system  
                
                # Export System CapEx $US
                export_system_cost = export_system_cost_kw * forced_wind_size * 1000
                export_system_installation_cost = export_system_installation_cost_kw * forced_wind_size * 1000
                total_export_system_cost = export_system_cost + export_system_installation_cost

                # Rough OpEx Estimation 
                # https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub = 0.5% CapEx per lifetime for offshore cables 
                export_om_cost_kw = 0.5/100 * total_export_system_cost_kw / useful_life  # US/kW-yr (assume 30 year lifetime)
                
                wind_om_cost_kw = wind_om_cost_kw - export_om_cost_kw # Wind System OM Cost with no HVDC OM cost estimates
                
                total_export_om_cost = 0.5/100 * total_export_system_cost / useful_life # $US total (assume 30 year lifetime))
                
                #Assign site location and details
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


                #Plot Wind Data to ensure offshore data is sound
                wind_data = site.wind_resource._data['data']
                wind_data2 = site.wind_resource.data['data']
                print(np.shape(wind_data))

                if plot_wind:
                    wind_speed = [x[2] for x in wind_data]
                    wind_speed2 = [x[6] for x in wind_data]
                    ws_avg = [x[6] - 0.5*(x[6]+x[2]) for x in wind_data]
                    ws_diff = [x[6]-x[2] for x in wind_data]
                    # plt.plot(wind_data2)
                    # plt.plot(wind_speed)
                    # plt.plot(wind_speed2)
                    plt.plot(ws_diff)
                    plt.plot(ws_avg)
                    plt.title('Wind Speed (m/s) for selected location \n {} \n lat, lon: {} \n Average Wind Speed (m/s) {}'.format(site_name,latlon,np.average(wind_speed)))
                    plt.savefig(os.path.join(results_dir,'Average Wind Speed_{}'.format(site_name)),bbox_inches='tight')
                    # plt.show()
                    # jkjkjkjk

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
                print('ORBIT Cost Contributions for {}_{}'.format(site_name,turbine_model))
                plt.savefig(os.path.join(results_dir,'BOS Cost Figure {}_{}.jpg'.format(site_name,turbine_model)),bbox_inches='tight')
                # plt.show()

                #Display Future Cost Reduction Estimates
                # Floating Wind Cost Reductions
                floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions.csv'))

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
                print("wind om cost ORBIT:",wind_om_cost_kw)
                
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
                                    'wind': {
                                        'num_turbines': nTurbs,
                                        'turbine_rating_kw': turbine_rating_mw*1000,
                                        'model_name': 'floris',
                                        'timestep': [0,8759],
                                        'floris_config': floris_config # if not specified, use default SAM models
                                    },
                                    'battery': {
                                        'system_capacity_kwh': storage_size_mwh * 1000,
                                        'system_capacity_kw': storage_size_mw * 1000
                                        }
                                    }
                else:
                            technologies = {#'pv':
                                        #{'system_capacity_kw': solar_size_mw * 1000},
                                    'wind': {
                                        'num_turbines': nTurbs,
                                        'turbine_rating_kw': turbine_rating_mw*1000,
                                        'model_name': 'floris',
                                        'timestep': [0,8759],
                                        'floris_config': floris_config # if not specified, use default SAM models
                                    }}

                hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
                energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe, lcoe_nom =  \
                    hopp_for_h2(site, scenario, technologies,
                                wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                    wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                    kw_continuous, load,
                    custom_powercurve,
                    electrolyzer_size_mw, grid_connected_hopp=False, wind_om_cost_kw=wind_om_cost_kw)

                wind_installed_cost = hybrid_plant.wind.total_installed_cost
                if solar_size_mw > 0:
                    solar_installed_cost = hybrid_plant.pv.total_installed_cost
                else:
                    solar_installed_cost = 0
                hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

                print("HOPP run complete")
                #print(hybrid_plant.om_capacity_expenses)
                # np.savetxt('floris_losses_turbine_power.txt', combined_pv_wind_power_production_hopp)

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
                # Not utilized in this analysis
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

                # Not utilized in this analysis. Off-grid system.
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
                    plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],"--",label="after buy from grid")
                    plt.plot(energy_to_electrolyzer[200:300],"--",label="energy to electrolyzer")
                    plt.legend()
                    plt.title('Power available after purchasing from grid (if enabled)')
                    # plt.show()

                #Step 6: Run the H2_PEM model
                #TODO: Refactor H2A model call
                # H2A model is unreliable!!                
                # Should take as input (electrolyzer size, cost, electrical timeseries, total system electrical usage (kwh/kg),
                # Should give as ouptut (h2 costs by net cap cost, levelized, total_unit_cost of hydrogen etc)   )

                # electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp
                electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
                electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

                adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
                #NB: adjusted_installed_cost does NOT include the electrolyzer cost
                useful_life = scenario['Useful Life']
                net_capital_costs = 0

                # system_rating = electrolyzer_size
                system_rating = wind_size_mw + solar_size_mw
                H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size_mw,
                                kw_continuous,electrolyzer_capex_kw,lcoe,adjusted_installed_cost,useful_life,
                                net_capital_costs)


                H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']
                H2_Results['cap_factor'] = H2_Results['cap_factor']
                
                print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
                print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
                print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))

                if plot_h2:
                    hydrogen_hourly_production = H2_Results['hydrogen_hourly_production']
                    plt.figure(figsize=(8,8))
                    plt.subplot(411)
                    plt.plot(electrical_generation_timeseries[200:300])
                    plt.ylim(0,max(electrical_generation_timeseries[200:300])*1.2)
                    plt.plot(load[200:300],label="electrolyzer rating")
                    plt.legend()
                    plt.title("Energy to electrolyzer (kW)")

                    plt.subplot(412)
                    plt.plot(hydrogen_hourly_production[200:300])
                    plt.ylim(0,max(hydrogen_hourly_production[200:300])*1.2)
                    plt.title("Hydrogen production rate (kg/hr)")

                    
                    plt.subplot(413)
                    plt.plot(H2_Results['electrolyzer_total_efficiency'][200:300])
                    plt.ylim(0,1)
                    plt.title("Electrolyzer Total Efficiency (%)")
                    
                    
                    plt.subplot(414)
                    plt.plot(H2_Results['water_hourly_usage'][200:300],"--",label="Hourly Water Usage")
                    plt.legend()
                    plt.title('Hourly Water Usage (kg/hr) \n' 'Total Annual Water Usage: {0:,.0f}kg'.format(H2_Results['water_annual_usage']))
                    plt.tight_layout()
                    plt.xlabel('Time (hours)')
                    plt.savefig(os.path.join(results_dir,'Electrolyzer Flows_{}_{}_{}'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                    # plt.show()

                #Electrolyzer financial model
                if h2_model == 'H2A':
                    #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
                    pass  
                elif h2_model == 'Simple':
                    from examples.H2_Analysis.H2_cost_model import basic_H2_cost_model
                    
                    cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc = \
                        basic_H2_cost_model(electrolyzer_size_mw, useful_life, atb_year,
                        electrical_generation_timeseries, H2_Results['hydrogen_annual_output'], scenario['H2 PTC'], scenario['Wind ITC'])

                #Step 6b: Run desal model
                from examples.H2_Analysis.desal_model import RO_desal

                water_usage_electrolyzer = H2_Results['water_hourly_usage']
                m3_water_per_kg_h2 = 0.01
                desal_system_size_m3_hr = electrolyzer_size_mw * (1000/55.5) * m3_water_per_kg_h2
                est_const_desal_power_mw_hr = desal_system_size_m3_hr * 4.2 /1000 # 4.2kWh/m^3 desal efficiency estimate
                # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
                Power = electrical_generation_timeseries
                fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals, desal_itc = RO_desal(Power, desal_system_size_m3_hr, useful_life, useful_life, scenario['Wind ITC'])
                print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size_mw, desal_system_size_m3_hr))
                print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
                print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
                print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
                # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
                print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

                if plot_desal:
                    plt.figure(figsize=(10,5))
                    plt.subplot(1,2,1)
                    plt.plot(fresh_water_flowrate[200:300],"--",label="Freshwater flowrate from desal")
                    plt.plot(feed_water_flowrate[200:300],"--",label="Feedwater flowrate to desal")
                    plt.legend()
                    plt.title('Freshwater flowrate (m^3/hr) from desal  (Snapshot)')
                    # plt.show()

                    plt.subplot(1,2,2)
                    plt.plot(operational_flags[200:300],"--",label="Operational Flag")
                    plt.legend()
                    plt.title('Desal Equipment Operational Status (Snapshot) \n 0 = Not enough power to operate \n 1 = Operating at reduced capacity \n 2 = Operating at full capacity')
                    plt.savefig(os.path.join(results_dir,'Desal Flows_{}_{}_{}'.format(site_name,atb_year,turbine_model)),bbox_inches='tight')
                    # plt.show()

                #Compressor Model
                #Not currently used in analysis (there is a built in compressor cost in electrolyzer but not other compression is considered)
                from examples.H2_Analysis.compressor import Compressor
                in_dict = dict()
                in_dict['flow_rate_kg_hr'] = 89
                in_dict['P_outlet'] = 100
                in_dict['compressor_rating_kWe'] = 802
                in_dict['mean_time_between_failure'] = 200
                in_dict['total_hydrogen_throughput'] = 18750
                compressor_results = dict()
                compressor = Compressor(in_dict, compressor_results)
                compressor.compressor_power()
                compressor.compressor_costs()
                print("compressor_power (kW): ", compressor_results['compressor_power'])
                print("Compressor capex [USD]: ", compressor_results['compressor_capex'])
                print("Compressor opex [USD/yr]: ", compressor_results['compressor_opex'])

                #Pressure Vessel Model Example
                #No end use so pressure vessel is not utilized
                from examples.H2_Analysis.underground_pipe_storage import Underground_Pipe_Storage
                storage_input = dict()
                storage_input['H2_storage_kg'] = 18750
                # storage_input['storage_duration_hrs'] = 4
                # storage_input['flow_rate_kg_hr'] = 89        #[kg-H2/hr]
                storage_input['compressor_output_pressure'] = 100
                storage_output = dict()
                underground_pipe_storage = Underground_Pipe_Storage(storage_input, storage_output)
                underground_pipe_storage.pipe_storage_costs()

                print('Underground pipe storage capex: ${0:,.0f}'.format(storage_output['pipe_storage_capex']))
                print('Underground pipe storage opex: ${0:,.0f}/yr'.format(storage_output['pipe_storage_opex']))

                #Pipeline Model
                print("Distance to port: ",site_df['Approx. distance to port'])
                dist_to_port_value = site_df['Approx. distance to port']
                pipe_flow_rate = max(H2_Results['hydrogen_hourly_production'])
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
                in_dict['plant_life'] = useful_life
                in_dict['useful_life'] = useful_life
                in_dict['dist_to_h2_load_km'] = int(dist_to_port)
                in_dict['site_depth_m'] = int(site_depth)
                in_dict['steel_cost_ton'] = 900.0 # $ US/ton searching for seamless FBE X52 carbon steel > $500-$1000 per ton
                in_dict['pressure_bar'] = storage_input['compressor_output_pressure']
                
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
                print("Pipeline Length (km):", out_dict['total_pipeline_length_km'])
                print("Pipeline Design Pressure (bar):",in_dict['pressure_bar'])
                print("Pipeline Diameter: {} in, Thickness {} in".format(out_dict['design_diam_in'][0],out_dict['design_thic_in'][0]))
                print("Pipeline CapEx ($US): ", capex_pipeline)
                print("Pipeline Opex ($US/year)", opex_pipeline)
                print("Substation CapEx ($US): ", capex_substation)
                print("Total H2-Export CapEx:", total_h2export_system_cost)
                
                
                # create data
                if plot_hvdcpipe:
                    barx = ['HVDC', 'Pipeline']
                    cost_comparison_hvdc_pipeline = [total_export_system_cost, total_h2export_system_cost]
                    plt.figure(figsize=(9,6))
                    plt.bar(barx, cost_comparison_hvdc_pipeline)

                    plt.ylabel("$USD")
                    plt.legend(["Total CAPEX"])
                    plt.title("H2 Pipeline vs HVDC cost\n {}\n Model: ASME Pipeline".format(site_name))
                    plt.savefig(os.path.join(results_dir,'Pipeline Vs HVDC Cost_{}_{}_{}_{}'.format(site_name,atb_year,dist_to_port_value,turbine_model)))
                    #plt.show()

                # Step 6.5: Intermediate financial calculation

                total_elec_production = np.sum(electrical_generation_timeseries)
                total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
                total_electrolyzer_cost = electrolyzer_total_capital_cost
                total_desal_cost = desal_capex
                total_system_installed_cost_pipeline = total_hopp_installed_cost + total_electrolyzer_cost + total_desal_cost + total_h2export_system_cost
                total_system_installed_cost_hvdc = total_hopp_installed_cost + total_electrolyzer_cost + total_desal_cost + total_export_system_cost
                annual_operating_cost_wind = np.average(hybrid_plant.wind.om_total_expense)
                print("Wind OM: ", annual_operating_cost_wind)
                annual_operating_cost_h2 = electrolyzer_OM_cost
                annual_operating_cost_desal = desal_opex
                total_annual_operating_costs_pipeline =  annual_operating_cost_wind + annual_operating_cost_h2 + annual_operating_cost_desal + opex_pipeline + cost_to_buy_from_grid - profit_from_selling_to_grid
                total_annual_operating_costs_hvdc = annual_operating_cost_wind + annual_operating_cost_h2 + annual_operating_cost_desal + total_export_om_cost + cost_to_buy_from_grid - profit_from_selling_to_grid

                h_lcoe_no_op_cost_pipeline = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_pipeline,
                                    0, discount_rate, useful_life)
                h_lcoe_no_op_cost_hvdc = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_hvdc,
                                    0, discount_rate, useful_life)                                
                
                print(annual_energies)
                lcoe_test = lcoe_calc((annual_energies.wind/1000),total_hopp_installed_cost, annual_operating_cost_wind, discount_rate, useful_life)
                print('LCOE energy: ',lcoe_test, '$/MWh')

                #Requires capital costs and operating cost to be seperate just a check
                #****Only works when there is no policy options (capex in this calc is the same irregardless of ITC)
                h_lcoe_pipeline = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_pipeline,
                                    total_annual_operating_costs_pipeline, discount_rate, useful_life)
                h_lcoe_hvdc = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_hvdc,
                                    total_annual_operating_costs_hvdc, discount_rate, useful_life)                                    
                print('Pipeline H_LCOE no op cost', h_lcoe_no_op_cost_pipeline,'Pipeline H_LCOE w/op cost',h_lcoe_pipeline)
                print('HVDC H_LCOE no op cost', h_lcoe_no_op_cost_hvdc,'Pipeline H_LCOE w/op cost',h_lcoe_hvdc)


                # Cashflow Financial Calculation
                discount_rate = scenario['Discount Rate']

                # Create cashflows for pipeline and HVDC
                cf_pipeline_annuals = -simple_cash_annuals(useful_life,useful_life,total_h2export_system_cost,opex_pipeline,0.03)
                cf_hvdc_annuals = - simple_cash_annuals(useful_life,useful_life,total_export_system_cost,total_export_om_cost,0.03)

                #Implement ITC for pipeline and hvdc 
                pipeline_itc = (scenario['Wind ITC']/100) * total_h2export_system_cost
                cf_pipeline_itc = [0]*30
                cf_pipeline_itc[1] = pipeline_itc
                cf_pipeline_annuals = np.add(cf_pipeline_annuals,cf_pipeline_itc)

                hvdc_itc = (scenario['Wind ITC']/100) * total_export_system_cost
                cf_hvdc_itc = [0]*30
                cf_hvdc_itc[1] = hvdc_itc
                cf_hvdc_annuals = np.add(cf_hvdc_annuals,cf_hvdc_itc)
                # print("capex pipeline, hvdc:", total_h2export_system_cost,total_export_system_cost)
                # print('opex pipeline, hvdc', opex_pipeline,total_export_om_cost)
                # print('pipeline annuals:', cf_pipeline_annuals)
                # print('hvdc annauls', cf_hvdc_annuals)

                cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
                print('wind cf annuals',cf_wind_annuals)
                if solar_size_mw > 0:
                    cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
                else:
                    cf_solar_annuals = np.zeros(30)
                cf_desal_annuals = desal_annuals

                cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals, cf_desal_annuals],['Wind', 'Solar', 'H2', 'Desal'])

                cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}_{}MW.csv".format(site_name, scenario_choice, atb_year, discount_rate,turbine_rating_mw)))

                #Basic steps in calculating the LCOH
                #More nuanced calculation than h_lcoe b/c it uses yearly cashflows which change year to year rather than total capex and opex
                #file:///Applications/SAM_2021.12.02/SAM.app/Contents/runtime/help/html/index.html?mtf_lcoe.htm

                #Calculate total lifecycle cost for each technology (TLCC)
                tlcc_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
                print('npv wind: ',tlcc_wind_costs)
                tlcc_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
                tlcc_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
                print("NPV H2 Costs using {} model: {}".format(h2_model,tlcc_h2_costs))
                tlcc_desal_costs = -npf.npv(discount_rate, cf_desal_annuals)
                print("NPV desal: ", tlcc_desal_costs)
                tlcc_pipeline_costs = npf.npv(discount_rate, cf_pipeline_annuals)
                tlcc_hvdc_costs = npf.npv(discount_rate, cf_hvdc_annuals)

                tlcc_total_costs = tlcc_wind_costs+tlcc_solar_costs+tlcc_h2_costs + tlcc_desal_costs + tlcc_hvdc_costs
                tlcc_total_costs_pipeline = tlcc_wind_costs + tlcc_solar_costs + tlcc_h2_costs + tlcc_desal_costs + tlcc_pipeline_costs
                
                # Manipulate h2 production for LCOH calculation
                # Note. This equation makes it appear that the energy term in the denominator is discounted. 
                # That is a result of the algebraic solution of the equation, not an indication of the physical performance of the system.
                discounted_h2_production = npf.npv(discount_rate, [H2_Results['hydrogen_annual_output']]*30)
                print('discounted h2 production',discounted_h2_production)

                #Individual technology LCOH contribution
                LCOH_cf_method_wind = -tlcc_wind_costs / discounted_h2_production
                LCOH_cf_method_pipeline = -tlcc_pipeline_costs / discounted_h2_production
                LCOH_cf_method_hvdc = -tlcc_hvdc_costs / discounted_h2_production
                LCOH_cf_method_solar = -tlcc_solar_costs / discounted_h2_production
                LCOH_cf_method_h2_costs = -tlcc_h2_costs / discounted_h2_production
                LCOH_cf_method_desal_costs = -tlcc_desal_costs / discounted_h2_production
                
                # Total LCOH for pipeline and hvdc export scenarios
                LCOH_cf_method_total_hvdc = -tlcc_total_costs / discounted_h2_production
                LCOH_cf_method_total_pipeline = -tlcc_total_costs_pipeline / discounted_h2_production

                financial_summary_df = pd.DataFrame([policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw, electrolyzer_capex_kw,
                                                        scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],
                                                        discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_h2_costs, tlcc_desal_costs, tlcc_pipeline_costs, tlcc_hvdc_costs,
                                                        LCOH_cf_method_wind,LCOH_cf_method_h2_costs,LCOH_cf_method_desal_costs, LCOH_cf_method_pipeline, LCOH_cf_method_hvdc, 
                                                        LCOH_cf_method_total_hvdc, LCOH_cf_method_total_pipeline],
                                                    ['Policy Option','Turbine Model','Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'Electrolyzer Cost KW', 'Debt Equity',
                                                        'ATB Year', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses', 
                                                        'NPV Solar Expenses', 'NPV H2 Expenses','NPV Desal Expenses', 'NPV Pipeline Expenses','NPV HVDC Expenses',
                                                        'LCOH Wind HVDC', 'LCOH H2', 'LCOH Desal', 'LCOH Pipeline','LCOH HVDC',
                                                        'LCOH cf method HVDC','LCOH cf method Pipeline'])
                financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,option)))

                # Gut Check H2 calculation Pipeline (non-levelized)
                total_installed_and_operational_lifetime_cost_pipeline = total_system_installed_cost_pipeline + (30 * total_annual_operating_costs_pipeline)
                lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
                gut_check_h2_cost_kg_pipeline = total_installed_and_operational_lifetime_cost_pipeline / lifetime_h2_production
                
                total_installed_and_operational_lifetime_cost_hvdc = total_system_installed_cost_hvdc + (30 * total_annual_operating_costs_hvdc)
                lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
                gut_check_h2_cost_kg_hvdc = total_installed_and_operational_lifetime_cost_hvdc / lifetime_h2_production

                # Total amount of ITC [USD]
                wind_itc_total = hybrid_plant.wind._financial_model.Outputs.itc_total
                total_itc_pipeline = wind_itc_total + pipeline_itc + desal_itc + h2_itc
                total_itc_hvdc = wind_itc_total + hvdc_itc + desal_itc + h2_itc

                print("Gut Check H2 Cost Pipeline:",gut_check_h2_cost_kg_pipeline)
                print("Gut Check H2 Cost HVDC:",gut_check_h2_cost_kg_hvdc)
                print("HVDC Scenario: LCOH for H2, Desal, Grid Electrical Cost:", LCOH_cf_method_total_hvdc)
                
                print("Pipeline Scenario: LCOH for H2, Desal, Grid Electrical Cost:", LCOH_cf_method_total_pipeline)

                # Step 7: Plot Results
                
                # create data
                #x = ['HVDC', 'Pipeline']
                
                # plot bars in stack manner
                if plot_hvdcpipe_lcoh:
                    plt.figure(figsize=(9,6))
                    plt.bar(barx, [LCOH_cf_method_wind,LCOH_cf_method_wind], color='blue')
                    plt.bar(barx, [LCOH_cf_method_h2_costs,LCOH_cf_method_h2_costs], bottom=[LCOH_cf_method_wind,LCOH_cf_method_wind], color='orange')
                    plt.bar(barx, [LCOH_cf_method_desal_costs,LCOH_cf_method_desal_costs], bottom =[(LCOH_cf_method_wind + LCOH_cf_method_h2_costs), (LCOH_cf_method_wind + LCOH_cf_method_h2_costs)], color='g')
                    plt.bar(barx, [LCOH_cf_method_hvdc,LCOH_cf_method_pipeline], bottom =[(LCOH_cf_method_wind + LCOH_cf_method_h2_costs + LCOH_cf_method_desal_costs), (LCOH_cf_method_wind + LCOH_cf_method_h2_costs+LCOH_cf_method_desal_costs)], color='black')
                    
                    plt.ylabel("LCOH")
                    plt.legend(["Wind", "Electrolyzer", "Desalination","Export System"])
                    plt.title("Levelized Cost of hydrogen - Cost Contributors\n {}\n {}\n {} \n{}".format(site_name,atb_year,turbine_model,option))
                    plt.savefig(os.path.join(results_dir,'LCOH Barchart_{}_{}_{}_{}.jpg'.format(site_name,atb_year,turbine_model,option)),bbox_inches='tight')
                    # plt.show()

                print_results = False
                print_h2_results = True
                save_outputs_dict = save_the_things()
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
                    print("Total Yearly Hydrogen Production: {}".format(H2_Results['hydrogen_annual_output']))
                    #print("Levelized Cost H2/kg (new method - no operational costs)".format(h_lcoe_no_op_cost))
                    print("Capacity Factor of Electrolyzer: {}".format(H2_Results['cap_factor']))

                if print_h2_results:
                    print('Total Lifetime H2(kg) produced: {}'.format(lifetime_h2_production))
                    print("Gut-check H2 cost/kg: {}".format(gut_check_h2_cost_kg_pipeline))
                #     print("h_lcoe: ", h_lcoe)
                   # print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
                    #     H2_Results['feedstock_cost_h2_levelized_hopp']))
                    # print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(H2A_Results['Total Hydrogen Cost ($/kgH2)']))
                    # print("Total unit cost of H2 ($/kg) : {}".format(H2_Results['total_unit_cost_of_hydrogen']))
                    # print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(
                    #     H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp']))

                    #Step 9: Summarize Results
                    print('For a {}MW Offshore Wind Plant of turbine size {} with {}MW onshore electrolyzer \n located at {} \n (average wind speed {}m/s) in {} \n with a Wind CAPEX cost of {}$/kW,  and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} /kg '.
                                format(forced_wind_size,turbine_model,electrolyzer_size_mw,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_capex_kw,LCOH_cf_method_total_hvdc))
                    print('For a {}MW Offshore Wind Plant of turbine size {} with {}MW offshore electrolyzer \n located at {} \n (average wind speed {}m/s) in {} \n with a Wind CAPEX cost of {}$/kW,  and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} /kg '.
                                format(forced_wind_size,turbine_model,electrolyzer_size_mw,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_total_capital_cost,LCOH_cf_method_total_pipeline))
                   
save_outputs = True
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    save_all_runs_df.to_csv(os.path.join(results_dir, "H2_Analysis_OSW_All_test.csv"))


print('Done')

