# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:29:30 2022

@author: mkoleva
"""

# Specify file path to PyFAST
import sys
import os
import glob
sys.path.insert(1,'../PyFAST/')
import pandas as pd
sys.path.append('../PyFAST/')
import run_pyfast_for_hydrogen_SMR
sys.path.append('')
import warnings
warnings.filterwarnings("ignore")
import hopp_tools_steel


dir1 = os.getcwd()
dirin_el_prices = 'examples/H2_Analysis/'
el_prices_files = glob.glob(os.path.join(dir1 + dirin_el_prices, 'annual_average_retail_prices.csv'))

lcoh_check_all = []
lcoh = []
years_all = []
sites_all = []
policy_all = []
NG_price_all = []
scenario = dict()
scenario['Wind PTC'] = 0.0
SMR_LCOH_dic = {'Year':[], 'Location':[], 'Policy': [], 'NG price case': [], 'LCOH':[], 'LCOA':[], 'LCOS':[]}

atb_years = [2020,2025,2030,2035]
    
site_locations = ["IN","TX","IA","MS"]

policy_cases = ['no', 'base', 'max']

NG_price_cases = ['default']

policy_option = {
    'no policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0, 'Storage ITC': 0},
    #'base': {'Wind ITC': 0, 'Wind PTC': 0.006, "H2 PTC": 0.6, 'Storage ITC': 6},
    #'max': {'Wind ITC': 0, 'Wind PTC': 0.029, "H2 PTC": 3.0, 'Storage ITC': 50},
    # 'max on grid hybrid': {'Wind ITC': 0, 'Wind PTC': 0.006, "H2 PTC": 0.60, 'Storage ITC': 6},
    # 'max on grid hybrid': {'Wind ITC': 0, 'Wind PTC': 0.029, "H2 PTC": 0.60, 'Storage ITC': 50},
    # 'option 3': {'Wind ITC': 6, 'Wind PTC': 0, "H2 PTC": 0.6},
    # 'option 4': {'Wind ITC': 30, 'Wind PTC': 0, "H2 PTC": 3},
    # 'option 5': {'Wind ITC': 50, 'Wind PTC': 0, "H2 PTC": 3},
}


site_df = pd.read_excel('examples/H2_Analysis/green_steel_site_renewable_costs_ATB.xlsx')
site_df = pd.DataFrame(site_df, columns =['Parameter', 'Site 1', 'Site 2', 'Site 3', 'Site 4', 'Site 5'])
site_df.set_index('Parameter', inplace=True)


for atb_year in atb_years:
    for i, site_location in enumerate(site_locations):
        for NG_price_case in NG_price_cases:
            for policy_case in policy_cases:

                hydrogen_annual_production, hydrogen_storage_duration_hr, lcoh, lcoh_breakdown, lcoe, plant_life, NG_cost = \
                    run_pyfast_for_hydrogen_SMR.run_pyfast_for_hydrogen_SMR(atb_year,site_location,policy_case,NG_price_case)

#                lcoh_breakdown = pd.DataFrame(lcoh_breakdown)
#                lcoh_breakdown = lcoh_breakdown.append
#                print(lcoh_breakdown)
                SMR_LCOH_dic['Year'].append(atb_year)
                SMR_LCOH_dic['Location'].append(site_location)
                SMR_LCOH_dic['Policy'].append(policy_case)
                SMR_LCOH_dic['NG price case'].append(NG_price_case)
                SMR_LCOH_dic['LCOH'].append(lcoh)

#                hopp_dict = hoppDict(save_model_input_yaml, save_model_output_yaml)
#                hopp_dict, site_df, sample_site = hopp_tools_steel.set_site_info(hopp_dict, site_df, sample_site)

                lime_unit_cost = site_df.loc['Lime ($/metric tonne)'] + site_df.loc['Lime Transport ($/metric tonne)']
                carbon_unit_cost = site_df.loc['Carbon ($/metric tonne)'] + site_df.loc['Carbon Transport ($/metric tonne)']
                iron_ore_pellets_unit_cost = site_df.loc['Iron Ore Pellets ($/metric tonne)'] + site_df.loc['Iron Ore Pellets Transport ($/metric tonne)']
                
                tmp, steel_economics_from_pyfast, steel_economics_summary, steel_breakeven_price, steel_annual_production_mtpy,steel_price_breakdown = \
                                                                               hopp_tools_steel.steel_LCOS_SMR([],lcoh,hydrogen_annual_production,
                                                                                                                        lime_unit_cost[i],
                                                                                                                        carbon_unit_cost[i],
                                                                                                                        iron_ore_pellets_unit_cost[i],
                                                                                                                        lcoe, scenario, NG_cost)
                
                cooling_water_cost = 0.000113349938601175 # $/Gal
                iron_based_catalyst_cost = 23.19977341 # $/kg
                oxygen_cost = 0.0285210891617726       # $/kg 
                tmp, ammonia_economics_from_pyfast, ammonia_economics_summary, ammonia_breakeven_price, ammonia_annual_production_kgpy,ammonia_price_breakdown = hopp_tools_steel.levelized_cost_of_ammonia_SMR([],lcoh,hydrogen_annual_production,
                                                                                                                        cooling_water_cost,
                                                                                                                        iron_based_catalyst_cost,
                                                                                                                        oxygen_cost, 
                                                                                                                        lcoe, scenario)
                
                # print(' ========================================================= ')
                # print('Plant life = ', plant_life)
                # print('ATB year = ', atb_year)
                # print('Policy_option = ', policy_option)
                # print('Policy case = ', policy_case)
                # print('Scenario = ', scenario)
                # print('Site location = ', site_location)   
                # print('LCOE = ', lcoe)                                    
                # print('Hydrogen storage duration = ', hydrogen_storage_duration_hr)
                # print('hydrogen_annual_production = ', hydrogen_annual_production)
                # print('LCOH = ', lcoh)
                # print('LCOH breakdown = ', lcoh_breakdown)
                # print('steel_annual_production_mtpy = ', steel_annual_production_mtpy)
                # print('steel_breakeven_price = ', steel_breakeven_price)
                # print('steel_price_breakdown = ', steel_price_breakdown)
                # print('ammonia_annual_production_kgpy = ', ammonia_annual_production_kgpy)
                # print('ammonia_breakeven_price = ', ammonia_breakeven_price)
                # print('ammonia_price_breakdown = ', ammonia_price_breakdown) 
                
                SMR_LCOH_dic['LCOA'].append(ammonia_breakeven_price)
                SMR_LCOH_dic['LCOS'].append(steel_breakeven_price)

SMR_LCOH_dic = pd.DataFrame(SMR_LCOH_dic)

