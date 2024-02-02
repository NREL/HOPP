# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:31:28 2022

@author: ereznic2
"""
# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')
import numpy as np
import pandas as pd
# sys.path.insert(1,sys.path[0] + '/ProFAST-main/') #ESG
import ProFAST

from greenheart.to_organize.H2_Analysis import LCA_single_scenario_ProFAST

# sys.path.append('../ProFAST/')

pf = ProFAST.ProFAST()


def run_profast_for_hydrogen(hopp_dict,electrolyzer_size_mw,H2_Results,\
                            electrolyzer_system_capex_kw,user_defined_time_between_replacement,electrolyzer_energy_kWh_per_kg,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            capex_desal,opex_desal,plant_life,water_cost,wind_size_mw,solar_size_mw,storage_size_mw,renewable_plant_cost_info,wind_om_cost_kw,hybrid_plant,\
                            grid_connection_scenario, atb_year, site_name, policy_option, policy, energy_to_electrolyzer, combined_pv_wind_power_production_hopp,combined_pv_wind_curtailment_hopp,\
                            energy_shortfall_hopp, elec_price,grid_prices_interpolated_USDperkwh, grid_price_scenario,user_defined_stack_replacement_time,use_optimistic_pem_efficiency,wind_annual_energy_MWh,solar_annual_energy_MWh,solar_ITC): 
    # mwh_to_kwh = 0.001
    # plant_life=useful_life
    # electrolyzer_system_capex_kw = electrolyzer_capex_kw
    energy_used_per_year_kWhpryr = H2_Results['Performance Schedules']['Annual Energy Used [kWh/year]'].values
    elec_efficiency_per_yr_kWhprkg=H2_Results['Performance Schedules']['Annual Average Efficiency [kWh/kg]'].values
    # H2_Results['Performance Schedules']['Annual Average Efficiency [%-HHV]']
    h2prod_per_year_kgpryr = H2_Results['Performance Schedules']['Annual H2 Production [kg/year]'].values
    water_consumption_avg_kgH2O_prkgH2 = 10 #ESG, we should make this input 09/05
    electrolysis_plant_capacity_kgperday = 24*H2_Results['new_H2_Results']['Rated BOL: H2 Production [kg/hr]']
    # debt_equity_split = 57.983
    replace_stacks_individually = False #TODO, make input (changes refurbishment period)

    water_consumption_avg_galH2O_prkgH2 = water_consumption_avg_kgH2O_prkgH2/3.79

    # Design point electricity consumption
    if use_optimistic_pem_efficiency:
        elec_consumption_kWhprkg_design = electrolyzer_energy_kWh_per_kg
    else:
        elec_consumption_kWhprkg_design=H2_Results['new_H2_Results']['Rated BOL: Efficiency [kWh/kg]'] #new 09/05
        # elec_consumption_kWhprkg_design=H2_Results['Rated kWh/kg-H2'] #new 09/05

    # Calculate electrolyzer production capacity
    # electrolysis_plant_capacity_kgperday=   electrolyzer_size_mw/elec_consumption_kWhprkg_design*1000*24
    #electrolysis_plant_capacity_kgperday = electrolyzer_size_mw*electrolyzer_design_efficiency_HHV/h2_HHV*3600*24

    # Installed capital cost
    electrolyzer_installation_factor = 12/100  #[%] for stack cost

    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land_cost = 250000   #[$]

    stack_replacement_cost = 15/100  #[% of installed capital cost]

    # Calculate electrolyzer installation cost
    total_direct_electrolyzer_cost_kw = (electrolyzer_system_capex_kw * (1+electrolyzer_installation_factor)) \

    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000

    electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)

    compressor_capex_USDprkWe_of_electrolysis = 39

    # Calculate capital costs
    capex_electrolyzer_overnight = electrolyzer_total_installed_capex + electrolyzer_indirect_cost
    capex_storage_installed = hydrogen_storage_capacity_kg*hydrogen_storage_cost_USDprkg
    capex_compressor_installed = compressor_capex_USDprkWe_of_electrolysis*electrolyzer_size_mw*1000
    #capex_hybrid_installed = hybrid_plant.grid.total_installed_cost
    # capex_hybrid_installed = revised_renewable_cost

    # Fixed and variable costs
    fixed_OM = 12.8 #[$/kW-y]
    fixed_cost_electrolysis_total = fixed_OM*electrolyzer_size_mw*1000
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    variable_OM = 1.30  #[$/MWh]

    H2_PTC_duration = 10 # years the tax credit is active
    Ren_PTC_duration = 10 # years the tax credit is active

    electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid\
          = LCA_single_scenario_ProFAST.hydrogen_LCA_singlescenario_ProFAST(grid_connection_scenario,atb_year,site_name,[],[],H2_Results,[],solar_size_mw,storage_size_mw,hopp_dict,H2_PTC_duration)

    AEP_renewables = sum(hopp_dict.main_dict['Models']['grid']['ouput_dict']['energy_from_renewables'])#kWh/year
    grid_cost_keys = list(grid_prices_interpolated_USDperkwh.keys())
    grid_price_per_yr = np.array(list(grid_prices_interpolated_USDperkwh.values()))
    if grid_connection_scenario =='grid-only':
        # grid_electricity_useage_kWhpkg=np.mean(H2_Results['Performance Schedules']['Annual Average Efficiency [kWh/kg]'].values) #new 09/05
        ren_electricity_useage_kWhpkg = 0
        ren_frac = 0

        grid_cost_pr_yr_USDprkg = grid_price_per_yr*energy_used_per_year_kWhpryr/h2prod_per_year_kgpryr
        grid_prices_interpolated_USDperkg = dict(zip(grid_cost_keys,grid_cost_pr_yr_USDprkg))
        
    elif grid_connection_scenario == 'hybrid-grid':
        
        energy_from_grid_pr_yr = energy_used_per_year_kWhpryr - AEP_renewables
        # grid_electricity_useage_kWhpkg=np.mean(energy_from_grid_pr_yr/h2prod_per_year_kgpryr)
        ren_electricity_useage_kWhpkg = AEP_renewables/h2prod_per_year_kgpryr
        wind_electricity_useage_kWhpkg = wind_annual_energy_MWh*1000/h2prod_per_year_kgpryr
        ren_frac = AEP_renewables/energy_used_per_year_kWhpryr
        grid_cost_pr_yr_USDprkg = grid_price_per_yr*energy_from_grid_pr_yr/h2prod_per_year_kgpryr
        grid_prices_interpolated_USDperkg = dict(zip(grid_cost_keys,grid_cost_pr_yr_USDprkg))
        # grid_price_per_yr = np.array(list(grid_prices_interpolated_USDperkwh.values()))
        pass
    elif grid_connection_scenario == 'off-grid':
        # grid_electricity_useage_kWhpkg=0
        ren_frac = 1
        ren_electricity_useage_kWhpkg =AEP_renewables/h2prod_per_year_kgpryr
        wind_electricity_useage_kWhpkg = wind_annual_energy_MWh*1000/h2prod_per_year_kgpryr
        grid_prices_interpolated_USDperkg = dict(zip(grid_cost_keys,np.zeros(len(grid_cost_keys))))
    #new changes end here


    # Establish Ren PTC and assign total emission intensity
    start_year = atb_year + 5
    endofincentives_year = start_year + H2_PTC_duration
    Ren_PTC = {}
    y_idx = 0
    electrolysis_total_EI_policy = {}
    for year in range(start_year,endofincentives_year):
        if atb_year < 2035:
            if grid_connection_scenario == 'grid-only':
                Ren_PTC[year] = 0
                electrolysis_total_EI_policy[year] = electrolysis_total_EI_policy_grid[year]
            elif grid_connection_scenario == 'off-grid':
                electrolysis_total_EI_policy[year] = electrolysis_total_EI_policy_offgrid[year]
                if policy_option == 'no-policy':
                    Ren_PTC[year] = 0
                elif policy_option == 'base':
                    if solar_ITC == True:
                        Ren_PTC[year] = policy['Wind PTC'] * wind_electricity_useage_kWhpkg[y_idx]
                    else:
                        Ren_PTC[year] = policy['Wind PTC'] * ren_electricity_useage_kWhpkg[y_idx]#np.sum(energy_to_electrolyzer)/ (H2_Results['hydrogen_annual_output'])
                elif policy_option == 'max':
                    if solar_ITC == True:
                        Ren_PTC[year] = policy['Wind PTC'] * wind_electricity_useage_kWhpkg[y_idx]
                    else:
                        Ren_PTC[year] = policy['Wind PTC'] * ren_electricity_useage_kWhpkg[y_idx]#np.sum(energy_to_electrolyzer)/ (H2_Results['hydrogen_annual_output'])
            elif grid_connection_scenario == 'hybrid-grid':
                electrolysis_total_EI_policy[year] = 0 # Basically this is not used for hybrid-grid
                if policy_option == 'no-policy':
                    Ren_PTC[year] = 0
                elif policy_option == 'base':
                    if solar_ITC == True:
                        Ren_PTC[year] = policy['Wind PTC'] * wind_electricity_useage_kWhpkg[y_idx]
                    else:
                        Ren_PTC[year] = policy['Wind PTC']  * ren_electricity_useage_kWhpkg[y_idx]#energy_from_renewables / (H2_Results['hydrogen_annual_output'])
                    #Ren_PTC = 0.0051  * np.sum(energy_to_electrolyzer)/ (H2_Results['hydrogen_annual_output']) # We will need to fix this by introducing ren_frac multiplier to denominator when HOPP changes to dealing with grid cases are changed
                elif policy_option == 'max':
                    if solar_ITC == True:
                        Ren_PTC[year] = policy['Wind PTC'] * wind_electricity_useage_kWhpkg[y_idx]
                    else:
                        Ren_PTC[year] = policy['Wind PTC'] * ren_electricity_useage_kWhpkg[y_idx]#energy_from_renewables/ (H2_Results['hydrogen_annual_output'])
                    # Ren_PTC = 0.03072 * np.sum(energy_to_electrolyzer)/ (H2_Results['hydrogen_annual_output']) # We will need to fix this by introducing ren_frac multiplier to denominator when HOPP changes to dealing with grid cases are changed
        elif atb_year == 2035:
                Ren_PTC[year]=0
        y_idx +=1
    # add in electrolzyer replacement schedule
    electrolyzer_refurbishment_schedule = np.zeros(plant_life)
    if user_defined_stack_replacement_time:
        refturb_period = round(user_defined_time_between_replacement/(24*365))
        electrolyzer_refurbishment_schedule[refturb_period:plant_life:refturb_period]=stack_replacement_cost
    else:
        refturb_period = int(np.floor(H2_Results['new_H2_Results']['Time Until Replacement [hrs]']/(24*365)))
        #int(np.floor(H2_Results['avg_time_between_replacement']/(24*365)))
        if replace_stacks_individually:
            percent_of_capacity_replaced = H2_Results['Performance Schedules']['Refurbishment Schedule [MW replaced/year]'].values/electrolyzer_size_mw
            electrolyzer_refurbishment_schedule = percent_of_capacity_replaced*stack_replacement_cost
        else:
            electrolyzer_refurbishment_schedule[refturb_period:plant_life:refturb_period]=stack_replacement_cost
    
    # electrolyzer_refurbishment_schedule[refturb_period:plant_life:refturb_period]=stack_replacement_cost

    # total_variable_OM_perkg = variable_OM*elec_avg_consumption_kWhprkg/1000 #09/05
    annual_variable_OM_perkg = (variable_OM/1000)*elec_efficiency_per_yr_kWhprkg #09/05

    if grid_connection_scenario != 'grid-only':
        wind_om_cost_kw =  renewable_plant_cost_info['wind']['o&m_per_kw']
        fixed_cost_wind = wind_om_cost_kw*wind_size_mw*1000
        capex_wind_installed_init = renewable_plant_cost_info['wind']['capex_per_kw'] * wind_size_mw*1000
        wind_cost_adj = [val for val in renewable_plant_cost_info['wind_savings_dollars'].values()]
        wind_revised_cost=np.sum(wind_cost_adj)

        solar_om_cost_kw = renewable_plant_cost_info['pv']['o&m_per_kw']
        fixed_cost_solar = solar_om_cost_kw*solar_size_mw*1000
        capex_solar_installed = renewable_plant_cost_info['pv']['capex_per_kw'] * solar_size_mw*1000
        battery_hrs=renewable_plant_cost_info['battery']['storage_hours']
        battery_capex_per_kw= renewable_plant_cost_info['battery']['capex_per_kwh']*battery_hrs +  renewable_plant_cost_info['battery']['capex_per_kw']
        capex_battery_installed = battery_capex_per_kw * renewable_plant_cost_info['battery']['size_mw']*1000
        fixed_cost_battery = renewable_plant_cost_info['battery']['o&m_percent'] * capex_battery_installed

    else:
        capex_wind_installed_init=0
        wind_revised_cost = 0
        wind_om_cost_kW = 0
        fixed_cost_solar=0
        capex_solar_installed=0
        capex_battery_installed=0
        fixed_cost_battery=0

    capex_wind_installed=capex_wind_installed_init+wind_revised_cost


    #Calculate H2 and combined PTC
    cambium_year = atb_year + 5
    endofincentives_year = cambium_year + H2_PTC_duration
    H2_PTC = {}
    y_idx = 0
    for year in range(cambium_year,endofincentives_year):
        
        if atb_year < 2035:
        
            if grid_connection_scenario == 'grid-only' or grid_connection_scenario == 'off-grid':

                if policy_option == 'no-policy':
                    ITC = 0
                    H2_PTC[year] = 0 # $/kg H2
                    Ren_PTC[year] = 0 # $/kWh

                elif policy_option == 'max':

                    ITC = 0.5

                    if electrolysis_total_EI_policy[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC[year] = 3 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 0.45 and electrolysis_total_EI_policy[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC[year] = 1 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 1.5 and electrolysis_total_EI_policy[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC[year] = 0.75 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 2.5 and electrolysis_total_EI_policy[year] <= 4: # kg CO2e/kg H2
                        H2_PTC[year] = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 4:
                        H2_PTC[year] = 0

                elif policy_option == 'base':

                    ITC = 0.06

                    if electrolysis_total_EI_policy[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC[year] = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 0.45 and electrolysis_total_EI_policy[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC[year] = 0.2 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 1.5 and electrolysis_total_EI_policy[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC[year] = 0.15 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 2.5 and electrolysis_total_EI_policy[year] <= 4: # kg CO2e/kg H2
                        H2_PTC[year] = 0.12 # $/kg H2
                    elif electrolysis_total_EI_policy[year] > 4:
                        H2_PTC[year] = 0


            if grid_connection_scenario == 'hybrid-grid':

                if policy_option == 'no-policy':
                    H2_PTC_grid = 0
                    H2_PTC_offgrid = 0
                    ITC = 0.0

                elif policy_option == 'max':

                    if electrolysis_total_EI_policy_grid[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC_grid = 3 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 0.45 and electrolysis_total_EI_policy_grid[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC_grid = 1 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 1.5 and electrolysis_total_EI_policy_grid[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC_grid = 0.75 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 2.5 and electrolysis_total_EI_policy_grid[year] <= 4: # kg CO2e/kg H2
                        H2_PTC_grid = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 4:
                        H2_PTC_grid = 0

                    if electrolysis_total_EI_policy_offgrid[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC_offgrid = 3 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 0.45 and electrolysis_total_EI_policy_offgrid[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC_offgrid = 1 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 1.5 and electrolysis_total_EI_policy_offgrid[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.75 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 2.5 and electrolysis_total_EI_policy_offgrid[year] <= 4: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 4:
                        H2_PTC_offgrid = 0

                    ITC = 0.5

                elif policy_option == 'base':

                    if electrolysis_total_EI_policy_grid[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC_grid = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 0.45 and electrolysis_total_EI_policy_grid[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC_grid = 0.2 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 1.5 and electrolysis_total_EI_policy_grid[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC_grid = 0.15 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 2.5 and electrolysis_total_EI_policy_grid[year] <= 4: # kg CO2e/kg H2
                        H2_PTC_grid = 0.12 # $/kg H2
                    elif electrolysis_total_EI_policy_grid[year] > 4:
                        H2_PTC_grid = 0


                    if electrolysis_total_EI_policy_offgrid[year] <= 0.45: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.6 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 0.45 and electrolysis_total_EI_policy_offgrid[year] <= 1.5: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.2 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 1.5 and electrolysis_total_EI_policy_offgrid[year] <= 2.5: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.15 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 2.5 and electrolysis_total_EI_policy_offgrid[year] <= 4: # kg CO2e/kg H2
                        H2_PTC_offgrid = 0.12 # $/kg H2
                    elif electrolysis_total_EI_policy_offgrid[year] > 4:
                        H2_PTC_offgrid = 0
                    
                    ITC = 0.06

                #H2_PTC =  ren_frac * H2_PTC_offgrid + (elec_cf - ren_frac) * H2_PTC_grid
                H2_PTC[year] =  ren_frac[y_idx] * H2_PTC_offgrid + (1 - ren_frac[y_idx]) * H2_PTC_grid
            #combined_PTC[year] = H2_PTC[year]+Ren_PTC[year]
        elif atb_year == 2035:
            H2_PTC[year]=0
        y_idx +=1
    # Reassign PTC values to zero for atb year 2035
    if atb_year == 2035: # need to clarify with Matt when exactly the H2 PTC would end
        #H2_PTC = 0
        #Ren_PTC = 0
        ITC = 0.0
    # if grid_price_scenario == 'retail-flat':
    #     elec_price_perkWh = mwh_to_kwh*elec_price # convert $/MWh to $/kWh
    # Set up ProFAST

    financial_assumptions = pd.read_csv('H2_Analysis/financial_inputs.csv',index_col=None,header=0)
    financial_assumptions.set_index(["Parameter"], inplace = True)

    fraction_capex_vre = (capex_wind_installed + capex_solar_installed + capex_battery_installed)\
                        /(capex_wind_installed + capex_solar_installed + capex_battery_installed + capex_electrolyzer_overnight + capex_desal + capex_compressor_installed + capex_storage_installed)

    fraction_debt_financing_vre = 1/(1+1/financial_assumptions.loc['debt equity ratio of initial financing','Wind/Solar/Bat'])
    fraction_debt_financing_h2 = 1/(1+1/financial_assumptions.loc['debt equity ratio of initial financing','Hydrogen/Steel/Ammonia'])
    fraction_equity_financing_vre = 1 - fraction_debt_financing_vre
    fraction_equity_financing_h2 = 1 - fraction_debt_financing_h2

    real_roe_vre = financial_assumptions.loc['leverage after tax nominal discount rate','Wind/Solar/Bat']
    real_roe_h2 = financial_assumptions.loc['leverage after tax nominal discount rate','Hydrogen/Steel/Ammonia']

    real_interest_vre = financial_assumptions.loc['debt interest rate','Wind/Solar/Bat']
    real_interest_h2 = financial_assumptions.loc['debt interest rate','Hydrogen/Steel/Ammonia']

    real_roe_combined = (fraction_equity_financing_vre*fraction_capex_vre*real_roe_vre + fraction_equity_financing_h2*(1-fraction_capex_vre)*real_roe_h2)\
                        /(fraction_equity_financing_vre*fraction_capex_vre + fraction_equity_financing_h2*(1-fraction_capex_vre))

    real_interest_combined = (real_interest_vre*fraction_capex_vre*fraction_debt_financing_vre + real_interest_h2*(1-fraction_capex_vre)*fraction_debt_financing_h2)\
                             /(fraction_capex_vre*fraction_debt_financing_vre + (1-fraction_capex_vre)*fraction_debt_financing_h2)

    debt_equity_ratio_combined = financial_assumptions.loc['debt equity ratio of initial financing','Wind/Solar/Bat']*fraction_capex_vre + financial_assumptions.loc['debt equity ratio of initial financing','Hydrogen/Steel/Ammonia']*(1-fraction_capex_vre)

    total_income_tax_rate_combined = financial_assumptions.loc['total income tax rate','Wind/Solar/Bat']*fraction_capex_vre + financial_assumptions.loc['total income tax rate','Hydrogen/Steel/Ammonia']*(1-fraction_capex_vre)
    capitalgains_tax_rate_combined = financial_assumptions.loc['capital gains tax rate','Wind/Solar/Bat']*fraction_capex_vre + financial_assumptions.loc['capital gains tax rate','Hydrogen/Steel/Ammonia']*(1-fraction_capex_vre)
    gen_inflation = 0.00

    nominal_roe_combined = (real_roe_combined+1)*(1+gen_inflation)-1
    nominal_interest_combined = (real_interest_combined+1)*(1+gen_inflation)-1
   # total_income_tax_rate = 

    pf = ProFAST.ProFAST()

    # Fill these in - can have most of them as 0 also
    
    install_years = 3
    analysis_start = atb_year + 5 - install_years
    #09/05 (5 lines below)
    operation_start = analysis_start + install_years
    operational_years = np.arange(operation_start,operation_start+plant_life,1)
    year_keys = ['{}'.format(y) for y in operational_years]
    cf_per_year_vals = H2_Results['Performance Schedules']['Capacity Factor [-]'].values #new 09/05
    elec_cf_per_year_PF = dict(zip(year_keys,cf_per_year_vals))
    life_average_cf = H2_Results['Performance Schedules']['Capacity Factor [-]'].mean()
    total_variable_OM_perkg = dict(zip(year_keys,annual_variable_OM_perkg))
    # elec_cf = np.mean(cf_per_year_vals)

    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',analysis_start)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',install_years*12)
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    # pf.set_params('long term utilization',elec_cf)
    pf.set_params('long term utilization',elec_cf_per_year_PF) #09/05
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0)
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance',property_tax_insurance)
    pf.set_params('admin expense',0)
    pf.set_params('total income tax rate',total_income_tax_rate_combined)
    pf.set_params('capital gains tax rate',capitalgains_tax_rate_combined)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',nominal_roe_combined)
    pf.set_params('debt equity ratio of initial financing',debt_equity_ratio_combined)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',nominal_interest_combined)
    pf.set_params('cash onhand',1)
    if solar_ITC == True:
        pf.set_params('one time cap inct',{'value':ITC*(capex_storage_installed+capex_battery_installed+capex_solar_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    else:
        pf.set_params('one time cap inct',{'value':ITC*(capex_storage_installed+capex_battery_installed),'depr type':'MACRS','depr period':7,'depreciable':True})
    #pf.set_params('one time cap inct',{'value':ITC*capex_solar_installed,'depr type':'MACRS','depr period':7,'depreciable':True})
    #pf.set_params('one time cap inct',{'value':ITC*capex_battery_installed,'depr type':'MACRS','depr period':7,'depreciable':True})

    #----------------------------------- Add capital items to ProFAST ----------------
    #pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=7,refurb=list(electrolyzer_refurbishment_schedule))
    pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=7,refurb=[0])

    if grid_connection_scenario == 'grid-only':
        pf.add_capital_item(name = "Wind Plant",cost = 0,depr_type = "MACRS",depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Solar Plant",cost = 0,depr_type = "MACRS",depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Battery Storage",cost = 0,depr_type = "MACRS",depr_period = 7,refurb = [0])
    else:
        pf.add_capital_item(name = "Wind Plant",cost = capex_wind_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Solar Plant",cost = capex_solar_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
        pf.add_capital_item(name = "Battery Storage",cost = capex_battery_installed,depr_type = "MACRS",depr_period = 7,refurb = [0])
        # pf.add_capital_item(name = "Renewable Plant",cost = capex_hybrid_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    #replacement_capex = np.sum(electrolyzer_total_installed_capex*electrolyzer_refurbishment_schedule)
    #NOTE TOTAL CAPEX DOES NOT REFLECT STACK REPLACEMENT COSTS AND HOW THEY CHANGE OVER TIME!
    total_capex = capex_electrolyzer_overnight+capex_compressor_installed+capex_storage_installed+capex_desal+capex_wind_installed+capex_solar_installed + capex_battery_installed #+ replacement_capex#capex_hybrid_installed
    # total_capex = capex_electrolyzer_overnight+capex_compressor_installed+capex_storage_installed+capex_desal+ capex_hybrid_installed
    # capex_fraction = {'Electrolyzer':capex_electrolyzer_overnight/total_capex,
    #                   'Compression':capex_compressor_installed/total_capex,
    #                   'Hydrogen Storage':capex_storage_installed/total_capex,
    #                   'Desalination':capex_desal/total_capex,
    #                   'Wind Plant':capex_wind_installed/total_capex,
    #                   'Solar Plant':capex_solar_installed/total_capex,
    #                   'Battery Storage':capex_battery_installed/total_capex,
    #                   #'Stack Replacement': replacement_capex/total_capex
    #                   #'Renewable Plant':capex_hybrid_installed/total_capex
    #                   }

    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_electrolysis_total,escalation=gen_inflation)
    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)
    #pf.add_fixed_cost(name="Renewable Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_renewables,escalation=gen_inflation)

    if grid_connection_scenario == 'grid-only':
        # pf.add_fixed_cost(name="Renewable Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=0,escalation=gen_inflation)
        pf.add_fixed_cost(name="Wind Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=0,escalation=gen_inflation)
        pf.add_fixed_cost(name="Solar Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=0,escalation=gen_inflation)
        pf.add_fixed_cost(name="Battery Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=0,escalation=gen_inflation)
    else:
        # pf.add_fixed_cost(name="Renewable Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_renewables,escalation=gen_inflation)
        pf.add_fixed_cost(name="Wind Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_wind,escalation=gen_inflation)
        pf.add_fixed_cost(name="Solar Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_solar,escalation=gen_inflation)
        pf.add_fixed_cost(name="Battery Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_battery,escalation=gen_inflation)

    
    #---------------------- Add feedstocks, note the various cost options-------------------
    #pf.add_feedstock(name='Electricity',usage=elec_avg_consumption_kWhprkg,unit='kWh',cost=lcoe/100,escalation=gen_inflation)
    pf.add_feedstock(name='Water',usage=water_consumption_avg_galH2O_prkgH2,unit='gallon-water',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=gen_inflation)

    #TODO: include changes in energy required/year for grid usage
    # pf.add_feedstock(name='Grid Electricity Cost',usage=grid_electricity_useage_kWhpkg,unit='$/kWh',cost=grid_prices_interpolated_USDperkwh,escalation=gen_inflation)
    pf.add_feedstock(name='Grid Electricity Cost',usage=1.0,unit='$/kg',cost=grid_prices_interpolated_USDperkg,escalation=gen_inflation)
    
    # pf.add_feedstock(name='Grid Electricity Cost',usage=min_grid_usage,unit='$/kWh',cost=grid_prices_interpolated_USDperkwh,escalation=gen_inflation)
    #---------------------- Add various tax credit incentives -------------------
    pf.add_incentive(name ='Renewable PTC credit', value=Ren_PTC, decay = 0, sunset_years = Ren_PTC_duration, tax_credit = True)
    pf.add_incentive(name ='Hydrogen PTC credit', value=H2_PTC, decay = 0, sunset_years = H2_PTC_duration, tax_credit = True)

    sol = pf.solve_price()

    summary = pf.summary_vals

    price_breakdown = pf.get_cost_breakdown()

    # Calculate contribution of equipment to breakeven price
    total_price_capex = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]\
                      + price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]\

    capex_fraction = {'Electrolyzer':price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0]/total_price_capex,
                  'Compression':price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]/total_price_capex,
                  'Hydrogen Storage':price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]/total_price_capex,
                  'Desalination':price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]/total_price_capex,
                  'Wind Plant':price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0]/total_price_capex,
                  'Solar Plant':price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0]/total_price_capex,
                  'Battery Storage':price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0]/total_price_capex}

    # Calculate financial expense associated with equipment
    cap_expense = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]

    # Calculate remaining financial expenses
    remaining_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]

    # Calculate LCOH breakdown and assign capital expense to equipment costs
    price_breakdown_electrolyzer = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0] + cap_expense*capex_fraction['Electrolyzer']
    price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
    price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']
    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0] + cap_expense*capex_fraction['Desalination']
    # price_breakdown_renewables = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Renewable Plant']
    price_breakdown_wind = price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Wind Plant']
    price_breakdown_solar = price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Solar Plant']
    price_breakdown_battery = price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0] + cap_expense*capex_fraction['Battery Storage']


    price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
    # price_breakdown_renewables_FOM = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_wind_FOM = price_breakdown.loc[price_breakdown['Name']=='Wind Plant Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_solar_FOM = price_breakdown.loc[price_breakdown['Name']=='Solar Plant Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_battery_FOM = price_breakdown.loc[price_breakdown['Name']=='Battery Storage Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\

    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

    price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]

    price_breakdown_grid_elec_price = price_breakdown.loc[price_breakdown['Name']=='Grid Electricity Cost','NPV'].tolist()[0]

    # price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]
    price_breakdown_renewables=price_breakdown_wind + price_breakdown_solar +price_breakdown_battery
    price_breakdown_renewables_FOM = price_breakdown_wind_FOM + price_breakdown_solar_FOM + price_breakdown_battery_FOM
    lcoh_check = price_breakdown_electrolyzer+price_breakdown_compression+price_breakdown_storage+price_breakdown_electrolysis_FOM\
        + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
            +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_taxes+price_breakdown_water+price_breakdown_grid_elec_price+remaining_financial\
                #+ price_breakdown_stack_replacement

    lcoh_breakdown = {'LCOH: Compression & storage ($/kg)':price_breakdown_storage+price_breakdown_compression,\
                      'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                      'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                      'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,\
                      'LCOH: Wind Plant ($/kg)':price_breakdown_wind,'LCOH: Wind Plant FOM ($/kg)':price_breakdown_wind_FOM,\
                      'LCOH: Solar Plant ($/kg)':price_breakdown_solar,'LCOH: Solar Plant FOM ($/kg)':price_breakdown_solar_FOM,\
                      'LCOH: Battery Storage ($/kg)':price_breakdown_battery,'LCOH: Battery Storage FOM ($/kg)':price_breakdown_battery_FOM,\
                      #'LCOH: Renewable plant ($/kg)':price_breakdown_renewables,'LCOH: Renewable FOM ($/kg)':price_breakdown_renewables_FOM,
                      'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                      'LCOH: Water consumption ($/kg)':price_breakdown_water,'LCOH: Grid electricity ($/kg)':price_breakdown_grid_elec_price,\
                      'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check,'LCOH Profast:':sol['price']}
    print('{} {} {} has LCOH = ${} /kg'.format(site_name,policy_option,grid_connection_scenario,sol['price']))
    price_breakdown = price_breakdown.drop(columns=['index','Amount'])
   
    # return(sol,summary,price_breakdown,lcoh_breakdown,capex_electrolyzer_overnight/electrolyzer_size_mw/1000,elec_cf,ren_frac,electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid,H2_PTC,Ren_PTC,total_capex)
    return(sol,summary,price_breakdown,lcoh_breakdown,capex_electrolyzer_overnight/electrolyzer_size_mw/1000,life_average_cf,ren_frac,electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid,H2_PTC,Ren_PTC,total_capex)