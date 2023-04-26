# -*- coding: utf-8 -*-

# Specify file path to PyFAST
import sys
import os
import glob
#sys.path.insert(1,'../PyFAST/')
import pandas as pd

#sys.path.append('../PyFAST/')
#import src.PyFAST as PyFAST

import ProFAST

dir1 = os.getcwd()
dirin_el_prices = '\\examples\\H2_Analysis\\'
el_prices_files = glob.glob(os.path.join(dir1 + dirin_el_prices, 'annual_average_retail_prices.csv'))
dircambium = 'Examples/H2_Analysis/Cambium_data/StdScen21_MidCase95by2035_hourly_' 

def run_profast_for_hydrogen_SMR(atb_year,site_name,policy_case,NG_price_case,CCS_option):

    # Toggles
    #------------------------------------------------------------------------------
    # policy_case = 'no'
    # #policy_case = ['no', 'base', 'max']
    # CO2_credit = 0
    # atb_years = 2020 
    # #[2020,2025,2030,2035,2040]
    # site_name = "IA"
    # #["IN","TX","IA","MS"]
    # NG_price_case = 'default'
    # #['default','min','max']
    NG_cost = 0.00536 # $2019/MJ 
    
    # Conversions
    #------------------------------------------------------------------------------
    mt_tokg_conv = 1000 # Metric tonne to kg conversion 
    hrs_in_year = 8760
    kg_to_MT_conv = 0.001 # Converion from kg to metric tonnes
    g_to_kg_conv  = 0.001  # Conversion from grams to kilograms
    kWh_to_MWh_conv = 0.001 # Conversion from kWh to MWh
    
    #------------------------------------------------------------------------------
    # Steam methane reforming (SMR) - Incumbent H2 production process
    #------------------------------------------------------------------------------

    smr_NG_combust = 56.2 # Natural gas combustion (g CO2e/MJ)
    smr_NG_consume = 167  # Natural gas consumption (MJ/kg H2)
    smr_PO_consume = 0    # Power consumption in SMR plant (kWh/kg H2)
    smr_steam_prod = 17.4 # Steam production on SMR site (MJ/kg H2)
    smr_HEX_eff    = 0.9  # Heat exchanger efficiency (-)
    smr_NG_supply  = 9    # Natural gas extraction and supply to SMR plant assuming 2% CH4 leakage rate (g CO2e/MJ)
    ccs_PO_consume = 0.2  # Power consumption for CCS (kWh/kg CO2)
    ccs_perc_capture = 0.95 # Carbon capture rate (-)
    
    # Data
    #------------------------------------------------------------------------------  
    plant_life = 30
    hydrogen_storage_cost_USDprkg = 540
    land_cost = 0 # $/acre
    water_cost = 0 # $/gal H2O
    CO2_credit = 0 # $/ton CO2
    #capex_desal = 
    #opex_desal = 
    capacity_factor = 0.9  
    h2_plant_capacity_kgpd = 189003 # kg H2/day
    h2_plant_capacity_kgpy = 68986000 # kg H2/yr
    hydrogen_production_kgpd = h2_plant_capacity_kgpd * capacity_factor # kg H2/day; The number is based on annual demand of 1 MMT steel; 
    hydrogen_production_kgpy = h2_plant_capacity_kgpy * capacity_factor # kg H2/year
    fom_SMR_perc = 0.03 # fraction of capital cost
    electricity_cost = 0.076 # $/kWh; If electricity prices file missing, this is the cost which will be taken
    hydrogen_storage_duration = 4 # hours, which have been chosen based on RODeO runs with grid connection
    lhv_h2 = 33 # kWh/kg H2
    water_consumption = 10 # gal H2O/kg H2 - for feedstock and process water
    compressor_capex_USDprkWe = 39 # $/kWe
    model_year_CEPCI = 607.5
    year2018_CEPCI = 603.1
    # policy credit
    CO2_per_H2 = 8.3 # kg CO2e/kg H2 -> change if the capture rate is changed
    policy_credit_45Q_duration = 12 # years
    policy_credit_PTC_duration = 10 # years
    energy_demand_process = 0.13 # kWh/kgH2 defaulted to SMR without CCS
    energy_demand_process_ccs = 1.5 # kWh/kg H2
    total_plant_cost = 0
    NG_consumption = 176 # MJ/kgH2
    energy_demand_NG = 0.51
    total_energy_demand =  0.64 # kWh/kgH2
    
    electricity_prices = pd.read_csv('examples/H2_Analysis/annual_average_retail_prices.csv')
    for el_prices_file in el_prices_files:
        electricity_prices = pd.read_csv(el_prices_file, header=0, index_col=0)
        
    if atb_year == 2020:
        cambium_year = 2025
    elif atb_year == 2025:
        cambium_year = 2030
    elif atb_year == 2030:
        cambium_year =2035
    elif atb_year == 2035:
        cambium_year = 2040
    
    # Read in Cambium data  
    cambiumdata_filepath = dircambium + site_name + '_'+str(cambium_year) + '.csv'
    cambium_data = pd.read_csv(cambiumdata_filepath,index_col = None,header = 4,usecols = ['lrmer_co2_c','lrmer_ch4_c','lrmer_n2o_c','lrmer_co2_p','lrmer_ch4_p','lrmer_n2o_p','lrmer_co2e_c','lrmer_co2e_p','lrmer_co2e'])
    
    cambium_data = cambium_data.reset_index().rename(columns = {'index':'Interval','lrmer_co2_c':'LRMER CO2 combustion (kg-CO2/MWh)','lrmer_ch4_c':'LRMER CH4 combustion (g-CH4/MWh)','lrmer_n2o_c':'LRMER N2O combustion (g-N2O/MWh)',\
                                                  'lrmer_co2_p':'LRMER CO2 production (kg-CO2/MWh)','lrmer_ch4_p':'LRMER CH4 production (g-CH4/MWh)','lrmer_n2o_p':'LRMER N2O production (g-N2O/MWh)','lrmer_co2e_c':'LRMER CO2 equiv. combustion (kg-CO2e/MWh)',\
                                                  'lrmer_co2e_p':'LRMER CO2 equiv. production (kg-CO2e/MWh)','lrmer_co2e':'LRMER CO2 equiv. total (kg-CO2e/MWh)'})
    
    cambium_data['Interval']=cambium_data['Interval']+1
    cambium_data = cambium_data.set_index('Interval')   
 
    
    # Calculate hourly grid emissions factors of interest. If we want to use different GWPs, we can do that here. The Grid Import is an hourly data i.e., in MWh
    cambium_data['Total grid emissions (kg-CO2e)'] = total_energy_demand * kWh_to_MWh_conv * hydrogen_production_kgpy * cambium_data['LRMER CO2 equiv. total (kg-CO2e/MWh)']
    cambium_data['Scope 2 (combustion) grid emissions (kg-CO2e)'] = total_energy_demand * kWh_to_MWh_conv * hydrogen_production_kgpy * cambium_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)']
    cambium_data['Scope 3 (production) grid emissions (kg-CO2e)'] = total_energy_demand * kWh_to_MWh_conv * hydrogen_production_kgpy * cambium_data['LRMER CO2 equiv. production (kg-CO2e/MWh)']
    
    # Sum total emissions
    scope2_grid_emissions_sum = cambium_data['Scope 2 (combustion) grid emissions (kg-CO2e)'].sum()*plant_life*kg_to_MT_conv
    scope3_grid_emissions_sum = cambium_data['Scope 3 (production) grid emissions (kg-CO2e)'].sum()*plant_life*kg_to_MT_conv
    
    # Energy demand and plant costs
    if CCS_option == 'wCCS':
        energy_demand_process_ccs = 1.5 # kWh/kgH2
        total_plant_cost = model_year_CEPCI/year2018_CEPCI*(0.0836 * (h2_plant_capacity_kgpd**0.687)) * 1000000 # $ ; the correlation takes daily capacity
        energy_demand_NG = 0.51 # 2.01-1.50 # kWh/kgH2
        NG_consumption = 176 # MJ/kgH2 XXX Using same value as SMR only case for now as a placeholder
        total_energy_demand = energy_demand_process_ccs + energy_demand_NG 

    elif CCS_option == 'woCCS':
        energy_demand_process = 0.13 # kWh/kgH2
        total_plant_cost = model_year_CEPCI/year2018_CEPCI*13301 * (h2_plant_capacity_kgpd**0.746) # $
        energy_demand_NG = 0.51 # 0.64-0.13 kWh/kgH2
        NG_consumption = 176 # MJ/kgH2
        total_energy_demand = energy_demand_process + energy_demand_NG        
    
    if atb_year == 2020: 
        electricity_prices = electricity_prices.iloc[1]
    elif atb_year == 2025: 
        electricity_prices = electricity_prices.iloc[2]
    elif atb_year == 2030: 
        electricity_prices = electricity_prices.iloc[3]
    elif atb_year == 2035: 
        electricity_prices = electricity_prices.iloc[4]
    
    
    # Indirect capital cost as a percentage of installed capital cost
    if site_name == 'IN': # Indiana
        land_cost = 6696 # $2019/acre 
        water_cost = 0.00634
        electricity_cost = electricity_prices['IN'] #$/MWh
    elif site_name == 'TX': # Texas
        land_cost = 2086 # $2019/acre
        water_cost = 0.00811
        electricity_cost = electricity_prices['TX'] #$/MWh
    elif site_name == 'IA': # Iowa
        land_cost = 7398 # $2019/acre
        water_cost = 0.00612
        electricity_cost = electricity_prices['IA'] #$/MWh
    elif site_name == 'MS': # Mississippi
        land_cost = 2788 # $2019/acre
        water_cost = 0.00844 
        electricity_cost = electricity_prices['MS'] #$/MWh
    elif site_name == 'WY': # Wyoming
        land_cost =7392 #$/MW
        water_cost = 0.003033 # $/gal
        electricity_cost = electricity_prices['WY'] #$/MWh
        
    if NG_price_case == 'default':
        NG_cost = 0.00536 # $2019/MJ 
    elif NG_price_case == 'min':
        NG_cost = 0.0024 # $2019/MJ
    elif NG_price_case == 'max':
        NG_cost = 0.0109 # $2019/MJ
    
    # Calculations
    #------------------------------------------------------------------------------
    # CAPEX
    #------------------------------------------------------------------------------
    # Calculate storage capital costs. Storage duration arbitrarily chosen at 4 hours capacity
    if CCS_option == 'wCCS': 
        hydrogen_storage_capacity_kg = hydrogen_storage_duration * energy_demand_process_ccs * hydrogen_production_kgpy / (hrs_in_year  * lhv_h2)
    elif CCS_option == 'woCCS': 
        hydrogen_storage_capacity_kg = hydrogen_storage_duration * energy_demand_process * hydrogen_production_kgpy / (hrs_in_year  * lhv_h2)
    capex_storage_installed = hydrogen_storage_capacity_kg * hydrogen_storage_cost_USDprkg
    capex_compressor_installed = compressor_capex_USDprkWe * h2_plant_capacity_kgpy * lhv_h2 / hrs_in_year 
    
    # Fixed and variable costs
    #------------------------------------------------------------------------------
    fom_SMR_total = fom_SMR_perc * total_plant_cost # $/year
    vom_SMR_total_perkg = NG_cost * NG_consumption  + total_energy_demand * electricity_cost / 1000  # $/kgH2
 
        
    # Calculate SMR emissions
    smr_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv + energy_demand_process * cambium_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
    smr_Scope2_EI = energy_demand_process * cambium_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
    smr_Scope1_EI = smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
    smr_total_EI  = smr_Scope1_EI + smr_Scope2_EI + smr_Scope3_EI    
               
    # Calculate SMR + CCS emissions
    smr_ccs_Scope3_EI = smr_NG_supply * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv + energy_demand_process_ccs * cambium_data['LRMER CO2 equiv. combustion (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
    smr_ccs_Scope2_EI = energy_demand_process_ccs * cambium_data['LRMER CO2 equiv. production (kg-CO2e/MWh)'].mean() * kWh_to_MWh_conv # kg CO2e/kg H2
    smr_ccs_Scope1_EI = (1-ccs_perc_capture)* smr_NG_combust * (smr_NG_consume - smr_steam_prod/smr_HEX_eff) * g_to_kg_conv # kg CO2e/kg H2
    smr_ccs_total_EI  = smr_ccs_Scope1_EI + smr_ccs_Scope2_EI + smr_ccs_Scope3_EI  
    
    
    H2_PTC = 0 
    # Policy credit
    if CCS_option == 'wCCS':    
        if policy_case == 'no policy':
            CO2_credit = 0
            H2_PTC = 0 
        elif policy_case == 'base':
            CO2_credit = 17 # $/ton CO2
            if smr_ccs_total_EI <= 0.45: # kg CO2e/kg H2
                H2_PTC = 0.6 # $/kg H2
            elif smr_ccs_total_EI > 0.45 and smr_ccs_total_EI <= 1.5: # kg CO2e/kg H2
                H2_PTC = 0.2 # $/kg H2
            elif smr_ccs_total_EI > 1.5 and smr_ccs_total_EI <= 2.5: # kg CO2e/kg H2     
                H2_PTC = 0.15 # $/kg H2
            elif smr_ccs_total_EI > 2.5 and smr_ccs_total_EI <= 4: # kg CO2e/kg H2    
                H2_PTC = 0.12 # $/kg H2
        elif policy_case == 'max':
            CO2_credit = 85 # $/ton CO2                   
            if smr_ccs_total_EI <= 0.45: # kg CO2e/kg H2
                H2_PTC = 3 # $/kg H2
            elif smr_ccs_total_EI > 0.45 and smr_ccs_total_EI <= 1.5: # kg CO2e/kg H2
                H2_PTC = 1 # $/kg H2
            elif smr_ccs_total_EI > 1.5 and smr_ccs_total_EI <= 2.5: # kg CO2e/kg H2     
                H2_PTC = 0.75 # $/kg H2
            elif smr_ccs_total_EI > 2.5 and smr_ccs_total_EI <= 4: # kg CO2e/kg H2    
                H2_PTC = 0.6 # $/kg H2
            
    if atb_year == 2035: 
       CO2_credit = 0 
       H2_PTC = 0
            
    CCS_credit_45Q = CO2_credit * (smr_Scope1_EI - smr_ccs_Scope1_EI)  / (mt_tokg_conv)  # $/kgH2

    policy_credit = max(H2_PTC,CCS_credit_45Q) # $/kgH2
    
    # Set up ProFAST
    pf = ProFAST.ProFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00 # keep the zeroes after the decimal otherwise script will throw an error
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":10.0,"escalation":gen_inflation})
    pf.set_params('capacity',h2_plant_capacity_kgpd) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',capacity_factor)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',0)
    pf.set_params('admin expense percent',0)
    pf.set_params('total income tax rate',0.27)
    pf.set_params('capital gains tax rate',0.15)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.0824)
    pf.set_params('debt equity ratio of initial financing',1.38)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',0.0489)
    pf.set_params('cash onhand percent',1)
    
    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="SMR Plant Cost",cost=total_plant_cost,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=7,refurb=[0])
    #    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=5,refurb=[0])
    
    total_capex = total_plant_cost+capex_storage_installed+capex_compressor_installed

    capex_fraction = {'SMR Plant Cost':total_plant_cost/total_capex,
                      'Compression':capex_compressor_installed/total_capex,
                      'Hydrogen Storage':capex_storage_installed/total_capex}
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="SMR FOM Cost",usage=1.0,unit='$/year',cost=fom_SMR_total,escalation=gen_inflation)
    #    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)
    
    #---------------------- Add feedstocks, note the various cost options-------------------
    #pf.add_feedstock(name='Electricity',usage=total_energy_demand,unit='kWh',cost=electricity_cost,escalation=gen_inflation)
    #pf.add_feedstock(name='Natural Gas',usage=NG_consumption,unit='MJ/kg-H2',cost=NG_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Water Charges',usage=water_consumption,unit='gallons of water per kg-H2',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='SMR VOM Cost',usage=1.0,unit='$/kg-H2',cost=vom_SMR_total_perkg,escalation=gen_inflation)
    
    if (CO2_credit > H2_PTC):
        pf.add_incentive(name ='Policy credit', value=policy_credit, decay = 0, sunset_years = policy_credit_45Q_duration, tax_credit = True)
    else:
        pf.add_incentive(name ='Policy credit', value=policy_credit, decay = 0, sunset_years = policy_credit_PTC_duration, tax_credit = True)

    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    price_breakdown = pf.get_cost_breakdown()
    
    # Calculate financial expense associated with equipment
    cap_expense = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
        
    remaining_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
    
    price_breakdown_SMR_plant = price_breakdown.loc[price_breakdown['Name']=='SMR Plant Cost','NPV'].tolist()[0] + cap_expense*capex_fraction['SMR Plant Cost']
    price_breakdown_H2_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0] + cap_expense*capex_fraction['Hydrogen Storage']
    price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
    #    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]
    #    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]

    price_breakdown_proptax_ins = price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()
    price_breakdown_SMR_FOM = price_breakdown.loc[price_breakdown['Name']=='SMR FOM Cost','NPV'].tolist()[0]
    price_breakdown_SMR_VOM = price_breakdown.loc[price_breakdown['Name']=='SMR VOM Cost','NPV'].tolist()[0]
    price_breakdown_water_charges = price_breakdown.loc[price_breakdown['Name']=='Water Charges','NPV'].tolist()[0] 
    #    price_breakdown_natural_gas = price_breakdown.loc[price_breakdown['Name']=='Natural Gas','NPV'].tolist()[0]
    #    price_breakdown_electricity = price_breakdown.loc[price_breakdown['Name']=='Electricity','NPV'].tolist()[0]
    
    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]
        
    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]
    import numpy as np 
    # price_breakdown_financial = np.array(price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0])\
    #     + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]
        
    lcoh_check = price_breakdown_SMR_plant + price_breakdown_H2_storage + price_breakdown_compression \
                        + price_breakdown_SMR_FOM + price_breakdown_SMR_VOM +  price_breakdown_water_charges \
                        + price_breakdown_taxes + remaining_financial\
                  

                       # + price_breakdown_desalination + price_breakdown_desalination_FOM
                         
    lcoh_breakdown = {'LCOH: Hydrogen Storage ($/kg)':price_breakdown_H2_storage,\
                      'LCOH: Compression ($/kg)':price_breakdown_compression,\
                      'LCOH: SMR Plant CAPEX ($/kg)':price_breakdown_SMR_plant,\
                      'LCOH: SMR Plant FOM ($/kg)':price_breakdown_SMR_FOM,
                      'LCOH: SMR Plant VOM ($/kg)':price_breakdown_SMR_VOM,\
                      'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                      'LCOH: Water charges ($/kg)':price_breakdown_water_charges,\
                      'LCOH: Finances ($/kg)':remaining_financial,
                      'LCOH: total ($/kg)':lcoh_check}

    hydrogen_annual_production=hydrogen_production_kgpy
    lcoh = lcoh_check
    lcoe = electricity_cost
    hydrogen_storage_duration_hr = hydrogen_storage_duration
    price_breakdown_storage = price_breakdown_H2_storage
    natural_gas_cost = NG_cost

    price_breakdown = price_breakdown.drop(columns=['index','Amount'])

    return(hydrogen_annual_production, hydrogen_storage_duration_hr, lcoh, lcoh_breakdown, price_breakdown,lcoe,  plant_life, natural_gas_cost,  price_breakdown_storage,price_breakdown_compression,
                         price_breakdown_SMR_plant,
                         price_breakdown_SMR_FOM, price_breakdown_SMR_VOM,\
                         price_breakdown_taxes,\
                         price_breakdown_water_charges,\
                         remaining_financial,\
                         total_capex
                         #policy_credit_45Q
                         )

