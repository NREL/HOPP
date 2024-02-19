# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 10:02:07 2022

@author: ereznic2
"""

# import sys
import numpy as np
import pandas as pd

# # Specify file path to PyFAST
# sys.path.append('../PyFAST/')

# import src.PyFAST as PyFAST

import ProFAST

import os.path

def run_profast_for_h2_transmission(project_dir, max_hydrogen_production_rate_kg_hr,max_hydrogen_delivery_rate_kg_hr,pipeline_length_km,electrolyzer_capacity_factor,enduse_capacity_factor,before_after_storage,plant_life,elec_price,grid_prices_interpolated_USDperkWh,site_name,atb_year):

    dir0 = os.path.join(project_dir, "H2_Analysis")

    # max_hydrogen_production_rate_kg_hr = 14852.8
    # max_hydrogen_delivery_rate_kg_hr = 6023.84
    # pipeline_length_km = 50
    # electrolyzer_capacity_factor = 0.33
    # enduse_capacity_factor = 0.9
    # before_after_storage = 'before'
    # plant_life = 30
    # lcoe = 4.7

    # Nameplate capacity of transmission
    if before_after_storage == 'before':
        #hydrogen_flow_capacity_kg_day = max_hydrogen_production_rate_kg_hr*24
        hydrogen_flow_capacity_kg_yr = max_hydrogen_production_rate_kg_hr*8760
        transmission_capacity_factor = electrolyzer_capacity_factor
    if before_after_storage == 'after':
        #hydrogen_flow_capacity_kg_day = max_hydrogen_delivery_rate_kg_hr*24
        hydrogen_flow_capacity_kg_yr =  max_hydrogen_delivery_rate_kg_hr*8760
        transmission_capacity_factor = enduse_capacity_factor

    hydrogen_flow_capacity_kg_day = hydrogen_flow_capacity_kg_yr/365

    # Read in compressor and pipeline scaling csv
    pipeline_compressor_cost_data = pd.read_csv(os.path.join(dir0, "HDSAM4_compressor_costs.csv"),index_col = None,header = 0)
    pipeline_pipe_cost_data = pd.read_csv(os.path.join(dir0, "HDSAM4_pipecapex_USDperkm.csv"),index_col = None,header = 0)
    pipeline_pipe_fixedopex_data = pd.read_csv(os.path.join(dir0, "HDSAM4_pipefixedopex_percentperyear.csv"),index_col = None,header = 0)

    # Select region for pipe capex and FOM
    regions={'IN':'Great Lakes','TX':'Southwest','IA':'Great Plains','MS':'Southeast','MN':'Great Plains','WY':'Rocky Mountains'}
    region=regions[site_name]

    # Downselect compressor data
    if atb_year <=2025:
        pipeline_compressor_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Production Volume']=='Low']
    else:
        pipeline_compressor_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Production Volume']=='Mid']

    # Interpolate compressor and pipe costs based on capacity. Note that it would be preferable to interpolate the scaling factor and then apply the scaling factor,
    # but this will take more time. Could do it at some point
    pipe_capex_USDperkm =  np.interp(hydrogen_flow_capacity_kg_yr,pipeline_pipe_cost_data['Capacity (kg/yr)'].to_numpy(),pipeline_pipe_cost_data[region].to_numpy())
    pipeline_capex = pipe_capex_USDperkm*pipeline_length_km
    pipe_fixedopex_percentperyear=  np.interp(hydrogen_flow_capacity_kg_yr,pipeline_pipe_fixedopex_data['Capacity (kg/yr)'].to_numpy(),pipeline_pipe_fixedopex_data[region].to_numpy())
    pipeline_FOM_USD_yr = pipe_fixedopex_percentperyear*pipeline_capex
    compressor_capex = np.interp(hydrogen_flow_capacity_kg_yr,pipeline_compressor_cost_data['Nameplate Capacity [kg/yr]'].to_numpy(),pipeline_compressor_cost_data['Capital Cost [$]'].to_numpy())
    compressor_FOM_frac = np.interp(hydrogen_flow_capacity_kg_yr,pipeline_compressor_cost_data['Nameplate Capacity [kg/yr]'].to_numpy(),pipeline_compressor_cost_data['Fixed Operating Cost [fraction of CapCost/y]'].to_numpy())
    compressor_FOM_USD_yr = compressor_FOM_frac*compressor_capex

    compressor_energy_usage_kwhperkw = np.interp(hydrogen_flow_capacity_kg_yr,pipeline_compressor_cost_data['Nameplate Capacity [kg/yr]'].to_numpy(),pipeline_compressor_cost_data['Electricity Use (kWh/kg)'].to_numpy())

    # pipeline_capex_perkm = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Capital Cost [$/km]'].to_numpy())
    # pipeline_capex = pipeline_capex_perkm*pipeline_length_km
    # pipeline_FOM_frac = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Fixed Operating Cost [fraction of OvernightCapCost/y]'].to_numpy())
    # pipeline_FOM_USD_yr = pipeline_FOM_frac*pipeline_capex

    financial_assumptions = pd.read_csv('H2_Analysis/financial_inputs.csv',index_col=None,header=0)
    financial_assumptions.set_index(["Parameter"], inplace = True)
    financial_assumptions = financial_assumptions['Hydrogen/Steel/Ammonia']

    # Set up ProFAST
    pf = ProFAST.ProFAST()

    install_years = 3
    analysis_start = list(grid_prices_interpolated_USDperkWh.keys())[0] - install_years

    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    pf.set_params('capacity',hydrogen_flow_capacity_kg_day) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',analysis_start)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',12*install_years)
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',0)
    pf.set_params('end of proj sale non depr assets',0)
    pf.set_params('demand rampup',0)
    pf.set_params('long term utilization',transmission_capacity_factor)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0)
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance',0)
    pf.set_params('admin expense',0)
    pf.set_params('total income tax rate',financial_assumptions['total income tax rate'])
    pf.set_params('capital gains tax rate',financial_assumptions['capital gains tax rate'])
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',financial_assumptions['leverage after tax nominal discount rate'])
    pf.set_params('debt equity ratio of initial financing',financial_assumptions['debt equity ratio of initial financing'])
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',financial_assumptions['debt interest rate'])
    pf.set_params('cash onhand',1)

    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="Pipeline",cost=pipeline_capex,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Compressor",cost=compressor_capex,depr_type="MACRS",depr_period=7,refurb=[0])

    total_capex = pipeline_capex + compressor_capex

    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Pipeline Fixed O&M Cost",usage=1.0,unit='$/year',cost=pipeline_FOM_USD_yr,escalation=gen_inflation)
    pf.add_fixed_cost(name="Compressor Fixed O&M Cost",usage=1.0,unit='$/year',cost=compressor_FOM_USD_yr,escalation=gen_inflation)

    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Electricity',usage=compressor_energy_usage_kwhperkw,unit='kWh',cost=grid_prices_interpolated_USDperkWh,escalation=gen_inflation)

    sol = pf.solve_price()

    summary = pf.summary_vals

    price_breakdown = pf.get_cost_breakdown()

    price_breakdown_compressor = price_breakdown.loc[price_breakdown['Name']=='Compressor','NPV'].tolist()[0]
    price_breakdown_pipeline = price_breakdown.loc[price_breakdown['Name']=='Pipeline','NPV'].tolist()[0]
    price_breakdown_compressor_FOM = price_breakdown.loc[price_breakdown['Name']=='Compressor Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_pipeline_FOM = price_breakdown.loc[price_breakdown['Name']=='Pipeline Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_electricity = price_breakdown.loc[price_breakdown['Name']=='Electricity','NPV'].tolist()[0]

    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]

    price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]

    price_check = price_breakdown_compressor+price_breakdown_pipeline+price_breakdown_compressor_FOM\
               + price_breakdown_pipeline_FOM+price_breakdown_electricity\
               + price_breakdown_taxes+price_breakdown_financial

    h2_transmission_price_breakdown = {'LCOHT: Compressor ($/kg)':price_breakdown_compressor,'LCOHT: Pipeline ($/kg)':price_breakdown_pipeline,\
                      'LCOHT: Compressor FOM ($/kg)':price_breakdown_compressor_FOM,'LCOHT: Pipeline FOM ($/kg)':price_breakdown_pipeline_FOM,\
                      'LCOHT: Compressor electricity ($/kg)':price_breakdown_electricity,'LCOHT: Taxes ($/kg)':price_breakdown_taxes,\
                      'LCOHT: Finances ($/kg)':price_breakdown_financial,'LCOHT: total ($/kg)':price_check}

    return(sol,summary,h2_transmission_price_breakdown,total_capex)
