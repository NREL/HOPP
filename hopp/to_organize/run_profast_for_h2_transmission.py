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

def run_profast_for_h2_transmission(project_dir, max_hydrogen_production_rate_kg_hr,max_hydrogen_delivery_rate_kg_hr,pipeline_length_km,electrolyzer_capacity_factor,enduse_capacity_factor,before_after_storage,plant_life,elec_price):

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
        hydrogen_flow_capacity_kg_day = max_hydrogen_production_rate_kg_hr*24
        transmission_capacity_factor = electrolyzer_capacity_factor
    if before_after_storage == 'after':
        hydrogen_flow_capacity_kg_day = max_hydrogen_delivery_rate_kg_hr*24
        transmission_capacity_factor = enduse_capacity_factor

    # Read in compressor and pipeline scaling csv
    pipeline_compressor_cost_data = pd.read_csv(os.path.join(dir0, "Pipeline and compressor sv 01.csv"),index_col = None,header = 0)
    compressor_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Technology'] == 'GH2 Pipeline Compressor'].drop(labels = ['Index'],axis=1)
    pipeline_cost_data = pipeline_compressor_cost_data.loc[pipeline_compressor_cost_data['Technology'] == 'GH2 Pipeline (transmission)'].drop(labels = ['Index'],axis=1)

    # Interpolate compressor and pipe costs based on capacity. Note that it would be preferable to interpolate the scaling factor and then apply the scaling factor,
    # but this will take more time. Could do it at some point
    compressor_capex = np.interp(hydrogen_flow_capacity_kg_day,compressor_cost_data['Nameplate kg/d'].to_numpy(),compressor_cost_data['Capital Cost [$]'].to_numpy())
    compressor_FOM_frac = np.interp(hydrogen_flow_capacity_kg_day,compressor_cost_data['Nameplate kg/d'].to_numpy(),compressor_cost_data['Fixed Operating Cost [fraction of OvernightCapCost/y]'].to_numpy())
    compressor_FOM_USD_yr = compressor_FOM_frac*compressor_capex

    pipeline_capex_perkm = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Capital Cost [$/km]'].to_numpy())
    pipeline_capex = pipeline_capex_perkm*pipeline_length_km
    pipeline_FOM_frac = np.interp(hydrogen_flow_capacity_kg_day,pipeline_cost_data['Nameplate kg/d'].to_numpy(),pipeline_cost_data['Fixed Operating Cost [fraction of OvernightCapCost/y]'].to_numpy())
    pipeline_FOM_USD_yr = pipeline_FOM_frac*pipeline_capex

    # Set up ProFAST
    pf = ProFAST.ProFAST('blank')

    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    pf.set_params('capacity',hydrogen_flow_capacity_kg_day) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)
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
    pf.set_params('total income tax rate',0.27)
    pf.set_params('capital gains tax rate',0.15)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    #pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.0824)
    pf.set_params('debt equity ratio of initial financing',1.38)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',0.0489)
    pf.set_params('cash onhand',1)

    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="Pipeline",cost=pipeline_capex,depr_type="MACRS",depr_period=7,refurb=[0])
    pf.add_capital_item(name="Compressor",cost=compressor_capex,depr_type="MACRS",depr_period=7,refurb=[0])

    total_capex = pipeline_capex + compressor_capex

    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Pipeline Fixed O&M Cost",usage=1.0,unit='$/year',cost=pipeline_FOM_USD_yr,escalation=gen_inflation)
    pf.add_fixed_cost(name="Compressor Fixed O&M Cost",usage=1.0,unit='$/year',cost=compressor_FOM_USD_yr,escalation=gen_inflation)

    #---------------------- Add feedstocks, note the various cost options-------------------
    pf.add_feedstock(name='Electricity',usage=0.5892,unit='kWh',cost=elec_price,escalation=gen_inflation)

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
