# -*- coding: utf-8 -*-

# Specify file path to PyFAST
import sys
import os
import glob
#sys.path.insert(1,'../PyFAST/')
import numpy as np
import pandas as pd

#sys.path.append('C:\\Users\\MKOLEVA\\Documents\\Masha\\Projects\\Green_steel\\Initial_stage\\Models\\H2OPP\\HOPP\\PyFAST')
sys.path.append('../PyFAST/')
import src.PyFAST as PyFAST

dir1 = os.getcwd()
dirin_el_prices = '\\examples\\H2_Analysis\\'
el_prices_files = glob.glob(os.path.join(dir1 + dirin_el_prices, 'annual_average_retail_prices.csv'))

 # def run_pyfast_for_hydrogen_SMR(atb_years,site_location,H2_Results,plant_life,\
 #                             hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
 #                             capex_desal,opex_desal,grid_connected_hopp, CCS_on):

# def run_pyfast_for_hydrogen_SMR():

#==============================================================================
# DATA
#==============================================================================    
CCS_on = 0
plant_life = 30
hydrogen_storage_cost_USDprkg = 540
#    capex_desal = 
#    opex_desal = 
atb_years = 2025 # [2020,2025,2030,2035,2040]
site_location = 'IN' #["IN","TX","IA","MS"]
MJ_per_mmBTu = 1055.06 # Megajoule per million british thermal units conversion
mt_tokg_conv = 1000 # Metric tonne to kg conversion 
hrs_in_day  = 24
capacity_factor = 0.9  
h2_plant_capacity_kgpd = 189003 # kgH2/day
h2_plant_capacity_kgpy = 68986000 #kgH2/yr
hydrogen_production = h2_plant_capacity_kgpd * capacity_factor # kgH2/day; The number is based on annual demand of 1 MMT steel; 
hydrogen_production_kgpy = h2_plant_capacity_kgpy * capacity_factor
NG_cost = 5972 # $2019/MJ 
base_case_prod_capacity = 483014 # kgH2/day
base_case_total_installed_cap_cost = 267578000 # $
fom_SMR_perkg = 0.07     #[$/kg H2] 
electricity_cost = 0.076 # $/kWh
hydrogen_storage_duration = 4 # hours 
lhv_h2 = 33 # kWh/kg H2
water_consumption = 10 # gal H2O/kg H2 - for feedstock and process water
compressor_capex_USDprkWe = 39 # $/kWe
kgH2_toscf = 423 # 423 standard cubic feet (scf) = 1 kilogram (kg) of hydrogen

#electricity_prices = pd.read_csv('C:\\Users\\MKOLEVA\\Documents\\Masha\\Projects\\Green_steel\\Initial_stage\\Models\\H2OPP\\HOPP\\HOPP\\examples\\H2_Analysis\\annual_average_retail_prices.csv', header=0, index_col=0)
for el_prices_file in el_prices_files:
    electricity_prices = pd.read_csv(el_prices_file, header=0, index_col=0)
        
locations = list(electricity_prices.columns)
years     = list(electricity_prices.index)

#==============================================================================
# CALCULATIONS
#==============================================================================
# Energy demand and plant cost
if CCS_on == 1:
    energy_demand_process = 1.5 #kWh/kgH2
    total_plant_cost = (0.0836 * (h2_plant_capacity_kgpd**0.687)) * 1000000 # $ ; the correlation takes daily capacity
    energy_demand_NG = 0.51 # 2.01-1.50 # kWh/kgH2
    NG_consumption = 176 # MJ/kgH2 XXX Using same value as SMR only case for now as a placeholder
    total_energy_demand = energy_demand_process + energy_demand_NG*0.278 # 0.278 is a conversion factor
else:
    energy_demand_process = 0.13 #kWh/kgH2
#    total_plant_cost = base_case_total_installed_cap_cost * (h2_plant_capacity_kgpd/base_case_prod_capacity**0.7) # $
    total_plant_cost = 3 * ((kgH2_toscf * h2_plant_capacity_kgpd)**0.3) * 1000000 # $
    energy_demand_NG = 0.51 # 0.64-0.13 kWh/kgH2
    NG_consumption = 176 # MJ/kgH2
    total_energy_demand = energy_demand_process + energy_demand_NG

  # Currently, no difference in future values but this chunk necessary to allow the opportunity to select of worst ($12,165/MJ) and best ($2131/MJ) case scenarios in future
if atb_years == 2020: 
    NG_cost = 0.00536 # $2019/MJ 
    electricity_prices = electricity_prices.iloc[1]
elif atb_years == 2025: 
    NG_cost = 0.00536 # $2019/MJ 
    electricity_prices = electricity_prices.iloc[2]
elif atb_years == 2030: 
    NG_cost = 0.00536 # $2019/MJ 
    electricity_prices = electricity_prices.iloc[3]
elif atb_years == 2035: 
    NG_cost = 0.00536 # $2019/MJ 
    electricity_prices = electricity_prices.iloc[4]

# Indirect capital cost as a percentage of installed capital cost
if site_location == 'IN': # Indiana
    land_cost = 6696 # $2019/acre 
    water_cost = 0.00634
    electricity_cost = electricity_prices['IN'] #$/MWh
elif site_location == 'TX': # Texas
    land_cost = 2086 # $2019/acre
    water_cost = 0.00811
    electricity_cost = electricity_prices['TX'] #$/MWh
elif site_location == 'IA': # Iowa
    land_cost = 7398 # $2019/acre
    water_cost = 0.00612
    electricity_cost = electricity_prices['IA'] #$/MWh
elif site_location == 'MS': # Mississippi
    land_cost = 2788 # $2019/acre
    water_cost = 0.00844 
    electricity_cost = electricity_prices['MS'] #$/MWh
  
# Calculate storage capital costs. Storage duration arbitrarily chosen at 4 hours capacity
hydrogen_storage_capacity_kg = hydrogen_storage_duration * energy_demand_process * hydrogen_production / (hrs_in_day  * lhv_h2)
capex_storage_installed = hydrogen_storage_capacity_kg * hydrogen_storage_cost_USDprkg
capex_compressor_installed = compressor_capex_USDprkWe * h2_plant_capacity_kgpd * lhv_h2 / hrs_in_day 

# Fixed and variable costs

fom_SMR_total = fom_SMR_perkg * hydrogen_production_kgpy  # $/year
vom_SMR_total_perkg = NG_cost * NG_consumption  + total_energy_demand * electricity_cost / 1000  # $/kgH2

# Incentives - my understanding is that Jen is implementing for all use cases. Will check with her 

# Set up PyFAST
pf = PyFAST.PyFAST('blank')

# Fill these in - can have most of them as 0 also
gen_inflation = 0.00 # keep the zeroes after the decimal otherwise script will throw an error
pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":10.0,"escalation":gen_inflation})
pf.set_params('capacity',h2_plant_capacity_kgpd) #units/day
pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
pf.set_params('analysis start year',2020)
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
pf.set_params('leverage after tax nominal discount rate',0.1)
pf.set_params('debt equity ratio of initial financing',0.5)
pf.set_params('debt type','Revolving debt')
pf.set_params('debt interest rate',0.06)
pf.set_params('cash onhand percent',1)

#----------------------------------- Add capital items to PyFAST ----------------
pf.add_capital_item(name="SMR Plant Cost",cost=total_plant_cost,depr_type="MACRS",depr_period=5,refurb=[0])
pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=5,refurb=[0])
pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=5,refurb=[0])
#    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=5,refurb=[0])

#-------------------------------------- Add fixed costs--------------------------------
pf.add_fixed_cost(name="SMR FOM Cost",usage=1.0,unit='$/year',cost=fom_SMR_total,escalation=gen_inflation)
#    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)

#---------------------- Add feedstocks, note the various cost options-------------------
#    pf.add_feedstock(name='Electricity',usage=total_energy_demand,unit='kWh',cost=electricity_cost,escalation=gen_inflation)
#    pf.add_feedstock(name='Natural Gas',usage=NG_consumption,unit='MJ/kg-H2',cost=NG_cost,escalation=gen_inflation)
pf.add_feedstock(name='Water Charges',usage=water_consumption,unit='gallons of water per kg-H2',cost=water_cost,escalation=gen_inflation)
pf.add_feedstock(name='SMR VOM Cost',usage=1.0,unit='$/kg-H2',cost=vom_SMR_total_perkg,escalation=gen_inflation)

#    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=gen_inflation)    
sol = pf.solve_price()

summary = pf.summary_vals

price_breakdown = pf.get_cost_breakdown()

price_breakdown_SMR_plant = price_breakdown.loc[price_breakdown['Name']=='SMR Plant Cost','NPV'].tolist()[0]
price_breakdown_H2_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]  
price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0]
#    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0]
#    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]

#    price_breakdown_labor_cost_annual = price_breakdown.loc[price_breakdown['Name']=='Annual Operating Labor Cost','NPV'].tolist()[0]  
#    price_breakdown_labor_cost_maintenance = price_breakdown.loc[price_breakdown['Name']=='Maintenance Labor Cost','NPV'].tolist()[0]  
#    price_breakdown_labor_cost_admin_support = price_breakdown.loc[price_breakdown['Name']=='Administrative & Support Labor Cost','NPV'].tolist()[0] 

#price_breakdown_proptax_ins = price_breakdown.loc[price_breakdown['Name']=='Property tax and insurance','NPV'].tolist()[0]
price_breakdown_SMR_FOM = price_breakdown.loc[price_breakdown['Name']=='SMR FOM Cost','NPV'].tolist()[0]
price_breakdown_SMR_VOM = price_breakdown.loc[price_breakdown['Name']=='SMR VOM Cost','NPV'].tolist()[0]
price_breakdown_water_charges = price_breakdown.loc[price_breakdown['Name']=='Water Charges','NPV'].tolist()[0] 
#    price_breakdown_natural_gas = price_breakdown.loc[price_breakdown['Name']=='Natural Gas','NPV'].tolist()[0]
#    price_breakdown_electricity = price_breakdown.loc[price_breakdown['Name']=='Electricity','NPV'].tolist()[0]


price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]\
    - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]
price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]
    
lcoh_check = price_breakdown_SMR_plant + price_breakdown_H2_storage + price_breakdown_compression \
                    + price_breakdown_SMR_FOM + price_breakdown_SMR_VOM +  price_breakdown_water_charges \
#                    + price_breakdown_taxes + price_breakdown_financial \
                     # + price_breakdown_desalination + price_breakdown_desalination_FOM
#                          + price_breakdown_electricity \
#                          + price_breakdown_natural_gas + price_breakdown_water_charges\
           
 
           
lcoh_breakdown = {'LCOH: Hydrogen Storage ($/kg)':price_breakdown_H2_storage,'LCOH: Compression ($/kg)':price_breakdown_compression,\
                  'LCOH: SMR Plant CAPEX ($/kg)':price_breakdown_SMR_plant,\
#                      'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                  'LCOH: SMR Plant FOM ($/kg)':price_breakdown_SMR_FOM,'LCOH: SMR Plant VOM ($/kg)':price_breakdown_SMR_VOM,\
#                      'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,\
                      'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                  'LCOH: Water charges ($/kg)':price_breakdown_water_charges,\
                      'LCOH: Finances ($/kg)':price_breakdown_financial,\
                  'LCOH: total ($/kg)':lcoh_check}

#return(sol,summary,hydrogen_production,lcoh_breakdown)

