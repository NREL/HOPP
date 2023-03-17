# from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT #OLD
import sys
# sys.path.append()
from hybrid.Electrolyzer_Models.run_h2_PEM_Basic import run_h2_basic #DOUBLE CHECK
from hybrid.Electrolyzer_Models.run_h2_clusters import run_PEM_clusters #DOUBLE CHECK
from hybrid.Electrolyzer_Models.PEM_H2_LT_electrolyzer_Basic import PEM_H2_Basic
from hybrid.Electrolyzer_Models.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters
from hybrid.Electrolyzer_Models.run_h2_distributed import run_PEM_distributed
from hybrid.Electrolyzer_Models.run_h2_power_powerElec import run_PEM_power_electronics
from matplotlib import pyplot as plt
from numpy.lib.function_base import average
import examples.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd

def reorg_cluster_data(h2_ts,h2_tot):
   all_h2_data={}
   output_keys = ['water_used_kg_hr','h2_produced_kg_hr_system','water_used_kg_annual',\
      'total_annual_efficiency','total_system_electrical_usage','time_until_replacement',\
         'elec_design_consumption_kWhprkg','elec_consumption_kWhprkg','cap_factor',\
            'total_efficiency','cf_per_stack','teod_per_stack'# 'electrolyzer_design_max_efficiency_HHV','electrolyzer_energy_kWh_per_kg'
            ]
   # h2_tot.loc['Total Uptime [sec]']/8760
   # h2_tot.loc['Total Off-Cycles']
   # ip_p=h2_ts.loc['Input Power [kW]'].sum()
   # con_p=h2_ts.loc['Power Consumed [kWh]'].sum()
   # ip_p=h2_ts.loc['Input Power [kW]'].sum()
   # con_p=h2_ts.loc['Power Consumed [kWh]'].sum()
   # h2_ts.loc['electrolyzer_total_efficiency_perc']
   e_h2=39.41
   

   h2_hr=h2_ts.loc['hydrogen_hourly_production'].sum()
   h20_hr=3.79*h2_ts.loc['water_hourly_usage_gal'].sum()
   h20_yr = np.sum(h20_hr)
   eff_perc_tot = h2_tot.loc['Final Efficiency [%]'].mean()
   elec_used=h2_tot.loc['Total Input Power [kWh]'].sum()
   avg_teod=h2_tot.loc['Avg [hrs] until Replacement Per Stack'].mean()
   best_kWhprkg=h2_tot.loc['Efficiency at Min Power [kWh/kgH2]']['Cluster #0']
   sys_kwh_per_kg=h2_tot.loc['Total Input Power [kWh]'].sum()/h2_tot.loc['Total H2 Production [kg]'].sum()
   pem_cf=h2_tot.loc['PEM Capacity Factor'].mean()
   ts_eff = (e_h2*h2_ts.loc['hydrogen_hourly_production'].sum())/h2_ts.loc['Input Power [kW]'].sum()
   ts_eff = np.nan_to_num(ts_eff)
   # avg_cf=h2_tot.loc['PEM Capacity Factor'].mean()

   data = [h20_hr,h2_hr,h20_yr,eff_perc_tot,elec_used,avg_teod,best_kWhprkg,sys_kwh_per_kg,pem_cf,ts_eff,h2_tot.loc['PEM Capacity Factor'].values,h2_tot.loc['Avg [hrs] until Replacement Per Stack'].values]
   h2_results = dict(zip(output_keys,data))
   return h2_results

def make_power_loss_figs(power_data,loss_data,h2_ts,h2_tot,nturbs,floris,lat):
   
   # hybrid_plant.site.data['lat']
   casename = 'Centralized_'
   
   if lat ==41.26:
      state = 'IN'
   elif lat == 29.92:
      state = 'TX'
   elif lat==42.55:
      state='IA'
   elif lat ==30.45:
      state = 'MS'
   elif lat == 41.659:
      state = 'WY'
   else:
      state='None'
   if floris:
      desc = casename + state + 'FlorisWind_{}nTurbs'.format(nturbs)
   else:
      desc = casename + state + 'PySamWind_{}nTurbs'.format(nturbs)
def determine_plant_desc(hybrid_plant,hopp_dict,electrolysis_scale,h2_model):
   lat=hybrid_plant.site.data['lat']
   if lat ==41.62:
      state = 'IN'
   elif lat == 29.92:
      state = 'TX'
   elif lat==42.55:
      state='IA'
   elif lat ==30.45:
      state = 'MS'
   elif lat == 41.659:
      state = 'WY'
   wind_size_mw=hopp_dict.main_dict['Configuration']['wind_size']
   solar_size_mw=hopp_dict.main_dict['Configuration']['solar_size']
   storage_size_mw=hopp_dict.main_dict['Configuration']['storage_size_mw']
   storage_size_mwh=hopp_dict.main_dict['Configuration']['storage_size_mwh']
   atb_year=hopp_dict.main_dict['Configuration']['atb_year']
   # solar_size_mw=round(hybrid_plant.pv._system_model.SystemDesign.system_capacity/1000)
   plant_desc = state + '_' + electrolysis_scale + '_{}_Wind{}_Solar{}_Battery_{}MW_{}MWH'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mw,storage_size_mwh) + h2_model
   # plant_desc = state +  '_{}_Wind{}_Solar{}_Battery_{}MW_{}MWH'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mw,storage_size_mwh) + h2_model

   return plant_desc
      
   


   

def run_basic(electrical_generation_timeseries,electrolyzer_size):
   print('Running Basic PEM Model ...')
   print('This model runs as many stacks as the input power allows')
   print('This model DOES NOT include the electrical loss calculation')
   pem = PEM_H2_Basic(electrical_generation_timeseries,electrolyzer_size,include_degradation_penalty=True,output_dict={},dt=3600)
   h2_ts_data,h2_tot_data = pem.run(electrical_generation_timeseries)
   h2_extra_data = pem.output_dict
   h2_results=reorg_cluster_data(h2_ts_data,h2_tot_data)
   return h2_results,h2_tot_data

def run_basic_clusters(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters,savemedir,desc):
   print('Running Clusters PEM Model ...')
   print('This model runs the user-defined number of PEM clusters')
   print('This model DOES NOT include the electrical loss calculation')
   
   pem = run_PEM_clusters(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters)
   h2_ts,h2_tot = pem.run()
   # if 'Wind1000_Solar750_Battery_0MW_0MWH' in desc:
   # hm=[pd.Series(h2_ts.loc['V_cell With Deg'][idx],name=idx) for idx in list(h2_ts.columns.values)]
   # hm_df=pd.DataFrame(hm).T
   hm=[pd.Series(h2_ts.loc['Stacks on'][idx],name=idx) for idx in list(h2_ts.columns.values)]
   hm_df=pd.DataFrame(hm).T
   #'Stacks on'
   
   hm_df.to_csv(savemedir + 'Deg_Info/Status_' + desc + '.csv')
   []

   h2_results=reorg_cluster_data(h2_ts,h2_tot)
   return h2_results,h2_tot

def run_pem_per_turb_withElecEff(hybrid_plant,electrical_generation_timeseries,electrolyzer_size):
   print('Running Distributed Case with some PEM per turb ...')
   print('Note that the installed wind capacity exceeds the electrolyzer capacity, so not all turbs will have an electrolyzer stack.')
   print('This case includes electrical losses from a rectifier only!')
   nTurbs=hybrid_plant.wind._system_model.nTurbs
   turb_rating_mw = hybrid_plant.wind.turb_rating/1e3
   total_turb_capac_mw = turb_rating_mw*nTurbs
   n_PEM_clusters = round(electrolyzer_size // turb_rating_mw)

   pem = run_PEM_distributed(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters)
   h2_ts,h2_tot = pem.run_distributed_layout_power_floris(hybrid_plant.wind)
   power_data = pem.power_data
   loss_data = 100*np.diff(pem.power_data.values)/(pem.power_data.values[0:-1])
   # saveme='/Users/egrant/Desktop/HOPP-GIT/PowerElecResults/Distributed/'
   #power_data.to_csv(saveme + 'IN_power_data_distributed.csv')
   h2_results=reorg_cluster_data(h2_ts,h2_tot)
   return h2_results,h2_tot

def run_centralized_pem_withElecEff(hybrid_plant,electrical_generation_timeseries,nTurbs,floris,electrolyzer_size=1000):
   print('Running Centralized Case with all PEM on a central load bus ...')
   print('This will (currently) throw an error if not using the floris wind plant model')
   print('There are a default of 8 PEM clusters due to rectifier sizing contraints')
   print('This case includes electrical losses from a step-up transformer, cables, and rectifier.')
   # nTurbs=hybrid_plant.wind._system_model.nTurbs
   power_elec=run_PEM_power_electronics(nTurbs)
   # run_PEM_power_electronics(electrical_generation_timeseries,electrolyzer_size,nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
   # prepem = run_PEM_distributed(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters=1)
   if nTurbs ==255:
      nturbs_per_cable = 15
      nturbs_dist2_load = [8,9]
      n_distances = 1

   elif nTurbs == 126:
      nturbs_per_cable = 6
      nturbs_dist2_load = [7]
      n_distances = 3

   elif nTurbs == 168:
      nturbs_per_cable = 6
      nturbs_dist2_load = [7]
      n_distances = 4

   if floris:
      power_2_pem,n_clusters=power_elec.run_power_elec_floris(nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
   else:
      power_2_pem,n_clusters=power_elec.run_power_elec_pysam(electrical_generation_timeseries,nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
   power_data = power_elec.power_data
   loss_data = power_elec.loss_data
   time_series_data = power_elec.ts_data
   power_components = power_elec.component_list
   power_components['Cables']['n_cables']
   power_components['Cables']['Cable Lengths']
   power_components['Cables']['Cost/m']
   cable_cost = np.sum(power_components['Cables']['Cable Lengths'])*2*power_components['Cables']['Cost/m']
   # cable_info=prepem.find_best_cable(nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
   # loss_sum,comp_sum,power_2_pem,n_clusters = prepem.get_fully_centralized_power_losses(hybrid_plant.wind,cable_info)
   pem= run_PEM_clusters(power_2_pem,electrolyzer_size,n_clusters)
   #saveme='/Users/egrant/Desktop/HOPP-GIT/PowerElecResults/Central/'
   #power_data.to_csv(saveme + 'IN_power_data_central.csv')
   #loss_data.to_csv(saveme + 'IN_loss_data_central.csv')
   h2_ts,h2_tot = pem.run()
   h2_results=reorg_cluster_data(h2_ts,h2_tot)
   return h2_results,loss_data,h2_tot
   []


def run_h2_PEM(electrical_generation_timeseries, electrolyzer_size,
                kw_continuous,forced_electrolyzer_cost_kw,lcoe,
                adjusted_installed_cost,useful_life,net_capital_costs, n_PEM_clusters,
                h2_model,electrolysis_scale,hybrid_plant,floris,hopp_dict):
   savemedir=hopp_dict.main_dict['ESGSAVEFOLDER']
   # savemedir='/Users/egrant/Desktop/HOPP-GIT/CF_Tests_ProFast/PowerLosses/'
   if not floris:
      if hybrid_plant.wind.rotor_diameter == 185:
         hybrid_plant.wind.turb_rating = 4000
         nTurbs = 255
         

      elif hybrid_plant.wind.rotor_diameter == 255:
         hybrid_plant.wind.turb_rating = 8000
         nTurbs = 126
         
      elif hybrid_plant.wind.rotor_diameter == 196:
         hybrid_plant.wind.turb_rating = 6000
         nTurbs = 168
         
      hybrid_plant.wind._system_model.nTurbs = nTurbs

   else:
      nTurbs=hybrid_plant.wind._system_model.nTurbs
      wind_size_mw = round(hybrid_plant.wind._system_model.system_capacity/1000)
   
   
   plant_desc = determine_plant_desc(hybrid_plant,hopp_dict,electrolysis_scale,h2_model)
   []
   # pysam_default_keys =['avail_bop','avail_grid','avail_turb','ops_env','ops_grid','ops_load','elec_eff','elec_parasitic','env_deg','env_exp','icing']
   # pysam_default_vals = [0.5,1.5,3.58,1,0.84,0.99,1.91,0.1,1.8,0.4,0.21]
   if h2_model =='Basic':
      out_dict,h2_tot = run_basic(electrical_generation_timeseries,electrolyzer_size)

   elif h2_model =='Clusters' and electrolysis_scale=='Centralized':
      # esg_main_dir = '/Users/egrant/Desktop/HOPP-GIT/CF_Tests_ProFast/PEM_Performance/'
      plant_desc = determine_plant_desc(hybrid_plant,hopp_dict,electrolysis_scale,h2_model)
      out_dict,h2_tot = run_basic_clusters(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters,savemedir=savemedir,desc=plant_desc)
      # h2_tot.to_csv(savemedir + 'PEM_Performance/' + plant_desc + '_PEMData.csv')

   elif electrolysis_scale=='Centralized' and h2_model == 'Efficiency':
      plant_desc = determine_plant_desc(hybrid_plant,hopp_dict,electrolysis_scale,h2_model)
      out_dict,loss_data,h2_tot = run_centralized_pem_withElecEff(hybrid_plant,electrical_generation_timeseries,nTurbs,floris)
      # loss_data.to_csv(savemedir + 'PowerLosses/' + plant_desc + '_PowerLosses.csv')
      # h2_tot.to_csv(savemedir + 'PEM_Performance/' + plant_desc + '_PEMData.csv')
      []
   elif electrolysis_scale=='Distributed' and h2_model == 'Efficiency':
      out_dict,h2_tot = run_pem_per_turb_withElecEff(hybrid_plant,electrical_generation_timeseries,electrolyzer_size)
   elif h2_model =='Compare Power Elec':
      h2_central,loss_data,h2_totc = run_centralized_pem_withElecEff(hybrid_plant,electrical_generation_timeseries,nTurbs,floris)
      h2_distr,h2_totd = run_pem_per_turb_withElecEff(hybrid_plant,electrical_generation_timeseries,electrolyzer_size)
      out_dict = h2_central
      h2_tot=h2_totc
      
      # # nTurbs=hybrid_plant.wind._system_model.nTurbs
      # turb_rating_mw = hybrid_plant.wind.turb_rating/1e3
      # total_turb_capac_mw = turb_rating_mw*nTurbs
      # n_PEM_clusters = round(electrolyzer_size // turb_rating_mw)
      #  #[kW] -> [MW]
      # #electrolyzer_size is in MW
      # pem = run_PEM_distributed(electrical_generation_timeseries,electrolyzer_size,n_PEM_clusters)
      # # pem = run_PEM_distributed(electrical_generation_timeseries,n_PEM_clusters*turb_rating_mw,n_PEM_clusters)
      # h2_ts,h2_tot = pem.run_distributed_layout_power_floris(hybrid_plant.wind)
      # []



   else:
      print('Invalid h2_model option! Running "Basic" PEM model ...')
      pem = PEM_H2_Basic(electrical_generation_timeseries,electrolyzer_size,include_degradation_penalty=True,output_dict={},dt=3600)
      h2_ts_data,h2_tot_data = pem.run(electrical_generation_timeseries)
      # h2_extra_data = pem.output_dict
      
      
   

   # pem = PEM_H2_Basic(electrical_generation_timeseries,electrolyzer_size,include_degradation_penalty=True,output_dict={},dt=3600)
   # h2_ts_data,h2_tot_data = pem.run(electrical_generation_timeseries)
   # h2_extra_data = pem.output_dict
   t_eod = out_dict['time_until_replacement']#h2_tot_data['Avg [hrs] until Replacement Per Stack']

   # el.h2_production_rate()
   # el.water_supply()

   avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
   # print("avg_generation: ", avg_generation)
   cap_factor = avg_generation / kw_continuous

   hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
   water_hourly_usage = out_dict['water_used_kg_hr']
   water_annual_usage = out_dict['water_used_kg_annual']
   electrolyzer_total_efficiency = out_dict['total_efficiency']

   # print("cap_factor: ", cap_factor)

   # Get Daily Hydrogen Production - Add Every 24 hours
   i = 0
   daily_H2_production = []
   while i < 8760:
      x = sum(hydrogen_hourly_production[i:i + 24])
      daily_H2_production.append(x)
      i = i + 24

   avg_daily_H2_production = np.mean(daily_H2_production)  # kgH2/day
   hydrogen_annual_output = sum(hydrogen_hourly_production)  # kgH2/year
   # elec_remainder_after_h2 = combined_pv_wind_curtailment_hopp

   H2A_Results = H2AModel.H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output, force_system_size=True,
                                 forced_system_size=electrolyzer_size, force_electrolyzer_cost=True,
                                 forced_electrolyzer_cost_kw=forced_electrolyzer_cost_kw, useful_life = useful_life)


   feedstock_cost_h2_levelized_hopp = lcoe * out_dict['total_system_electrical_usage'] / 100  # $/kg
   # Hybrid Plant - levelized H2 Cost - HOPP
   feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = adjusted_installed_cost / \
                                                         (hydrogen_annual_output * useful_life)  # $/kgH2

   # Total Hydrogen Cost ($/kgH2)
   h2a_costs = H2A_Results['Total Hydrogen Cost ($/kgH2)']
   total_unit_cost_of_hydrogen = h2a_costs + feedstock_cost_h2_levelized_hopp
   feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt = net_capital_costs / (
                              (kw_continuous / out_dict['total_system_electrical_usage']) * (8760 * useful_life))

   H2_Results = {'hydrogen_annual_output':
                     hydrogen_annual_output,
                  'feedstock_cost_h2_levelized_hopp':
                     feedstock_cost_h2_levelized_hopp,
                  'feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp':
                     feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp,
                  'feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt':
                     feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt,
                  'total_unit_cost_of_hydrogen':
                     total_unit_cost_of_hydrogen,
                  'cap_factor':
                     out_dict['cap_factor'],
                  'hydrogen_hourly_production':
                     hydrogen_hourly_production,
                  'water_hourly_usage':
                  water_hourly_usage,
                  'water_annual_usage':
                  water_annual_usage,
                  'electrolyzer_total_efficiency':
                  electrolyzer_total_efficiency,
                  'time_until_replacement':
                  t_eod
                  }

   return H2_Results, h2_tot#H2A_Results





