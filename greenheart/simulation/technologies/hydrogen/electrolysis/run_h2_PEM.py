
import numpy as np
import pandas as pd
from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master import run_PEM_clusters
   
def clean_up_final_outputs(h2_tot,h2_ts):
  
   new_h2_tot = h2_tot.drop(['Cluster Rated H2 Production [kg/hr]','Cluster Rated Power Consumed [kWh]','Cluster Rated H2 Production [kg/yr]',\
      'Stack Rated H2 Production [kg/hr]','Stack Rated Power Consumed [kWh]'])
   h2_ts.sum(axis=1)
   ts_sum_desc = ['Input Power [kWh]','Power Consumed [kWh]',\
      'hydrogen production no start-up time','hydrogen_hourly_production',\
      'water_hourly_usage_kg']
   
   # new_h2_ts = h2_ts.drop(['V_cell With Deg','Power Per Stack [kW]','Stack Current [A]'])
   new_h2_ts = h2_ts.loc[ts_sum_desc].sum(axis=1)
   return new_h2_ts,new_h2_tot
   # return new_h2_ts,new_h2_tot
def combine_cluster_annual_performance_info(h2_tot):
   clusters = h2_tot.loc['Performance By Year'].index.to_list()
   performance_metrics = list(h2_tot.loc['Performance By Year'].iloc[0].keys())
   vals_to_sum = [k for k in performance_metrics if '/year' in k]
   n_years = len(h2_tot.loc['Performance By Year'].iloc[0][performance_metrics[0]].values())
   yr_keys = list(h2_tot.loc['Performance By Year'].iloc[0][performance_metrics[0]].keys())

   vals_to_average = [k for k in performance_metrics if '/year' not in k]
   new_dict = {}
   # for k in vals_to_sum:
   for k in performance_metrics:
      vals = np.zeros(n_years)
      for c in clusters:
         vals += np.array(list(h2_tot.loc['Performance By Year'].loc[c][k].values()))
         # vals += np.array(h2_tot.loc['Performance By Year'].loc[c][k].values())
      
      if k in vals_to_average:
         vals = vals/len(clusters)
      new_dict[k]= dict(zip(yr_keys,vals))
   return new_dict

def run_h2_PEM(electrical_generation_timeseries, electrolyzer_size,
                useful_life, n_pem_clusters,  electrolysis_scale, 
                pem_control_type,electrolyzer_direct_cost_kw, user_defined_pem_param_dictionary,
                use_degradation_penalty, grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr,debug_mode = False,turndown_ratio = 0.1,
                verbose=True):
   #last modified by Elenya Grant on 9/21/2023
   
   pem=run_PEM_clusters(electrical_generation_timeseries,electrolyzer_size,n_pem_clusters,electrolyzer_direct_cost_kw,useful_life,user_defined_pem_param_dictionary,use_degradation_penalty,turndown_ratio,verbose=verbose)

   if grid_connection_scenario!='off-grid':
      h2_ts,h2_tot=pem.run_grid_connected_pem(electrolyzer_size,hydrogen_production_capacity_required_kgphr)
   else:
      if pem_control_type == 'optimize':
         h2_ts,h2_tot=pem.run(optimize=True)
      else:
         h2_ts,h2_tot=pem.run()
   #dictionaries of performance during each year of simulation, 
   #good to use for a more accurate financial analysis
   annual_avg_performance = combine_cluster_annual_performance_info(h2_tot)
   
   #time-series info (unchanged)
   energy_input_to_electrolyzer=h2_ts.loc['Input Power [kWh]'].sum()
   hydrogen_hourly_production = h2_ts.loc['hydrogen_hourly_production'].sum()
   hourly_system_electrical_usage=h2_ts.loc['Power Consumed [kWh]'].sum()
   water_hourly_usage = h2_ts.loc['water_hourly_usage_kg'].sum()
   avg_eff_perc=39.41*hydrogen_hourly_production/hourly_system_electrical_usage
   hourly_efficiency=np.nan_to_num(avg_eff_perc)
   #simulation based average performance (unchanged)
   average_uptime_hr=h2_tot.loc['Total Uptime [sec]'].mean()/3600
   water_annual_usage = np.sum(water_hourly_usage)
   total_system_electrical_usage = np.sum(hourly_system_electrical_usage)
   tot_avg_eff=39.41/h2_tot.loc['Total kWh/kg'].mean()
   cap_factor_sim = h2_tot.loc['PEM Capacity Factor (simulation)'].mean()
   
   #Beginning of Life (BOL) Rated Specs (attributes/system design)
   max_h2_pr_hr = h2_tot.loc['Cluster Rated H2 Production [kg/hr]'].sum()
   max_pwr_pr_hr = h2_tot.loc['Cluster Rated Power Consumed [kWh]'].sum()
   rated_kWh_pr_kg = h2_tot.loc['Stack Rated Efficiency [kWh/kg]'].mean()
   elec_rated_h2_capacity_kgpy =h2_tot.loc['Cluster Rated H2 Production [kg/yr]'].sum()
   gal_h20_pr_kg_h2 = h2_tot.loc['gal H20 per kg H2'].mean()

   atrribute_desc = ["Efficiency [kWh/kg]","H2 Production [kg/hr]","Power Consumed [kWh]","Annual H2 Production [kg/year]",'Gal H2O per kg-H2']
   attribute_specs = ['Rated BOL: '+s for s in atrribute_desc]
   attributes = [rated_kWh_pr_kg,max_h2_pr_hr,max_pwr_pr_hr,elec_rated_h2_capacity_kgpy,gal_h20_pr_kg_h2]
   
   #Plant Life Average Performance
   system_avg_life_capfac = pd.Series(annual_avg_performance['Capacity Factor [-]']).mean()
   system_total_annual_h2_kg_pr_year = pd.Series(annual_avg_performance['Annual H2 Production [kg/year]']).mean()
   system_avg_life_eff_kWh_pr_kg =pd.Series(annual_avg_performance['Annual Average Efficiency [kWh/kg]']).mean()
   system_avg_life_eff_perc = pd.Series(annual_avg_performance['Annual Average Efficiency [%-HHV]']).mean()
   system_avg_life_energy_kWh_pr_yr = pd.Series(annual_avg_performance['Annual Energy Used [kWh/year]']).mean()
   
   average_stack_life_hrs = np.nanmean(h2_tot.loc['Stack Life [hours]'].values)
   average_time_until_replacement = np.nanmean(h2_tot.loc['Time until replacement [hours]'].values)
   life_vals = [system_avg_life_capfac,system_total_annual_h2_kg_pr_year,average_stack_life_hrs,average_time_until_replacement,system_avg_life_eff_kWh_pr_kg,system_avg_life_eff_perc,system_avg_life_energy_kWh_pr_yr]
   life_desc = ["Life: Capacity Factor","Life: Annual H2 production [kg/year]","Stack Life [hrs]","Time Until Replacement [hrs]","Life: Efficiency [kWh/kg]","Life: Efficiency [%-HHV]",'Life: Annual Power Consumption [kWh/year]']
   
   
   #Simulation Results
   sim = ["Capacity Factor","Active Time / Sim Time","Total Input Power [kWh]",\
      "Total H2 Produced [kg]",\
      "Average Efficiency [%-HHV]","Total Stack Off-Cycles","H2 Warm-Up Losses [kg]"]
   
   sim_specs = ['Sim: '+s for s in sim]
   sim_performance = [cap_factor_sim, h2_tot.loc['Operational Time / Simulation Time (ratio)'].mean(),h2_tot.loc['Total Input Power [kWh]'].sum(),\
      h2_tot.loc['Total H2 Production [kg]'].sum(),\
      tot_avg_eff,h2_tot.loc['Total Off-Cycles'].sum(),h2_tot.loc['Warm-Up Losses on H2 Production'].sum()]
   
   
   new_H2_Results = dict(zip(attribute_specs,attributes))
   new_H2_Results.update(dict(zip(sim_specs,sim_performance)))
   new_H2_Results.update(dict(zip(life_desc,life_vals)))
   
   #can't change H2 results without messing up downstream workflow
   #embedded the "new" H2 Results that would be nice to switch to
   H2_Results = {'max_hydrogen_production [kg/hr]':
                  max_h2_pr_hr,
                  'hydrogen_annual_output':
                     system_total_annual_h2_kg_pr_year,
                  'cap_factor':
                  system_avg_life_capfac,
                  'cap_factor_sim':
                     cap_factor_sim ,
                  'hydrogen_hourly_production':
                     hydrogen_hourly_production,
                  'water_hourly_usage':
                  water_hourly_usage,
                  'water_annual_usage':
                  water_annual_usage,
                  'electrolyzer_avg_efficiency_percent':
                  system_avg_life_eff_perc,
                  # tot_avg_eff,
                  'electrolyzer_avg_efficiency_kWh_pr_kg':
                  system_avg_life_eff_kWh_pr_kg,
                  'total_electrical_consumption':
                  total_system_electrical_usage,
                  'electrolyzer_total_efficiency':
                  hourly_efficiency,
                  # 'time_between_replacement_per_stack':
                  # h2_tot.loc['Avg [hrs] until Replacement Per Stack'],
                  'avg_time_between_replacement':
                  average_time_until_replacement,
                  'avg_stack_life_hrs':
                  average_stack_life_hrs,
                  # h2_tot.loc['Avg [hrs] until Replacement Per Stack'].mean(),
                  'Rated kWh/kg-H2':rated_kWh_pr_kg,
                  'average_operational_time [hrs]':
                  average_uptime_hr,
                  'new_H2_Results':new_H2_Results,
                  'Performance Schedules':pd.DataFrame(annual_avg_performance),
                  }

   
   if not debug_mode:
      h2_ts,h2_tot = clean_up_final_outputs(h2_tot,h2_ts)
  
   return H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer   
   

def run_h2_PEM_IVcurve(
      energy_to_electrolyzer,
      electrolyzer_size_mw,
      kw_continuous,
      electrolyzer_capex_kw,
      lcoe,
      adjusted_installed_cost,
      useful_life,
      net_capital_costs=0,
):
   
   # electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp
   electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
   electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

   # system_rating = electrolyzer_size
   H2_Results, H2A_Results = kernel_PEM_IVcurve(
      electrical_generation_timeseries,
      electrolyzer_size_mw,
      useful_life,
      kw_continuous,
      electrolyzer_capex_kw,
      lcoe,
      adjusted_installed_cost,
      net_capital_costs)


   H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']
   H2_Results['cap_factor'] = H2_Results['cap_factor']

   print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
   print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
   print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))


   return H2_Results, H2A_Results # , electrical_generation_timeseries


