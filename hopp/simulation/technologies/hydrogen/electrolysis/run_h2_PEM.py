
from numpy.lib.function_base import average
import hopp.to_organize.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd
from hopp.to_organize.PEM_Model_2Push.run_PEM_master import run_PEM_clusters
from hopp.simulation.technologies.hydrogen.electrolysis.PEM_electrolyzer_IVcurve import PEM_electrolyzer_LT


def run_h2_PEM(electrical_generation_timeseries, electrolyzer_size,
                useful_life, n_pem_clusters,  electrolysis_scale,
                pem_control_type,electrolyzer_direct_cost_kw, user_defined_pem_param_dictionary,
                use_degradation_penalty, grid_connection_scenario,
                hydrogen_production_capacity_required_kgphr
                ):

   pem=run_PEM_clusters(electrical_generation_timeseries,electrolyzer_size,n_pem_clusters,electrolyzer_direct_cost_kw,useful_life,user_defined_pem_param_dictionary,use_degradation_penalty)

   if grid_connection_scenario!='off-grid':
      h2_ts,h2_tot=pem.run_grid_connected_pem(electrolyzer_size,hydrogen_production_capacity_required_kgphr)
   else:
      if pem_control_type == 'optimize':
         h2_ts,h2_tot=pem.run(optimize=True)
      else:
         h2_ts,h2_tot=pem.run()
   #avg_pem_cf = np.mean(h2_tot.loc['PEM Capacity Factor'].values)

   energy_used_by_electrolyzer=h2_ts.loc['Power Consumed [kWh]'].sum()
   energy_input_to_electrolyzer=h2_ts.loc['Input Power [kWh]'].sum()
   average_uptime_hr=h2_tot.loc['Total Uptime [sec]'].mean()/3600
   avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
   # print("avg_generation: ", avg_generation)
   elec_rated_h2_capacity_kgpy =h2_tot.loc['Cluster Rated H2 Production [kg/yr]'].sum()
   #cap_factor = h2_tot.loc['PEM Capacity Factor'].mean()#avg_generation / kw_continuous
   # This appears to give the same answer but it is a better definition
   cap_factor=h2_tot.loc['Total H2 Production [kg]'].sum()/elec_rated_h2_capacity_kgpy

   hydrogen_hourly_production = h2_ts.loc['hydrogen_hourly_production'].sum()
   water_hourly_usage = h2_ts.loc['water_hourly_usage_kg'].sum()
   water_annual_usage = np.sum(water_hourly_usage)
   hourly_system_electrical_usage=h2_ts.loc['Power Consumed [kWh]'].sum()
   total_system_electrical_usage = np.sum(hourly_system_electrical_usage)
   avg_eff_perc=39.41*hydrogen_hourly_production/hourly_system_electrical_usage #np.nan_to_num(h2_ts.loc['electrolyzer_total_efficiency_perc'].mean())
   hourly_efficiency=np.nan_to_num(avg_eff_perc)
   tot_avg_eff=39.41/h2_tot.loc['Total kWh/kg'].mean()
   #out_dict['water_used_kg_hr']
   # water_annual_usage = out_dict['water_used_kg_annual']
   #electrolyzer_total_efficiency = out_dict['total_efficiency']

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

#  H2A_Results = H2AModel.H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output, force_system_size=True,
#                                 forced_system_size=electrolyzer_size, force_electrolyzer_cost=True,
#                                  forced_electrolyzer_cost_kw=forced_electrolyzer_cost_kw, useful_life = useful_life)


   # feedstock_cost_h2_levelized_hopp = lcoe * total_system_electrical_usage / 100  # $/kg
   # # Hybrid Plant - levelized H2 Cost - HOPP
   # feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = adjusted_installed_cost / \
   #                                                       (hydrogen_annual_output * useful_life)  # $/kgH2

   # Total Hydrogen Cost ($/kgH2)
#  h2a_costs = H2A_Results['Total Hydrogen Cost ($/kgH2)']
#  total_unit_cost_of_hydrogen = h2a_costs + feedstock_cost_h2_levelized_hopp
   # feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt = net_capital_costs / (
   #                            (kw_continuous / total_system_electrical_usage) * (8760 * useful_life))
   rated_kWh_pr_kg=h2_tot.loc['Stack Rated Power Consumed [kWh]'].values[0]/h2_tot.loc['Stack Rated H2 Production [kg/hr]'].values[0]
   H2_Results = {'hydrogen_annual_output':
                     hydrogen_annual_output,
                  # 'feedstock_cost_h2_levelized_hopp':
                  #    feedstock_cost_h2_levelized_hopp,
                  # 'feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp':
                  #    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp,
                  # 'feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt':
                  #    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt,
               #   'total_unit_cost_of_hydrogen':
               #      total_unit_cost_of_hydrogen,
                  'cap_factor':
                     cap_factor,
                  'hydrogen_hourly_production':
                     hydrogen_hourly_production,
                  'water_hourly_usage':
                  water_hourly_usage,
                  'water_annual_usage':
                  water_annual_usage,
                  'electrolyzer_avg_efficiency':
                  tot_avg_eff,
                  'total_electrical_consumption':
                  total_system_electrical_usage,
                  'electrolyzer_total_efficiency':
                  hourly_efficiency,
                  'time_between_replacement_per_stack':
                  h2_tot.loc['Avg [hrs] until Replacement Per Stack'],
                  'avg_time_between_replacement':
                  h2_tot.loc['Avg [hrs] until Replacement Per Stack'].mean(),
                  'Rated kWh/kg-H2':rated_kWh_pr_kg,
                  'average_operational_time [hrs]':
                  average_uptime_hr
                  }

   return H2_Results, h2_ts, h2_tot,energy_input_to_electrolyzer #, H2A_Results




def kernel_PEM_IVcurve(
      electrical_generation_timeseries,
      electrolyzer_size,
      useful_life,
      kw_continuous,
      forced_electrolyzer_cost_kw,
      lcoe,
      adjusted_installed_cost,
      net_capital_costs,
      voltage_type="constant", stack_input_voltage_DC=250, min_V_cell=1.62,
      p_s_h2_bar=31, stack_input_current_lower_bound=500, cell_active_area=1250,
      N_cells=130, total_system_electrical_usage=55.5
):

    in_dict = dict()
    out_dict = dict()
    in_dict['P_input_external_kW'] = electrical_generation_timeseries
    in_dict['electrolyzer_system_size_MW'] = electrolyzer_size
    el = PEM_electrolyzer_LT(in_dict, out_dict)

    el.h2_production_rate()
    el.water_supply()

    avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
    cap_factor = avg_generation / kw_continuous

    hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
    water_hourly_usage = out_dict['water_used_kg_hr']
    water_annual_usage = out_dict['water_used_kg_annual']
    electrolyzer_total_efficiency = out_dict['total_efficiency']

    # Get Daily Hydrogen Production - Add Every 24 hours
    i = 0
    daily_H2_production = []
    while i <= 8760:
        x = sum(hydrogen_hourly_production[i:i + 24])
        daily_H2_production.append(x)
        i = i + 24

    avg_daily_H2_production = np.mean(daily_H2_production)  # kgH2/day
    hydrogen_annual_output = sum(hydrogen_hourly_production)  # kgH2/year
    # elec_remainder_after_h2 = combined_pv_wind_curtailment_hopp

    H2A_Results = H2AModel.H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output, force_system_size=True,
                                   forced_system_size=electrolyzer_size, force_electrolyzer_cost=True,
                                    forced_electrolyzer_cost_kw=forced_electrolyzer_cost_kw, useful_life = useful_life)


    feedstock_cost_h2_levelized_hopp = lcoe * total_system_electrical_usage / 100  # $/kg
    # Hybrid Plant - levelized H2 Cost - HOPP
    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = adjusted_installed_cost / \
                                                          (hydrogen_annual_output * useful_life)  # $/kgH2

    # Total Hydrogen Cost ($/kgH2)
    h2a_costs = H2A_Results['Total Hydrogen Cost ($/kgH2)']
    total_unit_cost_of_hydrogen = h2a_costs + feedstock_cost_h2_levelized_hopp
    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt = net_capital_costs / (
                                (kw_continuous / total_system_electrical_usage) * (8760 * useful_life))

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
                       cap_factor,
                    'hydrogen_hourly_production':
                        hydrogen_hourly_production,
                     'water_hourly_usage':
                     water_hourly_usage,
                     'water_annual_usage':
                     water_annual_usage,
                     'electrolyzer_total_efficiency':
                     electrolyzer_total_efficiency
                   }

    return H2_Results, H2A_Results


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


