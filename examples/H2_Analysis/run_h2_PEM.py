from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
from numpy.lib.function_base import average
import examples.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd


def run_h2_PEM(electrical_generation_timeseries, electrolyzer_size,
                kw_continuous,forced_electrolyzer_cost_kw,lcoe,
                adjusted_installed_cost,useful_life,net_capital_costs,
                voltage_type="constant", stack_input_voltage_DC=250, min_V_cell=1.62,
                p_s_h2_bar=31, stack_input_current_lower_bound=500, cell_active_area=1250, 
                N_cells=130, total_system_electrical_usage=55.5):

    in_dict = dict()
    out_dict = dict()
    in_dict['P_input_external_kW'] = electrical_generation_timeseries
    in_dict['electrolyzer_system_size_MW'] = electrolyzer_size
    el = PEM_electrolyzer_LT(in_dict, out_dict)

    # el.power_supply_rating_MW = electrolyzer_size
    # el.power_supply_rating_MW = power_supply_rating_MW
   #  print("electrolyzer size: ", electrolyzer_size)
   #  el.electrolyzer_system_size_MW = electrolyzer_size
   #  el.input_dict['voltage_type'] = voltage_type
   #  el.stack_input_voltage_DC = stack_input_voltage_DC
   # el.stack_input_voltage_DC = 
    # Assumptions:
   #  el.min_V_cell = min_V_cell  # Only used in variable voltage scenario
   #  el.p_s_h2_bar = p_s_h2_bar   # H2 outlet pressure
   #  el.stack_input_current_lower_bound = stack_input_current_lower_bound
   #  el.cell_active_area = cell_active_area
   #  el.N_cells = N_cells
   #  print("running production rate")
   #  el.h2_production_rate()

    el.h2_production_rate()

    avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
    # print("avg_generation: ", avg_generation)
    cap_factor = avg_generation / kw_continuous

    hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
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
                        hydrogen_hourly_production
                   }

    return H2_Results, H2A_Results





