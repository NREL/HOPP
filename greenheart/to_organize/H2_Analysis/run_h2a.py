import hopp.to_organize.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd


def run_h2a(electrical_generation_timeseries, kw_continuous, electrolyzer_size,
                                  hybrid_plant, reopt_results, scenario,
            combined_pv_wind_curtailment_hopp, lcoe, force_electrolyzer_cost,
            forced_electrolyzer_cost_kw, total_system_electrical_usage=55.5):

    avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation

    if avg_generation < kw_continuous:
        cap_factor = avg_generation / kw_continuous
    else:
        cap_factor = 0.97

    max_hourly_h2_production = (electrolyzer_size * 1000) * cap_factor / total_system_electrical_usage
    hydrogen_hourly_production = cap_factor * np.divide(electrical_generation_timeseries,
                                           total_system_electrical_usage)  # hourly hydrogen production (kg)
    hydrogen_hourly_production[hydrogen_hourly_production > max_hourly_h2_production] = max_hourly_h2_production

    # Get Daily Hydrogen Production - Add Every 24 hours
    i = 0
    daily_H2_production = []
    while i < 8760:
        x = sum(hydrogen_hourly_production[i:i + 25])
        daily_H2_production.append(x)
        i = i + 25

    avg_daily_H2_production = np.mean(daily_H2_production)  # kgH2/day
    hydrogen_annual_output = sum(hydrogen_hourly_production)  # kgH2/year
    elec_remainder_after_h2 = combined_pv_wind_curtailment_hopp

    # Run H2A model
    H2A_Results = H2AModel.H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output, force_system_size=True,
                                   forced_system_size=electrolyzer_size, force_electrolyzer_cost=True,
                                    forced_electrolyzer_cost_kw=forced_electrolyzer_cost_kw)

    print("Peak Daily Production Rate for H2 Electrolyzer {}".format(H2A_Results['peak_daily_production_rate']))
    h2a_costs = H2A_Results['Total Hydrogen Cost ($/kgH2)']
    determined_electrolyzer_size = H2A_Results['electrolyzer_size']
    determined_electrolyzer_plant_total_size = H2A_Results['total_plant_size']
    h2_scaled_total_installed_cost = H2A_Results['scaled_total_installed_cost']
    h2_scaled_total_installed_cost_kw = H2A_Results['scaled_total_installed_cost_kw']

    feedstock_cost_h2_levelized_hopp = lcoe * total_system_electrical_usage / 100  # $/kg
    # Hybrid Plant - levelized H2 Cost - HOPP
    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = hybrid_plant.grid.financial_model.Outputs.adjusted_installed_cost / \
                                                          (hydrogen_annual_output * scenario['Useful Life'])  # $/kgH2
    # Total Hydrogen Cost ($/kgH2)
    total_unit_cost_of_hydrogen = h2a_costs + feedstock_cost_h2_levelized_hopp

    # feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp = hybrid_plant.grid.financial_model.Outputs.adjusted_installed_cost /\
    #                                                     ((kw_continuous/55.5)*(8760 * useful_life))
    feedstock_cost_h2_via_net_cap_cost_lifetime_h2_reopt = reopt_results['outputs']['Scenario']['Site'] \
                                                               ['Financial']['net_capital_costs'] / (
                                                                       (kw_continuous / total_system_electrical_usage) * (8760 * scenario['Useful Life']))
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
                       cap_factor
                   }

    # resultsDF = pd.DataFrame(result_dict, index=[0])

    return H2_Results, H2A_Results