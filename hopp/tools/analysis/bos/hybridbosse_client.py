from hybridbosse.hybridbosse_api.run_hybridbosse import run as run_hybridbosse
import warnings

warnings.filterwarnings('ignore')   # ignore pandas FutureWarning

hybrids_input_dict = dict()

# shared infra inputs
hybrids_input_dict['shared_interconnection'] = True
hybrids_input_dict['distance_to_interconnect_mi'] = 1  # Input not used for projects <= 15 MW
hybrids_input_dict['new_switchyard'] = True
hybrids_input_dict['interconnect_voltage_kV'] = 15
hybrids_input_dict['shared_substation'] = True
hybrids_input_dict['hybrid_substation_rating_MW'] = 7.5

# Wind farm required inputs
hybrids_input_dict['num_turbines'] = 5
hybrids_input_dict['turbine_rating_MW'] = 1.5
hybrids_input_dict['wind_dist_interconnect_mi'] = 0    # Only used for calculating grid cost of wind only. Input not used for projects <= 15 MW
hybrids_input_dict['wind_construction_time_months'] = 5
hybrids_input_dict['project_id'] = 'hybrids'

# Solar farm required inputs
hybrids_input_dict['solar_system_size_MW_DC'] = 7.5
hybrids_input_dict['solar_construction_time_months'] = 5   # Optional. Overrides the internal scaling MW vs. construction time relationship
hybrids_input_dict['solar_dist_interconnect_mi'] = 0

# pre-processed input data:
hybrids_input_dict['wind_plant_size_MW'] = hybrids_input_dict['num_turbines'] * \
                                           hybrids_input_dict['turbine_rating_MW']

hybrids_input_dict['hybrid_plant_size_MW'] = hybrids_input_dict['wind_plant_size_MW'] + \
                                             hybrids_input_dict['solar_system_size_MW_DC']

hybrids_input_dict['hybrid_construction_months'] = \
    hybrids_input_dict['wind_construction_time_months'] + \
    hybrids_input_dict['solar_construction_time_months']

hybrids_input_dict['grid_interconnection_rating_MW'] = 100
hybrid_results, wind_only, solar_only = run_hybridbosse(hybrids_input_dict)

print('Hybrid Dictionary Results:')
print('')
print(hybrid_results)
print('')
print('Wind Only Dictionary Results:')
print('')
print(wind_only)
print('')
print('Solar Only Dictionary Results:')
print('')
print(solar_only)
