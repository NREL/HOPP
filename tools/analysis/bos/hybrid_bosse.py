import warnings
import math

from hybrid.log import bos_logger as logger
from hybridbosse.hybridbosse_api.run_hybridbosse import run as run_hybridbosse

from .bos_model import BOSCalculator


class HybridBOSSE(BOSCalculator):
    @staticmethod
    def _calculate(wind_size, solar_size, interconnection_rating, specify_construction_duration):
        warnings.filterwarnings('ignore')  # ignore pandas FutureWarning
        hybrids_input_dict = dict()

        # shared infra inputs
        hybrids_input_dict['shared_interconnection'] = True
        hybrids_input_dict['distance_to_interconnect_mi'] = 1  # Input not used for projects <= 15 MW
        hybrids_input_dict['new_switchyard'] = True
        grid_size_multiplier = 1
        grid_size = wind_size * grid_size_multiplier
        if grid_size > 15:
            hybrids_input_dict['distance_to_interconnect_mi'] = (0.0263 * grid_size) - 0.2632
        else:
            hybrids_input_dict['distance_to_interconnect_mi'] = 0
        if grid_size < 20:
            hybrids_input_dict['interconnect_voltage_kV'] = 15
        elif 20 < grid_size < 40:
            hybrids_input_dict['interconnect_voltage_kV'] = 34.5
        elif 40 <= grid_size < 75:
            hybrids_input_dict['interconnect_voltage_kV'] = 69  # should be 69
        elif grid_size >= 75:
            hybrids_input_dict['interconnect_voltage_kV'] = 138  # should be 138
        hybrids_input_dict['grid_interconnection_rating_MW'] = wind_size + solar_size
        hybrids_input_dict['shared_substation'] = True
        hybrids_input_dict['hybrid_substation_rating_MW'] = wind_size + solar_size

        # Wind farm required inputs
        hybrids_input_dict['turbine_rating_MW'] = 1.5
        num_turbines = math.ceil(wind_size/hybrids_input_dict['turbine_rating_MW'])
        hybrids_input_dict['num_turbines'] = num_turbines
        hybrids_input_dict[
            'wind_dist_interconnect_mi'] = 0  # Only used for calculating grid cost of wind only. Input not used for projects <= 15 MW
        hybrids_input_dict['wind_construction_time_months'] = 5
        hybrids_input_dict['project_id'] = 'hybrids'

        # Solar farm required inputs
        hybrids_input_dict['solar_system_size_MW_DC'] = solar_size
        hybrids_input_dict[
            'solar_construction_time_months'] = 5  # Optional. Overrides the internal scaling MW vs. construction time relationship
        hybrids_input_dict['solar_dist_interconnect_mi'] = 0

        # pre-processed input data:
        hybrids_input_dict['wind_plant_size_MW'] = hybrids_input_dict['num_turbines'] * \
                                                   hybrids_input_dict['turbine_rating_MW']

        hybrids_input_dict['hybrid_plant_size_MW'] = hybrids_input_dict['wind_plant_size_MW'] + \
                                                     hybrids_input_dict['solar_system_size_MW_DC']

        hybrids_input_dict['hybrid_construction_months'] = \
            hybrids_input_dict['wind_construction_time_months'] + \
            hybrids_input_dict['solar_construction_time_months']

        hybrid_results, wind_only, solar_only = run_hybridbosse(hybrids_input_dict)

        logger.info('Hybrid Dictionary Results: {}'.format(hybrid_results))
        logger.info('Wind Only Dictionary Results:'.format(wind_only))
        logger.info('Solar Only Dictionary Results:'.format(solar_only))

        wind_hybrid_bos_cost = hybrid_results['Wind_BOS_results']['total_bos_cost']
        solar_hybrid_bos_cost = hybrid_results['Solar_BOS_results']['total_bos_cost']
        hybrid_total_bos_cost = wind_hybrid_bos_cost + solar_hybrid_bos_cost #hybrid_results['hybrid']['hybrid_BOS_usd']

        return wind_hybrid_bos_cost, solar_hybrid_bos_cost, hybrid_total_bos_cost

    def _calculate_greenfield(self, wind_mw: float, solar_mw: float, interconnection_mw: float = 0):
        logger.info("Implemented")
        specify_construction_duration = False
        return HybridBOSSE._calculate(wind_mw, solar_mw, interconnection_mw, specify_construction_duration)

    def _calculate_solar_addition(self, wind_mw: float, solar_mw: float, interconnection_mw: float = 0):
        logger.error("Not yet implemented. Returning results for greenfield site")
        specify_construction_duration = False
        return HybridBOSSE._calculate(wind_mw, solar_mw, interconnection_mw, specify_construction_duration)

    def calculate_bos_costs(self, wind_mw, solar_mw, interconnection_mw, scenario_info):
        raise NotImplementedError
