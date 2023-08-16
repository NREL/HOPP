
import numpy as np
import traceback
from typing import Callable


# SIMULATION_ATTRIBUTES = ['annual_energies', 'generation_profile', 'internal_rate_of_returns',
#                          'lcoe_nom', 'lcoe_real', 'net_present_values', 'cost_installed',
#                          'total_revenues', 'capacity_payments', 'energy_purchases_values',
#                          'energy_sales_values', 'energy_values', 'benefit_cost_ratios']
#
# TOWER_ATTRIBUTES = ['dispatch', 'ssc_time_series']

def shrink_financial_model(model_dict):
    TIMESTEPS_YEAR = 8760
    shrink_keys = {'SystemOutput': ['gen', 'system_pre_curtailment_kwac'],
                   'Outputs':      ['gen_purchases', 'revenue_gen']}

    for main_key in shrink_keys.keys():
        for sub_key in  shrink_keys[main_key]:
            if sub_key in model_dict[main_key].keys():
                model_dict[main_key][sub_key] = model_dict[main_key][sub_key][:TIMESTEPS_YEAR]

    return model_dict


def expand_financial_model(model_dict):
    TIMESTEPS_YEAR = 8760
    ANALYSIS_PERIOD = len(model_dict['Outputs']['cf_annual_costs']) - 1
    shrink_keys = {'SystemOutput': ['gen', 'system_pre_curtailment_kwac'],
                   'Outputs': ['gen_purchases', 'revenue_gen']}

    for main_key in shrink_keys.keys():
        for sub_key in shrink_keys[main_key]:
            if sub_key in model_dict[main_key].keys():
                if len(model_dict[main_key][sub_key]) != TIMESTEPS_YEAR * ANALYSIS_PERIOD:
                    model_dict[main_key][sub_key] = model_dict[main_key][sub_key] * ANALYSIS_PERIOD

    return model_dict


class HybridSizingProblem():  # OptimizationProblem (unwritten base)
    """
    Problem class holding the design variable definitions and executing the HOPP simulation
    """
    sep = '__'
    DEFAULT_OPTIONS = dict(time_series_outputs=False, # add soc and curtailed series outputs
                           dispatch_factors=False,    # add dispatch factors to objective output
                           generation_profile=False,  # add technology generation profile to output
                           financial_model=False,     # add financial model dictionary to output
                           shrink_output=False,       # keep only the first year of output
                           )

    def __init__(self,
                 init_simulation: Callable,
                 design_variables: dict,
                 fixed_variables: dict = {},
                 output_options: dict = {}) -> None:
        """
        Create the problem instance, the simulation is not created until the objective is evaluated

        :param design_variables: Nested dictionary defining the design variables of the problem. 
            Each design variable requires an upper and lower bound
        
            Example:
            
            .. code-block::

                design_variables = dict(
                    pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
                              'tilt':                {'bounds':(30,      60)} 
                              },
                    battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
                              'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)}
                              })

        
        :param fixed_variables: Nested dictionary defining the fixed variables of the problem

            Example:

            .. code-block::

                fixed_variables = dict(pv = {'system_capacity_kw': 75*1e3})

        :return: None
        """

        self.simulation = None
        self.init_simulation = init_simulation
        self._parse_design_variables(design_variables, fixed_variables)
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(output_options)

    def _parse_design_variables(self,
                                design_variables: dict,
                                fixed_variables: dict) -> None:
        """
        Parse the nested dictionary structure into separate attributes

        :param design_variables: Nested dictionary defining the design variables of the problem
        :param fixed_variables: Nested dictionary defining the fixed variables of the problem
        :return: None
        """
        self.design_variables = design_variables
        self.fixed_variables = fixed_variables

        try:
            bounds = list()
            fields = list()
            field_set = set()
            fixed_values = list()

            for key, val in self.design_variables.items():
                for subkey, subval in val.items():

                    # Candidate field name, e.g., pv::tilt
                    field_name = self.sep.join([key, subkey])

                    # Check if field name has been repeated
                    if field_name in field_set:
                        raise Exception(f"{field_name} repeated in design variables")

                    # Assert that 'bounds' value is of length 2
                    field_bounds = subval['bounds']

                    num_bounds = len(field_bounds)
                    assert num_bounds == 2, f"{key}:{subkey} 'bounds' of length {num_bounds} not understood"

                    # Assert that 'bounds' first value (lower) is less than or equal to the second value (upper)
                    assert field_bounds[0] <= field_bounds[1], \
                        f"{key}:{subkey} invalid 'bounds': {field_bounds[0]}(lower) > {field_bounds[1]}(upper)"

                    field_set.add(field_name)
                    fields.append(field_name)
                    bounds.append(field_bounds)

            for key, val in self.fixed_variables.items():
                for subkey, subval in val.items():

                    # Candidate field name, e.g., pv::tilt
                    field_name = self.sep.join([key, subkey])

                    # # Check if field name has been repeated   (allow repeated in fixed variables)
                    # if field_name in field_set:
                    #     raise Exception(f"{field_name} repeated in design variables")

                    fields.append(field_name)
                    fixed_values.append(subval)

            self.candidate_fields = fields
            self.n_dim = len(fields)
            self.lower_bounds = np.array([bnd[0] for bnd in bounds])
            self.upper_bounds = np.array([bnd[1] for bnd in bounds])
            self.fixed_values = fixed_values

        except KeyError as error:
            raise KeyError(f"{key}:{subkey} needs simple bounds defined as 'bounds':(lower,upper)") from error

    def _check_candidate(self,
                         candidate: tuple) -> None:
        """
        Check if candidate tuple is valid for this problem.

        :param candidate: tuple containing field, value pairs in the order of self.fields and having values between the
            specified upper and lower bounds
        :return: None
        """
        # Assert that the correct number of field value pairs is present
        actual_length = len(candidate)
        assert actual_length == self.n_dim, \
            f"Expected candidate with {self.n_dim} (field,value) pairs, got candidate of length {actual_length}"

        # For each field value pair assert that the field name is correct and that the value is between the upper
        # and lower bounds
        for i, (field, value) in enumerate(candidate):
            if i == len(self.lower_bounds):
                break

            assert field == self.candidate_fields[i], \
                f"Expected field named {self.candidate_fields[i]} in position {i} of candidate, but found {field}"
            assert (value >= self.lower_bounds[i]) and (value <= self.upper_bounds[i]), \
                f"{field} invalid value ({value}), outside 'bounds':({self.lower_bounds[i]},{self.upper_bounds[i]})"

    def _set_simulation_to_candidate(self,
                                     candidate: tuple) -> None:
        """
        Set the simulation according to the provided design candidate

        :param candidate: Tuple containing field value pairs
        :return: None
        """
        for field,value in candidate:
            tech_key, key = field.split(self.sep)
            tech_model = getattr(self.simulation, tech_key)

            if hasattr(tech_model, key):
                setattr(tech_model, key, value)
            else:
                tech_model.value(key, value)
            
            # force consistent hybrid sizing
            # if tech_key == 'tower' and key == 'cycle_capacity_kw':
            #
            #     if 'battery' in self.simulation.power_sources.keys():
            #         batt_model = getattr(self.simulation, 'battery')
            #
            #         csp_cycle = value
            #         total_batt = (100*1e3) - csp_cycle
            #
            #         key = 'system_capacity_kw'
            #
            #         if hasattr(batt_model, key):
            #             setattr(batt_model, key, total_batt)
            #         else:
            #             batt_model.value(key, total_batt)
                        
            # elif tech_key == 'pv' and key == 'dc_ac_ratio':
            #
            #     total_pv = (100*1e3) * value
            #
            #     key = 'system_capacity_kw'
            #
            #     if hasattr(tech_model, key):
            #         setattr(tech_model, key, total_pv)
            #     else:
            #         tech_model.value(key, total_pv)
               

    def candidate_from_array(self, values: np.array) -> tuple:
        """
        Create a tuple of field value pairs according to the problem's design variables given an array of values

        :param values: An array of values in the problem's native units representing a design candidate
        :return: A candidate tuple of field value pairs, where values have been rounded to the variable's precision
        """
        # Round the values according to the provided precision value
        # rounded_values = [np.round(x, decimals=-self.precision[i]) for i,x in enumerate(values)] + self.fixed_values
        candidate = tuple([(field, val)
                           for field, val in zip(self.candidate_fields, np.append(values, self.fixed_values))])
        return candidate

    def candidate_from_unit_array(self, values: np.array) -> tuple:
        """
        Create a tuple of field value pairs according to the problem's design variables given an array of values

        :param values: An array of unit values representing a design candidate
        :return: A candidate tuple of field value pairs, where values have been rounded to the variable's precision
        """
        # Scale the unit input array to the problem's units
        scaled_values = values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

        # Round the values according to the provided precision value
        # rounded_values = [np.round(x, decimals=-self.precision[i]) for i, x in enumerate(scaled_values)] + self.fixed_values
        candidate = tuple([(field, val)
                           for field,val in zip(self.candidate_fields, np.append(scaled_values, self.fixed_values))])
        return candidate

    def evaluate_objective(self, candidate: tuple) ->  dict:
        """
        Set the simulation to the design candidate provided, evaluate the objective, build out a nested dictionary of
        results from that simulation. One or more of these results can then represent the objective of the problem.

        :param candidate: A tuple of field value pairs representing a design candidate for this problem
        :return:
        """
        try:
            # init dictionary to hold simulation output
            # result = dict()
            result = {field: val for field, val in candidate}

            ## We are doing this because it ensures we start from a clean plant state
            # if the simulation has been initialized, then delete
            if self.simulation is not None:
                del self.simulation

            # Initialize (or re-initialize the simulation)
            self.simulation = self.init_simulation()

            # Check if valid candidate, update simulation, execute simulation
            self._check_candidate(candidate)
            self._set_simulation_to_candidate(candidate)
            
            # return
            self.simulation.simulate()

            result.update(self.simulation.hybrid_simulation_outputs().copy())
            
            if self.options['time_series_outputs']:                
                # CSP TES SOC
                if 'tower' in self.simulation.power_sources.keys():
                    model = self.simulation.power_sources['tower']
                    result['tes_soc'] = model.outputs.ssc_time_series['e_ch_tes'][:8760]
                    result['rec_thermal'] = model.outputs.ssc_time_series['Q_thermal'][:8760]
                    
                # Battery SOC
                if 'battery' in self.simulation.power_sources.keys():
                    model = self.simulation.power_sources['battery']
                    result['bat_soc'] = model.Outputs.SOC[:8760]
                
                # Curtailment
                if 'grid' in self.simulation.power_sources.keys():
                    model = self.simulation.power_sources['grid']
                    result['curtailed'] = model.generation_curtailed[:8760]

            if self.options['generation_profile']:
                for source in self.simulation.power_sources.keys():
                    attr = 'generation_profile'
                    o_name = source.capitalize() + ' Generation Profile (MWh)'
                    try:
                        result[o_name] = getattr(getattr(self.simulation, attr), source) # list
                    except AttributeError:
                        continue

                if self.simulation.site.follow_desired_schedule:
                    result['Desired Schedule'] = self.simulation.site.desired_schedule

            if self.options['financial_model']:
                for source, model in self.simulation.power_sources.items():
                    attr = '_financial_model'
                    o_name = source.capitalize() + attr
                    try:
                        temp = getattr(model, attr).export() # dict

                        if self.options['shrink_output']:
                            result[o_name] = shrink_financial_model(temp)
                        else:
                            result[o_name] = temp
                    except AttributeError:
                        continue 

            # Add the dispatch factors, which are the pricing signal in the optimization problem
            if self.options['dispatch_factors']:
                result['dispatch_factors'] = self.simulation.dispatch_factors

        except Exception:
            # Some exception occurred while evaluating the objective, capture and document in the output
            err_str = traceback.format_exc()
            result['exception'] = err_str

            print(f'Candidate:\n{candidate}\n')
            print(f'produced an exception:\n{err_str}\n')


        return result