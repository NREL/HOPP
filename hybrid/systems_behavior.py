"""
systems_behavior.py

Defines what happens when various technology or financial models
are executed, using the SAM python interface.
"""

from collections import OrderedDict
import defaults.wind_singleowner as wind_defaults
from parameters.bos_json_lookup import bos_json_lookup_custom

import os
path_file = os.path.dirname(os.path.abspath(__file__))
path_parameters = os.path.join(path_file, '..', 'parameters')

def technologies():
    """
    Define which technologies are currently available to hybridize
    Technologies run serially.
    The Generic technology always runs after any other generator models.
    The StandAloneBattery technology runs after the generic system aggregates the power signal.
    The Grid technology runs last to enforce the interconnection agreement.
    The financial outputs are based upon the last model run in the chain.
    """
    return ['Solar', 'Wind', 'Geothermal', 'Generic', 'Battery', 'Grid']

def get_available_models():
    """
    Define the mapping of models in SAM with their respective names
    for technology models and financial model names defined through pySAM interface
    (currently all technologies use the Singleowner financial model in the hybrid project)
    """
    models = dict()
    for tech in technologies():
        models[tech] = dict()
        models[tech]['financial_model'] = 'Singleowner'
        if tech == 'Solar':
            models[tech]['technology_model'] = 'Pvsamv1'
        elif tech == 'Wind':
            models[tech]['technology_model'] = 'Windpower'
        elif tech == 'Geothermal':
            models[tech]['technology_model'] = 'Geothermal'
        elif tech == 'Generic':
            models[tech]['technology_model'] = 'GenericSystem'
        elif tech == 'Battery':
            models[tech]['technology_model'] = 'StandAloneBattery'
        elif tech == 'Grid':
            models[tech]['technology_model'] = 'Grid'

    return models

def get_system_behavior_fx(technologies_to_run):
    """
    Define the actual technologies that will get run

    technologies_to_run : list
       list of desired technologies to run, which is a subset of
        available models

    """
    systems = OrderedDict()
    if 'Solar' in technologies_to_run:
        systems['Solar'] = run_solar_models
    if 'Wind' in technologies_to_run:
        systems['Wind'] = run_wind_models
    if 'Geothermal' in technologies_to_run:
        systems['Geothermal'] = run_geothermal_models
    if 'Generic' in technologies_to_run:
        systems['Generic'] = run_hybrid_models
    if 'Battery' in technologies_to_run:
        systems['Battery'] = run_battery_models
    if 'Grid' in technologies_to_run:
        systems['Grid'] = run_grid_models
    return systems

def run_simple(systems, technology):
    """
    Define what models to call to run a simple technology
    Parameters
    ----------
    systems : dict
       Dictionary consisting of information about the model chain required for running a simulation
    technology : str
       String of technology to run
    """
    models = get_available_models()
    model = systems[technology]
    technology_model = model[models[technology]['technology_model']]
    financial_model = model[models[technology]['financial_model']]
    technology_model.execute()
    financial_model.SystemOutput.gen = technology_model.Outputs.gen
    financial_model.execute()

def run_solar_models(systems):
    """
    Define what models to call to run a PV system
    Parameters
    ----------
    systems : dict
        Dictionary consisting of information about the model chain required for running a PV simulation
    """
    run_simple(systems, 'Solar')

def run_wind_models(systems):
    """
    Define what models to call to run a Wind system
    Parameters
    ----------
    systems : dict
        Dictionary consisting of information about the model chain required for running a Wind simulation
    """
    technology = 'Wind'
    models = get_available_models()
    model = systems[technology]
    technology_model = model[models[technology]['technology_model']]
    financial_model = model[models[technology]['financial_model']]


    # calculate system capacity.  To evaluate other turbines, update the defaults dictionary
    technology_model.Turbine.calculate_powercurve(wind_defaults.wind_default_rated_output,
                                                  wind_defaults.wind_windsingleowner['Turbine']['wind_turbine_rotor_diameter'],
                                                  wind_defaults.wind_default_max_tip_speed,
                                                  wind_defaults.wind_default_max_tip_speed_ratio,
                                                  wind_defaults.wind_default_cut_in_speed,
                                                  wind_defaults.wind_default_cut_out_speed,
                                                  wind_defaults.wind_default_drive_train)

    # Note, the wind farm coordinates should be optimized and added to/removed before this point
    technology_model.Farm.system_capacity = max(technology_model.Turbine.wind_turbine_powercurve_powerout) \
                                     * len(technology_model.Farm.wind_farm_xCoordinates)
    
    technology_model.execute()
    financial_model.SystemOutput.gen = technology_model.Outputs.gen
    financial_model.execute()

def run_geothermal_models(systems):
    """
       Define what models to call to run a Geothermal system
       Parameters
       ----------
       systems : dict
           Dictionary consisting of information about the model chain required for running a Geothermal simulation
       """
    run_simple(systems, 'Geothermal')


def run_grid_models(systems):
    """
       Define what models to call to run the grid model, which enforces interconnection.
       The grid model requires the 'Generic' model to be enabled.
       Parameters
       ----------
       systems : dict
           Dictionary consisting of information about the model chain required for running a Geothermal simulation
    """
    models = get_available_models()
    model_generic = systems['Generic']
    financial_generic = model_generic[models['Generic']['financial_model']]

    technology = 'Grid'
    model = systems[technology]
    technology_model = model[models[technology]['technology_model']]
    financial_model = model[models[technology]['financial_model']]
    technology_model.Common.gen = financial_generic.SystemOutput.gen
    technology_model.execute()
    financial_model.SystemOutput.gen = technology_model.Outputs.gen
    financial_model.TaxCreditIncentives.ptc_fed_amount = financial_generic.TaxCreditIncentives.ptc_fed_amount
    financial_model.TaxCreditIncentives.itc_fed_percent = financial_generic.TaxCreditIncentives.itc_fed_percent
    financial_model.execute()

def run_battery_models(systems):
    """
    Define what models to call to run a battery system.
    The battery model requires the 'Generic' model to be enabled
    Parameters
    ----------
    systems : dict
        Dictionary consisting of information about the model chain required for running a Battery simulation
    """
    models = get_available_models()
    model_generic = systems['Generic']
    financial_generic = model_generic[models['Generic']['financial_model']]

    technology = 'Battery'
    model = systems[technology]
    technology_model = model[models[technology]['technology_model']]
    financial_model = model[models[technology]['financial_model']]
    technology_model.System.gen = financial_generic.SystemOutput.gen

    # ensure that battery model singleowner representation has been updated with appropriate prices
    technology_model.TimeOfDelivery.ppa_price_input = financial_model.PPAPrice.ppa_price_input
    technology_model.TimeOfDelivery.ppa_multiplier_model = financial_model.TimeOfDelivery.ppa_multiplier_model
    technology_model.TimeOfDelivery.dispatch_factors_ts = financial_model.TimeOfDelivery.dispatch_factors_ts
    technology_model.TimeOfDelivery.dispatch_sched_weekday = financial_model.TimeOfDelivery.dispatch_sched_weekday
    technology_model.TimeOfDelivery.dispatch_sched_weekend = financial_model.TimeOfDelivery.dispatch_sched_weekend
    dispatch_tod_factors = list()
    for i in range(1, 10):
        dispatch_tod_factors.append(getattr(financial_model.TimeOfDelivery,  'dispatch_factor' + str(i)))
    technology_model.TimeOfDelivery.dispatch_tod_factors = dispatch_tod_factors

    technology_model.execute()
    financial_model.SystemOutput.gen = technology_model.System.gen

    financial_generic.SystemCosts.total_installed_cost += financial_model.SystemCosts.total_installed_cost
    #financial_generic.TaxCreditIncentives.itc_fed_percent = hybrid_itc

    financial_model.execute()

def run_hybrid_models(systems):
    """
   Define model behavior.  Contains functions to merge the generation of
   technologies and their incentives

   Parameters
   ----------
   systems : OrderedDict
       dict consisting of information about the model chain required for running a generic simulation
       the last inserted item into the OrderedDict should be the 'Generic' sub-dict
    """
    models = get_available_models()
    applied_wind_solar_bos_model = False

    # ensure generic technology is run last and initialized
    if 'Generic' in systems:
        model_generic = systems['Generic']
        technology_generic = model_generic[models['Generic']['technology_model']]
        financial_generic = model_generic[models['Generic']['financial_model']]
        financial_generic.SystemCosts.total_installed_cost = 0
        technology_generic.Plant.system_capacity = 0
        technology_generic.Plant.spec_mode = 1
        technology_generic.Plant.energy_output_array = 8760 * [0] # could generalize for subhourly
        financial_generic.TaxCreditIncentives.ptc_fed_amount = [0]
        financial_generic.TaxCreditIncentives.itc_fed_amount = 0
        financial_generic.SystemCosts.total_installed_cost = 0

    for technology in systems:
        if technology != 'Generic' and technology != 'Grid' and technology != 'Battery':
            model = systems[technology]
            technology_model = model[models[technology]['technology_model']]
            financial_model = model[models[technology]['financial_model']]

            # do not need to rerun since already executed
            gen = 8760 * [0]
            if hasattr(technology_model.Outputs, 'gen'):
                gen = technology_model.Outputs.gen

            if 'Generic' in technologies():
                generation_kw = gen
                generation_generic_kw = technology_generic.Plant.energy_output_array
                generation_generic_kw = [generation_kw[i] + generation_generic_kw[i] for i in range(len(generation_kw))]
                technology_generic.Plant.system_capacity += max(gen)
                technology_generic.Plant.energy_output_array = generation_generic_kw

                if technology == 'Wind' or technology == 'Solar':
                    print('Calculating Wind and Solar Costs using bos_json_lookup')
                    wind_solar_total_installed_cost = calculate_wind_solar_costs(systems)
                    #wind_solar_total_installed_cost = 1
                    # apply the BOS model for wind and solar costs once
                    if not applied_wind_solar_bos_model:
                        if wind_solar_total_installed_cost > 0:
                            print("Wind and Solar costs were available in lookup and were used")
                            financial_generic.SystemCosts.total_installed_cost += wind_solar_total_installed_cost
                            applied_wind_solar_bos_model = True
                        else:
                            print("***Wind and Solar costs were not available in BOS lookup and were not used***")
                            financial_generic.SystemCosts.total_installed_cost += financial_model.SystemCosts.total_installed_cost
                else:
                    print("***Wind and Solar costs were not available in BOS lookup and were not used***")
                    financial_generic.SystemCosts.total_installed_cost += financial_model.SystemCosts.total_installed_cost

                financial_generic.SystemOutput.gen = technology_generic.Plant.energy_output_array


    # execute generic performance model to get annual energy and generation
    technology_generic.execute()

    # Scale the PTC/ITC contributions for the generic financial model
    hybrid_ptc = hybrid_itc = 0

    for technology in systems:
        if technology != 'Generic' and technology != 'Grid' and technology != 'Battery':
            model = systems[technology]
            technology_model = model[models[technology]['technology_model']]
            financial_model = model[models[technology]['financial_model']]

            annual_energy_percent = 0
            if technology_generic.Outputs.annual_energy > 0:
                annual_energy_percent = technology_model.Outputs.annual_energy / technology_generic.Outputs.annual_energy
            cost_percent = financial_model.SystemCosts.total_installed_cost / financial_generic.SystemCosts.total_installed_cost
            hybrid_ptc += financial_model.TaxCreditIncentives.ptc_fed_amount[0] * annual_energy_percent
            hybrid_itc += financial_model.TaxCreditIncentives.itc_fed_percent * cost_percent

        else:
            financial_generic.TaxCreditIncentives.ptc_fed_amount = [hybrid_ptc]
            financial_generic.TaxCreditIncentives.itc_fed_percent = hybrid_itc
            financial_generic.execute()

def calculate_wind_solar_costs(systems):

    wind_capacity = 0
    solar_capacity = 0
    scenario_type = 'Variable Ratio Wind and Solar Greenfield'
    models = get_available_models()

    # Assign capacity
    if 'Solar' in systems:
        technology = 'Solar'
        model = systems[technology]
        solar_capacity = int(model[models['Solar']['technology_model']].SystemDesign.system_capacity / 1000)
    if 'Wind' in systems:
        technology = 'Wind'
        model = systems[technology]
        #wind_capacity = int(model[models['Wind']['technology_model']].Farm.system_capacity / 1000)
        wind_capacity = int(model[models['Wind']['financial_model']].SystemOutput.system_capacity / 1000)
        print('Wind capacity detected by BOS cost lookup is: ', wind_capacity)
    else:
        scenario_type = 'Solar Only (Take BOS from Wind)'

    print("BOS model costs: Wind size: " + str(wind_capacity) + " Solar size: " + str(solar_capacity))


    search_matrix = {"Scenario Type": scenario_type,
                     "Wind Installed Capacity": [wind_capacity],
                     "Solar Installed Capacity": [solar_capacity]}  # Dictionary of search parameters and values
    desired_output_parameters = ["Total Project Cost"]  # List of desired output parameters
    #json_file_path = os.path.join(path_parameters, 'BOSSummaryResults.json')
    #json_file_path = os.path.join(path_parameters, 'BOSSummaryResults_new_scenarios_2.json')
    json_file_path = os.path.join(path_parameters, 'BOSSummaryResults_300MW.json')
    list_of_matches = bos_json_lookup_custom(json_file_path, search_matrix, desired_output_parameters)

    if len(list_of_matches) > 0:
        return list_of_matches[0][desired_output_parameters[0]]
    else:
        print('Found no matches in BOS model, returning 0')
        return 0
