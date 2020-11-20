"""
systems_behavior.py

Defines what happens when various technology or financial models
are executed, using the SAM python interface.
"""

from collections import OrderedDict
import defaults.wind_singleowner as wind_defaults
from parameters.bos_json_lookup import bos_json_lookup_custom
import os

#Landbosse functionality:
# from landbosse.model.Manager import Manager
# from defaults.defaults_data import windBOS_defaults
# from hybrid.optimal_collection_matrix import Graph
# import pandas as pd
# from openpyxl import load_workbook

path_file = os.path.dirname(os.path.abspath(__file__))
path_parameters = os.path.join(path_file, '..', 'parameters')

def update_BOS_excel(windBOS_outputs, plot_path):
    """ Append BOS costs to BOS excel file"""

    windBOS_out_df = pd.DataFrame({'Collection Cost': windBOS_outputs['sum_collection_cost'], 'Erection Cost': windBOS_outputs['sum_erection_cost'],
                                   'Development Cost': windBOS_outputs['sum_development_cost'], 'Foundation Cost': windBOS_outputs['sum_foundation_cost'],
                                   'Grid Connection Cost': windBOS_outputs['sum_grid_connection_cost'], 'Management Cost': windBOS_outputs['total_management_cost'],
                                   'Substation Cost': windBOS_outputs['sum_substation_cost'], 'Road Cost': windBOS_outputs['sum_road_cost']}, index=[0])
    windBOS_out_df['Total Cost'] = windBOS_out_df.sum(axis=1)
    writer = pd.ExcelWriter(plot_path, engine='openpyxl')
    writer.book = load_workbook(plot_path)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(plot_path)
    windBOS_out_df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1, startcol=1)
    writer.close()

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

def run_solar_models(systems,BOSCostSource):
    """
    Define what models to call to run a PV system
    Parameters
    ----------
    systems : dict
        Dictionary consisting of information about the model chain required for running a PV simulation
    """
    run_simple(systems, 'Solar')

def run_wind_models(systems,BOSCostSource):
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

def run_geothermal_models(systems,BOSCostSource):
    """
       Define what models to call to run a Geothermal system
       Parameters
       ----------
       systems : dict
           Dictionary consisting of information about the model chain required for running a Geothermal simulation
       """
    run_simple(systems, 'Geothermal')


def run_grid_models(systems,BOSCostSource):
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

def run_battery_models(systems,BOSCostSource):
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

def run_hybrid_models(systems,bos_details):
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
    # print(bos_details)
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
                    #print('Calculating Wind and Solar Costs using bos_json_lookup')
                    wind_solar_total_installed_cost = calculate_wind_solar_costs(systems, bos_details)

                    # apply the BOS model for wind and solar costs once
                    if not applied_wind_solar_bos_model:
                        if wind_solar_total_installed_cost > 0:
                            #print("Wind and Solar costs were available and were used")
                            financial_generic.SystemCosts.total_installed_cost += wind_solar_total_installed_cost
                            applied_wind_solar_bos_model = True
                        else:
                            #print("***Wind and Solar costs were not available in BOS lookup and were not used***")
                            financial_generic.SystemCosts.total_installed_cost += financial_model.SystemCosts.total_installed_cost
                else:
                    #print("***Wind and Solar costs were not available in BOS lookup and were not used***")
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

def calculate_wind_solar_costs(systems, bos_details):

    wind_capacity = 0
    solar_capacity = 0
    # scenario_type = 'Variable Ratio Wind and Solar Greenfield'
    models = get_available_models()
    fixedBOSCostWind = 15000000
    fixedBOSCostSolar = 5000000
    fixedBOSCostHybrid = 1000000
    solar_cost_per_mw = 1100000
    wind_cost_per_mw = 1450000
    # Assign capacity & Cost

    if 'Solar' in systems:
        technology = 'Solar'
        model = systems[technology]
        solar_capacity = int(model[models['Solar']['technology_model']].SystemDesign.system_capacity / 1000)
        solar_cost = int(model[models['Solar']['technology_model']].SystemDesign.system_capacity / 1000) * solar_cost_per_mw
    if 'Wind' in systems:
        technology = 'Wind'
        model = systems[technology]
        #wind_capacity = int(model[models['Wind']['technology_model']].Farm.system_capacity / 1000)
        wind_capacity = int(model[models['Wind']['financial_model']].SystemOutput.system_capacity / 1000)
        wind_cost = int(model[models['Wind']['financial_model']].SystemOutput.system_capacity / 1000) * wind_cost_per_mw
    else:
        scenario_type = 'Solar Only (Take BOS from Wind)'

    # print("BOS model costs: Wind size: " + str(wind_capacity) + " Solar size: " + str(solar_capacity))

    # bos_details['BOSSource'] = 'JSONLookup' #Options: Cost/MW, JSONLookup
    # bos_details['BOSFile'] = 'UPDATED_BOS_Summary_Results.json'
    # bos_details['BOSScenario'] = 'Wind Only' #Options: Wind Only, Solar Only, Variable Ratio Wind and Solar Greenfield, Solar Addition
    # bos_details['BOSScenarioDescription'] = '' #Options: Overbuild


    wind_solar_total_installed_cost = 0

    # Case 1: Cost/MW
    if bos_details['BOSSource'] == 'Cost/MW':
        #print('Determining total project cost using Cost/MW metric')
        if 'Solar' in systems:
            wind_solar_total_installed_cost = solar_cost + fixedBOSCostSolar
        if 'Wind' in systems:
            wind_solar_total_installed_cost = wind_cost + fixedBOSCostWind
        if 'Wind' in systems and 'Solar' in systems:
            wind_solar_total_installed_cost = solar_cost + wind_cost + fixedBOSCostHybrid
        return wind_solar_total_installed_cost

    # Case 2: JSON Lookup
    elif bos_details['BOSSource'] == 'JSONLookup':
        print('Using JSON Lookup')
        json_filename = bos_details['BOSFile']
        json_file_path = os.path.join(path_parameters, json_filename)
        search_matrix = {"Scenario Type": bos_details['BOSScenario'],
                         "Scenario Description": bos_details['BOSScenarioDescription'],
                         "Wind Installed Capacity": [wind_capacity],
                         "Solar Installed Capacity": [solar_capacity]}  # Dictionary of search parameters and values
        desired_output_parameters = ["Total Project Cost", "Wind Project Cost",
                                     "Solar Project Cost", "Wind BOS Cost",
                                     "Solar BOS Cost"]  # List of desired output parameters

        list_of_matches = bos_json_lookup_custom(json_file_path, search_matrix, desired_output_parameters)

        if len(list_of_matches) > 0:
            print('Found a BOS match')
            # Split to Individual Results
            total_project_cost = list_of_matches[0][desired_output_parameters[0]]
            wind_project_cost = list_of_matches[0][desired_output_parameters[1]]
            wind_bos_cost = list_of_matches[0][desired_output_parameters[3]]
            wind_installed_cost = wind_project_cost - wind_bos_cost
            solar_project_cost = list_of_matches[0][desired_output_parameters[2]]
            solar_bos_cost = list_of_matches[0][desired_output_parameters[4]]
            solar_installed_cost = solar_project_cost - solar_bos_cost
            print("Total Project Cost: {} Wind Project Cost: {} "
                  "Solar Project Cost: {} Wind BOS Cost: {} Solar BOS Cost {}".
                  format(total_project_cost, wind_project_cost, solar_project_cost, wind_bos_cost, solar_bos_cost))

            if bos_details['Modify Costs']:
                # Modify results using selected modifiers
                print("Total Project Cost Before Modifiers: {}".format(total_project_cost))
                if 'Solar' in systems:
                    wind_solar_total_installed_cost = ((1 - bos_details['solar_capex_reduction']) *
                                                       solar_installed_cost) + \
                                                      ((1 - bos_details['solar_bos_reduction']) * solar_bos_cost)
                if 'Wind' in systems:
                    wind_solar_total_installed_cost = ((1 - bos_details['wind_capex_reduction']) *
                                                       wind_installed_cost) + \
                                                      ((1 - bos_details['wind_bos_reduction']) * wind_bos_cost)
                if 'Wind' in systems and 'Solar' in systems:
                    wind_solar_total_installed_cost = ((1 - bos_details['solar_capex_reduction_hybrid']) *
                                                       solar_installed_cost) + \
                                                      ((1 - bos_details['solar_bos_reduction_hybrid']) * solar_bos_cost) +\
                                                      ((1 - bos_details['wind_capex_reduction_hybrid']) *
                                                       wind_installed_cost) + \
                                                      ((1 - bos_details['wind_bos_reduction_hybrid']) * wind_bos_cost)
                print("Total Project Cost After Modifiers: {}".format(wind_solar_total_installed_cost))
            else:
                # Not modifying wind or solar costs, return total project cost as provided by JSON
                wind_solar_total_installed_cost = total_project_cost

            return wind_solar_total_installed_cost

        else:
            print('Found no matches in BOS model, returning 0')
            return 0

    # Case 3: Cost/MW for Solar addition scenario
    elif bos_details['BOSSource'] == 'Solar Addition':
        # print('Determining total project cost using Cost/MW metric')
        if 'Solar' in systems:
            wind_solar_total_installed_cost += solar_cost
        if 'Wind' in systems:
            wind_solar_total_installed_cost += wind_cost + fixedBOSCostWind
        if 'Wind' in systems and 'Solar' in systems:
            wind_solar_total_installed_cost = solar_cost + wind_cost + fixedBOSCostHybrid
        return wind_solar_total_installed_cost

    # Case 4: HybridBOSSE
    elif bos_details['BOSSource'] == 'HybridBOSSE':
        print("<<<HybridBOSSE Costs>>>")
        print("Total Wind Cost: ", bos_details['total_wind_cost'])
        print("Total Solar Cost: ", bos_details['total_solar_cost'])
        print("Total Hybrid Cost: ", bos_details['total_hybrid_cost'])

        if 'Solar' in systems:
            scenario = 'Solar'
            wind_solar_total_installed_cost = bos_details['total_solar_cost']
        if 'Wind' in systems:
            scenario = 'Wind'
            wind_solar_total_installed_cost = bos_details['total_wind_cost']
        if 'Wind' in systems and 'Solar' in systems:
            scenario = 'Hybrid'
            wind_solar_total_installed_cost = bos_details['total_hybrid_cost']

        print("Total Project Cost for {} scenario using HybridBOSSE: {}".
              format(scenario, wind_solar_total_installed_cost))
        return wind_solar_total_installed_cost
