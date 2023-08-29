# extra function in the osw_h2 file
from math import floor
import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
import yaml
import re
# from yamlinclude import YamlIncludeConstructor
from pathlib import Path

# PATH = Path(__file__).parent
# YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / 'floris_input_files/')


# HOPP functionss
from hopp.to_organize.H2_Analysis.hopp_for_h2 import hopp_for_h2
from hopp.simulation.technologies.sites import SiteInfo
from hopp.to_organize.H2_Analysis.simple_dispatch import SimpleDispatch
from hopp.to_organize.H2_Analysis.compressor import Compressor
from hopp.simulation.technologies.hydrogen.desal.desal_model import RO_desal
from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage import Underground_Pipe_Storage
import hopp.simulation.technologies.hydrogen.electrolysis.run_h2_PEM as run_h2_PEM
from lcoe.lcoe import lcoe as lcoe_calc
import numpy_financial as npf

def set_site_info(site_df, sample_site):

    # turbinesheet = turbine_model[-4:]
    # scenario_df = xl.parse(turbinesheet)
    # scenario_df.set_index(["Parameter"], inplace = True)

    # site_df = scenario_df[site_location]

    latlon = site_df['Representative coordinates']
    lat, lon = (latlon.split(','))
    lat = float(lat)
    lon = float(lon)
    sample_site['lat'] = lat
    sample_site['lon'] = lon

    return site_df, sample_site


def set_financial_info(scenario,
                       debt_equity_split,
                       discount_rate):

    scenario['Debt Equity'] = debt_equity_split
    scenario['Discount Rate'] = discount_rate

    return scenario

def set_electrolyzer_info(atb_year):

    ### This is just a test!!!! Need to get exact numbers
    if atb_year == 2020:
        electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
        time_between_replacement = 40000    #[hrs]
    elif atb_year == 2025:
        electrolyzer_capex_kw = 300
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2030:
        electrolyzer_capex_kw = 150
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2050:
        electrolyzer_capex_kw = 100
        time_between_replacement = 80000    #[hrs]

    #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
    # if atb_year == 2022:
    #     electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
    #     time_between_replacement = 40000    #[hrs]
    # elif atb_year == 2025:
    #     electrolyzer_capex_kw = 300
    #     time_between_replacement = 80000    #[hrs]
    # elif atb_year == 2030:
    #     electrolyzer_capex_kw = 150
    #     time_between_replacement = 80000    #[hrs]
    # elif atb_year == 2035:
    #     electrolyzer_capex_kw = 100
    #     time_between_replacement = 80000    #[hrs]

    return electrolyzer_capex_kw, time_between_replacement

def set_turbine_model(turbine_model, scenario, parent_path, floris_dir, floris):
    if floris == True:
        # Define Turbine Characteristics based on user selected turbine.
        ########## TEMPERARY ###########
        site_number = 'base'
        site_number = 'singleT'
        site_number = 'lbw' #'osw'
        ################################

        turbine_file = floris_dir + 'floris_input' + turbine_model + '_' + site_number + '.yaml'
        with open(turbine_file, 'r') as f:
            floris_config = yaml.load(f, yaml.FullLoader)
            # floris_config = yaml.load(f, yaml.SafeLoader)
        nTurbs = len(floris_config['farm']['layout_x'])
        # turbine_type = floris_config['farm']['turbine_type'][0]
        turbine_type = floris_config['farm']['turbine_type'][0]['turbine_type']
        # print(floris_config['farm']['turbine_type'][0]['turbine_type'])
        turbine_rating_mw = float(re.findall('[0-9]+', turbine_type)[0])
    else:
        floris_config = 0
        turbine_rating_mw = float(re.findall('[0-9]+', turbine_model)[0])
        # TODO: replace nTurbs placeholder value with real value
        nTurbs = 0



    # Scaled from reference 15MW turbine: https://github.com/IEAWindTask37/IEA-15-240-RWT
    if turbine_model == '12MW':
        custom_powercurve_path = '2022atb_osw_12MW.csv'
        tower_height = 136
        rotor_diameter = 215

    elif turbine_model == '15MW':
        custom_powercurve_path = '2022atb_osw_15MW.csv'
        tower_height = 150
        rotor_diameter = 240

    elif turbine_model == '18MW':
        custom_powercurve_path = '2022atb_osw_18MW.csv'
        tower_height = 161
        rotor_diameter = 263

    elif turbine_model == '4MW':
        #TODO: replace with correct power curve
        custom_powercurve_path = '2020ATB_NREL_Reference_7MW_200.csv'
        tower_height = 130
        rotor_diameter = 185

    elif turbine_model == '6MW':
        #TODO: replace with correct power curve
        custom_powercurve_path = '2020ATB_NREL_Reference_7MW_200.csv'
        tower_height = 115
        rotor_diameter = 170

    elif turbine_model == '8MW':
        #TODO: replace with correct power curve
        custom_powercurve_path = '2020ATB_NREL_Reference_7MW_200.csv'
        tower_height = 160
        rotor_diameter = 225

    scenario['Tower Height'] = tower_height
    scenario['Turbine Rating'] = turbine_rating_mw
    scenario['Powercurve File'] = custom_powercurve_path
    scenario['Rotor Diameter'] = rotor_diameter

    # print("Powercurve Path: ", custom_powercurve_path)

    return scenario, nTurbs, floris_config #custom_powercurve_path, tower_height, rotor_diameter, turbine_rating_mw, wind_cost_kw, floating_cost_reductions_df, fixed_cost_reductions_df

def set_export_financials(wind_size_mw,
                        wind_cost_kw,
                        wind_om_cost_kw,
                        useful_life,
                        site_df):
    # HVDC Export System
    export_system_cost_kw = site_df['Export System'] + site_df['Offshore Substation']
    export_system_installation_cost_kw = site_df['Export System Installation'] + site_df['Offshore Substation Installation']
    total_export_system_cost_kw = export_system_cost_kw + export_system_installation_cost_kw

    wind_cost_kw = wind_cost_kw - total_export_system_cost_kw # Wind System Cost per KW ($US/kW) with no HVDC export system

    # Export System CapEx $US
    export_system_cost = export_system_cost_kw * wind_size_mw * 1000
    export_system_installation_cost = export_system_installation_cost_kw * wind_size_mw * 1000
    total_export_system_cost = export_system_cost + export_system_installation_cost

    # Rough OpEx Estimation
    # https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub = 0.5% CapEx per lifetime for offshore cables
    export_om_cost_kw = 0.5/100 * total_export_system_cost_kw / useful_life  # US/kW-yr (assume 30 year lifetime)

    wind_om_cost_kw = wind_om_cost_kw - export_om_cost_kw # Wind System OM Cost with no HVDC OM cost estimates

    total_export_om_cost = 0.5/100 * total_export_system_cost / useful_life # $US total (assume 30 year lifetime))

    return wind_cost_kw, wind_om_cost_kw, total_export_system_cost, total_export_om_cost

def set_turbine_financials(turbine_model,
                            fixed_or_floating_wind,
                            atb_year,
                            wind_cost_kw,
                            wind_om_cost_kw,
                            wind_net_cf,
                            parent_path):

    # Define Turbine Characteristics based on user selected turbine.
    # Scaled from reference 15MW turbine: https://github.com/IEAWindTask37/IEA-15-240-RWT
    if turbine_model == '12MW':
        # Future Cost Reduction Estimates - ATB 2022: Class 4 Fixed, Class 11 Float
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_12MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_12MW.csv'))

    elif turbine_model == '15MW':
        # Future Cost Reduction Estimates
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_15MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_15MW.csv'))

    elif turbine_model == '18MW':
        # Future Cost Reduction Estimates
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_18MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_18MW.csv'))

    #Display Future Cost Reduction Estimates per turbine
    # Fixed Wind Cost Reductions
    if fixed_or_floating_wind == 'Fixed - Monopile':
        capex_reduction = fixed_cost_reductions_df[str(atb_year)][0]
        opex_reduction = fixed_cost_reductions_df[str(atb_year)][1]
        net_cf_increase = fixed_cost_reductions_df[str(atb_year)][2]
    # Floating Wind Cost Reductions
    elif fixed_or_floating_wind == 'Floating - semisubmersible':
        capex_reduction = floating_cost_reductions_df[str(atb_year)][0]
        opex_reduction = floating_cost_reductions_df[str(atb_year)][1]
        net_cf_increase = floating_cost_reductions_df[str(atb_year)][2]

    # print("For {} wind in {}, capex reduction is estimated to be: {}, opex reduction is: {}, and net capacity factor increase is: {}.".format(fixed_or_floating_wind, str(atb_year), capex_reduction, opex_reduction, net_cf_increase))

    new_wind_cost_kw = wind_cost_kw * (100-float(capex_reduction[:-1]))/100
    new_wind_om_cost_kw = wind_om_cost_kw * (100-float(opex_reduction[:-1]))/100
    new_wind_net_cf = wind_net_cf * (100+float(net_cf_increase[:-1]))/100

    print_wind_info = False
    if print_wind_info:
        print("Wind Cost in baseline year was {}, reduced to {} in {}".format(wind_cost_kw, new_wind_cost_kw, atb_year))
        print("Operation and Maintain Cost, reduced from {} to {}".format(wind_om_cost_kw, new_wind_om_cost_kw))
        print("Net Capacity Factor increased from {} to {}".format(wind_net_cf, new_wind_net_cf))
        wind_cost_kw = new_wind_cost_kw
        wind_om_cost_kw = new_wind_om_cost_kw
        wind_net_cf = new_wind_net_cf
        print("wind om cost ORBIT:",wind_om_cost_kw)

    return new_wind_cost_kw, new_wind_om_cost_kw, new_wind_net_cf


def set_policy_values(scenario, policy, option):

   # Set policy values
    policy_option = option.__str__()
    scenario = policy[policy_option]

    return scenario, policy_option

def print_results2(scenario,
                  H2_Results,
                  wind_size_mw,
                  solar_size_mw,
                  storage_size_mw,
                  storage_size_mwh,
                  lcoe,
                  total_elec_production,
                  print_it):

    if print_it:
        # ------------------------- #
        #TODO: Tidy up these print statements
        print("Future Scenario: {}".format(scenario['Scenario Name']))
        print("Wind Cost per KW: {}".format(scenario['Wind Cost KW']))
        print("PV Cost per KW: {}".format(scenario['Solar Cost KW']))
        print("Storage Cost per KW: {}".format(scenario['Storage Cost kW']))
        print("Storage Cost per KWh: {}".format(scenario['Storage Cost kWh']))
        print("Wind Size built: {}".format(wind_size_mw))
        print("PV Size built: {}".format(solar_size_mw))
        print("Storage Size built: {}".format(storage_size_mw))
        print("Storage Size built: {}".format(storage_size_mwh))
        print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
        print("Total Yearly Electrical Output: {}".format(total_elec_production))
        print("Total Yearly Hydrogen Production: {}".format(H2_Results['hydrogen_annual_output']))
        #print("Levelized Cost H2/kg (new method - no operational costs)".format(h_lcoe_no_op_cost))
        print("Capacity Factor of Electrolyzer: {}".format(H2_Results['cap_factor']))

def print_h2_results2(lifetime_h2_production,
                     gut_check_h2_cost_kg,
                     LCOH_cf_method,
                     LCOH_cf_method_w_operating_costs,
                     forced_wind_size,
                     electrolyzer_size,
                     site_name,
                     wind_speed,
                     atb_year,
                     site_df,
                     electrolyzer_capex_kw,
                     print_it):

    if print_it:
        print('Total Lifetime H2(kg) produced: {}'.format(lifetime_h2_production))
        print("Gut-check H2 cost/kg: {}".format(gut_check_h2_cost_kg))
    #     print("h_lcoe: ", h_lcoe)
        print("LCOH CF Method (doesn't include grid electricity cost if used)", LCOH_cf_method)
        print("LCOH CF Method (includes operating costs + electricity)", LCOH_cf_method_w_operating_costs)
        # print("Levelized cost of H2 (electricity feedstock) (HOPP): {}".format(
        #     H2_Results['feedstock_cost_h2_levelized_hopp']))
        # print("Levelized cost of H2 (excl. electricity) (H2A): {}".format(H2A_Results['Total Hydrogen Cost ($/kgH2)']))
        # print("Total unit cost of H2 ($/kg) : {}".format(H2_Results['total_unit_cost_of_hydrogen']))
        # print("kg H2 cost from net cap cost/lifetime h2 production (HOPP): {}".format(
        #     H2_Results['feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp']))

        #Step 9: Summarize Results
        print('For a {}MW Offshore Wind Plant with {}MW electrolyzer located at {} \n (average wind speed {}m/s) in {}, with a Wind CAPEX cost of {},\n and an Electrolyzer cost of {}$/kW:\n The levelized cost of hydrogen was {} /kg '.
                    format(forced_wind_size,electrolyzer_size,site_name,np.average(wind_speed),atb_year,site_df['Total CapEx'],electrolyzer_capex_kw,LCOH_cf_method_w_operating_costs))

        print("LCOH CF Method (doesn't include grid electricity cost if used)", LCOH_cf_method)
        print("LCOH CF Method (includes operating costs + electricity)", LCOH_cf_method_w_operating_costs)

def run_HOPP(scenario,
             site,
             sample_site,
             forced_sizes,
             forced_solar_size,
             forced_wind_size,
             forced_storage_size_mw,
             forced_storage_size_mwh,
             wind_cost_kw,
             solar_cost_kw,
             storage_cost_kw,
             storage_cost_kwh,
             kw_continuous,
             load,
             electrolyzer_size,
             wind_om_cost_kw,
             nTurbs,
             floris_config,
             floris):

    if forced_sizes:
        solar_size_mw = forced_solar_size
        wind_size_mw = forced_wind_size
        storage_size_mw = forced_storage_size_mw
        storage_size_mwh = forced_storage_size_mwh
        storage_hours = 4

    turbine_rating_mw = scenario['Turbine Rating']
    tower_height = scenario['Tower Height']
    rotor_diameter = scenario['Rotor Diameter']

    if floris == False:
        if storage_size_mw > 0:
            technologies = {'pv':
                               {'system_capacity_kw': solar_size_mw * 1000},
                            'wind':
                                {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                    'turbine_rating_kw': turbine_rating_mw*1000,
                                    'hub_height': tower_height,
                                    'rotor_diameter': rotor_diameter},
                            'battery': {
                                'system_capacity_kwh': storage_size_mwh * 1000,
                                'system_capacity_kw': storage_size_mw * 1000
                                }
                            }
        else:
                    technologies = {'pv':
                                {'system_capacity_kw': solar_size_mw * 1000},
                            'wind':
                                {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                    'turbine_rating_kw': turbine_rating_mw*1000,
                                    'hub_height': tower_height,
                                    'rotor_diameter': rotor_diameter},
                            'battery': {
                                'system_capacity_kwh': storage_size_mwh * 1000,
                                'system_capacity_kw': storage_size_mw * 1000
                                }
                            }
        custom_powercurve=True
        hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp, \
           energy_shortfall_hopp,\
           annual_energies, wind_plus_solar_npv, npvs, lcoe, lcoe_nom =  \
        hopp_for_h2(site, scenario, technologies,
                    wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                    wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                    kw_continuous, load,
                    custom_powercurve,
                    electrolyzer_size, grid_connected_hopp=True, wind_om_cost_kw=wind_om_cost_kw)
    if floris == True:
        if storage_size_mw > 0:
            technologies = {'pv':
                               {'system_capacity_kw': solar_size_mw * 1000},
                            'wind': {
                                'num_turbines': nTurbs,
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'model_name': 'floris',
                                'timestep': [0,8760],
                                'floris_config': floris_config # if not specified, use default SAM models
                            },
                            'battery': {
                                'system_capacity_kwh': storage_size_mwh * 1000,
                                'system_capacity_kw': storage_size_mw * 1000
                                }
                            }
        else:
                    technologies = {'pv':
                                {'system_capacity_kw': solar_size_mw * 1000},
                            'wind': {
                                'num_turbines': nTurbs,
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'model_name': 'floris',
                                'timestep': [0,8760],
                                'floris_config': floris_config # if not specified, use default SAM models
                            }}

        from hopp.to_organize.H2_Analysis.hopp_for_h2_floris import hopp_for_h2_floris
        custom_powercurve=False
        hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
                energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe, lcoe_nom =  \
                    hopp_for_h2_floris(site, scenario, technologies,
                                wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                    wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                    kw_continuous, load,
                    custom_powercurve,
                    electrolyzer_size, grid_connected_hopp=False, wind_om_cost_kw=wind_om_cost_kw)



    wind_installed_cost = copy.deepcopy(hybrid_plant.wind.total_installed_cost)
    if solar_size_mw > 0:
        solar_installed_cost = copy.deepcopy(hybrid_plant.pv.total_installed_cost)
    else:
        solar_installed_cost = 0
    hybrid_installed_cost = copy.deepcopy(hybrid_plant.grid.total_installed_cost)

    # print("HOPP run complete")
    # print(hybrid_plant.om_capacity_expenses)

    return combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe

def run_battery(energy_shortfall_hopp,
                combined_pv_wind_curtailment_hopp,
                combined_pv_wind_power_production_hopp):

    bat_model = SimpleDispatch()
    bat_model.Nt = len(energy_shortfall_hopp)
    bat_model.curtailment = combined_pv_wind_curtailment_hopp
    bat_model.shortfall = energy_shortfall_hopp
    # print(combined_pv_wind_curtailment_hopp)
    # print(energy_shortfall_hopp)

    # bat_model.battery_storage = 100 * 1000
    # bat_model.charge_rate = 100 * 1000
    # bat_model.discharge_rate = 100 * 1000

    battery_used, excess_energy, battery_SOC = bat_model.run()
    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used

    return combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used, excess_energy

def compressor_model():

    #Compressor Model
    in_dict = dict()
    in_dict['flow_rate_kg_hr'] = 89
    in_dict['P_outlet'] = 100
    in_dict['compressor_rating_kWe'] = 802
    in_dict['mean_time_between_failure'] = 200
    in_dict['total_hydrogen_throughput'] = 18750
    compressor_results = dict()
    compressor = Compressor(in_dict, compressor_results)
    compressor.compressor_power()
    compressor.compressor_costs()
    # print("compressor_power (kW): ", compressor_results['compressor_power'])
    # print("Compressor capex [USD]: ", compressor_results['compressor_capex'])
    # print("Compressor opex [USD/yr]: ", compressor_results['compressor_opex'])

    return compressor, compressor_results

def pressure_vessel():

    #Pressure Vessel Model Example
    storage_input = dict()
    storage_input['H2_storage_kg'] = 18750
    # storage_input['storage_duration_hrs'] = 4
    # storage_input['flow_rate_kg_hr'] = 89        #[kg-H2/hr]
    storage_input['compressor_output_pressure'] = 100
    storage_output = dict()
    underground_pipe_storage = Underground_Pipe_Storage(storage_input, storage_output)
    underground_pipe_storage.pipe_storage_costs()

    # print('Underground pipe storage capex: ${0:,.0f}'.format(storage_output['pipe_storage_capex']))
    # print('Underground pipe storage opex: ${0:,.0f}/yr'.format(storage_output['pipe_storage_opex']))

    return storage_input, storage_output

def pipeline(site_df,
            H2_Results,
            useful_life,
            storage_input):

    print("Distance to port: ",site_df['Approx. distance to port'])
    dist_to_port_value = site_df['Approx. distance to port']
    pipe_flow_rate = max(H2_Results['hydrogen_hourly_production'])
    dist_to_port = ""
    for m in dist_to_port_value:
        if m.isdigit():
            dist_to_port = dist_to_port + m
    dist_to_port = int(dist_to_port)
    print("Water depth: ",site_df['Approx. water depth'])
    site_depth_value = site_df['Approx. water depth']
    site_depth = ""
    for m in site_depth_value:
        if m.isdigit():
            site_depth = site_depth + m
    site_depth = int(site_depth)

    #from examples.H2_Analysis.pipeline_model import Pipeline
    from hopp.to_organize.pipelineASME import PipelineASME
    in_dict = dict()
    #in_dict['pipeline_model'] = 'nrwl'
    #in_dict['pipeline_model'] = 'nexant'
    #in_dict['pipe_diam_in'] = 24.0
    in_dict['pipe_diam_in'] = np.linspace(12.0, 48.0, 20)
    in_dict['pipe_thic_in'] = np.linspace(0.1, 2.0, 50)
    #in_dict['offshore_bool'] = True
    in_dict['flow_rate_kg_hr'] = pipe_flow_rate
    in_dict['plant_life'] = useful_life
    in_dict['useful_life'] = useful_life
    in_dict['dist_to_h2_load_km'] = int(dist_to_port)
    in_dict['site_depth_m'] = int(site_depth)
    in_dict['steel_cost_ton'] = 900.0 # $ US/ton searching for seamless FBE X52 carbon steel > $500-$1000 per ton
    in_dict['pressure_bar'] = storage_input['compressor_output_pressure']

    out_dict = dict()

    print("Pipeline flow rate: ", pipe_flow_rate, "kg/hr")
    #pipeline_model = Pipeline(in_dict, out_dict)
    #capex_pipeline, opex_pipeline, pipeline_annuals = pipeline_model.pipeline_cost()
    #pipeline_cost_kw = capex_pipeline / (wind_size_mw*1000)
    #print("Pipeline CAPEX: ${0:,.0f}".format(capex_pipeline))
    #print("Pipeline Cost/kW: ${0:,.0f}/kW".format(pipeline_cost_kw))
    pipeline_model = PipelineASME(in_dict, out_dict)
    pipeline_model.pipelineDesign()
    pipeline_model.pipelineCost()
    capex_pipeline = np.min(out_dict['pipeline_capex'])
    opex_pipeline = np.min(out_dict['pipeline_opex'])
    capex_substation = out_dict['substation_capex']

    total_h2export_system_cost = capex_pipeline + capex_substation
    #print('Pipeline Model:', in_dict['pipeline_model'])
    #print('Pipeline length (miles):', out_dict['len_pipeline_miles'])
    #print('Pipeline CapEx Cost ($USD):', out_dict['pipeline_capex'])
    #print('Pipeline OpEx Cost ($USD):', out_dict['pipeline_opex'])
    print("Pipeline Length (km):", out_dict['total_pipeline_length_km'])
    print("Pipeline Design Pressure (bar):",in_dict['pressure_bar'])
    print("Pipeline Diameter: {} in, Thickness {} in".format(out_dict['design_diam_in'][0],out_dict['design_thic_in'][0]))
    print("Pipeline CapEx ($US): ", capex_pipeline)
    print("Pipeline Opex ($US/year)", opex_pipeline)
    print("Substation CapEx ($US): ", capex_substation)
    print("Total H2-Export CapEx:", total_h2export_system_cost)

    return total_h2export_system_cost, opex_pipeline, dist_to_port_value

def pipeline_vs_hvdc(site_df, wind_size_mw, total_h2export_system_cost):

    #Pipeline vs HVDC cost
    #Get Equivalent cost of HVDC export system from Orbit runs and remove it
    export_system_cost_kw = site_df['Export System'] + site_df['Offshore Substation']
    export_system_installation_cost_kw = site_df['Export System Installation'] + site_df['Offshore Substation Installation']
    total_export_system_cost_kw = export_system_cost_kw + export_system_installation_cost_kw
    export_system_cost = export_system_cost_kw * wind_size_mw * 1000
    export_system_installation_cost = export_system_installation_cost_kw * wind_size_mw * 1000
    total_export_system_cost = export_system_cost + export_system_installation_cost
    print("Total HVDC Export System Cost is ${0:,.0f} vs ${1:,.0f} for H2 Pipeline".format(total_export_system_cost, total_h2export_system_cost))

    return total_export_system_cost_kw, total_export_system_cost

def desal_model(H2_Results,
                electrolyzer_size,
                electrical_generation_timeseries,
                useful_life):

    water_usage_electrolyzer = H2_Results['water_hourly_usage']
    m3_water_per_kg_h2 = 0.01
    desal_system_size_m3_hr = electrolyzer_size * (1000/55.5) * m3_water_per_kg_h2
    est_const_desal_power_mw_hr = desal_system_size_m3_hr * 4.2 /1000 # 4.2kWh/m^3 desal efficiency estimate
    # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
    Power = copy.deepcopy(electrical_generation_timeseries)
    fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals = RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=30)
    print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size, desal_system_size_m3_hr))
    print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
    print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
    print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
    # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
    print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

    return desal_capex, desal_opex, desal_annuals

def run_H2_PEM_sim(hybrid_plant,
                energy_to_electrolyzer,
                scenario,
                wind_size_mw,
                solar_size_mw,
                electrolyzer_size_mw,
                kw_continuous,
                electrolyzer_capex_kw,
                lcoe):

    #TODO: Refactor H2A model call
    # Should take as input (electrolyzer size, cost, electrical timeseries, total system electrical usage (kwh/kg),
    # Should give as ouptut (h2 costs by net cap cost, levelized, total_unit_cost of hydrogen etc)   )

    # electrical_generation_timeseries = combined_pv_wind_storage_power_production_hopp
    electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
    electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

    adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
    #NB: adjusted_installed_cost does NOT include the electrolyzer cost
    useful_life = scenario['Useful Life']
    net_capital_costs = 0

    # system_rating = electrolyzer_size
    system_rating = wind_size_mw + solar_size_mw
    H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size_mw,
                    kw_continuous,electrolyzer_capex_kw,lcoe,adjusted_installed_cost,useful_life,
                    net_capital_costs)


    H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']
    H2_Results['cap_factor'] = H2_Results['cap_factor']

    # print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
    # print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
    # print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))


    return H2_Results, H2A_Results, electrical_generation_timeseries
def grid(combined_pv_wind_storage_power_production_hopp,
         sell_price,
         excess_energy,
         buy_price,
         kw_continuous,
         plot_grid):

    if plot_grid:
        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],label="before buy from grid")
        plt.suptitle("Power Signal Before Purchasing From Grid")

    if sell_price:
        profit_from_selling_to_grid = np.sum(excess_energy)*sell_price
    else:
        profit_from_selling_to_grid = 0.0

    # buy_price = False # if you want to force no buy from grid
    if buy_price:
        cost_to_buy_from_grid = buy_price

        for i in range(len(combined_pv_wind_storage_power_production_hopp)):
            if combined_pv_wind_storage_power_production_hopp[i] < kw_continuous:
                cost_to_buy_from_grid += (kw_continuous-combined_pv_wind_storage_power_production_hopp[i])*buy_price
                combined_pv_wind_storage_power_production_hopp[i] = kw_continuous
    else:
        cost_to_buy_from_grid = 0.0

    # capping power to electrolyzer
    energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]

    #Plot Dispatch Results

    if plot_grid:
        plt.figure(figsize=(9,6))
        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],"--",label="after buy from grid")
        plt.plot(energy_to_electrolyzer[200:300],"--",label="energy to electrolyzer")
        plt.legend()
        plt.title('Power available after purchasing from grid (if enabled)')
        # plt.show()

    return cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer

def calculate_financials(electrical_generation_timeseries,
                         hybrid_plant,
                         H2A_Results,
                         H2_Results,
                         desal_opex,
                         desal_annuals,
                         total_h2export_system_cost,
                         opex_pipeline,
                         total_export_system_cost,
                         total_export_om_cost,
                         electrolyzer_capex_kw,
                         time_between_replacement,
                         cost_to_buy_from_grid,
                         profit_from_selling_to_grid,
                         useful_life,
                         atb_year,
                         policy_option,
                         scenario,
                         h2_model,
                         desal_capex,
                         wind_cost_kw,
                         solar_cost_kw,
                         discount_rate,
                         solar_size_mw,
                         electrolyzer_size_mw,
                         results_dir,
                         site_name,
                         turbine_model,
                         scenario_choice):

    turbine_rating_mw = scenario['Turbine Rating']
    from hopp.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals

    #Electrolyzer financial model
    if h2_model == 'H2A':
        #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
        pass
    elif h2_model == 'Simple':

        from hopp.simulation.technologies.hydrogen.electrolysis.H2_cost_model import basic_H2_cost_model

        cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(electrolyzer_capex_kw,
                         time_between_replacement,electrolyzer_size_mw, useful_life, atb_year,
            electrical_generation_timeseries, H2_Results['hydrogen_annual_output'], scenario['H2 PTC'], scenario['Wind ITC'])

    total_elec_production = np.sum(electrical_generation_timeseries)
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
    total_electrolyzer_cost = electrolyzer_total_capital_cost
    total_desal_cost = desal_capex
    total_system_installed_cost_pipeline = total_hopp_installed_cost + total_electrolyzer_cost + total_desal_cost + total_h2export_system_cost
    total_system_installed_cost_hvdc = total_hopp_installed_cost + total_electrolyzer_cost + total_desal_cost + total_export_system_cost
    annual_operating_cost_wind = np.average(hybrid_plant.wind.om_total_expense)
    fixed_om_cost_wind = np.average(hybrid_plant.wind.om_fixed_expense)
    annual_operating_cost_h2 = electrolyzer_OM_cost
    annual_operating_cost_desal = desal_opex
    total_annual_operating_costs_pipeline =  annual_operating_cost_wind + annual_operating_cost_h2 + annual_operating_cost_desal + opex_pipeline + cost_to_buy_from_grid - profit_from_selling_to_grid
    total_annual_operating_costs_hvdc = annual_operating_cost_wind + annual_operating_cost_h2 + annual_operating_cost_desal + total_export_om_cost + cost_to_buy_from_grid - profit_from_selling_to_grid

    h_lcoe_no_op_cost_pipeline = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_pipeline,
                        0, discount_rate, useful_life)
    h_lcoe_no_op_cost_hvdc = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_hvdc,
                        0, discount_rate, useful_life)

    annual_energies = copy.deepcopy(hybrid_plant.annual_energies)
    lcoe_test = lcoe_calc((annual_energies.wind/1000),total_hopp_installed_cost, annual_operating_cost_wind, discount_rate, useful_life)
    # print('LCOE energy: ',lcoe_test, '$/MWh')

    #Requires capital costs and operating cost to be seperate just a check
    #****Only works when there is no policy options (capex in this calc is the same irregardless of ITC)
    h_lcoe_pipeline = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_pipeline,
                        total_annual_operating_costs_pipeline, discount_rate, useful_life)
    h_lcoe_hvdc = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost_hvdc,
                        total_annual_operating_costs_hvdc, discount_rate, useful_life)
    # print('Pipeline H_LCOE no op cost', h_lcoe_no_op_cost_pipeline,'Pipeline H_LCOE w/op cost',h_lcoe_pipeline)
    # print('HVDC H_LCOE no op cost', h_lcoe_no_op_cost_hvdc,'Pipeline H_LCOE w/op cost',h_lcoe_hvdc)


    # Cashflow Financial Calculation
    discount_rate = scenario['Discount Rate']

    # Create cashflows for pipeline and HVDC
    cf_pipeline_annuals = -simple_cash_annuals(useful_life,useful_life,total_h2export_system_cost,opex_pipeline,0.03)
    cf_hvdc_annuals = - simple_cash_annuals(useful_life,useful_life,total_export_system_cost,total_export_om_cost,0.03)

    #Implement ITC for pipeline and hvdc
    pipeline_itc = (scenario['Wind ITC']/100) * total_h2export_system_cost
    cf_pipeline_itc = [0]*30
    cf_pipeline_itc[1] = pipeline_itc
    cf_pipeline_annuals = np.add(cf_pipeline_annuals,cf_pipeline_itc)

    hvdc_itc = (scenario['Wind ITC']/100) * total_export_system_cost
    cf_hvdc_itc = [0]*30
    cf_hvdc_itc[1] = hvdc_itc
    cf_hvdc_annuals = np.add(cf_hvdc_annuals,cf_hvdc_itc)
    # print("capex pipeline, hvdc:", total_h2export_system_cost,total_export_system_cost)
    # print('opex pipeline, hvdc', opex_pipeline,total_export_om_cost)
    # print('pipeline annuals:', cf_pipeline_annuals)
    # print('hvdc annauls', cf_hvdc_annuals)

    cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
    if solar_size_mw > 0:
        cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
    else:
        cf_solar_annuals = np.zeros(30)
    cf_desal_annuals = -desal_annuals

    cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals, cf_desal_annuals],['Wind', 'Solar', 'H2', 'Desal'])

    cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}_{}MW.csv".format(site_name, scenario_choice, atb_year, discount_rate,turbine_rating_mw)))

    #Basic steps in calculating the LCOH
    #More nuanced calculation than h_lcoe b/c it uses yearly cashflows which change year to year rather than total capex and opex
    #file:///Applications/SAM_2021.12.02/SAM.app/Contents/runtime/help/html/index.html?mtf_lcoe.htm

    #Calculate total lifecycle cost for each technology (TLCC)
    tlcc_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
    #print('npv wind: ',tlcc_wind_costs)
    tlcc_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
    tlcc_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
    #print("NPV H2 Costs using {} model: {}".format(h2_model,tlcc_h2_costs))
    tlcc_desal_costs = -npf.npv(discount_rate, cf_desal_annuals)
    #print("NPV desal: ", tlcc_desal_costs)
    tlcc_pipeline_costs = npf.npv(discount_rate, cf_pipeline_annuals)
    tlcc_hvdc_costs = npf.npv(discount_rate, cf_hvdc_annuals)

    tlcc_total_costs = tlcc_wind_costs+tlcc_solar_costs+tlcc_h2_costs + tlcc_desal_costs + tlcc_hvdc_costs
    tlcc_total_costs_pipeline = tlcc_wind_costs + tlcc_solar_costs + tlcc_h2_costs + tlcc_desal_costs + tlcc_pipeline_costs

    # Manipulate h2 production for LCOH calculation
    # Note. This equation makes it appear that the energy term in the denominator is discounted.
    # That is a result of the algebraic solution of the equation, not an indication of the physical performance of the system.
    discounted_h2_production = npf.npv(discount_rate, [H2_Results['hydrogen_annual_output']]*30)
   # print('discounted h2 production',discounted_h2_production)

    #Individual technology LCOH contribution
    LCOH_cf_method_wind = -tlcc_wind_costs / discounted_h2_production
    LCOH_cf_method_pipeline = -tlcc_pipeline_costs / discounted_h2_production
    LCOH_cf_method_hvdc = -tlcc_hvdc_costs / discounted_h2_production
    LCOH_cf_method_solar = -tlcc_solar_costs / discounted_h2_production
    LCOH_cf_method_h2_costs = -tlcc_h2_costs / discounted_h2_production
    LCOH_cf_method_desal_costs = -tlcc_desal_costs / discounted_h2_production

    # Total LCOH for pipeline and hvdc export scenarios
    LCOH_cf_method_total_hvdc = -tlcc_total_costs / discounted_h2_production
    LCOH_cf_method_total_pipeline = -tlcc_total_costs_pipeline / discounted_h2_production

    financial_summary_df = pd.DataFrame([policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw, electrolyzer_capex_kw,
                                            scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],
                                            discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_h2_costs, tlcc_desal_costs, tlcc_pipeline_costs, tlcc_hvdc_costs,
                                            LCOH_cf_method_wind,LCOH_cf_method_h2_costs,LCOH_cf_method_desal_costs, LCOH_cf_method_pipeline, LCOH_cf_method_hvdc,
                                            LCOH_cf_method_total_hvdc, LCOH_cf_method_total_pipeline],
                                        ['Policy Option','Turbine Model','Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'Electrolyzer Cost KW', 'Debt Equity',
                                            'ATB Year', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses',
                                            'NPV Solar Expenses', 'NPV H2 Expenses','NPV Desal Expenses', 'NPV Pipeline Expenses','NPV HVDC Expenses',
                                            'LCOH Wind HVDC', 'LCOH H2', 'LCOH Desal', 'LCOH Pipeline','LCOH HVDC',
                                            'LCOH cf method HVDC','LCOH cf method Pipeline'])
    financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,policy_option)))

    # Gut Check H2 calculation Pipeline (non-levelized)
    total_installed_and_operational_lifetime_cost_pipeline = total_system_installed_cost_pipeline + (30 * total_annual_operating_costs_pipeline)
    lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
    gut_check_h2_cost_kg_pipeline = total_installed_and_operational_lifetime_cost_pipeline / lifetime_h2_production

    total_installed_and_operational_lifetime_cost_hvdc = total_system_installed_cost_hvdc + (30 * total_annual_operating_costs_hvdc)
    lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
    gut_check_h2_cost_kg_hvdc = total_installed_and_operational_lifetime_cost_hvdc / lifetime_h2_production

    # Total amount of ITC [USD]
    wind_itc_total = hybrid_plant.wind._financial_model.Outputs.itc_total
    total_itc_pipeline = wind_itc_total + pipeline_itc + h2_itc
    total_itc_hvdc = wind_itc_total + hvdc_itc + h2_itc

    # print("Gut Check H2 Cost Pipeline:",gut_check_h2_cost_kg_pipeline)
    # print("Gut Check H2 Cost HVDC:",gut_check_h2_cost_kg_hvdc)
    print("HVDC Scenario: LCOH for H2, Desal, Grid Electrical Cost:", LCOH_cf_method_total_hvdc)

    print("Pipeline Scenario: LCOH for H2, Desal, Grid Electrical Cost:", LCOH_cf_method_total_pipeline)


    return LCOH_cf_method_wind, LCOH_cf_method_pipeline, LCOH_cf_method_hvdc, LCOH_cf_method_solar,\
        LCOH_cf_method_h2_costs, LCOH_cf_method_desal_costs, LCOH_cf_method_total_hvdc, LCOH_cf_method_total_pipeline, \
        total_elec_production, lifetime_h2_production, gut_check_h2_cost_kg_pipeline, gut_check_h2_cost_kg_hvdc, \
        wind_itc_total, total_itc_pipeline, total_itc_hvdc, total_annual_operating_costs_hvdc, total_annual_operating_costs_pipeline, \
        h_lcoe_hvdc, h_lcoe_pipeline, tlcc_wind_costs, tlcc_solar_costs, tlcc_h2_costs, tlcc_desal_costs, tlcc_pipeline_costs,\
        tlcc_hvdc_costs, tlcc_total_costs, tlcc_total_costs_pipeline, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit, h2_itc

def write_outputs_RODeO(electrical_generation_timeseries,
                         hybrid_plant,
                         total_export_system_cost,
                         total_export_om_cost,
                         cost_to_buy_from_grid,
                         electrolyzer_capex_kw,
                         time_between_replacement,
                         profit_from_selling_to_grid,
                         useful_life,
                         atb_year,
                         policy_option,
                         scenario,
                         wind_cost_kw,
                         solar_cost_kw,
                         discount_rate,
                         solar_size_mw,
                         results_dir,
                         site_name,
                         turbine_model,
                         scenario_choice,
                         lcoe,
                         run_RODeO_selector,
                         lcoh,
                         electrolyzer_capacity_factor,
                         storage_duration_hr,
                         hydrogen_annual_production,
                         water_consumption_hourly,
                         RODeO_summary_results_dict,
                         steel_breakeven_price):

    turbine_rating_mw = scenario['Turbine Rating']
    from hopp.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals

    total_elec_production = np.sum(electrical_generation_timeseries)
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
    annual_operating_cost_wind = np.average(hybrid_plant.wind.om_total_expense)
    fixed_om_cost_wind = np.average(hybrid_plant.wind.om_fixed_expense)

    # Cashflow Financial Calculation
    discount_rate = scenario['Discount Rate']

    cf_hvdc_annuals = - simple_cash_annuals(useful_life,useful_life,total_export_system_cost,total_export_om_cost,0.03)

    hvdc_itc = (scenario['Wind ITC']/100) * total_export_system_cost
    cf_hvdc_itc = [0]*30
    cf_hvdc_itc[1] = hvdc_itc
    cf_hvdc_annuals = np.add(cf_hvdc_annuals,cf_hvdc_itc)

    cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
    if solar_size_mw > 0:
        cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
    else:
        cf_solar_annuals = np.zeros(30)

    cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals],['Wind', 'Solar'])

    cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}_{}MW.csv".format(site_name, scenario_choice, atb_year, discount_rate,turbine_rating_mw)))

    #Calculate total lifecycle cost for each technology (TLCC)
    tlcc_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
    #print('npv wind: ',tlcc_wind_costs)
    tlcc_solar_costs = npf.npv(discount_rate, cf_solar_annuals)

    tlcc_hvdc_costs = npf.npv(discount_rate, cf_hvdc_annuals)

    tlcc_total_costs = tlcc_wind_costs+tlcc_solar_costs + tlcc_hvdc_costs


    # Total amount of ITC [USD]
    wind_itc_total = hybrid_plant.wind._financial_model.Outputs.itc_total
    total_itc_hvdc = wind_itc_total + hvdc_itc

    financial_summary_df = pd.DataFrame([policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,
                                            scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],
                                            discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs,run_RODeO_selector,lcoe/100,lcoh,
                                            electrolyzer_capacity_factor,storage_duration_hr,hydrogen_annual_production,
                                            RODeO_summary_results_dict['Storage & compression cost (US$/kg)'],RODeO_summary_results_dict['Input CAPEX (US$/kg)'],
                                            RODeO_summary_results_dict['Input FOM (US$/kg)'],RODeO_summary_results_dict['Input VOM (US$/kg)'],
                                            RODeO_summary_results_dict['Renewable capital cost (US$/kg)'],RODeO_summary_results_dict['Renewable FOM (US$/kg)'],
                                            RODeO_summary_results_dict['Taxes (US$/kg)'],steel_breakeven_price],
                                        ['Policy Option','Turbine Model','Useful Life', 'Wind Cost ($/kW)', 'Solar Cost ($/kW)', 'Debt Equity',
                                            'ATB Year', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses',
                                            'NPV Solar Expenses', 'NPV HVDC Expenses','Used RODeO?','LCOE ($/MWh)','LCOH ($/kg)',
                                            'Electrolyzer CF (-)','Hydrogen storage duration (hr)','Hydrogen annual production (kg)',
                                            'LCOH: Storage and compression ($/kg)','LCOH: Electrolyzer CAPEX ($/kg)','LCOH: Electrolyzer FOM ($/kg)','LCOH: Electrolyzer VOM ($/kg)',
                                            'LCOH: Renewable CAPEX ($/kg)','LCOH: Renewable FOM ($/kg)','LCOH: Taxes ($/kg)','Steel break-even price ($/tonne)'])
    financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,policy_option)))

    return policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
           scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
           discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
           wind_itc_total, total_itc_hvdc\

def steel_LCOS(levelized_cost_hydrogen,
                hydrogen_annual_production,
                lime_unitcost,
                carbon_unitcost,
                iron_ore_pellet_unitcost):
    from hopp.to_organize.run_profast_for_steel import run_profast_for_steel
    # Specify file path to PyFAST
    import sys
    #sys.path.insert(1,'../PyFAST/')

    sys.path.append('../PyFAST/')

    #import src.PyFAST as PyFAST

    # Steel production break-even price analysis

    hydrogen_consumption_for_steel = 0.06596              # metric tonnes of hydrogen/metric tonne of steel productio
    # Could be good to make this more conservative, but it is probably fine if demand profile is flat
    max_steel_production_capacity_mtpy = hydrogen_annual_production/1000/hydrogen_consumption_for_steel

    # Could connect these to other things in the model
    steel_capacity_factor = 0.9
    steel_plant_life = 30

    # Should connect these to something (AEO, Cambium, etc.)
    natural_gas_cost = 4                        # $/MMBTU
    electricity_cost = 48.92                    # $/MWh

    steel_economics_from_pyfast,steel_economics_summary,steel_annual_capacity=\
        run_profast_for_steel(max_steel_production_capacity_mtpy,\
            steel_capacity_factor,steel_plant_life,levelized_cost_hydrogen,\
            electricity_cost,natural_gas_cost,lime_unitcost,
                carbon_unitcost,
                iron_ore_pellet_unitcost)

    steel_breakeven_price = steel_economics_from_pyfast.get('price')

    return steel_economics_from_pyfast, steel_economics_summary, steel_breakeven_price, steel_annual_capacity