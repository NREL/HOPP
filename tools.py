# extra function in the osw_h2 file

import numpy as np 
import pandas as pd
import copy  

# HOPP functionss
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from hybrid.sites import SiteInfo
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.compressor import Compressor
from examples.H2_Analysis.desal_model import RO_desal

def set_site_info(x1, turbine_model, site_location, sample_site):

    turbinesheet = turbine_model[-4:]
    scenario_df = xl.parse(turbinesheet)
    scenario_df.set_index(["Parameter"], inplace = True)

    site_df = scenario_df[site_location]

    latlon = site_df['Representative coordinates']
    lat, lon = (latlon.split(','))
    lat = float(lat)
    lon = float(lon)
    sample_site['lat'] = lat
    sample_site['lon'] = lon
    sample_site['no_solar'] = True

    return site_df, sample_site


def set_financial_info(scenario,
                       debt_equity_split,
                       discount_rate):

    scenario['Debt Equity'] = debt_equity_split
    scenario['Discount Rate'] = discount_rate

    return scenario

def set_electrolyzer_info(atb_year):

    #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
    if atb_year == 2022:
        electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
        time_between_replacement = 40000    #[hrs] 
    elif atb_year == 2025:
        electrolyzer_capex_kw = 300
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2030:
        electrolyzer_capex_kw = 150
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2035:
        electrolyzer_capex_kw = 100
        time_between_replacement = 80000    #[hrs]

def set_turbine_model(turbine_model, scenario, parent_path):

    # Define Turbine Characteristics based on user selected turbine. 
    # Scaled from reference 15MW turbine: https://github.com/IEAWindTask37/IEA-15-240-RWT
    if turbine_model == '2022ATB_12MW':
        custom_powercurve_path = '2022atb_osw_12MW.csv' 
        tower_height = 136
        rotor_diameter = 214
        turbine_rating_mw = 12
        wind_cost_kw = 1300
        # Future Cost Reduction Estimates - ATB 2022: Class 4 Fixed, Class 11 Float
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_12MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_12MW.csv'))

    elif turbine_model == '2022ATB_15MW':
        custom_powercurve_path = '2022atb_osw_15MW.csv' 
        tower_height = 150
        rotor_diameter = 240
        turbine_rating_mw = 15
        wind_cost_kw =  1300
        # Future Cost Reduction Estimates
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_15MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_15MW.csv'))

    elif turbine_model == '2022ATB_18MW':
        custom_powercurve_path = '2022atb_osw_18MW.csv' 
        tower_height = 161
        rotor_diameter = 263
        turbine_rating_mw = 18
        wind_cost_kw = 1300
        # Future Cost Reduction Estimates
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_18MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_18MW.csv'))

    scenario['Tower Height'] = tower_height
    scenario['Turbine Rating'] = turbine_rating_mw
    scenario['Powercurve File'] = custom_powercurve_path
    scenario['Rotor Diameter'] = rotor_diameter

    # print("Powercurve Path: ", custom_powercurve_path)

    return scenario #custom_powercurve_path, tower_height, rotor_diameter, turbine_rating_mw, wind_cost_kw, floating_cost_reductions_df, fixed_cost_reductions_df

def set_turbine_financials(turbine_model, 
                            fixed_or_floating_wind,
                            atb_year,
                            wind_om_cost_kw,
                            wind_net_cf, 
                            parent_path):

    # Define Turbine Characteristics based on user selected turbine. 
    # Scaled from reference 15MW turbine: https://github.com/IEAWindTask37/IEA-15-240-RWT
    if turbine_model == '2022ATB_12MW':
        custom_powercurve_path = '2022atb_osw_12MW.csv' 
        tower_height = 136
        rotor_diameter = 214
        turbine_rating_mw = 12
        wind_cost_kw = 1300
        # Future Cost Reduction Estimates - ATB 2022: Class 4 Fixed, Class 11 Float
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_12MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_12MW.csv'))

    elif turbine_model == '2022ATB_15MW':
        custom_powercurve_path = '2022atb_osw_15MW.csv' 
        tower_height = 150
        rotor_diameter = 240
        turbine_rating_mw = 15
        wind_cost_kw =  1300
        # Future Cost Reduction Estimates
        floating_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/floating_cost_reductions_15MW.csv'))
        fixed_cost_reductions_df = pd.read_csv(os.path.join(parent_path,'examples/H2_Analysis/fixed_cost_reductions_15MW.csv'))

    elif turbine_model == '2022ATB_18MW':
        custom_powercurve_path = '2022atb_osw_18MW.csv' 
        tower_height = 161
        rotor_diameter = 263
        turbine_rating_mw = 18
        wind_cost_kw = 1300
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


def set_policy_values(scenario, policy, count):

    # Set policy values
    scenario['Wind ITC'] = policy[count]['Wind ITC']
    scenario['Wind PTC'] = policy[count]['Wind PTC']
    scenario['H2 PTC'] = policy[count]['H2 PTC']

    return scenario

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
             custom_powercurve,
             electrolyzer_size,
             wind_om_cost_kw):

    if forced_sizes:
        solar_size_mw = forced_solar_size
        wind_size_mw = forced_wind_size
        storage_size_mw = forced_storage_size_mw
        storage_size_mwh = forced_storage_size_mwh
        storage_hours = 0

    turbine_rating_mw = scenario['Turbine Rating']
    tower_height = scenario['Tower Height']
    rotor_diameter = scenario['Rotor Diameter']

    if storage_size_mw > 0:
        technologies = {#'pv':
                        #   {'system_capacity_kw': solar_size_mw * 1000},
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
                technologies = {#'pv':
                            #{'system_capacity_kw': solar_size_mw * 1000},
                        'wind':
                            {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'hub_height': tower_height,
                                'rotor_diameter': rotor_diameter},
        #                 'battery': {
        #                     'system_capacity_kwh': storage_size_mwh * 1000,
        #                     'system_capacity_kw': storage_size_mw * 1000
        #                     }
                        }
    site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])
    hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
    energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
        hopp_for_h2(site, scenario, technologies,
                    wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                    wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                    kw_continuous, load,
                    custom_powercurve,
                    electrolyzer_size, grid_connected_hopp=True, wind_om_cost_kw=wind_om_cost_kw)

    wind_installed_cost = copy.deepcopy(hybrid_plant.wind.total_installed_cost)
    if solar_size_mw > 0:
        solar_installed_cost = copy.deepcopy(hybrid_plant.pv.total_installed_cost)
    else:
        solar_installed_cost = 0
    hybrid_installed_cost = copy.deepcopy(hybrid_plant.grid.total_installed_cost)

    # print("HOPP run complete")
    # print(hybrid_plant.om_capacity_expenses)

    return combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw

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

    return combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used

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
    from examples.H2_Analysis.h2_storage.pipe_storage.underground_pipe_storage import Underground_Pipe_Storage
    storage_input = dict()
    storage_input['H2_storage_kg'] = 18750
    # storage_input['storage_duration_hrs'] = 4
    # storage_input['flow_rate_kg_hr'] = 89        #[kg-H2/hr]
    storage_input['compressor_output_pressure'] = 100
    storage_output = dict()
    underground_pipe_storage = Underground_Pipe_Storage(storage_input, storage_output)
    underground_pipe_storage.pipe_storage_costs()

    print('Underground pipe storage capex: ${0:,.0f}'.format(storage_output['pipe_storage_capex']))
    print('Underground pipe storage opex: ${0:,.0f}/yr'.format(storage_output['pipe_storage_opex']))

def pipeline():

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
    from examples.H2_Analysis.pipelineASME import PipelineASME
    in_dict = dict()
    #in_dict['pipeline_model'] = 'nrwl'
    #in_dict['pipeline_model'] = 'nexant'
    #in_dict['pipe_diam_in'] = 24.0
    in_dict['pipe_diam_in'] = np.linspace(12.0, 48.0, 20)
    in_dict['pipe_thic_in'] = np.linspace(0.1, 2.0, 50)
    #in_dict['offshore_bool'] = True 
    in_dict['flow_rate_kg_hr'] = pipe_flow_rate
    in_dict['plant_life'] = 30
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

    return total_h2export_system_cost, opex_pipeline

def pipeline_vs_hvdc():

    #Pipeline vs HVDC cost
    #Get Equivalent cost of HVDC export system from Orbit runs and remove it
    export_system_cost_kw = site_df['Export System'] + site_df['Offshore Substation']
    export_system_installation_cost_kw = site_df['Export System Installation'] + site_df['Offshore Substation Installation']
    total_export_system_cost_kw = export_system_cost_kw + export_system_installation_cost_kw
    export_system_cost = export_system_cost_kw * wind_size_mw * 1000
    export_system_installation_cost = export_system_installation_cost_kw * wind_size_mw * 1000
    total_export_system_cost = export_system_cost + export_system_installation_cost
    print("Total HVDC Export System Cost is ${0:,.0f} vs ${1:,.0f} for H2 Pipeline".format(total_export_system_cost, total_h2export_system_cost))

    return total_export_system_cost_kw

def desal_model():

    water_usage_electrolyzer = H2_Results['water_hourly_usage']
    m3_water_per_kg_h2 = 0.01
    desal_system_size_m3_hr = electrolyzer_size * (1000/55.5) * m3_water_per_kg_h2
    est_const_desal_power_mw_hr = desal_system_size_m3_hr * 2.928 /1000 # 4kWh/m^3 desal efficiency estimate
    # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
    Power = copy.deepcopy(electrical_generation_timeseries)
    fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals = RO_desal(Power, desal_system_size_m3_hr, useful_life, plant_life=30)
    print("For {}MW Electrolyzer, implementing {}m^3/hr desal system".format(electrolyzer_size, desal_system_size_m3_hr))
    print("Estimated constant desal power usage {0:.3f}MW".format(est_const_desal_power_mw_hr))
    print("Desal System CAPEX ($): {0:,.02f}".format(desal_capex))
    print("Desal System OPEX ($): {0:,.02f}".format(desal_opex))
    # print("Freshwater Flowrate (m^3/hr): {}".format(fresh_water_flowrate))
    print("Total Annual Feedwater Required (m^3): {0:,.02f}".format(np.sum(feed_water_flowrate)))

    return desal_opex

def run_H2_PEM():

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
    H2_Results, H2A_Results = run_h2_PEM.run_h2_PEM(electrical_generation_timeseries,electrolyzer_size,
                    kw_continuous,electrolyzer_capex_kw,lcoe,adjusted_installed_cost,useful_life,
                    net_capital_costs)


    H2_Results['hydrogen_annual_output'] = H2_Results['hydrogen_annual_output']
    H2_Results['cap_factor'] = H2_Results['cap_factor']
    
    print("Total power input to electrolyzer: {}".format(np.sum(electrical_generation_timeseries)))
    print("Hydrogen Annual Output (kg): {}".format(H2_Results['hydrogen_annual_output']))
    print("Water Consumption (kg) Total: {}".format(H2_Results['water_annual_usage']))

    return H2_Results, H2A_Results, electrical_generation_timeseries

def grid():

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

    energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]

    #Plot Dispatch Results

    if plot_grid:
        plt.figure(figsize=(9,6))
        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],"--",label="after buy from grid")
        plt.plot(energy_to_electrolyzer[200:300],"--",label="energy to electrolyzer")
        plt.legend()
        plt.title('Power available after purchasing from grid (if enabled)')
        # plt.show()

    return cost_to_buy_from_grid, profit_from_selling_to_grid

def calculate_financials():

    total_elec_production = np.sum(electrical_generation_timeseries)
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost
    total_hopp_installed_cost_pipeline = hybrid_plant_pipeline.grid._financial_model.SystemCosts.total_installed_cost
    total_electrolyzer_cost = H2A_Results['scaled_total_installed_cost']
    print(H2A_Results['scaled_total_installed_cost_kw'])
    total_system_installed_cost = total_hopp_installed_cost + total_electrolyzer_cost
    total_system_installed_cost_pipeline = total_hopp_installed_cost_pipeline + total_electrolyzer_cost
    annual_operating_cost_h2 = H2A_Results['Fixed O&M'] * H2_Results['hydrogen_annual_output']
    annual_operating_cost_desal = desal_opex
    total_annual_operating_costs =  annual_operating_cost_h2 + annual_operating_cost_desal + cost_to_buy_from_grid - profit_from_selling_to_grid

    # h_lcoe_no_op_cost = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
    #                    0, 0.07, useful_life)

    h_lcoe = lcoe_calc((H2_Results['hydrogen_annual_output']), total_system_installed_cost,
                        total_annual_operating_costs, discount_rate, useful_life)

    # Cashflow Financial Calculation
    discount_rate = scenario['Discount Rate']
    cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
    cf_wind_annuals_pipeline = hybrid_plant_pipeline.wind._financial_model.Outputs.cf_annual_costs
    if solar_size_mw > 0:
        cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
    else:
        cf_solar_annuals = np.zeros(30)


    if h2_model == 'H2A':
        #cf_h2_annuals = H2A_Results['expenses_annual_cashflow'] # This is unreliable.
        pass  
    elif h2_model == 'Simple':
        from examples.H2_Analysis.H2_cost_model import basic_H2_cost_model
        
        cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement = \
            basic_H2_cost_model(electrolyzer_size, useful_life, atb_year,
            electrical_generation_timeseries, H2_Results['hydrogen_annual_output'], scenario['H2 PTC'])

    cf_operational_annuals = [-total_annual_operating_costs for i in range(30)]

    cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals, cf_h2_annuals],['Wind', 'Solar', 'H2'])

    cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}_{}MW.csv".format(site_name, scenario_choice, atb_year, discount_rate,turbine_rating_mw)))

    #NPVs of wind, solar, H2
    npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
    
    npv_wind_costs_pipeline = npf.npv(discount_rate, cf_wind_annuals_pipeline)
    npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)
    npv_h2_costs = npf.npv(discount_rate, cf_h2_annuals)
    print("NPV H2 Costs using {} model: {}".format(h2_model,npv_h2_costs))
    npv_operating_costs = npf.npv(discount_rate, cf_operational_annuals)
    npv_desal_costs = -desal_capex
    print("Desal CAPEX: ",desal_capex)

    npv_total_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs
    npv_total_costs_pipeline = npv_wind_costs_pipeline + npv_solar_costs + npv_h2_costs
    npv_total_costs_w_operating_costs = npv_wind_costs+npv_solar_costs+npv_h2_costs+npv_operating_costs
    npv_total_costs_w_operating_costs_pipeline = npv_wind_costs_pipeline+npv_solar_costs+npv_h2_costs+npv_operating_costs

    LCOH_cf_method_wind = -npv_wind_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_wind_pipeline = -npv_wind_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_solar = -npv_solar_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_h2_costs = -npv_h2_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_desal_costs = -npv_desal_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_operating_costs = -npv_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)

    LCOH_cf_method = -npv_total_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_pipeline = -npv_total_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_w_operating_costs = -npv_total_costs_w_operating_costs / (H2_Results['hydrogen_annual_output'] * useful_life)
    LCOH_cf_method_w_operating_costs_pipeline = -npv_total_costs_w_operating_costs_pipeline / (H2_Results['hydrogen_annual_output'] * useful_life)
    financial_summary_df = pd.DataFrame([scenario['Useful Life'], wind_cost_kw, solar_cost_kw, electrolyzer_capex_kw,
                                            scenario['Debt Equity'], atb_year, scenario['Wind PTC'], scenario['H2 PTC'],scenario['Wind ITC'],
                                            discount_rate, npv_wind_costs, npv_solar_costs, npv_h2_costs, LCOH_cf_method, LCOH_cf_method_pipeline, LCOH_cf_method_w_operating_costs, LCOH_cf_method_w_operating_costs_pipeline],
                                        ['Useful Life', 'Wind Cost KW', 'Solar Cost KW', 'Electrolyzer Cost KW', 'Debt Equity',
                                            'ATB Year', 'Wind PTC', 'H2 PTC', 'Wind ITC', 'Discount Rate', 'NPV Wind Expenses', 'NPV Solar Expenses', 'NPV H2 Expenses', 'LCOH cf method HVDC','LCOH cf method Pipeline','LCOH cf method HVDC w/operating cost','LCOH cf method Pipeline w/operating cost'])
    financial_summary_df.to_csv(os.path.join(results_dir, 'Financial Summary_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model)))

    # Gut Check H2 calculation (non-levelized)
    total_installed_and_operational_lifetime_cost = total_system_installed_cost + (30 * total_annual_operating_costs)
    lifetime_h2_production = 30 * H2_Results['hydrogen_annual_output']
    gut_check_h2_cost_kg = total_installed_and_operational_lifetime_cost / lifetime_h2_production

    print("Gut Check H2 Cost:",gut_check_h2_cost_kg)
    print("HVDC Scenario: LCOH w/o Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method)
    print("HVDC Scenario: LCOH WITH Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_w_operating_costs)

    print("Pipeline Scenario: LCOH w/o Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_pipeline)
    print("Pipeline Scenario: LCOH WITH Operating Cost for H2, Desal, Pressure Vessel, Grid Electrical Cost:", LCOH_cf_method_w_operating_costs_pipeline)
