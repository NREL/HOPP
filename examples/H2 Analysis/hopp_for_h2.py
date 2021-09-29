from simple_dispatch import SimpleDispatch
from hybrid.hybrid_simulation import HybridSimulation
import json
from tools.analysis import create_cost_calculator

def hopp_for_h2(site, scenario, technologies, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                kw_continuous, load,
                custom_powercurve,
                interconnection_size_mw, grid_connected_hopp=True):
    '''
    Runs HOPP for H2 analysis purposes
    @return:
    # '''
    # solar_size_mw = 0
    # storage_size_mw = 0
    # storage_size_mwh = 0
    # wind_size_mw = forced_system_size


    # Create model
    if not grid_connected_hopp:
        interconnection_size_mw = kw_continuous / 1000

    hybrid_plant = HybridSimulation(technologies, site, scenario['Rotor Diameter'], scenario['Tower Height'],
                                    interconnect_kw=technologies['grid'] * 1000,
                                    storage_kw=storage_size_mw * 1000,
                                    storage_kwh=storage_size_mwh * 1000,
                                    storage_hours=storage_hours)
    hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                              bos_cost_source='CostPerMW',
                                                              wind_installed_cost_mw=wind_cost_kw * 1000,
                                                              solar_installed_cost_mw=solar_cost_kw * 1000,
                                                              storage_installed_cost_mw=storage_cost_kw * 1000,
                                                              storage_installed_cost_mwh=storage_cost_kwh * 1000
                                                              ))
    hybrid_plant.wind.system_model.Turbine.wind_resource_shear = 0.33
    if solar_size_mw > 0:
        hybrid_plant.solar.financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.solar.financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['ITC Available']:
            hybrid_plant.solar.financial_model.TaxCreditIncentives.itc_fed_percent = 26
        else:
            hybrid_plant.solar.financial_model.TaxCreditIncentives.itc_fed_percent = 0

    if 'wind' in technologies:
        hybrid_plant.wind.financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hybrid_plant.wind.financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        if scenario['PTC Available']:
            ptc_val = 0.022
        else:
            ptc_val = 0.0

        interim_list = list(
            hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hybrid_plant.wind.financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        hybrid_plant.wind.system_model.Turbine.wind_turbine_hub_ht = scenario['Tower Height']

    if custom_powercurve:
        powercurve_file = open(scenario['Powercurve File'])
        powercurve_data = json.load(powercurve_file)
        powercurve_file.close()
        hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_windspeeds = \
            powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
        hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_powerout = \
            powercurve_data['turbine_powercurve_specification']['turbine_power_output']

    hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hybrid_plant.ppa_price = 0.05
    hybrid_plant.simulate(scenario['Useful Life'])

    # HOPP Specific Energy Metrics
    combined_pv_wind_power_production_hopp = hybrid_plant.grid.system_model.Outputs.system_pre_interconnect_kwac[0:8759]
    energy_shortfall_hopp = [x - y for x, y in
                             zip(load,combined_pv_wind_power_production_hopp)]
    energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
    combined_pv_wind_curtailment_hopp = [x - y for x, y in
                             zip(combined_pv_wind_power_production_hopp,load)]
    combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]

    # super simple dispatch battery model with no forecasting TODO: add forecasting
    # print("Length of 'energy_shortfall_hopp is {}".format(len(energy_shortfall_hopp)))
    # print("Length of 'combined_pv_wind_curtailment_hopp is {}".format(len(combined_pv_wind_curtailment_hopp)))
    # TODO: Fix bug in dispatch model that errors when first curtailment >0
    combined_pv_wind_curtailment_hopp[0] = 0

    # Save the outputs
    annual_energies = hybrid_plant.annual_energies
    wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.solar
    npvs = hybrid_plant.net_present_values
    lcoe = hybrid_plant.lcoe_real.hybrid

    return hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp, \
           energy_shortfall_hopp,\
           annual_energies, wind_plus_solar_npv, npvs, lcoe