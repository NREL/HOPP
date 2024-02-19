import os
import pathlib
from hopp.simulation.hybrid_simulation import HybridSimulation, TechnologiesConfig
import json
from hopp.tools.analysis import create_cost_calculator
from hopp.simulation.hopp_interface import HoppInterface
import pandas as pd

def hopp_for_h2_floris(site, scenario, technologies, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, storage_hours,
                wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
                kw_continuous, load,
                custom_powercurve,
                interconnection_size_mw, grid_connected_hopp=True, wind_om_cost_kw=42, solar_om_cost_kw=0, turbine_parent_path=None, ppa_price=0.0,
                wave_cost_dict={}):
    '''
    Runs HOPP for H2 analysis purposes
    :param site: :class:`hybrid.sites.site_info.SiteInfo`,
        Hybrid plant site information which includes layout, location and resource data
    :param scenario: ``dict``,
        Dictionary of scenario options, includes location, year, technology pricing
    :param technologies: nested ``dict``; i.e., ``{'pv': {'system_capacity_kw': float}}``
        Names of technologies to include and configuration dictionaries
        For details on configurations dictionaries see:
            ===============   =============================================
            Technology key    Class for reference
            ===============   =============================================
            ``pv``            :class:`hybrid.pv_source.PVPlant`
            ``wind``          :class:`hybrid.wind_source.WindPlant`
            ``tower``         :class:`hybrid.tower_source.TowerPlant`
            ``trough``        :class:`hybrid.trough_source.TroughPlant`
            ``battery``       :class:`hybrid.battery.Battery`
            ===============   =============================================
    :param wind_size_mw: ``float``,
        Wind technology size in MW
    :param solar_size_mw: ``float``,
        Solar technology size in MW
    :param storage_size_mw: ``float``,
        Storage technology size in MW
    :param storage_size_mwh: ``float``,
        Storage technology size in MWh
    :param storage_hours: ``float``,
        Number of hours of storage at maximum output rating.
    :param wind_cost_kw: ``float``,
        Wind installed cost in $/kw
    :param solar_cost_kw: ``float``,
        Solar installed cost in $/kW
    :param storage_cost_kw: ``float``,
        Storage cost in $/kW
    :param storage_cost_kwh: ``float``,
        Storage cost in $/kWh
    :param kw_continuous: ``float``,
        kW rating of electrolyzer
    :param load: ``list``,
        (8760) hourly load profile of electrolyzer in kW. Default is continuous load at kw_continuous rating
    :param custom_powercurve: ``bool``,
        Flag to determine if custom wind turbine powercurve file is loaded
    :param interconnection_size_mw: ``float``,
        Interconnection size in MW
    :param grid_connected_hopp: ``bool``,
        Flag for on-grid operation. Enables buying/selling of energy to grid.
    :returns:

    :param hybrid_plant: :class: `hybrid.hybrid_simulation.HybridSimulation`,
        Base class for simulation a Hybrid Plant
    :param combined_hybrid_power_production_hopp: ``list``,
        (8760x1) hourly sequence of combined pv and wind power in kW
    :param combined_pv_wind_curtailment_hopp: ``list``,
        (8760x1) hourly sequence of combined pv and wind curtailment/spilled energy in kW
    :param energy_shortfall_hopp: ``list``,
        (8760x1) hourly sequence of energy shortfall vs. load in kW
    :param annual_energies: ``dict``,
        Dictionary of AEP for each technology
    :param wind_plus_solar_npv: ``float``,
        Combined Net present value of wind + solar technologies
    :param npvs: ``dict``,
        Dictionary of net present values of technologies
    :param lcoe: ``float``
        Levelized cost of electricity for hybrid plant
    '''

    # Create model
    dispatch_options = {'battery_dispatch': 'heuristic'}
    if 'grid' not in technologies.keys():
        technologies['grid'] = {'interconnect_kw': interconnection_size_mw * 1e3}
    
    data = {"technologies": technologies,
            "site": site}
    # print(data)
    # create HOPP instance
    hi = HoppInterface(data)
    # hybrid_plant = HybridSimulation(site, technologies, dispatch_options=dispatch_options)
    hi.system.setup_cost_calculator(create_cost_calculator(interconnection_size_mw,
                                                              bos_cost_source='CostPerMW',
                                                              wind_installed_cost_mw=wind_cost_kw * 1000,
                                                              solar_installed_cost_mw=solar_cost_kw * 1000,
                                                              storage_installed_cost_mw=storage_cost_kw * 1000,
                                                              storage_installed_cost_mwh=storage_cost_kwh * 1000
                                                              ))
    
    if 'wave' in technologies.keys():
        hi.system.wave.create_mhk_cost_calculator(wave_cost_dict)

    hi.system.set_om_costs_per_kw(pv_om_per_kw=solar_om_cost_kw, wind_om_per_kw=wind_om_cost_kw, hybrid_om_per_kw=None)
    if solar_size_mw > 0:
        hi.system.pv._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hi.system.pv._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        # hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
        # if scenario['ITC Available']:
        #     hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 26
        # else:
        #     hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0

    # TODO clean this up for better function with custom financial model
    if 'wind' in technologies and 'fin_model' in technologies["grid"].keys() and type(technologies["grid"]["fin_model"]) == {}:
        # hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33
        # hybrid_plant.wind.wake_model = 3
        # hybrid_plant.wind.value("wake_int_loss", 3)
        hi.system.wind._financial_model.FinancialParameters.analysis_period = scenario['Useful Life']
        hi.system.wind._financial_model.FinancialParameters.system_capacity = wind_size_mw * 1000
        # hybrid_plant.wind.om_capacity =
        hi.system.wind._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        hi.system.wind._financial_model.value("debt_option", 0)
        hi.system.wind._financial_model.FinancialParameters.debt_percent = scenario['Debt Equity']
        hi.system.wind._financial_model.value("debt_option", 0)
        ptc_val = scenario['Wind PTC']

        interim_list = list(
            hi.system.wind._financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hi.system.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        #hybrid_plant.wind._system_model.Turbine.wind_turbine_hub_ht = scenario['Tower Height']

        hi.system.wind._financial_model.TaxCreditIncentives.itc_fed_percent = scenario['Wind ITC']
        hi.system.wind._financial_model.FinancialParameters.real_discount_rate = 7
    if custom_powercurve:
        if turbine_parent_path == None:
            turbine_parent_path = os.path.abspath(os.path.dirname(__file__))
        powercurve_file = open(os.path.join(turbine_parent_path, scenario['Powercurve File']))
        powercurve_file_extension = pathlib.Path(os.path.join(turbine_parent_path, scenario['Powercurve File'])).suffix
        if powercurve_file_extension == '.csv':
            curve_data = pd.read_csv(os.path.join(turbine_parent_path, scenario['Powercurve File']))
            wind_speed = curve_data['Wind Speed [m/s]'].values.tolist()
            curve_power = curve_data['Power [kW]']
            hi.system.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
            hi.system.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power

        else:
            powercurve_data = json.load(powercurve_file)
            powercurve_file.close()
            hi.system.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = \
                powercurve_data['turbine_powercurve_specification']['wind_speed_ms']
            hi.system.wind._system_model.Turbine.wind_turbine_powercurve_powerout = \
                powercurve_data['turbine_powercurve_specification']['turbine_power_output']

    if 'wind' in technologies: # this was a contested line in a refactor merge - may cause issue
        hi.system.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hi.system.ppa_price = ppa_price
    hi.system.simulate(scenario['Useful Life'])

    # HOPP Specific Energy Metrics
    ### wind wind losses, for wind only farms ###
    combined_hybrid_power_production_hopp = hi.system.grid._system_model.Outputs.system_pre_interconnect_kwac[0:8759]
    energy_shortfall_hopp = [x - y for x, y in
                             zip(load,combined_hybrid_power_production_hopp)]
    energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
    combined_pv_wind_curtailment_hopp = [x - y for x, y in
                             zip(combined_hybrid_power_production_hopp,load)]
    combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]

    # super simple dispatch battery model with no forecasting TODO: add forecasting
    # print("Length of 'energy_shortfall_hopp is {}".format(len(energy_shortfall_hopp)))
    # print("Length of 'combined_pv_wind_curtailment_hopp is {}".format(len(combined_pv_wind_curtailment_hopp)))
    # TODO: Fix bug in dispatch model that errors when first curtailment >0
    combined_pv_wind_curtailment_hopp[0] = 0

    # Save the outputs
    annual_energies = hi.system.annual_energies
    wind_plus_solar_npv = hi.system.net_present_values.hybrid
    npvs = hi.system.net_present_values
    lcoe = hi.system.lcoe_real
    lcoe_nom = hi.system.lcoe_nom
    # print('lcoe nominal: ', lcoe_nom)
    # print('annual energy',annual_energies)
    # print('discount rate', hybrid_plant.wind._financial_model.FinancialParameters.real_discount_rate)

    return hi.system, combined_hybrid_power_production_hopp, combined_pv_wind_curtailment_hopp, \
           energy_shortfall_hopp,\
           annual_energies, wind_plus_solar_npv, npvs, lcoe, lcoe_nom