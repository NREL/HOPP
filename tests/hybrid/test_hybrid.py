from pytest import approx, fixture
from pathlib import Path

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.layout.hybrid_layout import WindBoundaryGridParameters, PVGridParameters
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.keys import set_nrel_key_dot_env

set_nrel_key_dot_env()

@fixture
def site():
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)


interconnection_size_kw = 15000
pv_kw = 5000
wind_kw = 10000
batt_kw = 5000
technologies = {'pv': {
                    'system_capacity_kw': pv_kw,
                    'layout_params': PVGridParameters(x_position=0.5,
                                                      y_position=0.5,
                                                      aspect_power=0,
                                                      gcr=0.5,
                                                      s_buffer=2,
                                                      x_buffer=2)
                },
                'wind': {
                    'num_turbines': 5,
                    'turbine_rating_kw': wind_kw / 5,
                    'layout_mode': 'boundarygrid',
                    'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                                border_offset=0.5,
                                                                grid_angle=0.5,
                                                                grid_aspect_power=0.5,
                                                                row_phase_offset=0.5)
                },
                'trough': {
                    'cycle_capacity_kw': 15 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0
                },
                'tower': {
                    'cycle_capacity_kw': 15 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0
                },
                'battery': {
                    'system_capacity_kwh': batt_kw * 4,
                    'system_capacity_kw': batt_kw
                }}


def test_hybrid_wind_only(site):
    wind_only = {'wind': technologies['wind']}
    hybrid_plant = HybridSimulation(wind_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.simulate(25)
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.wind == approx(33615479, 1e3)
    assert aeps.hybrid == approx(33615479, 1e3)

    assert npvs.wind == approx(-13692784, 1e3)
    assert npvs.hybrid == approx(-13692784, 1e3)


def test_hybrid_pv_only(site):
    solar_only = {'pv': technologies['pv']}
    hybrid_plant = HybridSimulation(solar_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(9884106.55, 1e-3)
    assert aeps.hybrid == approx(9884106.55, 1e-3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.hybrid == approx(-5121293, 1e3)


def test_hybrid(site):
    """
    Performance from Wind is slightly different from wind-only case because the solar presence modified the wind layout
    """
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind')}
    hybrid_plant = HybridSimulation(solar_wind_hybrid, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    # plt.show()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(8703525.94, 13)
    assert aeps.wind == approx(33615479.57, 1e3)
    assert aeps.hybrid == approx(41681662.63, 1e3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.wind == approx(-13909363, 1e3)
    assert npvs.hybrid == approx(-19216589, 1e3)


def test_wind_pv_with_storage_dispatch(site):
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    hybrid_plant = HybridSimulation(wind_pv_battery, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    taxes = hybrid_plant.federal_taxes
    apv = hybrid_plant.energy_purchases_values
    debt = hybrid_plant.debt_payment
    esv = hybrid_plant.energy_sales_values
    depr = hybrid_plant.federal_depreciation_totals
    insr = hybrid_plant.insurance_expenses
    om = hybrid_plant.om_total_expenses
    rev = hybrid_plant.total_revenues
    tc = hybrid_plant.tax_incentives

    assert aeps.pv == approx(9882421, rel=0.05)
    assert aeps.wind == approx(33637983, rel=0.05)
    assert aeps.battery == approx(-31287, rel=0.05)
    assert aeps.hybrid == approx(43489117, rel=0.05)

    assert npvs.pv == approx(-853226, rel=5e-2)
    assert npvs.wind == approx(-4380277, rel=5e-2)
    assert npvs.battery == approx(-6889961, rel=5e-2)
    assert npvs.hybrid == approx(-11861790, rel=5e-2)

    assert taxes.pv[1] == approx(94661, rel=5e-2)
    assert taxes.wind[1] == approx(413068, rel=5e-2)
    assert taxes.battery[1] == approx(297174, rel=5e-2)
    assert taxes.hybrid[1] == approx(804904, rel=5e-2)

    assert apv.pv[1] == approx(0, rel=5e-2)
    assert apv.wind[1] == approx(0, rel=5e-2)
    assert apv.battery[1] == approx(40158, rel=5e-2)
    assert apv.hybrid[1] == approx(3050, rel=5e-2)

    assert debt.pv[1] == approx(0, rel=5e-2)
    assert debt.wind[1] == approx(0, rel=5e-2)
    assert debt.battery[1] == approx(0, rel=5e-2)
    assert debt.hybrid[1] == approx(0, rel=5e-2)

    assert esv.pv[1] == approx(353105, rel=5e-2)
    assert esv.wind[1] == approx(956067, rel=5e-2)
    assert esv.battery[1] == approx(80449, rel=5e-2)
    assert esv.hybrid[1] == approx(1352445, rel=5e-2)

    assert depr.pv[1] == approx(762811, rel=5e-2)
    assert depr.wind[1] == approx(2651114, rel=5e-2)
    assert depr.battery[1] == approx(1486921, rel=5e-2)
    assert depr.hybrid[1] == approx(4900847, rel=5e-2)

    assert insr.pv[0] == approx(0, rel=5e-2)
    assert insr.wind[0] == approx(0, rel=5e-2)
    assert insr.battery[0] == approx(0, rel=5e-2)
    assert insr.hybrid[0] == approx(0, rel=5e-2)

    assert om.pv[1] == approx(74993, rel=5e-2)
    assert om.wind[1] == approx(420000, rel=5e-2)
    assert om.battery[1] == approx(75000, rel=5e-2)
    assert om.hybrid[1] == approx(569993, rel=5e-2)

    assert rev.pv[1] == approx(353105, rel=5e-2)
    assert rev.wind[1] == approx(956067, rel=5e-2)
    assert rev.battery[1] == approx(80449, rel=5e-2)
    assert rev.hybrid[1] == approx(1352445, rel=5e-2)

    assert tc.pv[1] == approx(1123104, rel=5e-2)
    assert tc.wind[1] == approx(504569, rel=5e-2)
    assert tc.battery[1] == approx(0, rel=5e-2)
    assert tc.hybrid[1] == approx(1646170, rel=5e-2)

def test_tower_pv_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'tower': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000}}

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw_test,
                                    dispatch_options={'is_test_start_year': True,
                                                      'is_test_end_year': True})
    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.tower.value('helio_width', 8.0)
    hybrid_plant.tower.value('helio_height', 8.0)

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104053614.17, 1e-3)
    assert aeps.tower == approx(3769716.50, 5e-2)
    assert aeps.hybrid == approx(107780622.67, 1e-2)

    # TODO: check npv for csp would require a full simulation
    assert npvs.pv == approx(45233832.23, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)


def test_trough_pv_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'trough': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000}}

    solar_hybrid = {key: technologies_test[key] for key in ('trough', 'pv')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw_test,
                                    dispatch_options={'is_test_start_year': True,
                                                      'is_test_end_year': True})

    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104053614.17, 1e-3)
    assert aeps.trough == approx(1871471.58, 2e-2)
    assert aeps.hybrid == approx(105926003.55, 1e-3)

    assert npvs.pv == approx(45233832.23, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)


def test_tower_pv_battery_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'tower': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000},
                         'battery': {'system_capacity_kwh': 40 * 1000,
                                     'system_capacity_kw': 20 * 1000}}

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv', 'battery')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw_test,
                                    dispatch_options={'is_test_start_year': True,
                                                      'is_test_end_year': True})
    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.tower.value('helio_width', 10.0)
    hybrid_plant.tower.value('helio_height', 10.0)

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104053614.17, 1e-3)
    assert aeps.tower == approx(3769716.50, 5e-2)
    assert aeps.battery == approx(-9449.70, 2e-1)
    assert aeps.hybrid == approx(107882747.80, 1e-2)

    assert npvs.pv == approx(45233832.23, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)


def test_hybrid_om_costs_error(site):
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    hybrid_plant = HybridSimulation(wind_pv_battery, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.battery._financial_model.SystemCosts.om_production = (1,)
    try:
        hybrid_plant.simulate()
    except ValueError as e:
        assert e


def test_hybrid_om_costs(site):
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    hybrid_plant = HybridSimulation(wind_pv_battery, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    # set all O&M costs to 0 to start
    hybrid_plant.wind.om_fixed = 0
    hybrid_plant.wind.om_capacity = 0
    hybrid_plant.wind.om_variable = 0
    hybrid_plant.pv.om_fixed = 0
    hybrid_plant.pv.om_capacity = 0
    hybrid_plant.pv.om_variable = 0
    hybrid_plant.battery.om_fixed = 0
    hybrid_plant.battery.om_capacity = 0
    hybrid_plant.battery.om_variable = 0

    # test variable costs
    hybrid_plant.wind.om_variable = 5
    hybrid_plant.pv.om_variable = 2
    hybrid_plant.battery.om_variable = 3
    hybrid_plant.simulate()
    var_om_costs = hybrid_plant.om_variable_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(var_om_costs.hybrid)):
        assert var_om_costs.pv[i] + var_om_costs.wind[i] + var_om_costs.battery[i] == approx(var_om_costs.hybrid[i], rel=1e-1)
        assert total_om_costs.pv[i] == approx(var_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(var_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(var_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(var_om_costs.hybrid[i])
    hybrid_plant.wind.om_variable = 0
    hybrid_plant.pv.om_variable = 0
    hybrid_plant.battery.om_variable = 0

    # test fixed costs
    hybrid_plant.wind.om_fixed = 5
    hybrid_plant.pv.om_fixed = 2
    hybrid_plant.battery.om_fixed = 3
    hybrid_plant.simulate()
    fixed_om_costs = hybrid_plant.om_fixed_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(fixed_om_costs.hybrid)):
        assert fixed_om_costs.pv[i] + fixed_om_costs.wind[i] + fixed_om_costs.battery[i] \
               == approx(fixed_om_costs.hybrid[i])
        assert total_om_costs.pv[i] == approx(fixed_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(fixed_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(fixed_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(fixed_om_costs.hybrid[i])
    hybrid_plant.wind.om_fixed = 0
    hybrid_plant.pv.om_fixed = 0
    hybrid_plant.battery.om_fixed = 0

    # test capacity costs
    hybrid_plant.wind.om_capacity = 5
    hybrid_plant.pv.om_capacity = 2
    hybrid_plant.battery.om_capacity = 3
    hybrid_plant.simulate()
    cap_om_costs = hybrid_plant.om_capacity_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(cap_om_costs.hybrid)):
        assert cap_om_costs.pv[i] + cap_om_costs.wind[i] + cap_om_costs.battery[i] \
               == approx(cap_om_costs.hybrid[i])
        assert total_om_costs.pv[i] == approx(cap_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(cap_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(cap_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(cap_om_costs.hybrid[i])
    hybrid_plant.wind.om_capacity = 0
    hybrid_plant.pv.om_capacity = 0
    hybrid_plant.battery.om_capacity = 0


def test_hybrid_tax_incentives(site):
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    hybrid_plant = HybridSimulation(wind_pv_battery, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0.0
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = (1,)
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_amount = (2,)
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_amount = (3,)
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.simulate()

    ptc_wind = hybrid_plant.wind._financial_model.value("cf_ptc_fed")[1]
    assert ptc_wind == approx(hybrid_plant.wind._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.wind.annual_energy_kwh, rel=1e-3)

    ptc_pv = hybrid_plant.pv._financial_model.value("cf_ptc_fed")[1]
    assert ptc_pv == approx(hybrid_plant.pv._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.pv.annual_energy_kwh, rel=1e-3)

    ptc_batt = hybrid_plant.battery._financial_model.value("cf_ptc_fed")[1]
    assert ptc_batt == approx(hybrid_plant.battery._financial_model.value("ptc_fed_amount")[0]
           * hybrid_plant.battery._financial_model.LCOS.batt_annual_discharge_energy[1], rel=1e-3)

    ptc_hybrid = hybrid_plant.grid._financial_model.value("cf_ptc_fed")[1]
    ptc_fed_amount = hybrid_plant.grid._financial_model.value("ptc_fed_amount")[0]
    assert ptc_fed_amount == approx(1.229, rel=1e-2)
    assert ptc_hybrid == approx(ptc_fed_amount * hybrid_plant.grid._financial_model.Outputs.cf_energy_net[1], rel=1e-3)
