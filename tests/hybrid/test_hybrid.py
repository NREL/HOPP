from pytest import approx, fixture

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.layout.hybrid_layout import WindBoundaryGridParameters, PVGridParameters
from hybrid.hybrid_simulation import HybridSimulation


@fixture
def site():
    return SiteInfo(flatirons_site)


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
                'battery': {
                    'system_capacity_kwh': batt_kw * 4,
                    'system_capacity_kw': 5000
                }}


def test_hybrid_wind_only(site):
    wind_only = {'wind': technologies['wind']}
    hybrid_plant = HybridSimulation(wind_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.simulate(25)
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == 0
    assert aeps.wind == approx(33615479, 1e3)
    assert aeps.hybrid == approx(33615479, 1e3)

    assert npvs.pv == 0
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
    assert aeps.wind == 0
    assert aeps.hybrid == approx(9884106.55, 1e-3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.wind == 0
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


def test_hybrid_with_storage_dispatch(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw)
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

    assert aeps.pv == approx(9860000, rel=1e-2)
    assert aeps.wind == approx(33300000, rel=1e-1)
    assert aeps.battery == approx(-131300, rel=1e-2)
    assert aeps.hybrid == approx(42700000, rel=1e-1)

    assert npvs.pv == approx(-1300000, rel=1e-2)
    assert npvs.wind == approx(-4060000, rel=1e-1)
    assert npvs.battery == approx(-11820000, rel=1e-2)
    assert npvs.hybrid == approx(-17200000, rel=1e-2)

    assert taxes.pv[1] == approx(106000, rel=1e-2)
    assert taxes.wind[1] == approx(404000, rel=1e-2)
    assert taxes.battery[1] == approx(511000, rel=1e-2)
    assert taxes.hybrid[1] == approx(1020000, rel=1e-2)

    assert apv.pv[1] == approx(0, rel=1e-2)
    assert apv.wind[1] == approx(0, rel=1e-2)
    assert apv.battery[1] == approx(158392, rel=1e-2)
    assert apv.hybrid[1] == approx(38300, rel=1e-2)

    assert debt.pv[1] == approx(0, 1e-3)
    assert debt.wind[1] == approx(0, 1e-3)
    assert debt.battery[1] == approx(0, 1e-3)
    assert debt.hybrid[1] == approx(0, 1e-3)

    assert esv.pv[1] == approx(294925, rel=1e-2)
    assert esv.wind[1] == approx(999000, rel=1e-1)
    assert esv.battery[1] == approx(168381, rel=1e-2)
    assert esv.hybrid[1] == approx(1330000, rel=1e-2)

    assert depr.pv[1] == approx(762811, rel=1e-2)
    assert depr.wind[1] == approx(2651114, rel=1e-2)
    assert depr.battery[1] == approx(2555389, rel=1e-2)
    assert depr.hybrid[1] == approx(5969315, rel=1e-2)

    assert insr.pv[0] == approx(0, rel=1e-2)
    assert insr.wind[0] == approx(0, rel=1e-2)
    assert insr.battery[0] == approx(0, rel=1e-2)
    assert insr.hybrid[0] == approx(0, rel=1e-2)

    assert om.pv[1] == approx(74993, rel=1e-2)
    assert om.wind[1] == approx(420000, rel=1e-2)
    assert om.battery[1] == approx(75000, rel=1e-2)
    assert om.hybrid[1] == approx(569993, rel=1e-2)

    assert rev.pv[1] == approx(295000, rel=1e-2)
    assert rev.wind[1] == approx(1000000, rel=1e-2)
    assert rev.battery[1] == approx(168000, rel=1e-2)
    assert rev.hybrid[1] == approx(1330000, rel=1e-2)

    assert tc.pv[1] == approx(1123104, rel=1e-2)
    assert tc.wind[1] == approx(500000, rel=1e-2)
    assert tc.battery[1] == approx(0, rel=1e-2)
    assert tc.hybrid[1] == approx(1659156, rel=1e-2)


def test_hybrid_om_costs_error(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.battery._financial_model.SystemCosts.om_production = (1,)
    try:
        hybrid_plant.simulate()
    except ValueError as e:
        assert e


def test_hybrid_om_costs(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
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
        assert var_om_costs.pv[i] + var_om_costs.wind[i] + var_om_costs.battery[i] \
               == approx(var_om_costs.hybrid[i], rel=1e-3)
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
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = (1,)
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_amount = (2,)
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_amount = (3,)
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.simulate()

    ptc_wind = hybrid_plant.wind._financial_model.value("cf_ptc_fed")[1]
    assert ptc_wind == hybrid_plant.wind._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.wind.annual_energy_kw

    ptc_pv = hybrid_plant.pv._financial_model.value("cf_ptc_fed")[1]
    assert ptc_pv == hybrid_plant.pv._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.pv.annual_energy_kw

    ptc_batt = hybrid_plant.battery._financial_model.value("cf_ptc_fed")[1]
    assert ptc_batt == hybrid_plant.battery._financial_model.value("ptc_fed_amount")[0]\
           * hybrid_plant.battery._financial_model.LCOS.batt_annual_discharge_energy[1]

    ptc_hybrid = hybrid_plant.grid._financial_model.value("cf_ptc_fed")[1]
    ptc_fed_amount = hybrid_plant.grid._financial_model.value("ptc_fed_amount")[0]
    assert ptc_fed_amount == approx(1.22941)
    assert ptc_hybrid == approx(ptc_fed_amount * hybrid_plant.grid._financial_model.Outputs.cf_energy_net[1], rel=1e-3)
