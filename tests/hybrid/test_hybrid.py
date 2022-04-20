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

    assert aeps.pv == approx(8018410.08, 1e-3) 
    assert aeps.wind == 0
    assert aeps.hybrid == approx(8018410.08, 1e-3) 

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

    assert aeps.pv == approx(8017896.19, 1e-3)
    assert aeps.wind == approx(14741474, 1e-3)
    assert aeps.battery == approx(-156489, 1e-3) 
    assert aeps.hybrid == approx(22602881, 1e-3)

    npvs = hybrid_plant.net_present_values
    assert npvs.pv == approx(-1728818, 1e-3)
    assert npvs.wind == approx(-10372732, 1e-3)
    assert npvs.battery == approx(-11772684, 1e-3)
    assert npvs.hybrid == approx(-23895751, 1e-3)

    taxes = hybrid_plant.federal_taxes
    assert taxes.pv[1] == approx(116646, 1e-3)
    assert taxes.wind[1] == approx(513418, 1e-3)
    assert taxes.battery[1] == approx(510419, 1e-3)
    assert taxes.hybrid[1] == approx(1144696, 1e-3)

    apv = hybrid_plant.energy_purchases_values
    assert apv.pv[1] == approx(0, 1e-3)
    assert apv.wind[1] == approx(0, 1e-3)
    assert apv.battery[1] == approx(158961, 1e-3)
    assert apv.hybrid[1] == approx(89995, 1e-2)

    debt = hybrid_plant.debt_payment
    assert debt.pv[1] == approx(0, 1e-3)
    assert debt.wind[1] == approx(0, 1e-3)
    assert debt.battery[1] == approx(0, 1e-3)
    assert debt.hybrid[1] == approx(0, 1e-3)

    esv = hybrid_plant.energy_sales_values
    assert esv.pv[1] == approx(296504, 1e3)
    assert esv.wind[1] == approx(1009139, 1e3)
    assert esv.battery[1] == approx(167015, 1e3)
    assert esv.hybrid[1] == approx(1340129, 1e3)

    depr = hybrid_plant.federal_depreciation_totals
    assert depr.pv[1] == approx(762811, 1e3)
    assert depr.wind[1] == approx(2651114, 1e3)
    assert depr.battery[1] == approx(2555389, 1e3)
    assert depr.hybrid[1] == approx(5969315, 1e3)

    insr = hybrid_plant.insurance_expenses
    assert insr.pv[0] == approx(0, 1e3)
    assert insr.wind[0] == approx(0, 1e3)
    assert insr.battery[0] == approx(0, 1e3)
    assert insr.hybrid[0] == approx(0, 1e3)

    om = hybrid_plant.om_total_expenses
    assert om.pv[1] == approx(74993, 1e3)
    assert om.wind[1] == approx(420000, 1e3)
    assert om.battery[1] == approx(75000, 1e3)
    assert om.hybrid[1] == approx(569993, 1e3)

    rev = hybrid_plant.total_revenues
    assert rev.pv[1] == approx(240536, 1e3)
    assert rev.wind[1] == approx(442244, 1e3)
    assert rev.battery[1] == approx(175835, 1e3)
    assert rev.hybrid[1] == approx(768082, 1e3)

    tc = hybrid_plant.tax_incentives
    assert tc.pv[1] == approx(1123104, 1e3)
    assert tc.wind[1] == approx(221122, 1e3)
    assert tc.battery[1] == approx(0, 1e3)
    assert tc.hybrid[1] == approx(1379131, 1e3)


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
    assert ptc_fed_amount == approx(1.35788, rel=1e-3) 
    assert ptc_hybrid == approx(ptc_fed_amount * hybrid_plant.grid._financial_model.Outputs.cf_energy_net[1], rel=1e-3)
