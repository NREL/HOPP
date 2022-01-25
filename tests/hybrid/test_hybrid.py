import pytest

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.layout.hybrid_layout import WindBoundaryGridParameters, PVGridParameters
from hybrid.hybrid_simulation import HybridSimulation


@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


interconnection_size_kw = 15000
technologies = {'pv': {
                    'system_capacity_kw': 5000,
                    'layout_params': PVGridParameters(x_position=0.5,
                                                      y_position=0.5,
                                                      aspect_power=0,
                                                      gcr=0.5,
                                                      s_buffer=2,
                                                      x_buffer=2)
                },
                'wind': {
                    'num_turbines': 5,
                    'turbine_rating_kw': 2000,
                    'layout_mode': 'boundarygrid',
                    'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                                border_offset=0.5,
                                                                grid_angle=0.5,
                                                                grid_aspect_power=0.5,
                                                                row_phase_offset=0.5)
                },
                'battery': {
                    'system_capacity_kwh': 20 * 1000,
                    'system_capacity_kw': 5 * 1000
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
    assert aeps.wind == pytest.approx(33615479, 1e3)
    assert aeps.hybrid == pytest.approx(33615479, 1e3)

    assert npvs.pv == 0
    assert npvs.wind == pytest.approx(-13692784, 1e3)
    assert npvs.hybrid == pytest.approx(-13692784, 1e3)


def test_hybrid_pv_only(site):
    solar_only = {'pv': technologies['pv']}
    hybrid_plant = HybridSimulation(solar_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == pytest.approx(8703525.94, 1e-3)
    assert aeps.wind == 0
    assert aeps.hybrid == pytest.approx(8703525.94, 1e-3)

    assert npvs.pv == pytest.approx(-5121293, 1e3)
    assert npvs.wind == 0
    assert npvs.hybrid == pytest.approx(-5121293, 1e3)


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

    assert aeps.pv == pytest.approx(8703525.94, 13)
    assert aeps.wind == pytest.approx(33615479.57, 1e3)
    assert aeps.hybrid == pytest.approx(41681662.63, 1e3)

    assert npvs.pv == pytest.approx(-5121293, 1e3)
    assert npvs.wind == pytest.approx(-13909363, 1e3)
    assert npvs.hybrid == pytest.approx(-19216589, 1e3)


def test_hybrid_with_storage_dispatch(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies

    assert aeps.pv == pytest.approx(8703525, 1e-3)
    assert aeps.wind == pytest.approx(33615479, 1e-3)
    assert aeps.battery == pytest.approx(-131373, 1e-3)
    assert aeps.hybrid == pytest.approx(42187675, 1e-3)

    npvs = hybrid_plant.net_present_values
    assert npvs.pv == pytest.approx(-1657066, 1e-3)
    assert npvs.wind == pytest.approx(-3975100, 1e-3)
    assert npvs.battery == pytest.approx(-11743285, 1e-3)
    assert npvs.hybrid == pytest.approx(-17567009, 1e-3)

    taxes = hybrid_plant.federal_taxes
    assert taxes.pv[1] == pytest.approx(114582, 1e-3)
    assert taxes.wind[1] == pytest.approx(402835, 1e-3)
    assert taxes.battery[1] == pytest.approx(509869, 1e-3)
    assert taxes.hybrid[1] == pytest.approx(1029949, 1e-3)

    apv = hybrid_plant.energy_purchases_values
    assert apv.pv[1] == pytest.approx(0, 1e-3)
    assert apv.wind[1] == pytest.approx(0, 1e-3)
    assert apv.battery[1] == pytest.approx(-158650, 1e-3)
    assert apv.hybrid[1] == pytest.approx(-40309, 1e-2)

    debt = hybrid_plant.debt_payment
    assert debt.pv[1] == pytest.approx(0, 1e-3)
    assert debt.wind[1] == pytest.approx(0, 1e-3)
    assert debt.battery[1] == pytest.approx(0, 1e-3)
    assert debt.hybrid[1] == pytest.approx(0, 1e-3)

    esv = hybrid_plant.energy_sales_values
    assert esv.pv[1] == pytest.approx(261105, 1e3)
    assert esv.wind[1] == pytest.approx(1008464, 1e3)
    assert esv.battery[1] == pytest.approx(168342, 1e3)
    assert esv.hybrid[1] == pytest.approx(1305940, 1e3)

    depr = hybrid_plant.federal_depreciation_totals
    assert depr.pv[1] == pytest.approx(762811, 1e3)
    assert depr.wind[1] == pytest.approx(2651114, 1e3)
    assert depr.battery[1] == pytest.approx(2555389, 1e3)
    assert depr.hybrid[1] == pytest.approx(5969315, 1e3)

    insr = hybrid_plant.insurance_expenses
    assert insr.pv[0] == pytest.approx(0, 1e3)
    assert insr.wind[0] == pytest.approx(0, 1e3)
    assert insr.battery[0] == pytest.approx(0, 1e3)
    assert insr.hybrid[0] == pytest.approx(0, 1e3)

    om = hybrid_plant.om_total_expenses
    assert om.pv[0] == pytest.approx(84992, 1e3)
    assert om.wind[0] == pytest.approx(420000, 1e3)
    assert om.battery[0] == pytest.approx(260000, 1e3)
    assert om.hybrid[0] == pytest.approx(569992, 1e3)

    rev = hybrid_plant.total_revenues
    assert rev.pv[0] == pytest.approx(261105, 1e3)
    assert rev.wind[0] == pytest.approx(1008464, 1e3)
    assert rev.battery[0] == pytest.approx(9691, 1e3)
    assert rev.hybrid[0] == pytest.approx(1265630, 1e3)

    tc = hybrid_plant.tax_incentives
    assert tc.pv[1] == pytest.approx(1123104, 1e3)
    assert tc.wind[1] == pytest.approx(504232, 1e3)
    assert tc.battery[1] == pytest.approx(0, 1e3)
    assert tc.hybrid[1] == pytest.approx(1629356, 1e3)


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


def test_hybrid_om_prod_costs(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.wind._financial_model.SystemCosts.om_production = (5,)
    hybrid_plant.pv._financial_model.SystemCosts.om_production = (2,)
    hybrid_plant.battery._financial_model.SystemCosts.om_batt_variable_cost = (3000,)   # $/MWh
    hybrid_plant.simulate()
    om_prod_wind = hybrid_plant.wind._financial_model.value("cf_om_production_expense")[1]
    om_prod_pv = hybrid_plant.pv._financial_model.value("cf_om_production_expense")[1]
    om_prod_battery = hybrid_plant.battery._financial_model.value("cf_om_production_expense")[1]
    om_prod_hybrid = hybrid_plant.grid._financial_model.value("cf_om_production_expense")[1]
    assert om_prod_pv + om_prod_battery + om_prod_wind == pytest.approx(om_prod_hybrid)

    om_var_batt = hybrid_plant.battery._financial_model.Outputs.cf_om_production1_expense[1]
    assert om_var_batt == pytest.approx(hybrid_plant.battery._financial_model.LCOS.batt_annual_discharge_energy[0] * 3)


def test_hybrid_batt_prod_costs(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.battery._financial_model.SystemCosts.om_batt_variable_cost = (0,)   # $/MWh
    hybrid_plant.simulate()
    om_prod_wind = hybrid_plant.wind._financial_model.value("cf_om_production_expense")[1]
    om_prod_pv = hybrid_plant.pv._financial_model.value("cf_om_production_expense")[1]
    om_prod_battery = hybrid_plant.battery._financial_model.value("cf_om_production_expense")[1]
    om_prod_hybrid = hybrid_plant.grid._financial_model.value("cf_om_production_expense")[1]
    assert om_prod_pv + om_prod_battery + om_prod_wind == pytest.approx(om_prod_hybrid)


def test_hybrid_tax_incentives(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = (1,)
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_amount = (2,)
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_amount = (3,)
    hybrid_plant.simulate()
    ptc_wind = hybrid_plant.wind._financial_model.value("cf_ptc_fed")[1]
    ptc_pv = hybrid_plant.pv._financial_model.value("cf_ptc_fed")[1]
    ptc_batt = hybrid_plant.battery._financial_model.value("cf_ptc_fed")[1]
    ptc_hybrid = hybrid_plant.grid._financial_model.value("cf_ptc_fed")[1]
    assert ptc_wind + ptc_pv + ptc_batt == pytest.approx(ptc_hybrid)

