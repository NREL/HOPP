import pytest

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.layout.hybrid_layout import WindBoundaryGridParameters, PVGridParameters
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.keys import set_nrel_key_dot_env

set_nrel_key_dot_env()

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


def test_wind_pv_with_storage_dispatch(site):
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    hybrid_plant = HybridSimulation(wind_pv_battery, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    gen_profiles = hybrid_plant.generation_profile

    aeps = hybrid_plant.annual_energies
    assert aeps.pv == pytest.approx(8703525, 1e3)
    assert aeps.wind == pytest.approx(32978136, 1e3)
    assert aeps.battery == pytest.approx(-218034, 1e3)
    assert aeps.hybrid == pytest.approx(41463627, 1e3)

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

    om = hybrid_plant.om_expenses
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


def test_tower_pv_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'tower': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000},
                         'grid': 50000}

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv', 'grid')}
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

    assert aeps.pv == pytest.approx(87692005.68, 1e-3)
    assert aeps.tower == pytest.approx(3650674.52, 5e-2)
    assert aeps.hybrid == pytest.approx(91231861.2, 1e-2)

    # TODO: check npv for csp would require a full simulation
    assert npvs.pv == pytest.approx(45233832.23, 1e3)
    #assert npvs.tower == pytest.approx(-13909363, 1e3)
    #assert npvs.hybrid == pytest.approx(-19216589, 1e3)


def test_trough_pv_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'trough': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000},
                         'grid': 50000}

    solar_hybrid = {key: technologies_test[key] for key in ('trough', 'pv', 'grid')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw_test,
                                    dispatch_options={'is_test_start_year': True,
                                                      'is_test_end_year': True})
    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == pytest.approx(87692005.68, 1e-3)
    assert aeps.trough == pytest.approx(1827310.47, 2e-2)
    assert aeps.hybrid == pytest.approx(89508343.80, 1e-3)

    assert npvs.pv == pytest.approx(45233832.23, 1e3)
    #assert npvs.tower == pytest.approx(-13909363, 1e3)
    #assert npvs.hybrid == pytest.approx(-19216589, 1e3)


def test_tower_pv_battery_hybrid(site):
    interconnection_size_kw_test = 50000
    technologies_test = {'tower': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000},
                         'battery': {'system_capacity_kwh': 40 * 1000,
                                     'system_capacity_kw': 20 * 1000},
                         'grid': 50000}

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw_test,
                                    dispatch_options={  # 'solver': 'cbc',
                                                      'is_test_start_year': True,
                                                      'is_test_end_year': True})
    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.tower.value('helio_width', 10.0)
    hybrid_plant.tower.value('helio_height', 10.0)

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == pytest.approx(87692005.68, 1e-3)
    assert aeps.tower == pytest.approx(3769716.50, 5e-2)
    assert aeps.battery == pytest.approx(-7276.63, 2e-1)
    assert aeps.hybrid == pytest.approx(91448182.18, 1e-2)

    assert npvs.pv == pytest.approx(45233832.23, 1e3)
    #assert npvs.tower == pytest.approx(-13909363, 1e3)
    #assert npvs.hybrid == pytest.approx(-19216589, 1e3)

