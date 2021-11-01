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
    assert aeps.battery == pytest.approx(-99244, 1e-3)
    assert aeps.hybrid == pytest.approx(42219764, 1e-3)

    npvs = hybrid_plant.net_present_values
    assert npvs.pv == pytest.approx(-1657066, 1e-3)
    assert npvs.wind == pytest.approx(-3975100, 1e-3)
    assert npvs.battery == pytest.approx(-13479325, 1e-3)
    assert npvs.hybrid == pytest.approx(-19795614, 1e-3)

    taxes = hybrid_plant.federal_taxes
    assert taxes.pv[1] == pytest.approx(114582, 1e-3)
    assert taxes.wind[1] == pytest.approx(402835, 1e-3)
    assert taxes.battery[1] == pytest.approx(548350, 1e-3)
    assert taxes.hybrid[1] == pytest.approx(1050083, 1e-3)

    apv = hybrid_plant.energy_purchases_values
    assert apv.pv[1] == pytest.approx(0, 1e-3)
    assert apv.wind[1] == pytest.approx(0, 1e-3)
    assert apv.battery[1] == pytest.approx(-126309, 1e-3)
    assert apv.hybrid[1] == pytest.approx(-5178, 1e-3)

    debt = hybrid_plant.debt_payment
    assert debt.pv[1] == pytest.approx(0, 1e-3)
    assert debt.wind[1] == pytest.approx(0, 1e-3)
    assert debt.battery[1] == pytest.approx(0, 1e-3)
    assert debt.hybrid[1] == pytest.approx(0, 1e-3)

    esv = hybrid_plant.energy_sales_values
    assert esv.pv[1] == pytest.approx(261105, 1e3)
    assert esv.wind[1] == pytest.approx(1008464, 1e3)
    assert esv.battery[1] == pytest.approx(130218, 1e3)
    assert esv.hybrid[1] == pytest.approx(1271686, 1e3)

    depr = hybrid_plant.federal_depreciation_totals
    assert depr.pv[1] == pytest.approx(762811, 1e3)
    assert depr.wind[1] == pytest.approx(2651114, 1e3)
    assert depr.battery[1] == pytest.approx(2555389, 1e3)
    assert depr.hybrid[1] == pytest.approx(6073287, 1e3)

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
    assert rev.battery[0] == pytest.approx(4147, 1e3)
    assert rev.hybrid[0] == pytest.approx(1266507, 1e3)
