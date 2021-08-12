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
                },
                'grid': 15000}


def test_hybrid_wind_only(site):
    wind_only = {key: technologies[key] for key in ('wind', 'grid')}
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
    solar_only = {key: technologies[key] for key in ('pv', 'grid')}
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
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'grid')}
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
    npvs = hybrid_plant.net_present_values

    print(aeps)
    print(npvs)

    assert aeps.pv == pytest.approx(8703525, 1e3)
    assert aeps.wind == pytest.approx(32978136, 1e3)
    assert aeps.battery == pytest.approx(-218034, 1e3)
    assert aeps.hybrid == pytest.approx(41463627, 1e3)

    assert npvs.pv == pytest.approx(-3557479, 1e3)
    assert npvs.wind == pytest.approx(-7736427, 1e3)
    assert npvs.battery == pytest.approx(-21635232, 1e3)
    assert npvs.hybrid == pytest.approx(-11841323, 1e3)
