from pytest import approx, fixture
import json

from hopp import ROOT_DIR
from hopp.simulation import HoppInterface
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from tests.hopp.utils import create_default_site_info, DEFAULT_FIN_CONFIG

pvsamv1_defaults_file = ROOT_DIR.parent / "tests" / "hopp" / "pvsamv1_basic_params.json"

@fixture
def site():
    return create_default_site_info()


def test_custom_financial():
    discount_rate = 0.0906       # [1/year]
    cash_flow = [
        -4.8274e+07, 3.57154e+07, 7.7538e+06, 4.76858e+06, 2.96768e+06,
        2.94339e+06, 1.5851e+06, 227235, 202615, 176816,
        149414, 120856, 90563.4, 58964.5, 25609.9,
        378270, 1.20607e+06, 633062, 3.19583e+06, 6.01239e+06,
        5.78599e+06, 5.53565e+06, 5.49998e+06, 5.4857e+06, 5.47012e+06,
        6.84512e+06]
    npv = CustomFinancialModel.npv(discount_rate, cash_flow)
    assert npv == approx(7412807, 1e-3)


def test_detailed_pv(site, subtests):
    # Run detailed PV model (pvsamv1) using a custom financial model
    annual_energy_expected = 108833068
    npv_expected = -39094449

    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

    interconnect_kw = 150e6
    hopp_config = {
        "site": site,
        "technologies": {
            "pv": {
                'use_pvwatts': False,
                'tech_config': tech_config,
                'layout_params': {
                    "x_position": 0.5,
                    "y_position": 0.5,
                    "aspect_power": 0,
                    "gcr": 0.3,
                    "s_buffer": 2,
                    "x_buffer": 2
                },
                'fin_model': DEFAULT_FIN_CONFIG,
                'dc_degradation': [0] * 25,
            },
            "grid": {
                'interconnect_kw': interconnect_kw,
                'fin_model': DEFAULT_FIN_CONFIG,
                'ppa_price': 0.01
            }
        },
        # "config": {
        #     "cost_info": {
        #         # based on 2023 ATB moderate case for utility-scale pv
        #         "solar_installed_cost_mw": 1331.353 * 1000
        #     }
        # }
    }

    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system

    # Verify technology and financial parameters are linked, specifically testing 'analysis_period'

    hybrid_plant.layout.plot()

    hybrid_plant.simulate()
    with subtests.test("with minimal params"):
        analysis_period_orig = hybrid_plant.pv.value('analysis_period')
        assert analysis_period_orig == hybrid_plant.pv._system_model.value('analysis_period')
        assert analysis_period_orig == hybrid_plant.pv._financial_model.value('analysis_period')
        analysis_period_new = 7
        assert analysis_period_orig != analysis_period_new
        hybrid_plant.pv.value('analysis_period', analysis_period_new)                   # modify via plant setter
        assert analysis_period_new == hybrid_plant.pv._system_model.value('analysis_period')
        assert analysis_period_new == hybrid_plant.pv._financial_model.value('analysis_period')
        hybrid_plant.pv._system_model.value('analysis_period', analysis_period_orig)    # modify via system model setter
        assert analysis_period_orig == hybrid_plant.pv.value('analysis_period')
        assert analysis_period_orig != hybrid_plant.pv._financial_model.value('analysis_period')    # NOTE: this is updated just before execute
        hybrid_plant.pv._financial_model.value('analysis_period', analysis_period_new)  # modify via financial model setter
        assert analysis_period_new == hybrid_plant.pv.value('analysis_period')
        assert analysis_period_new == hybrid_plant.pv._system_model.value('analysis_period')
        hybrid_plant.pv.value('analysis_period', analysis_period_orig)                  # reset value

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npvs.pv, 1e-10)


def test_hybrid_simple_pv_with_wind(site, subtests):
    # Run wind + simple PV (pvwattsv8) hybrid plant with custom financial model
    annual_energy_expected_pv = 98653103
    annual_energy_expected_wind = 32440267
    annual_energy_expected_hybrid = 131067896
    npv_expected_pv = -39925445
    npv_expected_wind = -11884863
    npv_expected_hybrid = -51812393

    interconnect_kw = 150e6
    pv_kw = 50000
    wind_kw = 10000

    power_sources = {
        'pv': {
            'system_capacity_kw': pv_kw,
            'layout_params': {
                "x_position": 0.5, 
                "y_position": 0.5, 
                "aspect_power": 0, 
                "gcr": 0.5, 
                "s_buffer": 2, 
                "x_buffer": 2
            },
            'fin_model': DEFAULT_FIN_CONFIG,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": 0.5, 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            }, 
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
            'ppa_price': 0.01
        },
    }

    hopp_config = {
        "site": site,
        "technologies": power_sources
    }
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    with subtests.test("with minimal params"):
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)


def test_hybrid_detailed_pv_with_wind(site, subtests):
    # Test wind + detailed PV (pvsamv1) hybrid plant with custom financial model
    annual_energy_expected_pv = 21541876
    annual_energy_expected_wind = 32296230
    annual_energy_expected_hybrid = 53838106
    npv_expected_pv = -7844643
    npv_expected_wind = -11896652
    npv_expected_hybrid = -19733945

    interconnect_kw = 150e6
    wind_kw = 10000

    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)
    
    # NOTE: PV array shrunk to avoid problem associated with flicker calculation
    tech_config['system_capacity'] = 10000
    tech_config['inverter_count'] = 10
    tech_config['subarray1_nstrings'] = 2687

    layout_params = {
        "x_position": 0.5, 
        "y_position": 0.5, 
        "aspect_power": 0, 
        "gcr": 0.3, 
        "s_buffer": 2, 
        "x_buffer": 2
    }

    power_sources = {
        'pv': {
            'use_pvwatts': False,
            'tech_config': tech_config,
            'layout_params': layout_params,
            'fin_model': DEFAULT_FIN_CONFIG,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": 0.5, 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
            'ppa_price': 0.01
        }
    }
    hopp_config = {
        "site": site,
        "technologies": power_sources
    }

    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    with subtests.test("with minimal params"):
        assert sizes.pv == approx(10000, 1e-3)
        assert sizes.wind == approx(wind_kw, 1e-3)
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)

def test_hybrid_simple_pv_with_wind_storage_dispatch(site, subtests):
    # Test wind + simple PV (pvwattsv8) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 9857584
    annual_energy_expected_wind = 31951719
    annual_energy_expected_battery = -96912
    annual_energy_expected_hybrid = 41709692
    npv_expected_pv = -1905544
    npv_expected_wind = -5159400
    npv_expected_battery = -8183543
    npv_expected_hybrid = -15249189

    interconnect_kw = 15000
    pv_kw = 5000
    wind_kw = 10000
    batt_kw = 5000

    power_sources = {
        'pv': {
            'system_capacity_kw': pv_kw,
            'layout_params': {
                "x_position": 0.5, 
                "y_position": 0.5, 
                "aspect_power": 0, 
                "gcr": 0.5, 
                "s_buffer": 2, 
                "x_buffer": 2
            },
            'fin_model': DEFAULT_FIN_CONFIG,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": 0.5, 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
            'ppa_price': 0.03
        }
    }
    hopp_config = {
        "site": site,
        "technologies": power_sources
    }
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    with subtests.test("with minimal params"):
        assert sizes.pv == approx(pv_kw, 1e-3)
        assert sizes.wind == approx(wind_kw, 1e-3)
        assert sizes.battery == approx(batt_kw, 1e-3)
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
        assert aeps.battery == approx(annual_energy_expected_battery, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
        assert npvs.battery == approx(npv_expected_battery, 1e-3)
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)


def test_hybrid_detailed_pv_with_wind_storage_dispatch(site, subtests):
    # Test wind + detailed PV (pvsamv1) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 20416252
    annual_energy_expected_wind = 32321927
    annual_energy_expected_battery = -91312
    annual_energy_expected_hybrid = 52645082
    npv_expected_pv = -3606490
    npv_expected_wind = -5050712
    npv_expected_battery = -8181700
    npv_expected_hybrid = -16839535

    interconnect_kw = 15000
    wind_kw = 10000
    batt_kw = 5000

    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)
    
    # NOTE: PV array shrunk to avoid problem associated with flicker calculation
    tech_config['system_capacity'] = 10000
    tech_config['inverter_count'] = 10
    tech_config['subarray1_nstrings'] = 2687

    power_sources = {
        'pv': {
            'use_pvwatts': False,
            'tech_config': tech_config,
            'layout_params': {
                "x_position": 0.5, 
                "y_position": 0.5, 
                "aspect_power": 0, 
                "gcr": 0.5, 
                "s_buffer": 2, 
                "x_buffer": 2
            },
            'fin_model': DEFAULT_FIN_CONFIG,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": 0.5, 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
            'ppa_price': 0.03
        }
    }
    hopp_config = {
        "site": site,
        "technologies": power_sources
    } 
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    with subtests.test("with minimal params"):
        assert sizes.pv == approx(10000, 1e-3)
        assert sizes.wind == approx(wind_kw, 1e-3)
        assert sizes.battery == approx(batt_kw, 1e-3)
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
        assert aeps.battery == approx(annual_energy_expected_battery, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
        assert npvs.battery == approx(npv_expected_battery, 1e-3)
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)
