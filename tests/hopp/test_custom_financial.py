from pytest import approx, fixture
import json

from hopp import ROOT_DIR
from hopp.tools.hopp_interface import HoppInterface
from hopp.simulation.technologies.layout.hybrid_layout import WindBoundaryGridParameters
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.technologies.detailed_pv_plant import DetailedPVPlant, DetailedPVConfig
from hopp.simulation.technologies.grid import Grid
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


def test_detailed_pv(site):
    # Run detailed PV model (pvsamv1) using a custom financial model
    annual_energy_expected = 108239401
    npv_expected = -39144853

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
            },
            "grid": {
                'interconnect_kw': interconnect_kw,
                'fin_model': DEFAULT_FIN_CONFIG,
            }
        }
    }

    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system

    # Verify technology and financial parameters are linked, specifically testing 'analysis_period'
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

    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    assert aeps.pv == approx(annual_energy_expected, 1e-3)
    assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
    assert npvs.pv == approx(npv_expected, 1e-3)
    assert npvs.hybrid == approx(npv_expected, 1e-3)
    assert npvs.hybrid == approx(npvs.pv, 1e-10)


def test_hybrid_simple_pv_with_wind(site):
    # Run wind + simple PV (pvwattsv8) hybrid plant with custom financial model
    annual_energy_expected_pv = 98653103
    annual_energy_expected_wind = 33584937
    annual_energy_expected_hybrid = 132238041
    npv_expected_pv = -39925445
    npv_expected_wind = -11791174
    npv_expected_hybrid = -51716620

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
        }
    }

    hopp_config = {
        "site": site,
        "technologies": power_sources
    }
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
    assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
    assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
    assert npvs.pv == approx(npv_expected_pv, 1e-3)
    assert npvs.wind == approx(npv_expected_wind, 1e-3)
    assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)


def test_hybrid_detailed_pv_with_wind(site):
    # Test wind + detailed PV (pvsamv1) hybrid plant with custom financial model
    annual_energy_expected_pv = 21452080
    annual_energy_expected_wind = 33433774
    annual_energy_expected_hybrid = 54885854
    npv_expected_pv = -7844643
    npv_expected_wind = -11803547
    npv_expected_hybrid = -19648190

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
        }
    }
    hopp_config = {
        "site": site,
        "technologies": power_sources
    }

    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    assert sizes.pv == approx(10000, 1e-3)
    assert sizes.wind == approx(wind_kw, 1e-3)
    assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
    assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
    assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
    assert npvs.pv == approx(npv_expected_pv, 1e-3)
    assert npvs.wind == approx(npv_expected_wind, 1e-3)
    assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)


def test_hybrid_simple_pv_with_wind_storage_dispatch(site):
    # Test wind + simple PV (pvwattsv8) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 9857584
    annual_energy_expected_wind = 33074859
    annual_energy_expected_battery = -97180
    annual_energy_expected_hybrid = 42835263
    npv_expected_pv = -1905544
    npv_expected_wind = -4829660
    npv_expected_battery = -8183543
    npv_expected_hybrid = -14918736

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
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
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


def test_hybrid_detailed_pv_with_wind_storage_dispatch(site):
    # Test wind + detailed PV (pvsamv1) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 20365655
    annual_energy_expected_wind = 33462743
    annual_energy_expected_battery = -90903
    annual_energy_expected_hybrid = 53736299
    npv_expected_pv = -3621345
    npv_expected_wind = -4715783
    npv_expected_battery = -8181700
    npv_expected_hybrid = -16519167

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
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
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
