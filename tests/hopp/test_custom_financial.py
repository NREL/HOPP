from pytest import approx, fixture
import json

from hopp import ROOT_DIR
from hopp.simulation import HoppInterface
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel

from tests.hopp.utils import create_default_site_info, DEFAULT_FIN_CONFIG
import copy
import numpy as np

DEFAULT_FIN_CONFIG_LOCAL = copy.deepcopy(DEFAULT_FIN_CONFIG)
DEFAULT_FIN_CONFIG_LOCAL.pop("revenue") # these tests were written before the revenue section was added to the default financial config

from hopp.utilities import load_yaml

from hopp.simulation.technologies.financial.mhk_cost_model import MHKCostModelInputs

pvsamv1_defaults_file = ROOT_DIR.parent / "tests" / "hopp" / "pvsamv1_basic_params.json"

mhk_yaml_path = (
    ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "wave" / "wave_device.yaml"
)
mhk_config = load_yaml(mhk_yaml_path)

wave_resource_file = (
    ROOT_DIR / "simulation" / "resource_files" / "wave" / "Wave_resource_timeseries.csv"
)

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
    annual_energy_expected = 8884369
    npv_expected = -4066396

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
                'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
                'dc_degradation': [0] * 25,
            },
            "grid": {
                'interconnect_kw': interconnect_kw,
                'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
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
    annual_energy_expected_pv = 107705270
    annual_energy_expected_wind = 32440267
    annual_energy_expected_hybrid = 140145538
    npv_expected_pv = -39184550
    npv_expected_wind = -11884863
    npv_expected_hybrid = -51069413

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
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": np.rad2deg(0.5), 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            }, 
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
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
    with subtests.test("minimal params pv aep"):
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
    with subtests.test("minimal params wind aep"):
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
    with subtests.test("minimal params hybrid aep"):
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
    with subtests.test("minimal params pv npv"):
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
    with subtests.test("minimal params wind npv"):
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
    with subtests.test("minimal params hybrid npv"):
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)


def test_hybrid_detailed_pv_with_wind(site, subtests):
    # Test wind + detailed PV (pvsamv1) hybrid plant with custom financial model
    annual_energy_expected_pv = 8863135
    annual_energy_expected_wind = 31453286
    annual_energy_expected_hybrid = 40316422
    npv_expected_pv = -4068134
    npv_expected_wind = -11965644
    npv_expected_hybrid = -16033778

    interconnect_kw = 150e6
    wind_kw = 10000

    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

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
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": np.rad2deg(0.5), 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
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
        assert sizes.pv == approx(4993, 1e-3)
        assert sizes.wind == approx(wind_kw, 1e-3)
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)

def test_hybrid_simple_pv_with_wind_wave_storage_dispatch(subtests):

    site_internal = create_default_site_info(wave=True, wave_resource_file=wave_resource_file)
    # Test wind + simple PV (pvwattsv8) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 10761987
    annual_energy_expected_wind = 31951719
    annual_energy_expected_wave = 12132526
    annual_energy_expected_battery = -103752
    annual_energy_expected_hybrid = 54747904

    npv_expected_pv = -1640023
    npv_expected_wind = -5159400
    npv_expected_wave = -57994156.
    npv_expected_battery = -8183543
    npv_expected_hybrid = -72978515.

    lcoe_expected_pv = 3.104064331441355
    lcoe_expected_wind = 3.162940789633178
    lcoe_expected_wave = 33.09696662806905
    lcoe_expected_battery = 13.333128855903514
    lcoe_expected_hybrid = 10.756369801042606

    total_installed_cost_expected = 89050689.65833203 

    interconnect_kw = 20000
    pv_kw = 5000
    wind_kw = 10000
    batt_kw = 5000
    wave_kw = 2860

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
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": np.rad2deg(0.5), 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        "wave": {
            "device_rating_kw": wave_kw/10,
            "num_devices": 10,
            "wave_power_matrix": mhk_config["wave_power_matrix"],
            "fin_model": DEFAULT_FIN_CONFIG_LOCAL,
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
            'ppa_price': 0.03
        }
    }
    config = {
        "simulation_options": {
            "wind": {
                "skip_financial": False # test that setting this to false allows financial calculations to run
            }
        }
    }
    hopp_config = {
        "site": site_internal,
        "technologies": power_sources,
        "config": config
    }

    mhk_cost_model_inputs = MHKCostModelInputs.from_dict(
        {
            "reference_model_num": 3,
            "water_depth": 100,
            "distance_to_shore": 80,
            "number_rows": 10,
            "device_spacing": 600,
            "row_spacing": 600,
            "cable_system_overbuild": 20,
        }
    )
    
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    hybrid_plant.layout.plot()
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01
    hybrid_plant.battery._financial_model.om_batt_variable_cost = [0.75]
    hybrid_plant.wave.create_mhk_cost_calculator(mhk_cost_model_inputs)

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    lcoes = hybrid_plant.lcoe_nom # cents/kWh

    with subtests.test("with minimal params pv size"):
        assert sizes.pv == approx(pv_kw, 1e-3)
    with subtests.test("with minimal params wind size"):
        assert sizes.wind == approx(wind_kw, 1e-3)
    with subtests.test("with minimal params wave size"):
        assert sizes.wave == approx(wave_kw, 1e-3)
    with subtests.test("with minimal params batt kw size"):
        assert sizes.battery == approx(batt_kw, 1e-3)

    with subtests.test("with minimal params pv aep"):
        assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
    with subtests.test("with minimal params wind aep"):
        assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
    with subtests.test("with minimal params wave aep"):
        assert aeps.wave == approx(annual_energy_expected_wave, 1e-3)
    with subtests.test("with minimal params battery aep"):
        assert aeps.battery == approx(annual_energy_expected_battery, 1e-3)
    with subtests.test("with minimal params hybrid aep"):
        assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)

    with subtests.test("with minimal params pv npv"):
        assert npvs.pv == approx(npv_expected_pv, 1e-3)
    with subtests.test("with minimal params wind npv"):
        assert npvs.wind == approx(npv_expected_wind, 1e-3)
    with subtests.test("with minimal params wave npv"):
        assert npvs.wave == approx(npv_expected_wave, 1e-3)
    with subtests.test("with minimal params batt npv"):
        assert npvs.battery == approx(npv_expected_battery, 1e-3)
    with subtests.test("with minimal params hybrid npv"):
        assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)

    with subtests.test("lcoe pv"):
        assert lcoes.pv == approx(lcoe_expected_pv, 1e-3)
    with subtests.test("lcoe wind"):
        assert lcoes.wind == approx(lcoe_expected_wind, 1e-3)
    with subtests.test("lcoe wave"):
        assert lcoes.wave == approx(lcoe_expected_wave, 1e-3)
    with subtests.test("lcoe battery"): ############## left commented since I'm not sure calculating LCOE for battery this way makes sense
        assert lcoes.battery == approx(lcoe_expected_battery, 1e-3)
    with subtests.test("lcoe hybrid"):
        assert lcoes.hybrid == approx(lcoe_expected_hybrid, 1e-3)

    with subtests.test("total installed cost"):
        assert hybrid_plant.grid.total_installed_cost == approx(total_installed_cost_expected, 1E-6)


def test_hybrid_detailed_pv_with_wind_storage_dispatch(site, subtests):
    # Test wind + detailed PV (pvsamv1) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 8851251
    annual_energy_expected_wind = 31559803
    annual_energy_expected_battery = -102220
    annual_energy_expected_hybrid = 40308834
    npv_expected_pv = -2194945
    npv_expected_wind = -5274461
    npv_expected_battery = -8181700
    npv_expected_hybrid = -15654417

    interconnect_kw = 15000
    wind_kw = 10000
    batt_kw = 5000

    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

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
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
            'dc_degradation': [0] * 25
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': {
                "border_spacing": 2, 
                "border_offset": 0.5, 
                "grid_angle": np.rad2deg(0.5), 
                "grid_aspect_power": 0.5, 
                "row_phase_offset": 0.5
            },
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': DEFAULT_FIN_CONFIG_LOCAL,
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
    hybrid_plant.battery._financial_model.om_batt_variable_cost = [0.75]

    hybrid_plant.simulate()

    sizes = hybrid_plant.system_capacity_kw
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    with subtests.test("with minimal params"):
        assert sizes.pv == approx(4993, 1e-3)
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
