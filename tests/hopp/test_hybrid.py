from pathlib import Path
from copy import deepcopy

from pytest import approx, fixture, raises

import numpy as np
import json

from hopp.simulation import HoppInterface

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.pv.detailed_pv_plant import DetailedPVPlant, DetailedPVConfig
from hopp.simulation.technologies.layout.pv_design_utils import size_electrical_parameters
from hopp.simulation.technologies.financial.mhk_cost_model import MHKCostModelInputs
from tests.hopp.utils import create_default_site_info, DEFAULT_FIN_CONFIG
from hopp import ROOT_DIR
from hopp.utilities import load_yaml


@fixture
def hybrid_config():
    """Loads the config YAML and updates site info to use resource files."""
    hybrid_config_path = ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "hybrid_run.yaml"
    hybrid_config = load_yaml(hybrid_config_path)

    return hybrid_config

@fixture
def site():
    return create_default_site_info()

wave_resource_file = ROOT_DIR.parent / "resource_files" / "wave" / "Wave_resource_timeseries.csv"

@fixture
def wavesite():
    data = {
        "lat": 44.6899,
        "lon": 124.1346,
        "year": 2010,
        "tz": -7
    }
    return SiteInfo(
        data,
        wave_resource_file=wave_resource_file, 
        solar=False, 
        wind=False, 
        wave=True
    )

mhk_yaml_path = ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "wave" / "wave_device.yaml"
mhk_config = load_yaml(mhk_yaml_path)

interconnection_size_kw = 15000
pv_kw = 5000
wind_kw = 10000
batt_kw = 5000

detailed_pv = {
    'tech_config': {
        'system_capacity_kw': pv_kw
    },
    'layout_params': {
        "x_position": 0.5,
        "y_position": 0.5,
        "aspect_power": 0,
        "gcr": 0.5,
        "s_buffer": 2,
        "x_buffer": 2
    }
}
# From a Cambium midcase BA10 2030 analysis (Jan 1 = 1):
capacity_credit_hours_of_year = [4604,4605,4606,4628,4629,4630,4652,4821,5157,5253,
                                 5254,5277,5278,5299,5300,5301,5302,5321,5323,5324,
                                 5325,5326,5327,5347,5348,5349,5350,5369,5370,5371,
                                 5372,5374,5395,5396,5397,5398,5419,5420,5421,5422,
                                 5443,5444,5445,5446,5467,5468,5469,5493,5494,5517,
                                 5539,5587,5589,5590,5661,5757,5781,5803,5804,5805,
                                 5806,5826,5827,5830,5947,5948,5949,5995,5996,5997,
                                 6019,6090,6091,6092,6093,6139,6140,6141,6163,6164,
                                 6165,6166,6187,6188,6211,6212,6331,6354,6355,6356,
                                 6572,6594,6595,6596,6597,6598,6618,6619,6620,6621]
# List length 8760, True if the hour counts for capacity payments, False otherwise
capacity_credit_hours = [hour in capacity_credit_hours_of_year for hour in range(1,8760+1)]

def test_hybrid_wave_only(hybrid_config, wavesite, subtests):
    hybrid_config["site"]["wave"] = True
    hybrid_config["site"]["wave_resource_file"] = wave_resource_file
    wave_only_technologies = {
        'wave': {
            'device_rating_kw': mhk_config['device_rating_kw'], 
            'num_devices': 10, 
            'wave_power_matrix': mhk_config['wave_power_matrix'],
            'fin_model': DEFAULT_FIN_CONFIG
        },
        'grid': {
            'interconnect_kw': interconnection_size_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        }
    }

    hybrid_config["technologies"] = wave_only_technologies
    
    # TODO once the financial model is implemented, romove the line immediately following this comment and un-indent the rest of the test    
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system
    # hybrid_plant = HybridSimulation(wave_only_technologies, wavesite)
    cost_model_inputs = MHKCostModelInputs.from_dict({
        'reference_model_num':3,
        'water_depth': 100,
        'distance_to_shore': 80,
        'number_rows': 10,
        'device_spacing':600,
        'row_spacing': 600,
        'cable_system_overbuild': 20
	})
    assert hybrid_plant.wave is not None
    hybrid_plant.wave.create_mhk_cost_calculator(cost_model_inputs)

    hi.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    # check that wave and grid match when only wave is in the hybrid system
    with subtests.test("financial parameters"):
        assert hybrid_plant.wave._financial_model.FinancialParameters == approx(hybrid_plant.grid._financial_model.FinancialParameters)
    with subtests.test("Revenue"):
        assert hybrid_plant.wave._financial_model.Revenue == approx(hybrid_plant.grid._financial_model.Revenue)
    with subtests.test("SystemCosts"):
        assert hybrid_plant.wave._financial_model.SystemCosts == approx(hybrid_plant.grid._financial_model.SystemCosts)

    # with subtests.test("SystemOutput.__dict__"):
    #     skip(reason="this test will not be consistent until the code is more type stable. Outputs may be tuple or list")
    #     assert hybrid_plant.wave._financial_model.SystemOutput.__dict__ == hybrid_plant.grid._financial_model.SystemOutput.__dict__
    with subtests.test("SystemOutput.gen"):
        assert hybrid_plant.wave._financial_model.SystemOutput.gen == approx(hybrid_plant.grid._financial_model.SystemOutput.gen)
    with subtests.test("SystemOutput.system_capacity"):
        assert hybrid_plant.wave._financial_model.SystemOutput.system_capacity == approx(hybrid_plant.grid._financial_model.SystemOutput.system_capacity)
    with subtests.test("SystemOutput.degradation"):
        assert hybrid_plant.wave._financial_model.SystemOutput.degradation == approx(hybrid_plant.grid._financial_model.SystemOutput.degradation)
    with subtests.test("SystemOutput.system_pre_curtailment_kwac"):
        assert hybrid_plant.wave._financial_model.SystemOutput.system_pre_curtailment_kwac == approx(hybrid_plant.grid._financial_model.SystemOutput.system_pre_curtailment_kwac)
    with subtests.test("SystemOutput.annual_energy_pre_curtailment_ac"):
        assert hybrid_plant.wave._financial_model.SystemOutput.annual_energy_pre_curtailment_ac == approx(hybrid_plant.grid._financial_model.SystemOutput.annual_energy_pre_curtailment_ac)

    with subtests.test("Outputs"):
        assert hybrid_plant.wave._financial_model.Outputs == approx(hybrid_plant.grid._financial_model.Outputs)
    with subtests.test("net cash flow"):
        wave_period = hybrid_plant.wave._financial_model.value('analysis_period')
        grid_period = hybrid_plant.grid._financial_model.value('analysis_period')
        assert hybrid_plant.wave._financial_model.net_cash_flow(wave_period) == approx(hybrid_plant.grid._financial_model.net_cash_flow(grid_period))
    
    with subtests.test("degradation"):
        assert hybrid_plant.wave._financial_model.value("degradation") == approx(hybrid_plant.grid._financial_model.value("degradation"))
    with subtests.test("total_installed_cost"):
        assert hybrid_plant.wave._financial_model.value("total_installed_cost") == approx(hybrid_plant.grid._financial_model.value("total_installed_cost"))
    with subtests.test("inflation_rate"):
        assert hybrid_plant.wave._financial_model.value("inflation_rate") == approx(hybrid_plant.grid._financial_model.value("inflation_rate"))
    with subtests.test("annual_energy"):
        assert hybrid_plant.wave._financial_model.value("annual_energy") == approx(hybrid_plant.grid._financial_model.value("annual_energy"))
    with subtests.test("ppa_price_input"):
        assert hybrid_plant.wave._financial_model.value("ppa_price_input") == approx(hybrid_plant.grid._financial_model.value("ppa_price_input"))
    with subtests.test("ppa_escalation"):
        assert hybrid_plant.wave._financial_model.value("ppa_escalation") == approx(hybrid_plant.grid._financial_model.value("ppa_escalation"))

    # test hybrid outputs
    with subtests.test("wave aep"):
        assert aeps.wave == approx(12132526.0,1e-2)
    with subtests.test("hybrid wave only aep"):
        assert aeps.hybrid == approx(aeps.wave)
    with subtests.test("wave cf"):
        assert cf.wave == approx(48.42,1e-2)
    with subtests.test("hybrid wave only cf"):
        assert cf.hybrid == approx(cf.wave)
    with subtests.test("wave npv"):
        #TODO check/verify this test value somehow, not sure how to do it right now
        assert npvs.wave == approx(-53731805.52113224)
    with subtests.test("hybrid wave only npv"):
        assert npvs.hybrid == approx(npvs.wave)

def test_hybrid_wave_battery(hybrid_config, wavesite, subtests):
    hybrid_config["site"]["wave"] = True
    hybrid_config["site"]["wave_resource_file"] = wave_resource_file
    wave_only_technologies = {
        'wave': {
            'device_rating_kw': mhk_config['device_rating_kw'], 
            'num_devices': 10, 
            'wave_power_matrix': mhk_config['wave_power_matrix'],
            'fin_model': DEFAULT_FIN_CONFIG
        },
        'battery': {
            'system_capacity_kwh': 20000,
            'system_capacity_kw': 80000,
            'fin_model': DEFAULT_FIN_CONFIG
        },
        'grid': {
            'interconnect_kw': interconnection_size_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        }
    }

    hybrid_config["technologies"] = wave_only_technologies
    
    # TODO once the financial model is implemented, romove the line immediately following this comment and un-indent the rest of the test    
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system
    # hybrid_plant = HybridSimulation(wave_only_technologies, wavesite)
    cost_model_inputs = MHKCostModelInputs.from_dict({
        'reference_model_num':3,
        'water_depth': 100,
        'distance_to_shore': 80,
        'number_rows': 10,
        'device_spacing':600,
        'row_spacing': 600,
        'cable_system_overbuild': 20
	})
    assert hybrid_plant.wave is not None
    hybrid_plant.wave.create_mhk_cost_calculator(cost_model_inputs)

    hi.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    with subtests.test("battery aep"):
        assert aeps.battery == approx(87.84, 1e3)

def test_hybrid_wind_only(hybrid_config):
    technologies = hybrid_config["technologies"]
    wind_only = {key: technologies[key] for key in ('wind', 'grid')}
    hybrid_config["technologies"] = wind_only
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate(25)

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    assert aeps.wind == approx(33615479, 1e3)
    assert aeps.hybrid == approx(33615479, 1e3)

    assert npvs.wind == approx(-13692784, 1e3)
    assert npvs.hybrid == approx(-13692784, 1e3)


def test_hybrid_pv_only(hybrid_config):
    technologies = hybrid_config["technologies"]
    solar_only = {key: technologies[key] for key in ('pv', 'grid')}
    hybrid_config["technologies"] = solar_only
    hi = HoppInterface(hybrid_config)

    hybrid_plant = hi.system

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    assert cf.hybrid == approx(cf.pv)

    assert aeps.pv == approx(9884106.55, 1e-3)
    assert aeps.hybrid == approx(9884106.55, 1e-3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.hybrid == approx(-5121293, 1e3)

def test_hybrid_pv_only_custom_fin(hybrid_config, subtests):
    solar_only = {
        'pv': {
            'system_capacity_kw': 5000,
            'layout_params': {
                'x_position': 0.5,
                'y_position': 0.5,
                'aspect_power': 0,
                'gcr': 0.5,
                's_buffer': 2,
                'x_buffer': 2,
               },
            'dc_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'fin_model': DEFAULT_FIN_CONFIG
        },
        'grid':{
            'interconnect_kw': interconnection_size_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        }
    }
    hybrid_config["technologies"] = solar_only
    hybrid_config["config"] = {
        "cost_info": {
            'solar_installed_cost_mw': 400 * 1000,
        }
    }
    hi = HoppInterface(hybrid_config)

    hybrid_plant = hi.system
    hybrid_plant.set_om_costs_per_kw(pv_om_per_kw=20)

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    with subtests.test("total installed cost"):
        assert hybrid_plant.pv.total_installed_cost == approx(2000000,1e-3)

    with subtests.test("om cost"):
        assert hybrid_plant.pv.om_capacity == (20,)

    with subtests.test("capacity factor"):
        assert cf.hybrid == approx(cf.pv)
    
    with subtests.test("aep"):
        assert aeps.pv == approx(9884106.55, 1e-3)
        assert aeps.hybrid == aeps.pv

def test_hybrid_pv_battery_custom_fin(hybrid_config, subtests):
    tech = {
        'pv': {
            'system_capacity_kw': 5000,
            'layout_params': {
                'x_position': 0.5,
                'y_position': 0.5,
                'aspect_power': 0,
                'gcr': 0.5,
                's_buffer': 2,
                'x_buffer': 2,
               },
            'dc_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'fin_model': DEFAULT_FIN_CONFIG
        },
          'battery': {
            'system_capacity_kw': 5000,
            'system_capacity_kwh': 20000,
            'fin_model': DEFAULT_FIN_CONFIG
          },
        'grid':{
            'interconnect_kw': interconnection_size_kw,
            'fin_model': DEFAULT_FIN_CONFIG,
        }
    }
    hybrid_config["technologies"] = tech
    hybrid_config["config"] = {
        "cost_info": {
            'solar_installed_cost_mw': 400 * 1000,
            'storage_installed_cost_mw': 200 * 1000,
            'storage_installed_cost_mwh': 300 * 1000
        }
    }
    hi = HoppInterface(hybrid_config)

    hybrid_plant = hi.system
    # hybrid_plant.pv.set_overnight_capital_cost(400)
    # hybrid_plant.battery.set_overnight_capital_cost(300,200)
    hybrid_plant.set_om_costs_per_kw(pv_om_per_kw=20,battery_om_per_kw=30)

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    cf = hybrid_plant.capacity_factors

    with subtests.test("pv total installed cost"):
        assert hybrid_plant.pv.total_installed_cost == approx(2000000,1e-3)

    with subtests.test("pv om cost"):
        assert hybrid_plant.pv.om_capacity == (20,)

    with subtests.test("battery total installed cost"):
        assert hybrid_plant.battery.total_installed_cost == approx(7000000,1e-3)

    with subtests.test("battery om cost"):
        assert hybrid_plant.battery.om_capacity == (30,)

def test_detailed_pv_system_capacity(hybrid_config, subtests):
    with subtests.test("Detailed PV model (pvsamv1) using defaults except the top level system_capacity_kw parameter"):
        annual_energy_expected = 11128604
        npv_expected = -2436229
        technologies = hybrid_config["technologies"]
        solar_only = deepcopy({key: technologies[key] for key in ('pv', 'grid')})   # includes system_capacity_kw parameter
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model but don't change any defaults
        solar_only['grid']['interconnect_kw'] = 150e3
        hybrid_config["technologies"] = solar_only
        hi = HoppInterface(hybrid_config)
        hybrid_plant = hi.system
        assert hybrid_plant.pv.value('subarray1_nstrings') == 1343
        hybrid_plant.layout.plot()

        hi.simulate()

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)

    
    with subtests.test("Detailed PV model (pvsamv1) using parameters from file except the top level system_capacity_kw parameter"):
        pvsamv1_defaults_file = Path(__file__).absolute().parent / "pvsamv1_basic_params.json"
        with open(pvsamv1_defaults_file, 'r') as f:
            tech_config = json.load(f)
        solar_only = deepcopy({key: technologies[key] for key in ('pv', 'grid')})   # includes system_capacity_kw parameter
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model
        solar_only['pv']['tech_config'] = tech_config       # specify parameters
        solar_only['grid']['interconnect_kw'] = 150e3
        hybrid_config["technologies"] = solar_only
        with raises(Exception) as context:
            hi = HoppInterface(hybrid_config)
        assert "The specified system capacity of 5000 kW is more than 5% from the value calculated" in str(context.value)

        # Run detailed PV model (pvsamv1) using file parameters, minus the number of strings, and the top level system_capacity_kw parameter
        annual_energy_expected = 8955045
        npv_expected = -2622684
        pvsamv1_defaults_file = Path(__file__).absolute().parent / "pvsamv1_basic_params.json"
        with open(pvsamv1_defaults_file, 'r') as f:
            tech_config = json.load(f)
        tech_config.pop('subarray1_nstrings')
        solar_only = deepcopy({key: technologies[key] for key in ('pv', 'grid')})   # includes system_capacity_kw parameter
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model
        solar_only['pv']['tech_config'] = tech_config       # specify parameters
        solar_only['grid']['interconnect_kw'] = 150e3
        hybrid_config["technologies"] = solar_only
        hi = HoppInterface(hybrid_config)
        hybrid_plant = hi.system
        assert hybrid_plant.pv.value('subarray1_nstrings') == 1343
        hybrid_plant.layout.plot()

        hi.simulate()
        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)


def test_hybrid_detailed_pv_only(site, hybrid_config, subtests):
    with subtests.test("standalone detailed PV model (pvsamv1) using defaults"):
        annual_energy_expected = 11128604
        config = DetailedPVConfig.from_dict(detailed_pv)
        pv_plant = DetailedPVPlant(site=site, config=config)
        assert pv_plant.system_capacity_kw == approx(pv_kw, 1e-2)
        pv_plant.simulate_power(1, False)
        assert pv_plant.system_capacity_kw == approx(pv_kw, 1e-2)
        assert pv_plant._system_model.Outputs.annual_energy == approx(annual_energy_expected, 1e-2)
        assert pv_plant._system_model.Outputs.capacity_factor == approx(25.66, 1e-2)

    with subtests.test("detailed PV model (pvsamv1) using defaults"):
        technologies = hybrid_config["technologies"]
        npv_expected = -2436229
        solar_only = {
            'pv': detailed_pv,
            'grid': technologies['grid']
        }
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model but don't change any defaults
        solar_only['grid']['interconnect_kw'] = 150e3
        hybrid_config["technologies"] = solar_only
        hi = HoppInterface(hybrid_config)
        hybrid_plant = hi.system
        hybrid_plant.layout.plot()

        hi.simulate()

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)

    with subtests.test("Detailed PV model (pvsamv1) using parameters from file"):
        annual_energy_expected = 102997528
        npv_expected = -25049424
        pvsamv1_defaults_file = Path(__file__).absolute().parent / "pvsamv1_basic_params.json"
        with open(pvsamv1_defaults_file, 'r') as f:
            tech_config = json.load(f)
        solar_only = deepcopy({key: technologies[key] for key in ('pv', 'grid')})
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model
        solar_only['pv']['tech_config'] = tech_config       # specify parameters
        solar_only['grid']['interconnect_kw'] = 150e3
        solar_only['pv']['system_capacity_kw'] = 50000      # use another system capacity
        hybrid_config["technologies"] = solar_only
        hi = HoppInterface(hybrid_config)
        hybrid_plant = hi.system
        hybrid_plant.layout.plot()

        hi.simulate()

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)

    # # Run user-instantiated or user-defined detailed PV model (pvsamv1) using parameters from file
    # config = DetailedPVConfig.from_dict(solar_only['pv'])
    # power_sources = {
    #     'pv': {
    #         'pv_plant': DetailedPVPlant(site=site, config=config),
    #     },
    #     'grid': {
    #         'interconnect_kw': 150e3
    #     }
    # }
    # hybrid_plant = HybridSimulation(power_sources, site)
    # hybrid_plant.layout.plot()
    # hi.simulate()
    # aeps = hybrid_plant.annual_energies
    # npvs = hybrid_plant.net_present_values
    # assert aeps.pv == approx(annual_energy_expected, 1e-3)
    # assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
    # assert npvs.pv == approx(npv_expected, 1e-3)
    # assert npvs.hybrid == approx(npv_expected, 1e-3)

    with subtests.test("Detailed PV model using parameters from file and autosizing electrical parameters"):
        annual_energy_expected = 102319358
        npv_expected = -25110524
        pvsamv1_defaults_file = Path(__file__).absolute().parent / "pvsamv1_basic_params.json"
        with open(pvsamv1_defaults_file, 'r') as f:
            tech_config = json.load(f)
        solar_only = deepcopy({key: technologies[key] for key in ('pv', 'grid')})
        solar_only['pv']['use_pvwatts'] = False             # specify detailed PV model
        solar_only['pv']['tech_config'] = tech_config       # specify parameters
        solar_only['grid']['interconnect_kw'] = 150e3
        solar_only['pv'].pop('system_capacity_kw')          # use default system capacity instead

        # autosize number of strings, number of inverters and adjust system capacity
        n_strings, n_combiners, n_inverters, calculated_system_capacity = size_electrical_parameters(
            target_system_capacity=solar_only['pv']['tech_config']['system_capacity'],
            target_dc_ac_ratio=1.34,
            modules_per_string=solar_only['pv']['tech_config']['subarray1_modules_per_string'],
            module_power= \
                solar_only['pv']['tech_config']['cec_i_mp_ref'] \
                * solar_only['pv']['tech_config']['cec_v_mp_ref'] \
                * 1e-3,
            inverter_power=solar_only['pv']['tech_config']['inv_snl_paco'] * 1e-3,
            n_inputs_inverter=50,
            n_inputs_combiner=32
        )
        assert n_strings == 13435
        assert n_combiners == 420
        assert n_inverters == 50
        assert calculated_system_capacity == approx(50002.2, 1e-3)
        solar_only['pv']['tech_config']['subarray1_nstrings'] = n_strings
        solar_only['pv']['tech_config']['inverter_count'] = n_inverters
        solar_only['pv']['tech_config']['system_capacity'] = calculated_system_capacity

        hybrid_config["technologies"] = solar_only
        hi = HoppInterface(hybrid_config)
        hybrid_plant = hi.system
        hybrid_plant.layout.plot()

        hi.simulate()

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert hybrid_plant.pv.system_capacity_kw == approx(50002.2, 1e-2)
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)


def test_hybrid_user_instantiated(site, subtests):
    # Run detailed PV model (pvsamv1) using defaults and user-instantiated financial models
    annual_energy_expected = 11128604
    npv_expected = -2436229
    system_capacity_kw = 5000
    system_capacity_kw_expected = 4998
    interconnect_kw = 150e3

    layout_params = {
        "x_position": 0.5, 
        "y_position": 0.5, 
        "aspect_power": 0, 
        "gcr": 0.5, 
        "s_buffer": 2, 
        "x_buffer": 2
    }

    # Run non-user-instantiated to compare against
    with subtests.test("baseline comparison"):
        solar_only = {
            'pv': {
                'use_pvwatts': False,
                'tech_config': {'system_capacity_kw': system_capacity_kw},
                "layout_params": layout_params,
                'dc_degradation': [0] * 25
            },
            'grid': {
                'interconnect_kw': interconnect_kw,
                'ppa_price': 0.01
            }
        }
        hopp_config = {
            "site": site,
            "technologies": solar_only
        }
        hi = HoppInterface(hopp_config)
        hybrid_plant = hi.system
        hybrid_plant.layout.plot()
        hybrid_plant.simulate()
        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert hybrid_plant.pv.system_capacity_kw == approx(system_capacity_kw, 1e-2)
        assert aeps.pv == approx(annual_energy_expected, 1e-2)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-2)
        assert npvs.pv == approx(npv_expected, 1e-2)
        assert npvs.hybrid == approx(npv_expected, 1e-2)


    with subtests.test("detailed PV plant, grid and respective financial models"):
        # Run 
        power_sources = {
            'pv': {
                'use_pvwatts': False,
                'system_capacity_kw': system_capacity_kw,
                'layout_params': layout_params,
                'fin_model': 'FlatPlatePVSingleOwner',
                'dc_degradation': [0] * 25
            },
            'grid': {
                'interconnect_kw': interconnect_kw,
                'fin_model': 'GenericSystemSingleOwner',
                'ppa_price': 0.01
            }
        }
        hopp_config = {
            "site": site,
            "technologies": power_sources
        }
        hi = HoppInterface(hopp_config)
        hybrid_plant = hi.system
        assert hybrid_plant.pv is not None
        hybrid_plant.layout.plot()

        hybrid_plant.simulate()

        aeps = hybrid_plant.annual_energies
        npvs = hybrid_plant.net_present_values
        assert hybrid_plant.pv._system_model.value("system_capacity") == approx(system_capacity_kw_expected, 1e-3)
        assert hybrid_plant.pv._financial_model.value("system_capacity") == approx(system_capacity_kw_expected, 1e-3)
        assert aeps.pv == approx(annual_energy_expected, 1e-3)
        assert aeps.hybrid == approx(annual_energy_expected, 1e-3)
        assert npvs.pv == approx(npv_expected, 1e-3)
        assert npvs.hybrid == approx(npv_expected, 1e-3)


def test_hybrid(hybrid_config):
    """
    Performance from Wind is slightly different from wind-only case because the solar presence modified the wind layout
    """
    technologies = hybrid_config["technologies"]
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'grid')}
    hybrid_config["technologies"] = solar_wind_hybrid
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(8703525.94, 13)
    assert aeps.wind == approx(33615479.57, 1e3)
    assert aeps.hybrid == approx(41681662.63, 1e3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.wind == approx(-13909363, 1e3)
    assert npvs.hybrid == approx(-19216589, 1e3)


def test_wind_pv_with_storage_dispatch(hybrid_config):
    technologies = hybrid_config["technologies"]
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    hybrid_config["technologies"] = wind_pv_battery
    hybrid_config["technologies"]["grid"]["ppa_price"] = 0.03
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    taxes = hybrid_plant.federal_taxes
    apv = hybrid_plant.energy_purchases
    debt = hybrid_plant.debt_payment
    esv = hybrid_plant.energy_sales
    depr = hybrid_plant.federal_depreciation_totals
    insr = hybrid_plant.insurance_expenses
    om = hybrid_plant.om_total_expenses
    rev = hybrid_plant.total_revenues
    tc = hybrid_plant.tax_incentives

    assert aeps.pv == approx(9882421, rel=0.05)
    assert aeps.wind == approx(31951719, rel=0.05)
    assert aeps.battery == approx(-99103, rel=0.05)
    assert aeps.hybrid == approx(43489117, rel=0.05)

    assert npvs.pv == approx(-719826, rel=5e-2)
    assert npvs.wind == approx(-2573090, rel=5e-2)
    assert npvs.battery == approx(-4871034, rel=5e-2)
    assert npvs.hybrid == approx(-8254104, rel=5e-2)

    assert taxes.pv[1] == approx(94661, rel=5e-2)
    assert taxes.wind[1] == approx(413068, rel=5e-2)
    assert taxes.battery[1] == approx(248373, rel=5e-2)
    assert taxes.hybrid[1] == approx(804904, rel=5e-2)

    assert apv.pv[1] == approx(0, rel=5e-2)
    assert apv.wind[1] == approx(0, rel=5e-2)
    assert apv.battery[1] == approx(-4070354, rel=5e-2)
    assert apv.hybrid[1] == approx(-348443, rel=5e-2)

    assert debt.pv[1] == approx(0, rel=5e-2)
    assert debt.wind[1] == approx(0, rel=5e-2)
    assert debt.battery[1] == approx(0, rel=5e-2)
    assert debt.hybrid[1] == approx(0, rel=5e-2)

    assert esv.pv[1] == approx(9854885, rel=5e-2)
    assert esv.wind[1] == approx(31951719, rel=5e-2)
    assert esv.battery[1] == approx(3973442, rel=5e-2)
    assert esv.hybrid[1] == approx(42058135, rel=5e-2)

    assert depr.pv[1] == approx(745532, rel=5e-2)
    assert depr.wind[1] == approx(2651114, rel=5e-2)
    assert depr.battery[1] == approx(1266736, rel=5e-2)
    assert depr.hybrid[1] == approx(4663383, rel=5e-2)

    assert insr.pv[0] == approx(0, rel=5e-2)
    assert insr.wind[0] == approx(0, rel=5e-2)
    assert insr.battery[0] == approx(0, rel=5e-2)
    assert insr.hybrid[0] == approx(0, rel=5e-2)

    assert om.pv[1] == approx(74993, rel=5e-2)
    assert om.wind[1] == approx(430000, rel=5e-2)
    assert om.battery[1] == approx(75000, rel=5e-2)
    assert om.hybrid[1] == approx(569993, rel=5e-2)

    assert rev.pv[1] == approx(352218, rel=5e-2)
    assert rev.wind[1] == approx(904283, rel=5e-2)
    assert rev.battery[1] == approx(167939, rel=5e-2)
    assert rev.hybrid[1] == approx(1334802, rel=5e-2)

    assert tc.pv[1] == approx(1295889, rel=5e-2)
    assert tc.wind[1] == approx(830744, rel=5e-2)
    assert tc.battery[1] == approx(2201850, rel=5e-2)
    assert tc.hybrid[1] == approx(4338902, rel=5e-2)


def test_tower_pv_hybrid(hybrid_config):
    interconnection_size_kw_test = 50000
    technologies_test = {
        'tower': {
            'cycle_capacity_kw': 50 * 1000, 
            'solar_multiple': 2.0, 
            'tes_hours': 12.0
        },
        'pv': {'system_capacity_kw': 50 * 1000},
        'grid': {
            'interconnect_kw': interconnection_size_kw_test,
            'ppa_price': 0.12
        }
    }

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv', 'grid')}
    hybrid_config["technologies"] = solar_hybrid
    dispatch_options={'is_test_start_year': True, 'is_test_end_year': True}
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system
    hybrid_plant.tower.value('helio_width', 8.0)
    hybrid_plant.tower.value('helio_height', 8.0)

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104286701.28, 1e-3)
    assert aeps.tower == approx(3769716.50, 5e-2)
    assert aeps.hybrid == approx(107780622.67, 1e-2)

    # TODO: check npv for csp would require a full simulation
    assert npvs.pv == approx(45233832.23, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)


def test_trough_pv_hybrid(hybrid_config):
    interconnection_size_kw_test = 50000
    technologies_test = {
        'trough': {
            'cycle_capacity_kw': 50 * 1000, 
            'solar_multiple': 2.0, 
            'tes_hours': 12.0
        },
        'pv': {'system_capacity_kw': 50 * 1000},
        'grid': {
            'interconnect_kw': interconnection_size_kw_test,
            'ppa_price': 0.12
        },
    }

    solar_hybrid = {key: technologies_test[key] for key in ('trough', 'pv', 'grid')}
    hybrid_config["technologies"] = solar_hybrid
    dispatch_options={'is_test_start_year': True, 'is_test_end_year': True}
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104286701.17, 1e-3)
    assert aeps.trough == approx(1858279.58, 2e-2)
    assert aeps.hybrid == approx(106111732.52, 1e-3)

    assert npvs.pv == approx(80738107, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)


def test_tower_pv_battery_hybrid(hybrid_config):
    interconnection_size_kw_test = 50000
    technologies_test = {
        'tower': {
            'cycle_capacity_kw': 50 * 1000, 
            'solar_multiple': 2.0, 
            'tes_hours': 12.0
        },
        'pv': {'system_capacity_kw': 50 * 1000},
        'battery': {
            'system_capacity_kwh': 40 * 1000,
            'system_capacity_kw': 20 * 1000
        },
        'grid': {
            'interconnect_kw': interconnection_size_kw_test,
            'ppa_price': 0.12
        }
    }

    solar_hybrid = {key: technologies_test[key] for key in ('tower', 'pv', 'battery', 'grid')}
    dispatch_options={'is_test_start_year': True, 'is_test_end_year': True}
    hybrid_config["technologies"] = solar_hybrid
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system
    hybrid_plant.tower.value('helio_width', 10.0)
    hybrid_plant.tower.value('helio_height', 10.0)

    hi.simulate()

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(104286701, 1e-3)
    assert aeps.tower == approx(3783849, 5e-2)
    assert aeps.battery == approx(-9477, 2e-1)
    assert aeps.hybrid == approx(107903653, 1e-2)

    assert npvs.pv == approx(80738107, 1e3)
    #assert npvs.tower == approx(-13909363, 1e3)
    #assert npvs.hybrid == approx(-19216589, 1e3)

def test_hybrid_om_costs_error(hybrid_config):
    technologies = hybrid_config["technologies"]
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'}
    hybrid_config["technologies"] = wind_pv_battery
    hybrid_config["technologies"]["grid"]["ppa_price"] = 0.03
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system
    hybrid_plant.battery._financial_model.value('om_production', (1,))

    try:
        hi.simulate()
    except ValueError as e:
        assert e

def test_hybrid_om_costs(hybrid_config):
    technologies = hybrid_config["technologies"]
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'}
    hybrid_config["technologies"] = wind_pv_battery
    hybrid_config["technologies"]["grid"]["ppa_price"] = 0.03
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

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

    hi.simulate()

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
    hi.simulate()
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
    hi.simulate()
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

def test_hybrid_tax_incentives(hybrid_config):
    technologies = hybrid_config["technologies"]
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'}
    hybrid_config["technologies"] = wind_pv_battery
    hybrid_config["technologies"]["grid"]["ppa_price"] = 0.03
    hybrid_config["config"]["dispatch_options"] = dispatch_options
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    hybrid_plant.pv._financial_model.value('itc_fed_percent', [0.0])
    hybrid_plant.wind._financial_model.value('ptc_fed_amount', (1,))
    hybrid_plant.pv._financial_model.value('ptc_fed_amount', (2,))
    hybrid_plant.battery._financial_model.value('ptc_fed_amount', (3,))
    hybrid_plant.wind._financial_model.value('ptc_fed_escal', 0)
    hybrid_plant.pv._financial_model.value('ptc_fed_escal', 0)
    hybrid_plant.battery._financial_model.value('ptc_fed_escal', 0)

    hi.simulate()

    ptc_wind = hybrid_plant.wind._financial_model.value("cf_ptc_fed")[1]
    assert ptc_wind == approx(hybrid_plant.wind._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.wind.annual_energy_kwh, rel=1e-3)

    ptc_pv = hybrid_plant.pv._financial_model.value("cf_ptc_fed")[1]
    assert ptc_pv == approx(hybrid_plant.pv._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.pv.annual_energy_kwh, rel=1e-3)

    ptc_batt = hybrid_plant.battery._financial_model.value("cf_ptc_fed")[1]
    assert ptc_batt == approx(hybrid_plant.battery._financial_model.value("ptc_fed_amount")[0]
           * hybrid_plant.battery._financial_model.value('batt_annual_discharge_energy')[1], rel=1e-3)

    ptc_hybrid = hybrid_plant.grid._financial_model.value("cf_ptc_fed")[1]
    ptc_fed_amount = hybrid_plant.grid._financial_model.value("ptc_fed_amount")[0]
    assert ptc_fed_amount == approx(1.229, rel=1e-2)
    assert ptc_hybrid == approx(ptc_fed_amount * hybrid_plant.grid._financial_model.value('cf_energy_net')[1], rel=1e-3)


def test_capacity_credit(hybrid_config):
    technologies = hybrid_config["technologies"]
    site = create_default_site_info(capacity_hours=capacity_credit_hours)
    wind_pv_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery')}
    wind_pv_battery['grid'] = {
        'interconnect_kw': interconnection_size_kw,
        'ppa_price': 0.03
    }
    hybrid_config["technologies"] = wind_pv_battery
    hybrid_config["site"] = site
    hi = HoppInterface(hybrid_config)
    hybrid_plant = hi.system

    assert hybrid_plant.interconnect_kw == 15e3

    # Backup values for resetting before tests
    gen_max_feasible_orig = hybrid_plant.battery.gen_max_feasible
    capacity_hours_orig = hybrid_plant.site.capacity_hours
    interconnect_kw_orig = hybrid_plant.interconnect_kw
    def reinstate_orig_values():
        hybrid_plant.battery.gen_max_feasible = gen_max_feasible_orig
        hybrid_plant.site.capacity_hours = capacity_hours_orig
        hybrid_plant.interconnect_kw = interconnect_kw_orig

    # Test when 0 gen_max_feasible
    reinstate_orig_values()
    hybrid_plant.battery.gen_max_feasible = [0] * 8760
    capacity_credit_battery = hybrid_plant.battery.calc_capacity_credit_percent(hybrid_plant.interconnect_kw)
    assert capacity_credit_battery == approx(0, rel=0.05)
    # Test when representative gen_max_feasible
    reinstate_orig_values()
    hybrid_plant.battery.gen_max_feasible = [2500] * 8760
    capacity_credit_battery = hybrid_plant.battery.calc_capacity_credit_percent(hybrid_plant.interconnect_kw)
    assert capacity_credit_battery == approx(50, rel=0.05)
    # Test when no capacity hours
    reinstate_orig_values()
    hybrid_plant.battery.gen_max_feasible = [2500] * 8760
    hybrid_plant.site.capacity_hours = [False] * 8760
    capacity_credit_battery = hybrid_plant.battery.calc_capacity_credit_percent(hybrid_plant.interconnect_kw)
    assert capacity_credit_battery == approx(0, rel=0.05)
    # Test when no interconnect capacity
    reinstate_orig_values()
    hybrid_plant.battery.gen_max_feasible = [2500] * 8760
    hybrid_plant.interconnect_kw = 0
    capacity_credit_battery = hybrid_plant.battery.calc_capacity_credit_percent(hybrid_plant.interconnect_kw)
    assert capacity_credit_battery == approx(0, rel=0.05)

    # Test integration with system simulation
    reinstate_orig_values()
    cap_payment_mw = 100000
    hybrid_plant.assign({"cp_capacity_payment_amount": [cap_payment_mw]})

    assert hybrid_plant.interconnect_kw == 15e3

    hi.simulate()

    total_gen_max_feasible = np.array(hybrid_plant.pv.gen_max_feasible) \
                           + np.array(hybrid_plant.wind.gen_max_feasible) \
                           + np.array(hybrid_plant.battery.gen_max_feasible)
    assert sum(hybrid_plant.grid.gen_max_feasible) == approx(sum(np.minimum(hybrid_plant.grid.interconnect_kw * hybrid_plant.site.interval / 60, \
                                                                            total_gen_max_feasible)), rel=0.01)

    total_nominal_capacity = hybrid_plant.pv.calc_nominal_capacity(hybrid_plant.interconnect_kw) \
                           + hybrid_plant.wind.calc_nominal_capacity(hybrid_plant.interconnect_kw) \
                           + hybrid_plant.battery.calc_nominal_capacity(hybrid_plant.interconnect_kw)
    assert total_nominal_capacity == approx(18845.8, rel=0.01)
    assert total_nominal_capacity == approx(hybrid_plant.grid.hybrid_nominal_capacity, rel=0.01)
    
    capcred = hybrid_plant.capacity_credit_percent
    assert capcred['pv'][0] == approx(8.03, rel=0.05)
    assert capcred['wind'][0] == approx(33.25, rel=0.10)
    assert capcred['battery'][0] == approx(58.95, rel=0.05)
    assert capcred['hybrid'][0] == approx(43.88, rel=0.05)

    cp_pay = hybrid_plant.capacity_payments
    np_cap = hybrid_plant.system_nameplate_mw # This is not the same as nominal capacity...
    assert cp_pay['pv'][1]/(np_cap['pv'])/(capcred['pv'][0]/100) == approx(cap_payment_mw, 0.05)
    assert cp_pay['wind'][1]/(np_cap['wind'])/(capcred['wind'][0]/100) == approx(cap_payment_mw, 0.05)
    assert cp_pay['battery'][1]/(np_cap['battery'])/(capcred['battery'][0]/100) == approx(cap_payment_mw, 0.05)
    assert cp_pay['hybrid'][1]/(np_cap['hybrid'])/(capcred['hybrid'][0]/100) == approx(cap_payment_mw, 0.05)

    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    taxes = hybrid_plant.federal_taxes
    apv = hybrid_plant.energy_purchases
    debt = hybrid_plant.debt_payment
    esv = hybrid_plant.energy_sales
    depr = hybrid_plant.federal_depreciation_totals
    insr = hybrid_plant.insurance_expenses
    om = hybrid_plant.om_total_expenses
    rev = hybrid_plant.total_revenues
    tc = hybrid_plant.tax_incentives

    print("AEP", [aeps.pv, aeps.wind, aeps.battery, aeps.hybrid])
    print("NPV", [npvs.pv, npvs.wind, npvs.battery, npvs.hybrid])
    print("TAXES", [taxes.pv[1], taxes.wind[1], taxes.battery[1], taxes.hybrid[1]])
    print("APV", [apv.pv[1], apv.wind[1], apv.battery[1], apv.hybrid[1]])
    print("ESV", [esv.pv[1], esv.wind[1], esv.battery[1], esv.hybrid[1]])
    print("DEPR", [depr.pv[1], depr.wind[1], depr.battery[1], depr.hybrid[1]])
    print("OM", [om.pv[1], om.wind[1], om.battery[1], om.hybrid[1]])
    print("REV", [rev.pv[1], rev.wind[1], rev.battery[1], rev.hybrid[1]])
    print("TC", [tc.pv[1], tc.wind[1], tc.battery[1], tc.hybrid[1]])

    assert aeps.pv == approx(9882421, rel=0.05)
    assert aeps.wind == approx(31951719, rel=0.05)
    assert aeps.battery == approx(-97166, rel=0.05)
    assert aeps.hybrid == approx(43489117, rel=0.05)

    assert npvs.pv == approx(-435187, rel=5e-2)
    assert npvs.wind == approx(-369348, rel=5e-2)
    assert npvs.battery == approx(-2700460, rel=5e-2)
    assert npvs.hybrid == approx(-2129876, rel=5e-2)

    assert taxes.pv[1] == approx(83720, rel=5e-2)
    assert taxes.wind[1] == approx(365206, rel=5e-2)
    assert taxes.battery[1] == approx(189346, rel=5e-2)
    assert taxes.hybrid[1] == approx(598426, rel=5e-2)

    assert apv.pv[1] == approx(0, rel=5e-2)
    assert apv.wind[1] == approx(0, rel=5e-2)
    assert apv.battery[1] == approx(-4070354, rel=5e-2)
    assert apv.hybrid[1] == approx(-348443, rel=5e-2)

    assert debt.pv[1] == approx(0, rel=5e-2)
    assert debt.wind[1] == approx(0, rel=5e-2)
    assert debt.battery[1] == approx(0, rel=5e-2)
    assert debt.hybrid[1] == approx(0, rel=5e-2)

    assert esv.pv[1] == approx(9854885, rel=5e-2)
    assert esv.wind[1] == approx(31951719, rel=5e-2)
    assert esv.battery[1] == approx(3973442, rel=5e-2)
    assert esv.hybrid[1] == approx(42058135, rel=5e-2)

    assert depr.pv[1] == approx(745532, rel=5e-2)
    assert depr.wind[1] == approx(2651114, rel=5e-2)
    assert depr.battery[1] == approx(1266736, rel=5e-2)
    assert depr.hybrid[1] == approx(4663383, rel=5e-2)

    assert insr.pv[0] == approx(0, rel=5e-2)
    assert insr.wind[0] == approx(0, rel=5e-2)
    assert insr.battery[0] == approx(0, rel=5e-2)
    assert insr.hybrid[0] == approx(0, rel=5e-2)

    assert om.pv[1] == approx(74993, rel=5e-2)
    assert om.wind[1] == approx(430000, rel=5e-2)
    assert om.battery[1] == approx(75000, rel=5e-2)
    assert om.hybrid[1] == approx(579993, rel=5e-2)

    assert rev.pv[1] == approx(391851, rel=5e-2)
    assert rev.wind[1] == approx(1211138, rel=5e-2)
    assert rev.battery[1] == approx(470175, rel=5e-2)
    assert rev.hybrid[1] == approx(2187556, rel=5e-2)

    assert tc.pv[1] == approx(1295889, rel=5e-2)
    assert tc.wind[1] == approx(830744, rel=5e-2)
    assert tc.battery[1] == approx(2201850, rel=5e-2)
    assert tc.hybrid[1] == approx(4338902, rel=5e-2)
