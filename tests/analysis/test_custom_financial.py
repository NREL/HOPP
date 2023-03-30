from pytest import approx, fixture
from pathlib import Path
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.layout.hybrid_layout import PVGridParameters, WindBoundaryGridParameters
from hybrid.financial.custom_financial_model import CustomFinancialModel
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.detailed_pv_plant import DetailedPVPlant
from examples.Detailed_PV_Layout.detailed_pv_layout import DetailedPVParameters, DetailedPVLayout
from hybrid.grid import Grid
import json


solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"

@fixture
def site():
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)


default_fin_config = {
    'batt_replacement_schedule_percent': [0],
    'batt_bank_replacement': [0],
    'batt_replacement_option': 0,
    'batt_computed_bank_capacity': 0,
    'batt_meter_position': 0,
    'battery_per_kWh': 0,
    'en_batt': 0,
    'en_standalone_batt': 0,
    'om_fixed': [1],
    'om_production': [2],
    'om_capacity': (0,),
    'om_batt_fixed_cost': 0,
    'om_batt_variable_cost': [0],
    'om_batt_capacity_cost': 0,
    'om_batt_replacement_cost': 0,
    'om_batt_nameplate': 0,
    'om_replacement_cost_escal': 0,
    'system_use_lifetime_output': 0,
    'inflation_rate': 2.5,
    'real_discount_rate': 6.4,
    'cp_capacity_credit_percent': [0],
    'degradation': [0],
}


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
    npv_expected = -40021910

    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hybrid/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)

    layout_params = PVGridParameters(x_position=0.5,
                                     y_position=0.5,
                                     aspect_power=0,
                                     gcr=0.3,
                                     s_buffer=2,
                                     x_buffer=2)
    interconnect_kw = 150e6

    detailed_pvplant = DetailedPVPlant(
        site=site,
        pv_config={
            'tech_config': tech_config,
            'layout_params': layout_params,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    grid_source = Grid(
        site=site,
        grid_config={
            'interconnect_kw': interconnect_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    power_sources = {
        'pv': {
            'pv_plant': detailed_pvplant,
        },
        'grid': {
            'grid_source': grid_source
        }
    }
    hybrid_plant = HybridSimulation(power_sources, site)
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


def test_hybrid_simple_pv_with_wind(site):
    # Run wind + simple PV (pvwattsv8) hybrid plant with custom financial model
    annual_energy_expected_pv = 98821626
    annual_energy_expected_wind = 33637984
    annual_energy_expected_hybrid = 132459610
    npv_expected_pv = -40714053
    npv_expected_wind = -12059963
    npv_expected_hybrid = -52774017

    interconnect_kw = 150e6
    pv_kw = 50000
    wind_kw = 10000

    grid_source = Grid(
        site=site,
        grid_config={
            'interconnect_kw': interconnect_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    power_sources = {
        'pv': {
            'system_capacity_kw': pv_kw,
            'layout_params': PVGridParameters(x_position=0.5,
                                              y_position=0.5,
                                              aspect_power=0,
                                              gcr=0.5,
                                              s_buffer=2,
                                              x_buffer=2),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                        border_offset=0.5,
                                                        grid_angle=0.5,
                                                        grid_aspect_power=0.5,
                                                        row_phase_offset=0.5),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'grid': {
            'grid_source': grid_source,
        }
    }
    hybrid_plant = HybridSimulation(power_sources, site)
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
    annual_energy_expected_pv = 21500708
    annual_energy_expected_wind = 33637984
    annual_energy_expected_hybrid = 55138692
    npv_expected_pv = -8015242
    npv_expected_wind = -12059963
    npv_expected_hybrid = -20075205

    interconnect_kw = 150e6
    wind_kw = 10000

    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hybrid/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)
    
    # NOTE: PV array shrunk to avoid problem associated with flicker calculation
    tech_config['system_capacity'] = 10000
    tech_config['inverter_count'] = 10
    tech_config['subarray1_nstrings'] = 2687

    layout_params = PVGridParameters(x_position=0.5,
                                     y_position=0.5,
                                     aspect_power=0,
                                     gcr=0.3,
                                     s_buffer=2,
                                     x_buffer=2)

    detailed_pvplant = DetailedPVPlant(
        site=site,
        pv_config={
            'tech_config': tech_config,
            'layout_params': layout_params,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    grid_source = Grid(
        site=site,
        grid_config={
            'interconnect_kw': interconnect_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    power_sources = {
        'pv': {
            'pv_plant': detailed_pvplant,
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                        border_offset=0.5,
                                                        grid_angle=0.5,
                                                        grid_aspect_power=0.5,
                                                        row_phase_offset=0.5),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'grid': {
            'grid_source': grid_source,
        }
    }
    hybrid_plant = HybridSimulation(power_sources, site)
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


def test_hybrid_simple_pv_with_wind_storage_dispatch(site):
    # Test wind + simple PV (pvwattsv8) + storage with dispatch hybrid plant with custom financial model
    annual_energy_expected_pv = 9882421
    annual_energy_expected_wind = 33637983
    annual_energy_expected_battery = -31287
    annual_energy_expected_hybrid = 43489117
    npv_expected_pv = -2138980
    npv_expected_wind = -5483725
    npv_expected_battery = -8163435
    npv_expected_hybrid = -15786128

    interconnect_kw = 15000
    pv_kw = 5000
    wind_kw = 10000
    batt_kw = 5000

    power_sources = {
        'pv': {
            'system_capacity_kw': pv_kw,
            'layout_params': PVGridParameters(x_position=0.5,
                                              y_position=0.5,
                                              aspect_power=0,
                                              gcr=0.5,
                                              s_buffer=2,
                                              x_buffer=2),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                        border_offset=0.5,
                                                        grid_angle=0.5,
                                                        grid_aspect_power=0.5,
                                                        row_phase_offset=0.5),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    }
    hybrid_plant = HybridSimulation(power_sources, site)
    hybrid_plant.layout.plot()
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
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
    annual_energy_expected_pv = 20413333
    annual_energy_expected_wind = 33637984
    annual_energy_expected_battery = -30147
    annual_energy_expected_hybrid = 54020819
    npv_expected_pv = -4104598
    npv_expected_wind = -5483725
    npv_expected_battery = -8163128
    npv_expected_hybrid = -17751533

    interconnect_kw = 15000
    wind_kw = 10000
    batt_kw = 5000

    pvsamv1_defaults_file = Path(__file__).absolute().parent.parent / "hybrid/pvsamv1_basic_params.json"
    with open(pvsamv1_defaults_file, 'r') as f:
        tech_config = json.load(f)
    
    # NOTE: PV array shrunk to avoid problem associated with flicker calculation
    tech_config['system_capacity'] = 10000
    tech_config['inverter_count'] = 10
    tech_config['subarray1_nstrings'] = 2687

    detailed_pvplant = DetailedPVPlant(
        site=site,
        pv_config={
            'tech_config': tech_config,
            'layout_params': PVGridParameters(x_position=0.5,
                                              y_position=0.5,
                                              aspect_power=0,
                                              gcr=0.5,
                                              s_buffer=2,
                                              x_buffer=2),
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    )

    power_sources = {
        'pv': {
            'pv_plant': detailed_pvplant,
        },
        'wind': {
            'num_turbines': 5,
            'turbine_rating_kw': wind_kw / 5,
            'layout_mode': 'boundarygrid',
            'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                        border_offset=0.5,
                                                        grid_angle=0.5,
                                                        grid_aspect_power=0.5,
                                                        row_phase_offset=0.5),
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'battery': {
            'system_capacity_kwh': batt_kw * 4,
            'system_capacity_kw': batt_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        },
        'grid': {
            'interconnect_kw': interconnect_kw,
            'fin_model': CustomFinancialModel(default_fin_config),
        }
    }
    hybrid_plant = HybridSimulation(power_sources, site)
    hybrid_plant.layout.plot()
    hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.01
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    assert aeps.pv == approx(annual_energy_expected_pv, 1e-3)
    assert aeps.wind == approx(annual_energy_expected_wind, 1e-3)
    assert aeps.battery == approx(annual_energy_expected_battery, 1e-3)
    assert aeps.hybrid == approx(annual_energy_expected_hybrid, 1e-3)
    assert npvs.pv == approx(npv_expected_pv, 1e-3)
    assert npvs.wind == approx(npv_expected_wind, 1e-3)
    assert npvs.battery == approx(npv_expected_battery, 1e-3)
    assert npvs.hybrid == approx(npv_expected_hybrid, 1e-3)
