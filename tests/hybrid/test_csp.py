import pytest
import pandas as pd
import datetime
from pathlib import Path


from hybrid.sites import SiteInfo, flatirons_site
from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant
from hybrid.hybrid_simulation import HybridSimulation

@pytest.fixture
def site():
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

def test_pySSC_tower_model(site):
    """Testing pySSC tower model using heuristic dispatch method"""
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    expected_energy = 4029953.45

    csp = TowerPlant(site, tower_config)
    csp.generate_field()

    start_datetime, end_datetime = CspDispatch.get_start_end_datetime(293*24, 72)

    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})

    # csp.ssc.create_lk_inputs_file("test.lk", csp.site.solar_resource.filename)  # Energy output: 4029953.44
    tech_outputs = csp.ssc.execute()
    annual_energy = tech_outputs['annual_energy']

    print('Three days all at once starting 10/21, annual energy = {e:.0f} MWhe'.format(e=annual_energy * 1.e-3))

    # Testing if configuration was not overwritten
    assert csp.cycle_capacity_kw == tower_config['cycle_capacity_kw']
    assert csp.solar_multiple == tower_config['solar_multiple']
    assert csp.tes_hours == tower_config['tes_hours']

    assert annual_energy == pytest.approx(expected_energy, 1e-5)


def test_pySSC_tower_increment_simulation(site):
    """Testing pySSC tower model using heuristic dispatch method and incrementing simulation"""
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    csp = TowerPlant(site, tower_config)
    csp.generate_field()

    start_datetime, end_datetime = CspDispatch.get_start_end_datetime(293*24, 72)
    increment_duration = datetime.timedelta(hours=24)  # Time duration of each simulated horizon

    # Without Increments
    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})
    tech_outputs = csp.ssc.execute()
    wo_increments_annual_energy = tech_outputs['annual_energy']

    # With increments
    n = int((end_datetime - start_datetime).total_seconds() / increment_duration.total_seconds())
    for j in range(n):
        start_datetime_new = start_datetime + j * increment_duration
        end_datetime_new = start_datetime_new + increment_duration
        csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime_new)})
        csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime_new)})
        csp.update_ssc_inputs_from_plant_state()
        tech_outputs = csp.ssc.execute()
        csp.outputs.update_from_ssc_output(tech_outputs)
        csp.set_plant_state_from_ssc_outputs(tech_outputs, increment_duration.total_seconds())

    increments_annual_energy = csp.annual_energy_kwh

    assert increments_annual_energy == pytest.approx(wo_increments_annual_energy, 1e-5)


def test_pySSC_trough_model(site):
    """Testing pySSC trough model using heuristic dispatch method"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}

    expected_energy = 2116895.0210105316

    csp = TroughPlant(site, trough_config)

    start_datetime, end_datetime = CspDispatch.get_start_end_datetime(293*24, 72)

    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})

    # csp.ssc.create_lk_inputs_file("trough_test.lk", csp.site.solar_resource.filename)  # Energy output: 2100958.280693
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert csp.cycle_capacity_kw == trough_config['cycle_capacity_kw']
    assert csp.solar_multiple == trough_config['solar_multiple']
    assert csp.tes_hours == trough_config['tes_hours']

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)


def test_pySSC_trough_increment_simulation(site):
    """Testing pySSC trough model using heuristic dispatch method and incrementing simulation"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}

    csp = TroughPlant(site, trough_config)

    start_datetime, end_datetime = CspDispatch.get_start_end_datetime(293*24, 72)

    increment_duration = datetime.timedelta(hours=24)  # Time duration of each simulated horizon

    # Without Increments
    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})
    tech_outputs = csp.ssc.execute()
    wo_increments_annual_energy = tech_outputs['annual_energy']

    # With increments
    n = int((end_datetime - start_datetime).total_seconds() / increment_duration.total_seconds())
    for j in range(n):
        start_datetime_new = start_datetime + j * increment_duration
        end_datetime_new = start_datetime_new + increment_duration
        csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime_new)})
        csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime_new)})
        csp.update_ssc_inputs_from_plant_state()
        tech_outputs = csp.ssc.execute()
        csp.outputs.update_from_ssc_output(tech_outputs)
        csp.set_plant_state_from_ssc_outputs(tech_outputs, increment_duration.total_seconds())

    increments_annual_energy = csp.annual_energy_kwh

    assert increments_annual_energy == pytest.approx(wo_increments_annual_energy, 1e-5)


def test_value_csp_call(site):
    """Testing csp override of PowerSource value()"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}

    csp = TroughPlant(site, trough_config)

    # Testing value call get and set - system model
    assert csp.value('startup_time') == csp.ssc.get('startup_time')
    csp.value('startup_time', 0.25)
    assert csp.value('startup_time') == 0.25
    # financial model
    assert csp.value('inflation_rate') == csp._financial_model.FinancialParameters.inflation_rate
    csp.value('inflation_rate', 3.0)
    assert csp._financial_model.FinancialParameters.inflation_rate == 3.0
    # class setter and getter
    assert csp.value('tes_hours') == trough_config['tes_hours']
    csp.value('tes_hours', 6.0)
    assert csp.tes_hours == 6.0


def test_tower_with_dispatch_model(site):
    """Testing pySSC tower model using HOPP built-in dispatch model"""
    expected_energy = 3842225.688

    interconnection_size_kw = 50000
    technologies = {'tower': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 6.0,
                              'optimize_field_before_sim': False}}

    system = HybridSimulation(technologies, site,
                              interconnect_kw=interconnection_size_kw,
                              dispatch_options={'is_test_start_year': True,
                                                'is_test_end_year': True})

    system.tower.value('helio_width', 10.0)
    system.tower.value('helio_height', 10.0)

    system.tower.value('h_tower', 117.7)
    system.tower.value('rec_height', 11.3)
    system.tower.value('D_rec', 11.12)

    system.ppa_price = (0.12, )
    system.simulate()

    assert system.tower.annual_energy_kwh == pytest.approx(expected_energy, 1e-2)

    # Check dispatch targets
    disp_outputs = system.tower.outputs.dispatch
    ssc_outputs = system.tower.outputs.ssc_time_series
    for i in range(len(ssc_outputs['gen'])):
        # cycle start-up allowed
        target = 1 if (disp_outputs['is_cycle_generating'][i] + disp_outputs['is_cycle_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_pc_su_allowed'][i], 1e-5)
        # receiver start-up allowed
        target = 1 if (disp_outputs['is_field_generating'][i] + disp_outputs['is_field_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_rec_su_allowed'][i], 1e-5)
        # cycle thermal power
        start_power = system.tower.dispatch.allowable_cycle_startup_power if disp_outputs['is_cycle_starting'][i] else 0
        target = disp_outputs['cycle_thermal_power'][i] + start_power
        assert target == pytest.approx(ssc_outputs['q_dot_pc_target_on'][i], 1e-3)
        # thermal energy storage state-of-charge
        if i % system.dispatch_builder.options.n_roll_periods == 0:
            tes_estimate = disp_outputs['thermal_energy_storage'][i]
            tes_actual = ssc_outputs['e_ch_tes'][i]
            assert tes_estimate == pytest.approx(tes_actual, 0.01)
        # else:
        #     assert tes_estimate == pytest.approx(tes_actual, 0.15)


def test_trough_with_dispatch_model(site):
    """Testing pySSC tower model using HOPP built-in dispatch model"""
    expected_energy = 1873589.560

    interconnection_size_kw = 50000
    technologies = {'trough': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 6.0}}

    system = HybridSimulation(technologies, site,
                              interconnect_kw=interconnection_size_kw,
                              dispatch_options={'is_test_start_year': True,
                                                'is_test_end_year': True})
    system.ppa_price = (0.12,)
    system.simulate()

    assert system.trough.annual_energy_kwh == pytest.approx(expected_energy, 1e-2)

    # FIXME: This fails most likely due to poor estimates of trough thermal power input
    # Check dispatch targets
    # disp_outputs = system.trough.outputs.dispatch
    # ssc_outputs = system.trough.outputs.ssc_time_series
    # for i in range(len(ssc_outputs['gen'])):
    #     # cycle start-up allowed
    #     target = 1 if (disp_outputs['is_cycle_generating'][i] + disp_outputs['is_cycle_starting'][i]) > 0.01 else 0
    #     assert target == pytest.approx(ssc_outputs['is_pc_su_allowed'][i], 1e-5)
    #     # receiver start-up allowed
    #     target = 1 if (disp_outputs['is_field_generating'][i] + disp_outputs['is_field_starting'][i]) > 0.01 else 0
    #     assert target == pytest.approx(ssc_outputs['is_rec_su_allowed'][i], 1e-5)
    #     # cycle thermal power
    #     start_power = system.trough.dispatch.allowable_cycle_startup_power if disp_outputs['is_cycle_starting'][i] else 0
    #     target = disp_outputs['cycle_thermal_power'][i] + start_power
    #     assert target == pytest.approx(ssc_outputs['q_dot_pc_target'][i], 1e-3)
    #     thermal energy storage state-of-charge
    #     if i % system.dispatch_builder.options.n_roll_periods == 0:
    #         tes_estimate = disp_outputs['thermal_energy_storage'][i]
    #         tes_actual = ssc_outputs['e_ch_tes'][i]
    #         assert tes_estimate == pytest.approx(tes_actual, 0.01)
    #     else:
    #         assert tes_estimate == pytest.approx(tes_actual, 0.15)


def test_tower_field_optimize_before_sim(site):
    """Testing pySSC tower model using HOPP built-in dispatch model"""
    interconnection_size_kw = 50000
    technologies = {'tower': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 6.0,
                              'optimize_field_before_sim': True},
                    'grid': 50000}

    system = {key: technologies[key] for key in ('tower', 'grid')}
    system = HybridSimulation(system, site,
                              interconnect_kw=interconnection_size_kw,
                              dispatch_options={'is_test_start_year': True})
    system.ppa_price = (0.12,)

    system.tower.value('helio_width', 10.0)
    system.tower.value('helio_height', 10.0)

    system.tower.generate_field()

    # Get old field:
    field_parameters = ['N_hel', 'D_rec', 'rec_height', 'h_tower', 'land_area_base', 'A_sf_in']
    old_values = {}
    for k in field_parameters:
        old_values[k] = system.tower.value(k)

    system.simulate()

    new_values = {}
    for k in field_parameters:
        new_values[k] = system.tower.value(k)

    assert system.tower.optimize_field_before_sim

    for k in field_parameters:
        assert old_values[k] != new_values[k]


def test_trough_annual_financial(site):
    """Testing trough annual performance and financial models with heuristic dispatch """
    trough_config = {'cycle_capacity_kw': 80 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}  

    # Expected values from SAM UI (develop) built 9/24/2021 (default parameters except those in trough_config, weather file, and ppa_soln_mode = 1)
    # Note results should be close, but won't match exactly because daotk-develop ssc branch is used for performance simulations
    expected_energy = 180014701
    expected_lcoe_nom = 19.4445
    expected_ppa_nom = 19.0373

    csp = TroughPlant(site, trough_config)
    csp.ssc.set({'time_start': 0.0, 'time_stop': 8760*3600})
    tech_outputs = csp.ssc.execute()
    csp.outputs.update_from_ssc_output(tech_outputs)
    csp.simulate_financials(100*1e3, 25)

    assert csp.annual_energy_kwh == pytest.approx(expected_energy, 1e-4)
    assert csp._financial_model.value('lcoe_nom') == pytest.approx(expected_lcoe_nom, 1e-4)
    assert csp._financial_model.value('lppa_nom') == pytest.approx(expected_ppa_nom, 1e-4)


def test_tower_annual_financial(site):
    """Testing tower annual performance and financial models with heuristic dispatch """
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 2.0,
                     'tes_hours': 8.0}  

    # Expected values from SAM UI (develop) built 9/24/2021 (default parameters except those in tower_config, weather file, field_model_type = 1, ppa_soln_mode = 1)  
    # Note results should be close, but won't match exactly because daotk-develop ssc branch is used for performance simulations
    expected_Nhel = 6172
    expected_energy = 371737920
    expected_lcoe_nom = 15.2010
    expected_ppa_nom = 15.8016

    csp = TowerPlant(site, tower_config)
    csp.generate_field()
    csp.ssc.set({'time_start': 0.0, 'time_stop': 8760*3600})
    tech_outputs = csp.ssc.execute()
    csp.outputs.update_from_ssc_output(tech_outputs)
    csp.simulate_financials(120*1e3, 25)

    assert csp.ssc.get('N_hel') == pytest.approx(expected_Nhel, 1e-3)
    assert csp.annual_energy_kwh == pytest.approx(expected_energy, 2e-3)
    assert csp._financial_model.value('lcoe_nom') == pytest.approx(expected_lcoe_nom, 2e-3)
    assert csp._financial_model.value('lppa_nom') == pytest.approx(expected_ppa_nom, 2e-3)