from pathlib import Path

import pytest
import numpy as np
import pyomo.environ as pyomo
from pyomo.environ import units as u
from pyomo.opt import TerminationCondition
from pyomo.util.check_units import assert_units_consistent

from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp.simulation.technologies.battery import Battery, BatteryConfig, BatteryStateless, BatteryStatelessConfig
from hopp.simulation.technologies.dispatch import SimpleBatteryDispatch
from hopp.simulation.technologies.dispatch.hybrid_dispatch_builder_solver import HybridDispatchBuilderSolver, HybridDispatchOptions
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp import ROOT_DIR
from tests.hopp.utils import DEFAULT_FIN_CONFIG

solar_resource_file = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
site = SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

interconnect_mw = 50
technologies_input = {
    'pv': {
        'system_capacity_kw': 50 * 1000,
    },
    'battery': {
        'system_capacity_kwh': 200 * 1000,
        'system_capacity_kw': 50 * 1000,
        'tracking': False,
        'fin_model': CustomFinancialModel(DEFAULT_FIN_CONFIG, name="Test")
    },
    'grid': {
        'interconnect_kw': interconnect_mw * 1000
    }}

# Manually creating objective for testing
dispatch_n_look_ahead = 48
prices = {}
block_length = 8
index = 0
for i in range(int(dispatch_n_look_ahead / block_length)):
    for j in range(block_length):
        if i % 2 == 0:
            prices[index] = 30.0  # assuming low prices
        else:
            prices[index] = 100.0  # assuming high prices
        index += 1


def create_test_objective_rule(m):
    return sum((m.battery[t].time_duration * (
            (m.price[t] - m.battery[t].cost_per_discharge) * m.battery[t].discharge_power
            - (m.price[t] + m.battery[t].cost_per_charge) * m.battery[t].charge_power))
                for t in m.battery.index_set())


def test_batterystateless_dispatch(subtests):
    expected_objective = 28957.15

    # Run battery stateful as system model first
    technologies = technologies_input.copy()
    technologies['battery']['tracking'] = True
    model = pyomo.ConcreteModel(name='battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              initialize=prices,
                              mutable=True,
                              units=u.USD / u.MWh)
    
    config = BatteryConfig.from_dict(technologies['battery'])
    battery = Battery(site, config=config)
    battery._dispatch = SimpleBatteryDispatch(model,
                                              model.forecast_horizon,
                                              battery._system_model,
                                              battery._financial_model,
                                              'battery',
                                              HybridDispatchOptions())
    
    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery.dispatch.initialize_parameters()
    battery.dispatch.update_time_series_parameters(0)
    battery.dispatch.update_dispatch_initial_soc(battery.dispatch.minimum_soc)   # Set initial SOC to minimum
    assert_units_consistent(model)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    with subtests.test("TerminationCondition"):
        assert results.solver.termination_condition == TerminationCondition.optimal
    
    with subtests.test("expected_objective"):
        assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)

    sum_charge_power = sum(battery.dispatch.charge_power)
    sum_discharge_power = sum(battery.dispatch.discharge_power)

    with subtests.test("sum_charge_power"):
        assert sum(battery.dispatch.charge_power) == pytest.approx(sum_charge_power, 1e-2)

    with subtests.test("sum_discharge_power"):
        assert sum(battery.dispatch.discharge_power) == pytest.approx(sum_discharge_power, 1e-2)

    battery.simulate_with_dispatch(48, 0)
    for i in range(45):
        with subtests.test(f"battery.dispatch.power[{i}]"):
            dispatch_power = battery.dispatch.power[i] * 1e3
            assert battery.outputs.P[i] == pytest.approx(dispatch_power,  1e-3 * abs(dispatch_power))

    with subtests.test("sum_charge_power"):
        assert battery.outputs.dispatch_lifecycles_per_day[0:2] == pytest.approx([0.75048, 1.50096], rel=1e-3)

    with subtests.test("sum_charge_power"):
        assert battery.outputs.n_cycles[23] == 0

    with subtests.test("sum_charge_power"):
        assert battery.outputs.n_cycles[47] == 1

    # Run battery stateless as system model to compare
    technologies['battery']['tracking'] = False
    model_sl = pyomo.ConcreteModel(name='battery_stateless')
    model_sl.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    model_sl.price = pyomo.Param(model_sl.forecast_horizon,
                                 within=pyomo.Reals,
                                 initialize=prices,
                                 mutable=True,
                                 units=u.USD / u.MWh)
    
    config = BatteryStatelessConfig.from_dict(technologies['battery'])
    battery_sl = BatteryStateless(site, config=config)
    battery_sl._dispatch = SimpleBatteryDispatch(model_sl,
                                                 model_sl.forecast_horizon,
                                                 battery_sl._system_model,
                                                 battery_sl._financial_model,
                                                 'battery',
                                                 HybridDispatchOptions())
    
    model_sl.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery_sl.dispatch.initialize_parameters()
    battery_sl.dispatch.update_time_series_parameters(0)
    assert_units_consistent(model_sl)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model_sl)

    with subtests.test("sum_charge_power"):
        assert results.solver.termination_condition == TerminationCondition.optimal
    
    with subtests.test("sum_charge_power"):
        assert pyomo.value(model_sl.test_objective) == pytest.approx(expected_objective, 1e-5)

    with subtests.test("sum_charge_power"):
        assert sum(battery_sl.dispatch.charge_power) == pytest.approx(sum_charge_power, 1e-2)
    
    with subtests.test("sum_charge_power"):
        assert sum(battery_sl.dispatch.discharge_power) == pytest.approx(sum_discharge_power, 1e-2)

    battery_sl.simulate_with_dispatch(48, 0)
    for i in range(24):
        with subtests.test(f"battery_sl.dispatch.power[{i}]"):
            dispatch_power = battery_sl.dispatch.power[i] * 1e3
            assert battery_sl.outputs.P[i] == pytest.approx(dispatch_power, 1e-3 * abs(dispatch_power))

    battery_dispatch = np.array(battery.dispatch.power)[0:48]
    battery_actual = np.array(battery.generation_profile[0:dispatch_n_look_ahead]) * 1e-3   # convert to MWh
    battery_sl_dispatch = np.array(battery_sl.dispatch.power)[0:48]
    battery_sl_actual = np.array(battery_sl.generation_profile)[0:48] * 1e-3   # convert to MWh

    with subtests.test("battery_dispatch vs battery_sl_dispatch"):
        assert sum(battery_dispatch - battery_sl_dispatch) == 0
    
    with subtests.test("battery_actual vs battery_dispatch"):
        assert sum(abs(battery_actual - battery_dispatch)) <= 33
    
    with subtests.test("battery_sl_actual vs battery_sl_dispatch"):
        assert sum(abs(battery_sl_actual - battery_sl_dispatch)) == 0
    
    with subtests.test("battery_actual vs battery_sl_actual"):
        assert sum(abs(battery_actual - battery_sl_actual)) <= 33
    
    with subtests.test("lifecycles_per_day"):
        assert battery_sl.outputs.lifecycles_per_day[0:2] == pytest.approx([0.75048, 1.50096], rel=1e-3)


def test_batterystateless_cycle_limits(subtests):
    expected_objective = 22513      # objective is less than above due to cycling limits
    
    technologies = technologies_input.copy()
    technologies['battery']['tracking'] = False
    model_sl = pyomo.ConcreteModel(name='battery_stateless')
    model_sl.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    model_sl.price = pyomo.Param(model_sl.forecast_horizon,
                                 within=pyomo.Reals,
                                 initialize=prices,
                                 mutable=True,
                                 units=u.USD / u.MWh)
    
    config = BatteryConfig.from_dict(technologies['battery'])
    battery_sl = BatteryStateless(site, config=config)
    battery_sl._dispatch = SimpleBatteryDispatch(model_sl,
                                                 model_sl.forecast_horizon,
                                                 battery_sl._system_model,
                                                 battery_sl._financial_model,
                                                 'battery',
                                                 HybridDispatchOptions({'max_lifecycle_per_day': 1}))
    
    model_sl.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery_sl.dispatch.initialize_parameters()
    battery_sl.dispatch.update_time_series_parameters(0)
    assert_units_consistent(model_sl)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model_sl)

    
    with subtests.test("termination_condition"):
        assert results.solver.termination_condition == TerminationCondition.optimal
    
    with subtests.test("test_objective"):
        assert pyomo.value(model_sl.test_objective) == pytest.approx(expected_objective, 1e-3)

    battery_sl.simulate_with_dispatch(48, 0)

    with subtests.test("lifecycles_per_day"):
        assert battery_sl.outputs.lifecycles_per_day[0:2] == pytest.approx([0.75048, 1], rel=1e-3)

