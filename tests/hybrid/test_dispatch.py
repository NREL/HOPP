import pytest
from pathlib import Path
import pyomo.environ as pyomo
from pyomo.environ import units as u
from pyomo.opt import TerminationCondition
from pyomo.util.check_units import assert_units_consistent

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.wind_source import WindPlant
from hybrid.pv_source import PVPlant
from hybrid.battery import Battery
from hybrid.hybrid_simulation import HybridSimulation

from hybrid.dispatch import *
from hybrid.dispatch.hybrid_dispatch_builder_solver import HybridDispatchBuilderSolver


@pytest.fixture
def site():
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)



technologies = {'pv': {
                    'system_capacity_kw': 50 * 1000,
                },
                'wind': {
                    'num_turbines': 25,
                    'turbine_rating_kw': 2000
                },
                'battery': {
                    'system_capacity_kwh': 200 * 1000,
                    'system_capacity_kw': 50 * 1000
                }}
interconnect_mw = 50


def test_solar_dispatch(site):
    expected_objective = 27748

    dispatch_n_look_ahead = 48

    solar = PVPlant(site, technologies['pv'])

    model = pyomo.ConcreteModel(name='solar_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    solar._dispatch = PvDispatch(model,
                                 model.forecast_horizon,
                                 solar._system_model,
                                 solar._financial_model)

    # Manually creating objective for testing
    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              default=60.0,     # assuming flat PPA of $60/MWh
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return sum((m.pv[i].time_duration * m.price[i] * m.pv[i].generation
                    - m.pv[i].generation_cost) for i in m.pv.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    assert_units_consistent(model)

    solar.dispatch.initialize_dispatch_model_parameters()
    solar.simulate(1)

    solar.dispatch.update_time_series_dispatch_model_parameters(0)

    print("Total available generation: {}".format(sum(solar.dispatch.available_generation)))

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 10)
    available_resource = solar.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = solar.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_wind_dispatch(site):
    expected_objective = 20719.281

    dispatch_n_look_ahead = 48

    wind = WindPlant(site, technologies['wind'])

    model = pyomo.ConcreteModel(name='wind_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    wind._dispatch = WindDispatch(model,
                                  model.forecast_horizon,
                                  wind._system_model,
                                  wind._financial_model)

    # Manually creating objective for testing
    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              default=60.0,     # assuming flat PPA of $60/MWh
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return sum((m.wind[t].time_duration * m.price[t] * m.wind[t].generation
                    - m.wind[t].generation_cost) for t in m.wind.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    assert_units_consistent(model)

    wind.dispatch.initialize_dispatch_model_parameters()
    wind.simulate(1)

    wind.dispatch.update_time_series_dispatch_model_parameters(0)

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    available_resource = wind.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = wind.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_simple_battery_dispatch(site):
    expected_objective = 32011.8234
    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    battery._dispatch = SimpleBatteryDispatch(model,
                                              model.forecast_horizon,
                                              battery._system_model,
                                              battery._financial_model,
                                              include_lifecycle_count=False)

    # Manually creating objective for testing
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

    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              initialize=prices,
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return sum((m.battery[t].time_duration * m.price[t] * (m.battery[t].discharge_power - m.battery[t].charge_power)
                    - m.battery[t].discharge_cost - m.battery[t].charge_cost) for t in m.battery.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery.dispatch.initialize_dispatch_model_parameters()
    battery.dispatch.update_time_series_dispatch_model_parameters(0)
    battery.dispatch.update_dispatch_initial_soc(battery.dispatch.minimum_soc)   # Set initial SOC to minimum
    assert_units_consistent(model)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert (sum(battery.dispatch.charge_power) * battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(battery.dispatch.discharge_power)))

    battery._simulate_with_dispatch(48, 0)
    for i in range(24):
        dispatch_power = battery.dispatch.power[i] * 1e3
        assert battery.Outputs.P[i] == pytest.approx(dispatch_power, 1e-3 * abs(dispatch_power))


def test_simple_battery_dispatch_lifecycle_count(site):
    expected_objective = 27333.2634
    expected_lifecycles = 2.339

    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    battery._dispatch = SimpleBatteryDispatch(model,
                                              model.forecast_horizon,
                                              battery._system_model,
                                              battery._financial_model,
                                              include_lifecycle_count=True)

    # Manually creating objective for testing
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

    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              initialize=prices,
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return (sum((m.battery[t].time_duration
                     * m.price[t]
                     * (m.battery[t].discharge_power - m.battery[t].charge_power)
                     - m.battery[t].discharge_cost
                     - m.battery[t].charge_cost) for t in m.battery.index_set())
                - m.lifecycle_cost * m.lifecycles)

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery.dispatch.initialize_dispatch_model_parameters()
    battery.dispatch.update_time_series_dispatch_model_parameters(0)
    model.initial_SOC = battery.dispatch.minimum_soc   # Set initial SOC to minimum
    assert_units_consistent(model)

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    assert pyomo.value(battery.dispatch.lifecycles) == pytest.approx(expected_lifecycles, 1e-3)

    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert (sum(battery.dispatch.charge_power) * battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(battery.dispatch.discharge_power)))


def test_detailed_battery_dispatch(site):
    expected_objective = 39076.571
    expected_lifecycles = 0.36470
    # TODO: McCormick error is large enough to make objective 50% higher than
    #  the value of simple battery dispatch objective

    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='detailed_battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    battery._dispatch = ConvexLinearVoltageBatteryDispatch(model,
                                                           model.forecast_horizon,
                                                           battery._system_model,
                                                           battery._financial_model)

    # Manually creating objective for testing
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

    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              initialize=prices,
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return (sum((m.convex_LV_battery[t].time_duration
                     * m.price[t]
                     * (m.convex_LV_battery[t].discharge_power - m.convex_LV_battery[t].charge_power)
                     - m.convex_LV_battery[t].discharge_cost
                     - m.convex_LV_battery[t].charge_cost) for t in m.convex_LV_battery.index_set())
                - m.lifecycle_cost * m.lifecycles)

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    battery.dispatch.initialize_dispatch_model_parameters()
    battery.dispatch.update_time_series_dispatch_model_parameters(0)
    model.initial_SOC = battery.dispatch.minimum_soc   # Set initial SOC to minimum
    assert_units_consistent(model)

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    # TODO: trying to solve the nonlinear problem but solver doesn't work...
    #           Need to try another nonlinear solver
    # results = HybridDispatchBuilderSolver.mindtpy_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-3)
    assert pyomo.value(battery.dispatch.lifecycles) == pytest.approx(expected_lifecycles, 1e-3)
    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert sum(battery.dispatch.charge_current) >= sum(battery.dispatch.discharge_current) - 1e-7
    # assert sum(battery.dispatch.charge_power) > sum(battery.dispatch.discharge_power)
    # TODO: model cheats too much where last test fails


def test_hybrid_dispatch(site):
    expected_objective = 194599.572

    hybrid_plant = HybridSimulation(technologies, site, interconnect_mw * 1000,
                                    dispatch_options={'grid_charging': False})
    hybrid_plant.grid.value("federal_tax_rate", (0., ))
    hybrid_plant.grid.value("state_tax_rate", (0., ))
    hybrid_plant.pv.simulate(1)
    hybrid_plant.wind.simulate(1)

    hybrid_plant.dispatch_builder.dispatch.initialize_dispatch_model_parameters()
    hybrid_plant.dispatch_builder.dispatch.update_time_series_dispatch_model_parameters(0)
    hybrid_plant.battery.dispatch.initial_SOC = hybrid_plant.battery.dispatch.minimum_soc   # Set to min SOC

    results = HybridDispatchBuilderSolver.glpk_solve_call(hybrid_plant.dispatch_builder.pyomo_model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    gross_profit_objective = pyomo.value(hybrid_plant.dispatch_builder.dispatch.objective_value)
    assert gross_profit_objective == pytest.approx(expected_objective, 1e-3)
    n_look_ahead_periods = hybrid_plant.dispatch_builder.options.n_look_ahead_periods
    available_resource = hybrid_plant.pv.generation_profile[0:n_look_ahead_periods]
    dispatch_generation = hybrid_plant.pv.dispatch.generation
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    available_resource = hybrid_plant.wind.generation_profile[0:n_look_ahead_periods]
    dispatch_generation = hybrid_plant.wind.dispatch.generation
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    assert sum(hybrid_plant.battery.dispatch.charge_power) > 0.0
    assert sum(hybrid_plant.battery.dispatch.discharge_power) > 0.0
    assert (sum(hybrid_plant.battery.dispatch.charge_power)
            * hybrid_plant.battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(hybrid_plant.battery.dispatch.discharge_power)))

    transmission_limit = hybrid_plant.grid.value('grid_interconnection_limit_kwac')
    system_generation = hybrid_plant.grid.dispatch.system_generation
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        assert system_generation[t] * 1e3 <= transmission_limit
        assert system_generation[t] * 1e3 >= 0.0


def test_hybrid_dispatch_heuristic(site):
    dispatch_options = {'battery_dispatch': 'heuristic',
                        'grid_charging': False}
    hybrid_plant = HybridSimulation(technologies, site, interconnect_mw * 1000,
                                    dispatch_options=dispatch_options)
    fixed_dispatch = [0.0]*6
    fixed_dispatch.extend([-1.0]*6)
    fixed_dispatch.extend([1.0]*6)
    fixed_dispatch.extend([0.0]*6)

    hybrid_plant.battery.dispatch.user_fixed_dispatch = fixed_dispatch

    hybrid_plant.simulate(1)

    assert sum(hybrid_plant.battery.dispatch.charge_power) > 0.0
    assert sum(hybrid_plant.battery.dispatch.discharge_power) > 0.0


def test_hybrid_dispatch_one_cycle_heuristic(site):
    dispatch_options = {'battery_dispatch': 'one_cycle_heuristic',
                        'grid_charging': False}

    hybrid_plant = HybridSimulation(technologies, site, interconnect_mw * 1000,
                                    dispatch_options=dispatch_options)
    hybrid_plant.simulate(1)

    assert sum(hybrid_plant.battery.Outputs.P) < 0.0
    

def test_hybrid_solar_battery_dispatch(site):
    expected_objective = 36057.573

    solar_battery_technologies = {k: technologies[k] for k in ('pv', 'battery')}
    hybrid_plant = HybridSimulation(solar_battery_technologies, site, interconnect_mw * 1000,
                                    dispatch_options={'grid_charging': False})
    hybrid_plant.grid.value("federal_tax_rate", (0., ))
    hybrid_plant.grid.value("state_tax_rate", (0., ))
    hybrid_plant.pv.simulate(1)

    hybrid_plant.dispatch_builder.dispatch.initialize_dispatch_model_parameters()
    hybrid_plant.dispatch_builder.dispatch.update_time_series_dispatch_model_parameters(0)
    hybrid_plant.battery.dispatch.initial_SOC = hybrid_plant.battery.dispatch.minimum_soc   # Set to min SOC

    n_look_ahead_periods = hybrid_plant.dispatch_builder.options.n_look_ahead_periods
    # This was done because the default peak prices coincide with solar production...
    available_resource = hybrid_plant.pv.generation_profile[0:n_look_ahead_periods]
    prices = [0.] * len(available_resource)
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        if available_resource[t] > 0.0:
            prices[t] = 30.0
        else:
            prices[t] = 110.0
    hybrid_plant.grid.dispatch.electricity_sell_price = prices
    hybrid_plant.grid.dispatch.electricity_purchase_price = prices

    results = HybridDispatchBuilderSolver.glpk_solve_call(hybrid_plant.dispatch_builder.pyomo_model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    gross_profit_objective = pyomo.value(hybrid_plant.dispatch_builder.dispatch.objective_value)
    assert gross_profit_objective == pytest.approx(expected_objective, 1e-3)

    available_resource = hybrid_plant.pv.generation_profile[0:n_look_ahead_periods]
    dispatch_generation = hybrid_plant.pv.dispatch.generation
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    assert sum(hybrid_plant.battery.dispatch.charge_power) > 0.0
    assert sum(hybrid_plant.battery.dispatch.discharge_power) > 0.0
    assert (sum(hybrid_plant.battery.dispatch.charge_power)
            * hybrid_plant.battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(hybrid_plant.battery.dispatch.discharge_power)))

    transmission_limit = hybrid_plant.grid.value('grid_interconnection_limit_kwac')
    system_generation = hybrid_plant.grid.dispatch.system_generation
    for t in hybrid_plant.dispatch_builder.pyomo_model.forecast_horizon:
        assert system_generation[t] * 1e3 <= transmission_limit
        assert system_generation[t] * 1e3 >= 0.0


def test_hybrid_dispatch_financials(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_mw * 1000,
                                    dispatch_options={'grid_charging': True})
    hybrid_plant.simulate(1)

    assert sum(hybrid_plant.battery.Outputs.P) < 0.0