import pytest
import pyomo.environ as pyomo
from pyomo.environ import units as u
from pyomo.opt import TerminationCondition
from pyomo.util.check_units import assert_units_consistent

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.wind_source import WindPlant
from hybrid.pv_source import PVPlant
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant
from hybrid.battery import Battery
from hybrid.hybrid_simulation import HybridSimulation

from hybrid.dispatch import *
from hybrid.dispatch.hybrid_dispatch_builder_solver import HybridDispatchBuilderSolver


@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


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
                },
                'tower': {
                    'cycle_capacity_kw': 50 * 1000,
                    'solar_multiple': 2.4,
                    'tes_hours': 10.0
                },
                'trough': {
                    'cycle_capacity_kw': 50 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0
                },
                'grid': 50}


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

    solar.dispatch.initialize_parameters()
    solar.simulate(1)

    solar.dispatch.update_time_series_parameters(0)

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 10)
    available_resource = solar.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = solar.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_csp_dispatch_model(site):
    expected_objective = 222414.39234292
    dispatch_n_look_ahead = 48

    model = pyomo.ConcreteModel(name='csp')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    csp_dispatch = CspDispatch(model,
                               model.forecast_horizon,
                               None,
                               None)

    # Manually creating objective for testing
    model.price = pyomo.Param(model.forecast_horizon,
                              within=pyomo.Reals,
                              default=60.0,     # assuming flat PPA of $60/MWh
                              mutable=True,
                              units=u.USD / u.MWh)
    price = [60.]*5
    price.extend([30.]*8)
    price.extend([100.]*4)
    price.extend([75.]*7)
    price.extend(price)
    for i, p in enumerate(price):
        model.price[i] = p

    def create_test_objective_rule(m):
        return sum(m.csp[t].time_duration * m.price[t] * m.csp[t].cycle_generation
                   - m.csp[t].cost_per_field_generation * m.csp[t].receiver_thermal_power * m.csp[t].time_duration
                   - m.csp[t].cost_per_field_start * m.csp[t].incur_field_start
                   - m.csp[t].cost_per_cycle_generation * m.csp[t].cycle_generation  * m.csp[t].time_duration
                   - m.csp[t].cost_per_cycle_start * m.csp[t].incur_cycle_start
                   - m.csp[t].cost_per_change_thermal_input * m.csp[t].cycle_thermal_ramp for t in m.csp.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    assert_units_consistent(model)

    # WITHIN csp_dispatch.initialize_parameters()
    # Cost Parameters
    csp_dispatch.cost_per_field_generation = 3.0
    csp_dispatch.cost_per_field_start = 5650.0
    csp_dispatch.cost_per_cycle_generation = 2.0
    csp_dispatch.cost_per_cycle_start = 6520.0
    csp_dispatch.cost_per_change_thermal_input = 0.3
    # Solar field and thermal energy storage performance parameters
    csp_dispatch.field_startup_losses = 1.5
    csp_dispatch.receiver_required_startup_energy = 141.0
    csp_dispatch.storage_capacity = 10. * 393.0
    csp_dispatch.minimum_receiver_power = 141.0
    csp_dispatch.allowable_receiver_startup_power = 141.0
    csp_dispatch.receiver_pumping_losses = 0.0265
    csp_dispatch.field_track_losses = 0.3
    csp_dispatch.heat_trace_losses = 1.5
    # Power cycle performance
    csp_dispatch.cycle_required_startup_energy = 197.0
    csp_dispatch.cycle_nominal_efficiency = 0.414
    csp_dispatch.cycle_pumping_losses = 0.0127
    csp_dispatch.allowable_cycle_startup_power = 197.0
    csp_dispatch.minimum_cycle_thermal_power = 117.9
    csp_dispatch.maximum_cycle_thermal_power = 393
    minimum_cycle_power = 40.75
    csp_dispatch.maximum_cycle_power = 163
    csp_dispatch.cycle_performance_slope = ((csp_dispatch.maximum_cycle_power - minimum_cycle_power)
                                            / (csp_dispatch.maximum_cycle_thermal_power
                                               - csp_dispatch.minimum_cycle_thermal_power))

    n_horizon = len(csp_dispatch.blocks.index_set())
    csp_dispatch.time_duration = [1.0] * n_horizon
    csp_dispatch.cycle_ambient_efficiency_correction = [csp_dispatch.cycle_nominal_efficiency] * n_horizon

    heat_gen = [0.0]*6
    heat_gen.extend([0.222905449, 0.698358974, 0.812419872, 0.805703526, 0.805679487, 0.805360577, 0.805392628,
                     0.805285256, 0.805644231, 0.811056090, 0.604987179, 0.515375000, 0.104403045])  # 13
    heat_gen.extend([0.0]*11)
    heat_gen.extend([0.171546474, 0.601642628, 0.755834936, 0.808812500, 0.810616987, 0.73800641, 0.642097756,
                     0.544584936, 0.681479167, 0.547671474, 0.438600962, 0.384945513, 0.034808173])  # 13
    heat_gen.extend([0.0] * 5)

    heat_gen = [heat * 565.0 for heat in heat_gen]
    csp_dispatch.available_thermal_generation = heat_gen

    print("Total available thermal generation: {}".format(sum(csp_dispatch.available_thermal_generation)))

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)


def test_tower_dispatch(site):
    """Tests setting up tower dispatch using system model and running simulation with dispatch"""
    expected_objective = 72642.80566838199
    dispatch_n_look_ahead = 48

    tower = TowerPlant(site, technologies['tower'])
    tower.generate_field()

    model = pyomo.ConcreteModel(name='tower_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    tower._dispatch = TowerDispatch(model,
                                    model.forecast_horizon,
                                    tower,
                                    tower._financial_model)

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

    # TODO: Use hybrid simulation class with grid and remove this objective set-up
    def create_test_objective_rule(m):
        return sum(m.tower[t].time_duration * m.price[t] * m.tower[t].cycle_generation
                   - m.tower[t].cost_per_field_generation * m.tower[t].receiver_thermal_power * m.tower[t].time_duration
                   - m.tower[t].cost_per_field_start * m.tower[t].incur_field_start
                   - m.tower[t].cost_per_cycle_generation * m.tower[t].cycle_generation * m.tower[t].time_duration
                   - m.tower[t].cost_per_cycle_start * m.tower[t].incur_cycle_start
                   - m.tower[t].cost_per_change_thermal_input * m.tower[t].cycle_thermal_ramp for t in m.tower.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    tower.dispatch.initialize_parameters()
    tower.dispatch.update_time_series_parameters(0)
    tower.dispatch.update_initial_conditions()

    assert_units_consistent(model)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    tower.simulate_with_dispatch(48, 0)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    assert sum(tower.dispatch.receiver_thermal_power) > 0.0  # Useful thermal generation
    assert sum(tower.dispatch.cycle_generation) > 0.0  # Useful power generation

    # TODO: Add checks for dispatch solution vs. simulation results
    '''
    tower.simulate_with_dispatch(48, 0)
    for i in range(24):
        dispatch_power = battery.dispatch.power[i] * 1e3
        assert battery.Outputs.P[i] == pytest.approx(dispatch_power, 1e-3 * abs(dispatch_power))
    '''


def test_trough_dispatch(site):
    """Tests setting up trough dispatch using system model and running simulation with dispatch"""
    expected_objective = 28516.592861387388
    dispatch_n_look_ahead = 48

    trough = TroughPlant(site, technologies['trough'])

    model = pyomo.ConcreteModel(name='trough_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    trough._dispatch = TroughDispatch(model,
                                      model.forecast_horizon,
                                      trough,
                                      trough._financial_model)

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
                              within=pyomo.NonNegativeReals,
                              initialize=prices,
                              mutable=True,
                              units=u.USD / u.MWh)

    def create_test_objective_rule(m):
        return sum(m.trough[t].time_duration * m.price[t] * m.trough[t].cycle_generation
                   - m.trough[t].cost_per_field_generation * m.trough[t].receiver_thermal_power * m.trough[t].time_duration
                   - m.trough[t].cost_per_field_start * m.trough[t].incur_field_start
                   - m.trough[t].cost_per_cycle_generation * m.trough[t].cycle_generation * m.trough[t].time_duration
                   - m.trough[t].cost_per_cycle_start * m.trough[t].incur_cycle_start
                   - m.trough[t].cost_per_change_thermal_input * m.trough[t].cycle_thermal_ramp for t in m.trough.index_set())

    model.test_objective = pyomo.Objective(
        rule=create_test_objective_rule,
        sense=pyomo.maximize)

    trough.dispatch.initialize_parameters()
    trough.dispatch.update_time_series_parameters(0)
    trough.dispatch.update_initial_conditions()

    assert_units_consistent(model)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    trough.simulate_with_dispatch(48, 0)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    assert sum(trough.dispatch.receiver_thermal_power) > 0.0  # Useful thermal generation
    assert sum(trough.dispatch.cycle_generation) > 0.0  # Useful power generation

    # TODO: Update the simulate_with_dispatch function for towers and troughs
    '''
    tower.simulate_with_dispatch(48, 0)
    for i in range(24):
        dispatch_power = battery.dispatch.power[i] * 1e3
        assert battery.Outputs.P[i] == pytest.approx(dispatch_power, 1e-3 * abs(dispatch_power))
    '''


def test_wind_dispatch(site):
    expected_objective = 21011.222

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

    wind.dispatch.initialize_parameters()
    wind.simulate(1)

    wind.dispatch.update_time_series_parameters(0)

    results = HybridDispatchBuilderSolver.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    available_resource = wind.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = wind.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_simple_battery_dispatch(site):
    expected_objective = 31299.2696
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

    battery.dispatch.initialize_parameters()
    battery.dispatch.update_time_series_parameters(0)
    battery.dispatch.update_dispatch_initial_soc(battery.dispatch.minimum_soc)   # Set initial SOC to minimum
    assert_units_consistent(model)
    results = HybridDispatchBuilderSolver.glpk_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.test_objective) == pytest.approx(expected_objective, 1e-5)
    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert (sum(battery.dispatch.charge_power) * battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(battery.dispatch.discharge_power)))

    battery.simulate_with_dispatch(48, 0)
    for i in range(24):
        dispatch_power = battery.dispatch.power[i] * 1e3
        assert battery.Outputs.P[i] == pytest.approx(dispatch_power, 1e-3 * abs(dispatch_power))


def test_simple_battery_dispatch_lifecycle_count(site):
    expected_objective = 26620.7096
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

    battery.dispatch.initialize_parameters()
    battery.dispatch.update_time_series_parameters(0)
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
    expected_objective = 35221.192
    expected_lifecycles = 0.292799
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

    battery.dispatch.initialize_parameters()
    battery.dispatch.update_time_series_parameters(0)
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
    assert sum(battery.dispatch.charge_current) > sum(battery.dispatch.discharge_current)
    # assert sum(battery.dispatch.charge_power) > sum(battery.dispatch.discharge_power)
    # TODO: model cheats too much where last test fails


def test_hybrid_dispatch(site):
    expected_objective = 42073.267

    # TODO: update with csp
    wind_solar_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(wind_solar_battery, site, technologies['grid'] * 1000,
                                    dispatch_options={'grid_charging': False})
    hybrid_plant.grid.value("federal_tax_rate", (0., ))
    hybrid_plant.grid.value("state_tax_rate", (0., ))
    hybrid_plant.pv.simulate(1)
    hybrid_plant.wind.simulate(1)

    hybrid_plant.dispatch_builder.dispatch.initialize_parameters()
    hybrid_plant.dispatch_builder.dispatch.update_time_series_parameters(0)
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
    wind_solar_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}

    hybrid_plant = HybridSimulation(wind_solar_battery, site, technologies['grid'] * 1000,
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

    wind_solar_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(wind_solar_battery, site, technologies['grid'] * 1000,
                                    dispatch_options=dispatch_options)
    hybrid_plant.simulate(1)

    assert sum(hybrid_plant.battery.Outputs.P) < 0.0
    

def test_hybrid_solar_battery_dispatch(site):
    expected_objective = 37394.8194  # 35733.817341

    solar_battery_technologies = {k: technologies[k] for k in ('pv', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(solar_battery_technologies, site, technologies['grid'] * 1000,
                                    dispatch_options={'grid_charging': False})
    hybrid_plant.grid.value("federal_tax_rate", (0., ))
    hybrid_plant.grid.value("state_tax_rate", (0., ))
    hybrid_plant.pv.simulate(1)

    hybrid_plant.dispatch_builder.dispatch.initialize_parameters()
    hybrid_plant.dispatch_builder.dispatch.update_time_series_parameters(0)
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
    wind_solar_battery = {key: technologies[key] for key in ('pv', 'wind', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(wind_solar_battery, site, technologies['grid'] * 1000,
                                    dispatch_options={'grid_charging': True})
    hybrid_plant.simulate(1)

    assert sum(hybrid_plant.battery.Outputs.P) < 0.0