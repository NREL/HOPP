import pytest
import pyomo.environ as pyomo
from pyomo.opt import TerminationCondition
from pyomo.util.check_units import assert_units_consistent


from hybrid.sites import SiteInfo, flatirons_site
from hybrid.wind_source import WindPlant
from hybrid.solar_source import SolarPlant
from hybrid.storage import Battery
from hybrid.hybrid_simulation import HybridSimulation

from hybrid.dispatch.solar_dispatch import SolarDispatch
from hybrid.dispatch.wind_dispatch import WindDispatch
from hybrid.dispatch.battery_dispatch import BatteryDispatch
from hybrid.dispatch.hybrid_dispatch import HybridDispatch


@pytest.fixture
def site():
    return SiteInfo(flatirons_site)

technologies = {'solar': {
                    'system_capacity_kw': 50 * 1000,
                },
                'wind': {
                    'num_turbines': 25,
                    'turbine_rating_kw': 2000
                },
                'battery': 200 * 1000,
                'grid': 50}


def test_solar_dispatch(site):
    expected_objective = 51645.37566

    dispatch_n_look_ahead = 48

    solar = SolarPlant(site, technologies['solar'])

    model = pyomo.ConcreteModel(name='solar_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    solar._dispatch = SolarDispatch(model, model.forecast_horizon)
    solar.dispatch.create_gross_profit_objective()

    assert_units_consistent(model)

    solar.initialize_dispatch_model_parameters()
    solar.simulate()

    solar.update_time_series_dispatch_model_parameters(0)

    results = HybridDispatch.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    available_resource = solar.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = solar.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_wind_dispatch(site):
    expected_objective = 11043.84250

    dispatch_n_look_ahead = 48

    wind = WindPlant(site, technologies['wind'])

    model = pyomo.ConcreteModel(name='wind_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))

    wind._dispatch = WindDispatch(model, model.forecast_horizon)
    wind.dispatch.create_gross_profit_objective()

    assert_units_consistent(model)

    wind.initialize_dispatch_model_parameters()
    wind.simulate()

    wind.update_time_series_dispatch_model_parameters(0)

    results = HybridDispatch.glpk_solve_call(model)
    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(model.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    available_resource = wind.generation_profile[0:dispatch_n_look_ahead]
    dispatch_generation = wind.dispatch.generation
    for t in model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)


def test_battery_dispatch(site):
    expected_objective = 7852.67513

    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    battery._dispatch = BatteryDispatch(model, model.forecast_horizon)
    battery.dispatch.create_gross_profit_objective()

    battery.initialize_dispatch_model_parameters()
    battery.update_time_series_dispatch_model_parameters(0)
    model.initial_SOC = battery.dispatch.minimum_soc   # Set initial SOC to minimum
    assert_units_consistent(model)

    results = HybridDispatch.glpk_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert (sum(battery.dispatch.charge_power) * battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(battery.dispatch.discharge_power)))


def test_battery_dispatch_lifecycle_count(site):
    expected_lifecycles = 1.5178928
    expected_objective = 4816.889539

    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    battery._dispatch = BatteryDispatch(model, model.forecast_horizon, include_lifecycle_cost=True)
    battery.dispatch.create_gross_profit_objective()

    battery.initialize_dispatch_model_parameters()
    battery.update_time_series_dispatch_model_parameters(0)
    model.initial_SOC = battery.dispatch.minimum_soc   # Set initial SOC to minimum
    assert_units_consistent(model)

    results = HybridDispatch.glpk_solve_call(model)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    assert pyomo.value(battery.dispatch.lifecycles) == pytest.approx(expected_lifecycles, 1e-3)

    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert (sum(battery.dispatch.charge_power) * battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(battery.dispatch.discharge_power)))


def test_detailed_battery_dispatch(site):
    expected_objective = 15349.798
    # TODO: McCormick error is large enough to make objective twice the value of simple battery dispatch objective

    dispatch_n_look_ahead = 48

    battery = Battery(site, technologies['battery'])

    model = pyomo.ConcreteModel(name='detailed_battery_only')
    model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
    battery._dispatch = BatteryDispatch(model, model.forecast_horizon,
                                        use_simple_battery_dispatch=False,
                                        use_exp_voltage_point=False,
                                        use_nonlinear_formulation=True)
    battery.dispatch.create_gross_profit_objective()

    battery.initialize_dispatch_model_parameters()
    battery.update_time_series_dispatch_model_parameters(0)
    model.initial_SOC = battery.dispatch.minimum_soc   # Set initial SOC to minimum
    assert_units_consistent(model)

    # results = HybridDispatch.glpk_solve_call(model, log_name='detailed_battery.log')
    results = HybridDispatch.mindtpy_solve_call(model)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.plot([0, max(battery.dispatch.real_discharge_current_soc)],
    # #          [0, max(battery.dispatch.real_discharge_current_soc)],
    # #          'r--')
    # # plt.scatter(battery.dispatch.real_charge_current_soc, battery.dispatch.aux_charge_current_soc)
    # # plt.scatter(battery.dispatch.real_discharge_current_soc, battery.dispatch.aux_discharge_current_soc)
    # time = range(dispatch_n_look_ahead)
    # plt.plot(time, battery.dispatch.power)
    # plt.plot(time, [current * 1e3 for current in battery.dispatch.current])
    # plt.plot(time, [(real - aux) * 1e3 for real, aux in zip(battery.dispatch.real_charge_current_soc, battery.dispatch.aux_charge_current_soc)], 'r')
    # plt.plot(time, [(real - aux) * 1e3 for real, aux in zip(battery.dispatch.real_discharge_current_soc, battery.dispatch.aux_discharge_current_soc)], 'k')
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(battery.dispatch.soc, [(real - aux) for real, aux in zip(battery.dispatch.real_charge_current_soc, battery.dispatch.aux_charge_current_soc)], c='r')
    # plt.scatter(battery.dispatch.soc, [(real - aux) for real, aux in zip(battery.dispatch.real_discharge_current_soc, battery.dispatch.aux_discharge_current_soc)], c='k')
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(battery.dispatch.charge_current, [(real - aux) for real, aux in zip(battery.dispatch.real_charge_current_soc, battery.dispatch.aux_charge_current_soc)], c='r')
    # plt.scatter(battery.dispatch.discharge_current, [(real - aux) for real, aux in zip(battery.dispatch.real_discharge_current_soc, battery.dispatch.aux_discharge_current_soc)], c='k')
    # plt.show()

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert pyomo.value(model.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    assert sum(battery.dispatch.charge_power) > 0.0
    assert sum(battery.dispatch.discharge_power) > 0.0
    assert sum(battery.dispatch.charge_power) > sum(battery.dispatch.discharge_power)  # TODO: model cheats too much


def test_hybrid_dispatch(site):
    expected_objective = 66280.413787

    hybrid_plant = HybridSimulation(technologies, site, technologies['grid'] * 1000)

    hybrid_plant.solar.simulate()
    hybrid_plant.wind.simulate()

    hybrid_plant.dispatch.update_time_series_dispatch_model_parameters(0)
    hybrid_plant.battery.dispatch.initial_SOC = hybrid_plant.battery.dispatch.minimum_soc   # Set to min SOC

    results = HybridDispatch.glpk_solve_call(hybrid_plant.dispatch._pyomo_model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(hybrid_plant.dispatch.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    available_resource = hybrid_plant.solar.generation_profile[0:hybrid_plant.dispatch.options.n_look_ahead_periods]
    dispatch_generation = hybrid_plant.solar.dispatch.generation
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    available_resource = hybrid_plant.wind.generation_profile[0:hybrid_plant.dispatch.options.n_look_ahead_periods]
    dispatch_generation = hybrid_plant.wind.dispatch.generation
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    assert sum(hybrid_plant.battery.dispatch.charge_power) > 0.0
    assert sum(hybrid_plant.battery.dispatch.discharge_power) > 0.0
    assert (sum(hybrid_plant.battery.dispatch.charge_power)
            * hybrid_plant.battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(hybrid_plant.battery.dispatch.discharge_power)))

    transmission_limit = hybrid_plant.grid.get_variable('grid_interconnection_limit_kwac')
    system_generation = hybrid_plant.grid.dispatch.electricity_sold
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        assert system_generation[t] * 1e3 <= transmission_limit
        assert system_generation[t] * 1e3 >= 0.0


def test_hybrid_solar_battery_dispatch(site):
    expected_objective = 35733.817341

    solar_battery_technologies = {k: technologies[k] for k in ('solar', 'battery', 'grid')}
    hybrid_plant = HybridSimulation(solar_battery_technologies, site, technologies['grid'] * 1000)

    hybrid_plant.solar.simulate()

    hybrid_plant.dispatch.update_time_series_dispatch_model_parameters(0)
    hybrid_plant.battery.dispatch.initial_SOC = hybrid_plant.battery.dispatch.minimum_soc   # Set to min SOC

    # This was done because the default peak prices coincide with solar production...
    available_resource = hybrid_plant.solar.generation_profile[0:hybrid_plant.dispatch.options.n_look_ahead_periods]
    prices = [0.] * len(available_resource)
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        if available_resource[t] > 0.0:
            prices[t] = 30.0
        else:
            prices[t] = 110.0
    hybrid_plant.dispatch.update_electricity_prices(prices)

    results = HybridDispatch.glpk_solve_call(hybrid_plant.dispatch._pyomo_model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    assert pyomo.value(hybrid_plant.dispatch.gross_profit_objective) == pytest.approx(expected_objective, 1e-3)
    available_resource = hybrid_plant.solar.generation_profile[0:hybrid_plant.dispatch.options.n_look_ahead_periods]
    dispatch_generation = hybrid_plant.solar.dispatch.generation
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        assert dispatch_generation[t] * 1e3 == pytest.approx(available_resource[t], 1e-3)

    assert sum(hybrid_plant.battery.dispatch.charge_power) > 0.0
    assert sum(hybrid_plant.battery.dispatch.discharge_power) > 0.0
    assert (sum(hybrid_plant.battery.dispatch.charge_power)
            * hybrid_plant.battery.dispatch.round_trip_efficiency / 100.0
            == pytest.approx(sum(hybrid_plant.battery.dispatch.discharge_power)))

    transmission_limit = hybrid_plant.grid.get_variable('grid_interconnection_limit_kwac')
    system_generation = hybrid_plant.grid.dispatch.electricity_sold
    for t in hybrid_plant.dispatch._pyomo_model.forecast_horizon:
        assert system_generation[t] * 1e3 <= transmission_limit
        assert system_generation[t] * 1e3 >= 0.0

