import pyomo.environ as pyomo
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.battery import Battery
from hybrid.dispatch.power_storage.linear_voltage_nonconvex_battery_dispatch import NonConvexLinearVoltageBatteryDispatch
from hybrid.dispatch.hybrid_dispatch_builder_solver import HybridDispatch


# model = pyomo.ConcreteModel()
# model.x = pyomo.Var(bounds=(1.0,10.0),initialize=5.0)
# model.y = pyomo.Var(within=Binary)
# model.p = pyomo.Param(mutable=True)
# model.c1 = pyomo.Constraint(expr=(model.x-4.0)**2 - model.x <= model.p*(1-model.y))
# model.c2 = pyomo.Constraint(expr=model.x*log(model.x)+5.0 <= model.p*(model.y))
# model.objective = pyomo.Objective(expr=model.x, sense=minimize)
# model.p = 50.0
# results = pyomo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt')
# print(results)
#
# print(model.x.value)
# print(model.y.value)


technologies = {'solar': {
                    'system_capacity_kw': 50 * 1000,
                },
                'wind': {
                    'num_turbines': 25,
                    'turbine_rating_kw': 2000
                },
                'battery': {
                    'system_capacity_kwh': 200 * 1000,
                    'system_capacity_kw': 200 * 250
                },
                'grid': 50}

site = SiteInfo(flatirons_site)

expected_objective = 15349.798
# TODO: McCormick error is large enough to make objective twice the value of simple battery dispatch objective

dispatch_n_look_ahead = 48

battery = Battery(site, technologies['battery'])

model = pyomo.ConcreteModel(name='detailed_battery_only')
model.forecast_horizon = pyomo.Set(initialize=range(dispatch_n_look_ahead))
battery._dispatch = NonConvexLinearVoltageBatteryDispatch(model,
                                                          model.forecast_horizon,
                                                          battery._system_model,
                                                          battery._financial_model,
                                                          use_exp_voltage_point=False)
# battery.dispatch.create_gross_profit_objective()

# TODO uncomment out and update function calls to latest code
# battery.initialize_dispatch_model_parameters()
# battery.update_time_series_dispatch_model_parameters(0)
# model.initial_SOC = battery.dispatch.minimum_soc  # Set initial SOC to minimum
# assert_units_consistent(model)

# results = HybridDispatch.glpk_solve_call(model, log_name='detailed_battery.log')
# results = HybridDispatch.mindtpy_solve_call(model)