from pyomo.environ import *
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def optimize(wind_data, T=50, n_stacks=3, c_wp=0, c_sw=12, rated_power=500, dt=1):
    model = ConcreteModel()
    # Things to solve:
    C_INV = 1.47e6
    LT = 90000
    model.g_p = Var([i for i in range(n_stacks * T)], bounds=(0, 1e8))
    model.h_I = Var([i for i in range(n_stacks * T)], bounds=(0, 1e8))
    model.h_T = Var([i for i in range(n_stacks * T)], bounds=(0, 1e8))
    model.y_I = Var(
        [i for i in range(n_stacks * T)],
        initialize=[0 for i in range(n_stacks * T)],
        within=Binary,
    )
    model.y_T = Var(
        [i for i in range(n_stacks * T)],
        initialize=[0 for i in range(n_stacks * T)],
        within=Binary,
    )
    model.u = Var(
        [0],
        initialize=1e-9,
        bounds=(0, 1e8),
    )
    model.AC = Var(
        [0],
    )

    C_WP = c_wp * np.ones(
        T,
    )  # could vary with time
    C_SW = c_sw * np.ones(
        T,
    )  # could vary with time

    P_max = rated_power
    P_min = 0.1 * rated_power

    if isinstance(wind_data, pd.DataFrame):
        P_wind_t = wind_data["Wind + PV Generation"][:T]
        P_wind_t = P_wind_t / np.max(P_wind_t) * n_stacks * P_max
    else:
        P_wind_t[:T]

    def obj(model):
        return model.AC[0]

    def physical_constraint_AC(model):
        AC = 0
        for t in range(T):
            for stack in range(n_stacks):
                AC = (
                    AC
                    + C_WP[t] * model.g_p[t * n_stacks + stack]
                    + C_SW[t] * model.h_T[t * n_stacks + stack]
                )
        return model.AC[0] == AC + C_INV * n_stacks / LT * model.u[0]

    def physical_constraint_F_tot(model):
        """Objective function"""
        F_tot = 0
        for t in range(T):
            for stack in range(n_stacks):
                F_tot = (
                    F_tot
                    + (
                        0.0145 * model.g_p[t * n_stacks + stack]
                        + 0.3874 * model.h_I[t * n_stacks + stack] * rated_power / 500
                    )
                    * dt
                )
        return F_tot == 1.0

    def constraint_h_I(model, t, stack):
        return model.h_I[t * n_stacks + stack] <= model.u[0]

    def constraint_h_T(model, t, stack):
        return model.h_T[t * n_stacks + stack] <= model.u[0]

    def constraint_h_M_y_I_l(model, t, stack):
        return model.h_I[t * n_stacks + stack] <= 1e8 * model.y_I[t * n_stacks + stack]

    def constraint_h_M_y_I_u(model, t, stack):
        return (
            model.h_I[t * n_stacks + stack]
            >= model.u[0] - 1e8 * model.y_I[t * n_stacks + stack]
        )

    def constraint_h_M_y_T_l(model, t, stack):
        return model.h_T[t * n_stacks + stack] <= 1e8 * model.y_T[t * n_stacks + stack]

    def constraint_h_M_y_T_u(model, t, stack):
        return (
            model.h_T[t * n_stacks + stack]
            >= model.u[0] - 1e8 * model.y_T[t * n_stacks + stack]
        )

    def power_constraint(model, t):
        """Make sure sum of stack powers is below available wind power."""
        power_full_stack = 0
        for stack in range(n_stacks):
            power_full_stack = power_full_stack + model.g_p[t * n_stacks + stack]
        return power_full_stack <= P_wind_t[t] * model.u[0]

    def safety_bounds_lower(model, t, stack):
        """Make sure input powers don't exceed safety bounds."""
        return (
            P_min * model.h_I[t * n_stacks + stack] <= model.g_p[t * n_stacks + stack]
        )

    def safety_bounds_upper(model, t, stack):
        """Make sure input powers don't exceed safety bounds."""
        return (
            P_max * model.h_I[t * n_stacks + stack] >= model.g_p[t * n_stacks + stack]
        )

    def switching_constraint_pos(model, stack, t):
        trans = model.h_I[t * n_stacks + stack] - model.h_I[(t - 1) * n_stacks + stack]
        return model.h_T[t * n_stacks + stack] >= trans

    def switching_constraint_neg(model, stack, t):
        trans = model.h_I[t * n_stacks + stack] - model.h_I[(t - 1) * n_stacks + stack]
        return -model.h_T[t * n_stacks + stack] <= trans

    model.pwr_constraints = ConstraintList()
    model.safety_constraints = ConstraintList()
    model.switching_constraints = ConstraintList()
    model.physical_constraints = ConstraintList()
    model.h_constraints = ConstraintList()

    for t in range(T):
        model.pwr_constraints.add(power_constraint(model, t))
        for stack in range(n_stacks):
            model.safety_constraints.add(safety_bounds_lower(model, t, stack))
            model.safety_constraints.add(safety_bounds_upper(model, t, stack))
            model.h_constraints.add(constraint_h_I(model, t, stack))
            model.h_constraints.add(constraint_h_T(model, t, stack))
            model.h_constraints.add(constraint_h_M_y_I_l(model, t, stack))
            model.h_constraints.add(constraint_h_M_y_I_u(model, t, stack))
            model.h_constraints.add(constraint_h_M_y_T_l(model, t, stack))
            model.h_constraints.add(constraint_h_M_y_T_u(model, t, stack))

            if t > 0:
                model.switching_constraints.add(
                    switching_constraint_pos(model, stack, t)
                )
                model.switching_constraints.add(
                    switching_constraint_neg(model, stack, t)
                )
    model.physical_constraints.add(physical_constraint_F_tot(model))
    model.physical_constraints.add(physical_constraint_AC(model))
    model.objective = Objective(expr=obj(model), sense=minimize)
    results = SolverFactory("glpk").solve(model)

    print(results)

    I = np.array([model.y_I[i].value for i in range(n_stacks * T)])
    I_ = np.reshape(I, (T, n_stacks))
    P = np.array([model.g_p[i].value / model.u[0].value for i in range(n_stacks * T)])
    P_ = np.reshape(P, (T, n_stacks))
    Tr = np.array(
        [model.h_T[i].value / model.u[0].value for i in range(n_stacks * T)]
    ).reshape((T, n_stacks))
    Tr_ = np.sum(Tr, axis=0)
    P_tot_opt = np.sum(P_, axis=1)
    H2f = np.zeros((T, n_stacks))
    for stack in range(n_stacks):
        H2f[:, stack] = (
            0.0145 * P_[:, stack] + 0.3874 * I_[:, stack] * rated_power / 500
        ) * dt
    return P_tot_opt, P_, H2f, I_, Tr_, P_wind_t
