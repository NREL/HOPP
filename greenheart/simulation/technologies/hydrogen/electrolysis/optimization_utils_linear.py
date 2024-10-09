from pyomo.environ import *
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def optimize(
    P_wind_t,
    T=50,
    n_stacks=3,
    c_wp=0,
    c_sw=12,
    rated_power=500,
    dt=1,
    P_init=None,
    I_init=None,
    T_init=None,
    AC_init=0,
    F_tot_init=0,
):
    model = ConcreteModel()
    # Things to solve:
    C_INV = 1.47e6
    LT = 90000

    # Initializations
    if P_init is None:
        P_init = [0 for i in range(n_stacks * T)]
    else:
        P_init = P_init.flatten()
    if I_init is None:
        I_init = [0 for i in range(n_stacks * T)]
    else:
        I_init = I_init.flatten()
    if T_init is None:
        T_init = [0 for i in range(n_stacks * T)]
    else:
        T_init = T_init.flatten()

    model.p = Var(
        [i for i in range(n_stacks * T)],
        bounds=(-1e-2, rated_power),
        initialize=P_init,
    )
    model.I = Var(
        [i for i in range(n_stacks * T)],
        within=Binary,
        initialize=I_init,  # .astype(int),
    )
    model.T = Var(
        [i for i in range(n_stacks * T)],
        within=Binary,
        initialize=T_init,  # .astype(int),
    )
    model.AC = Var(
        [0], bounds=(1e-3, 1.2 * rated_power * n_stacks * T), initialize=float(AC_init)
    )
    model.F_tot = Var(
        [0], bounds=(1e-3, 8 * rated_power * n_stacks * T), initialize=float(F_tot_init)
    )
    model.eps = Param(initialize=1, mutable=True)

    C_WP = c_wp * np.ones(
        T,
    )  # could vary with time
    C_SW = c_sw * np.ones(
        T,
    )  # could vary with time

    P_max = rated_power
    P_min = 0.1 * rated_power

    def obj(model):
        return model.AC[0] - model.eps * model.F_tot[0]

    def physical_constraint_AC(model):
        AC = 0
        for t in range(T):
            for stack in range(n_stacks):
                AC = (
                    AC
                    + C_WP[t] * model.p[t * n_stacks + stack]
                    + C_SW[t] * model.T[t * n_stacks + stack]
                )
        return model.AC[0] == AC + C_INV * n_stacks / LT

    def physical_constraint_F_tot(model):
        """Objective function"""
        F_tot = 0
        for t in range(T):
            for stack in range(n_stacks):
                F_tot = (
                    F_tot
                    + (
                        0.0145 * model.p[t * n_stacks + stack]
                        + 0.3874 * model.I[t * n_stacks + stack] * rated_power / 500
                    )
                    * dt
                )
        return model.F_tot[0] == F_tot

    def power_constraint(model, t):
        """Make sure sum of stack powers is below available wind power."""
        power_full_stack = 0
        for stack in range(n_stacks):
            power_full_stack = power_full_stack + model.p[t * n_stacks + stack]
        return power_full_stack <= P_wind_t[t]

    def safety_bounds_lower(model, t, stack):
        """Make sure input powers don't exceed safety bounds."""
        return P_min * model.I[t * n_stacks + stack] <= model.p[t * n_stacks + stack]

    def safety_bounds_upper(model, t, stack):
        """Make sure input powers don't exceed safety bounds."""
        return P_max * model.I[t * n_stacks + stack] >= model.p[t * n_stacks + stack]

    def switching_constraint_pos(model, stack, t):
        trans = model.I[t * n_stacks + stack] - model.I[(t - 1) * n_stacks + stack]
        return model.T[t * n_stacks + stack] >= trans

    def switching_constraint_neg(model, stack, t):
        trans = model.I[t * n_stacks + stack] - model.I[(t - 1) * n_stacks + stack]
        return -model.T[t * n_stacks + stack] <= trans

    model.pwr_constraints = ConstraintList()
    model.safety_constraints = ConstraintList()
    model.switching_constraints = ConstraintList()
    model.physical_constraints = ConstraintList()

    for t in range(T):
        model.pwr_constraints.add(power_constraint(model, t))
        for stack in range(n_stacks):
            model.safety_constraints.add(safety_bounds_lower(model, t, stack))
            model.safety_constraints.add(safety_bounds_upper(model, t, stack))

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
    eps = 10
    cbc_path = os.path.abspath('')+'\\hybrid\\PEM_Model_2Push\\cbc.exe'
    solver = SolverFactory('cbc', executable=cbc_path) # Use this if you have a Windows machine; also make sure that cbc.exe is in the same folder as this script
    #solver = SolverFactory("cbc") # Use this if you don't have a windows machine
    j = 1
    while eps > 1e-3:
        start = time.process_time()
        results = solver.solve(model)
        print("time to solve", time.process_time() - start)

        model.eps = value(model.AC[0] / model.F_tot[0])  # optimal value
        eps = model.AC[0].value - model.eps.value * model.F_tot[0].value
        j = j + 1

    I = np.array([model.I[i].value for i in range(n_stacks * T)])
    I_ = np.reshape(I, (T, n_stacks))
    P = np.array([model.p[i].value for i in range(n_stacks * T)])
    P_ = np.reshape(P, (T, n_stacks))
    Tr = np.array([model.T[i].value for i in range(n_stacks * T)]).reshape(
        (T, n_stacks)
    )
    P_tot_opt = np.sum(P_, axis=1)
    H2f = np.zeros((T, n_stacks))
    for stack in range(n_stacks):
        H2f[:, stack] = (
            0.0145 * P_[:, stack] + 0.3874 * I_[:, stack] * rated_power / 500
        ) * dt
    return (
        P_tot_opt,
        P_,
        H2f,
        I_,
        Tr,
        P_wind_t,
        model.AC[0].value,
        model.F_tot[0].value,
    )
