import copy
import numpy as np
import random
import pandas as pd
# import pandapower as pp
# import pandapower.converter as pc
import matplotlib.pyplot as plt
import pypower as pp
from pypower.api import loadcase
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly

# Python Adaptation of VoltCont_Alg_Paper.m (Guido Cavraro) by Aaron Barker
# This file is voltage control without integrating the duals.
from define_powerflow_constants import *
# mpc = pc.from_mpc('case6.mat', f_hz=50, validate_conversion=True)
mpc = loadcase('case6.mat')
num_bus = 4
L = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[1,0,0,0,0]])
Vmin = 0.94
VMAX = 1.02
# generators and loads
gen = [0, 4]
num_gen = 1
loads = [1,2,3]
num_loads = 3
# control limits
x_max = ([[VMAX],[10],[10],[10],[VMAX]])
x_min = ([[0.97],[- 20],[- 20],[- 20],[0.97]])
# fixing rand seed
random.seed(128)
# augmenting reactive power demand to have a voltage violation
#*Differs from Matlab/Matpower Code. Loads are in mpc.load, not mpc.bus
# Cannot match random seed between matlab and python, so replicating initial load values instead.
# We can still perturb the system using the line above ^
# mpc['bus'][:,3] * 1.6 * [random.uniform(0,1) for x in range(0,len(mpc['bus'][:,3]))]
mpc['bus'][:,3] = [0,
140.979958318168,
127.967453454248,
139.101693580060, 0]

# # Alg parameters
T = 1000
lamb_M = np.zeros((5,T))
volt = np.zeros((5, T))
x = np.zeros((5, T))
x[0][0] = 1
x[4][0] = 1
xtil = copy.deepcopy(x)

# setting alg parameters
eta1 = 2 * np.array([[1],[1],[1],[10],[2]])
eta2 = 0.1 * np.array([[1],[10],[10],[10],[10]])
eta3 = 0.1 * np.array([[10],[1],[1],[1],[1]])
# vectors of function values
F_t = np.zeros((5, T))

# running the first PF to have initial conditions
mpc = pp.api.runpf(mpc)
mpc= mpc[0]
mpc['gen'] = mpc['gen'].astype(np.float64)
volt[:, 0] = mpc['bus'][:,7]

#Run for timesteps
empty_mat_5x5 = np.copy(np.zeros((5,5)))
eta1_diag = np.copy(empty_mat_5x5)
eta2_diag = np.copy(empty_mat_5x5)
eta3_diag = np.copy(empty_mat_5x5)
np.fill_diagonal(eta1_diag, eta1)
eta2_diag = eta2_diag
np.fill_diagonal(eta2_diag, eta2)
eta3_diag = eta3_diag
np.fill_diagonal(eta3_diag, eta3)
F_t = np.array(np.zeros((5,T)))

for tt in range(1, T-1):
    # value of f and g
    f_t_step = np.zeros((5,1))
    for load in loads:
        print("Voltage for load {} is {}".format((load),volt[(load), (tt-1)]))
        if volt[(load),(tt-1)] < Vmin:
            f_t_step[(load)] = Vmin - volt[(load), (tt-1)]
            print(f_t_step)
    # F_t = np.append(F_t, f_t_step.tolist(),1)
    F_t[:,[tt]] = f_t_step
    test_1 = x[:, tt-1]
    test_2 = eta1_diag
    test_3 = f_t_step
    test_4 = eta2_diag
    test_5 = L
    test_6 = lamb_M[:, (tt-1)]

    # Computation of control inputs
    # xtil[:,tt-1] = x[:,tt-2] + eta1_diag * (f_t_step + eta2_diag * L * lamb_M[:,(tt-2)])
    eq_pt1 = np.dot(L, lamb_M[:, [(tt-1)]]) #The bracket index syntax here is really important to recast ndarray vector to 2D
    eq_pt2 = np.dot(eta2_diag, eq_pt1)
    eq_pt3 = np.add(f_t_step, eq_pt2)
    eq_pt4 = np.dot(eta1_diag, eq_pt3)
    eq_end = np.add(x[:, [tt-1]], eq_pt4) #The bracket index syntax here is really important to recast ndarray vector to 2D
    xtil[:, [tt]] = eq_end

    # Lagrange Multipliers-like update
    test_b_1 = eta3_diag
    test_b_2 = xtil[:, [tt]] #The bracket index syntax here is really important to recast ndarray vector to 2D
    test_b_3 = x_max
    test_b_4 = xtil[:, [tt]] - x_max #The bracket index syntax here is really important to recast ndarray vector to 2D
    lamb_M[:, [tt]] = np.maximum(0, np.dot(eta3_diag, (xtil[:, [tt]] - x_max)))
#
    # Projection
    x[:, [tt]] = xtil[:, [tt]]

    for nn in range(0, 5):
        if x[nn, (tt)] > x_max[nn][0]:
            x[nn, (tt)] = x_max[nn][0]
        if x[nn, (tt)] < x_min[nn][0]:
            x[nn, (tt)] = x_min[nn][0]

    # Actuation
    for load in loads:
        # Qg (index=3), reactive power output (MVAr)
        # load/loads is just indexing to the loads (as opposed to the other
        # bus elements)

        mpc['gen'][[(load)], 2] = float(x[(load), (tt)])
        if 1 < (x[(load), (tt)]) < 10:
            print("x value is: {}".format((x[(load), (tt)])))

    # Generators are index 1 and 5 (0 and 4 python). Need to set the vm_pu value for the generators
    # Vg(index=6), voltage magnitude setpoint(p.u.)
    for gg in gen:
        print('Load ind {} X value being set is {}'.format(load, x[(gg), tt]))
        mpc['gen'][[(gg)], 5] = float(x[(gg), (tt)])
        # mpc.res_gen['vm_pu'][0] = x[gen[gg]-1, tt-1]

    mpc = pp.api.runpf(mpc)
    mpc = mpc[0]
    volt[:, [tt]] = mpc['bus'][:, [7]]

F_t_df = pd.DataFrame(F_t)
F_t_df.to_csv('F_t_CSV')
x_df = pd.DataFrame(x)
x_df.to_csv('x_CSV')
volt_df = pd.DataFrame(volt)
volt_df.to_csv('volt_CSV')
print('Stop and debug')


plt.plot(volt[0,:-1])
plt.plot(volt[1,:-1])
plt.plot(volt[2,:-1])
plt.plot(volt[3,:-1])
plt.plot(volt[4,:-1])
plt.legend(['Bus 1', 'Bus 2', 'Bus 3', 'Bus 4', 'Bus 5'])
plt.ylabel('Voltage')

plt.show()
