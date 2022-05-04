# implement a simple dispatch model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# curtail and energy short fall data
filename = 'curtail.csv'
combined_pv_wind_curtailment_hopp = pd.read_csv(filename)

filename = 'energy_shortfall.csv'
energy_shortfall_hopp = pd.read_csv(filename)

# number of timesteps to evaluate
N = 100
# N = len(combined_pv_wind_curtailment_hopp)

def simple_dispatch(combined_pv_wind_curtailment_hopp,energy_shortfall_hopp,N):

    # storage module
    rated_size = 4000 # kW -> 200 MW
    charge_rate = 4000 # kWH -> 200 MWh
    battery_SOC = np.zeros(N)
    battery_used = np.zeros(N)
    excess_energy = np.zeros(N)
    for i in range(1,N):

        # should you charge
        if combined_pv_wind_curtailment_hopp['0'].iloc[i] > 0:
            if i == 0:
                battery_SOC = np.min([combined_pv_wind_curtailment_hopp['0'].iloc[i], charge_rate])
            else:
                battery_SOC[i] = battery_SOC[i - 1]
            if battery_SOC[i-1] < rated_size:
                add_gen = np.min([combined_pv_wind_curtailment_hopp['0'].iloc[i], charge_rate])
                battery_SOC[i] = np.min([battery_SOC[i] + add_gen, rated_size])
        else:
            battery_SOC[i] = battery_SOC[i - 1]

        # should you discharge
        if energy_shortfall_hopp['0'].iloc[i] > 0 and battery_SOC[i] > 0:
            energy_used = np.min([energy_shortfall_hopp['0'].iloc[i], battery_SOC[i]])
            battery_SOC[i] = battery_SOC[i] - energy_used
            battery_used[i] = energy_used

        # overall the amount of energy you could have sold to the grid assuming perfect knowledge
        if battery_SOC[i] == rated_size and battery_SOC[i-1] == rated_size:
            excess_energy[i] = combined_pv_wind_curtailment_hopp['0'].iloc[i]
        else:
            excess_energy[i] = np.max([combined_pv_wind_curtailment_hopp['0'].iloc[i] - battery_SOC[i], 0.0])

    return battery_used, excess_energy, battery_SOC

battery_used, excess_energy, battery_SOC = simple_dispatch(combined_pv_wind_curtailment_hopp,energy_shortfall_hopp,N)

print('Battery Generation: ', np.sum(battery_used))
print('Amount of energy going to the grid: ', np.sum(excess_energy))


plt.figure()
plt.plot(combined_pv_wind_curtailment_hopp['0'].iloc[0:N], label='Available Curtailed Power')
plt.plot(excess_energy,label='Excess Energy')
plt.plot(energy_shortfall_hopp['0'].iloc[0:N], label='Energy Shortfall')
plt.plot(battery_SOC,label='Battery Charge')
plt.plot(battery_used,label='Battery Discharge')
plt.xlabel('Time (hr)',fontsize=15)
plt.ylabel('Power (kW)', fontsize=15)
plt.grid()
plt.legend()
plt.show()