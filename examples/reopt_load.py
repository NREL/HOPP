import pickle
import pandas as pd
import numpy as np
from simple_dispatch import SimpleDispatch

import matplotlib.pyplot as plt

# curtail and energy short fall data
filename = 'H2 Analysis/curtail.csv'
combined_pv_wind_curtailment_hopp = pd.read_csv(filename)

filename = 'H2 Analysis/energy_shortfall.csv'
energy_shortfall_hopp = pd.read_csv(filename)

# number of timesteps to evaluate
N = 100
# N = len(combined_pv_wind_curtailment_hopp)

size_battery = 4000

bat_model = SimpleDispatch(np.array(combined_pv_wind_curtailment_hopp['0']),
                           np.array(energy_shortfall_hopp['0']),N,size_battery)

battery_used, excess_energy, battery_SOC = bat_model.run()

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