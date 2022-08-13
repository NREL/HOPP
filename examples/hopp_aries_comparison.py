import json
from tkinter.ttk import Style
import pandas as pd
import matplotlib.pyplot as plt

# Import HOPP results
filepath = 'results/' + 'yearlong_outputs.json'
hopp_results = pd.read_json(filepath)
hopp_results.index = pd.date_range(start="2021-06-30 23:00:00", periods=8760, freq="h")
filepath = 'results/' + 'yearlong_outputs_uncorrected.json'
hopp_results_uncorrected = pd.read_json(filepath)
hopp_results_uncorrected.index = pd.date_range(start="2021-06-30 23:00:00", periods=8760, freq="h")
filepath = 'results/' + 'yearlong_wind_misalignment.json'
pitch_misalignment = pd.read_json(filepath)

# Import ARIES generation
filepath = 'resource_files/' + 'yearlong_aries_generation.csv'
aries_gen = pd.read_csv(filepath)
aries_gen.index = pd.date_range(start="2021-06-30 23:00:00", periods=8760, freq="h")
aries_gen.columns = ['Solar','Wind']

# Convert turbine status to codes
filepath = 'resource_files/' + 'yearlong_ge15_turbine_status.csv'
turbine_status = pd.read_csv(filepath)
turbine_status = turbine_status.loc[:,'Turbine Status'].values
for idx, value in enumerate(turbine_status):
    if value == 7:
        newvalue = 5
    elif value == 5 or value == 6:
        newvalue = 6
    elif value > 7:
        newvalue = 6
    else:
        newvalue = value
    turbine_status[idx] = newvalue;
codes = ['*NO DATA*',
    'Turbine OK',
    'Grid Connection',
    'Running/Idle',
    'Maintenance',
    'Weather Conditions',
    'Other']

# Parse results
hopp_solar = hopp_results.loc[:,'pv generation (kW)']
hopp_wind = hopp_results.loc[:,'wind generation (kW)']
hopp_wind_uncorrected = hopp_results_uncorrected.loc[:,'wind generation (kW)']
aries_solar = aries_gen.loc[:,'Solar']
aries_wind = aries_gen.loc[:,'Wind']

# Zero out negative solar from ARIES
zero_inds = aries_solar.values<0
aries_solar.iloc[zero_inds] = 0

# Plot results
start = '2022-06-05'
end = '2022-06-19'
mod_label = 'HOPP Modeled Output'
act_label = 'Actual Power Output'

plt.subplot(3,1,1)
plt.plot(hopp_solar.index,hopp_solar.values,label=mod_label)
plt.plot(aries_solar.index,aries_solar.values,label=act_label)
plt.ylabel("First Solar 430 kW PV [kW]")
plt.legend()
plt.xlim([pd.to_datetime(start),pd.to_datetime(end)])

plt.subplot(3,1,2)
plt.plot(aries_wind.index,aries_wind.values,label=act_label,color='C1')
plt.plot(hopp_wind.index,hopp_wind_uncorrected.values,label=mod_label+', uncorrected',color='C0')
plt.plot(hopp_wind.index,hopp_wind.values,'--',label=mod_label+', yaw/pitch corrected',color='C0')
plt.ylabel("GE 1.5 MW Turbine [kW]")
plt.legend(ncol=3)
plt.ylim([-100,1600])
plt.xlim([pd.to_datetime(start),pd.to_datetime(end)])

plt.subplot(3,1,3)

# plt.plot(hopp_wind.index,pitch_misalignment.loc['Pitch'].values,label='Pitch')
# plt.plot(hopp_wind.index,pitch_misalignment.loc['Yaw Misalignment'].values,label='Yaw Misalignment')
# plt.ylabel("Angle [degrees]")
# plt.legend()

plt.plot(hopp_wind.index,turbine_status)
plt.ylim([-.5,6.5])
plt.xlim([pd.to_datetime(start),pd.to_datetime(end)])
Ax = plt.gca()
Ax.set_yticks([0,1,2,3,4,5,6])
Ax.set_yticklabels(codes)
plt.grid(axis='y')
plt.ylabel('Turbine Status')

plt.show()