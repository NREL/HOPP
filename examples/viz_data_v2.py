# looking at more parameters

# visualize output of h2 analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'H2_Analysis_Plainview Bioenergy - Texas.csv'
df = pd.read_csv(filename)

plt.figure(figsize=(10, 7))
atb_year = ['2020', '2025', '2030', '2035'] #'Custom', '2020', '2025', '2030', 'Custom', '2020', '2025', '2030', 'Custom']
useful_life = ['25', '25', '25', '25', '30', '30', '30', '30', '35', '35', '35', '35',]
c = ['k', 'b', 'r','g', 'k--', 'b--', 'r--','g--', 'ko-', 'bs-', 'r^-','gv-']
count = 0
# for i in range(len(useful_life)):
#     strTest = atb_year[i] + ' ' + useful_life[i] + ' years'
#     plt.plot(df['Hub Height (m)'].iloc[count:count+5], df['H2 Elec Feedstock Cost/kW'].iloc[count:count+5],c[i],label=strTest)
#     count = count + 5
# plt.grid()
# plt.ylabel('H2 Elec Feedstock Cost/kW')
# plt.xlabel('Hub Height (m)')
# plt.legend()
# plt.plot([np.min(df['Hub Height (m)']),np.max(df['Hub Height (m)'])], [1.5, 1.5])
# plt.show()

for i in range(len(atb_year)):
    strTest = atb_year[i]
    plt.plot(df['ATB Year'].iloc[count:count+5], df['H2 Elec Feedstock Cost/kW'].iloc[count:count+5], c[0], label=strTest)
    count = count + 5
plt.grid()
plt.ylabel('H2 Elec Feedstock Cost/kW')
plt.xlabel('ATB Year')
# plt.legend()
plt.title("H2 Elec Feedstock Cost/kWh ATB Year 2020 to 2035")
# plt.plot([np.min(df['Hub Height (m)']),np.max(df['Hub Height (m)'])], [1.5, 1.5])
plt.show()
