# looking at more parameters

# visualize output of h2 analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'H2_Analysis_Plainview Bioenergy - Texas.csv'  #'H2_Analysis_Plainview Bioenergy - Texas.csv'
df = pd.read_csv(filename)

plt.figure(figsize=(10, 7))
atb_year = ['2020', '2025', '2030', '2035'] #'Custom', '2020', '2025', '2030', 'Custom', '2020', '2025', '2030', 'Custom']
useful_life = ['25', '25', '25', '25', '30', '30', '30', '30', '35', '35', '35', '35']
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

# Plot 1: Simple Feedstock Cost Vs. ATB year
# for i in range(len(atb_year)):
#     strTest = atb_year[i]
#     plt.plot(df['ATB Year'].iloc[count:count+5], df['Levelized H2 Elec Feedstock Cost/kg (HOPP)'].iloc[count:count+5], c[0], label=strTest)
#     count = count + 5
# plt.grid()
# plt.ylabel('H2 Elec Feedstock Cost/kg')
# plt.xlabel('ATB Year')
# # plt.legend()
# plt.title("H2 Elec Feedstock Cost/kg - ATB Year 2020 to 2035")
# # plt.plot([np.min(df['Hub Height (m)']),np.max(df['Hub Height (m)'])], [1.5, 1.5])
# plt.show()

# Plot 2
labels = ['2020']
x = np.arange(len(labels))  # the label locations
width = 0.4
no_helps = (df.loc[df['Scenario Choice'].isin(['No Help'])])
policy_helps = (df.loc[df['Scenario Choice'].isin(['Policy Help'])])
all_helps = (df.loc[df['Scenario Choice'].isin(['Advanced Technology Progression'])])
# df._get_value('')
fig, ax = plt.subplots()
for i in range(len(atb_year)):
    strTest = atb_year[i]
    # rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
    # rects2 = ax.bar(x + width / 2, women_means, width, label='Women')
    print(df['Levelized H2 Elec Feedstock Cost/kg (HOPP)'])

    p1 = ax.bar(x - (width / 2), no_helps, width, label='No Help')
    p2 = ax.bar(x + (width / 2), policy_helps, width, label='Policy Help')
    p3 = ax.bar(x + 2*(width / 2), all_helps, width, label='Advanced Technology Progression')
    count = count + 5

ax.grid()
plt.ylabel('H2 Elec Feedstock Cost/kg')
plt.xlabel('ATB Year')
ax.set_xticklabels(labels)
# plt.legend()
plt.title("H2 Elec Feedstock Cost/kg - ATB Year 2020 to 2035")
# plt.plot([np.min(df['Hub Height (m)']),np.max(df['Hub Height (m)'])], [1.5, 1.5])
plt.show()

