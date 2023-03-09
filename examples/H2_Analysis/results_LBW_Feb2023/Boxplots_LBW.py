#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:15:16 2023

@author: cclark2
"""

import os
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

output_dir = '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_VI/'
# c_label = ['No Policy', 'Base Policy', 'Maximum Policy']


# df_no = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_All_Feb2023.csv', usecols = ['ATB Year', 'Policy', 'LCOH ($/kg)'])
# # sns.boxplot(data=df_no, x="LCOH ($/kg)", y="Policy", hue="ATB Year")

# dims = (20, 15)
# plt.rcParams.update({'font.size': 24})
# fig, ax = plt.subplots(figsize=dims)
# ax = sns.boxplot(data=df_no, x="LCOH ($/kg)", y="Policy", hue="ATB Year")
# plt.axvline(x = 1, color = 'black', linestyle = 'dashed', label = '$1/kg')
# plt.axvline(x = 2, color = 'black', linestyle = 'dashed', label = '$2/kg')
# plt.savefig(os.path.join(output_dir, 'LCOH_Boxplot_withLines.png'), dpi=300)
# # plt.show()

# sites = [2, 3, 4, 5, 6, 7, 8]
# site_names = ['GA', 'TX', 'UT', 'CA', 'NY N', 'NY S', 'OR']

sites = [4, 5, 6]
site_names = ['UT', 'CA', 'NY'] #NY N


for site, site_name in list(zip(sites, site_names)):
    
    h2_file = '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_V/{}_H2Production.csv'.format(site)
    pp_file = '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_V/{}_PowerProduction.csv'.format(site)
    
    df_h2 = pd.read_csv(h2_file)
    df_pp = pd.read_csv(pp_file)
    
    #add site name to files
    site_name_list = [site_name]*len(df_h2)
    df_h2['Site Name'] = site_name_list
    site_name_list = [site_name]*len(df_pp)
    df_pp['Site Name'] = site_name_list
    
    df_h2.to_csv(h2_file, index=False)
    df_pp.to_csv(pp_file, index=False)

#%%combine csv
csv_files = [os.path.join('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_V/{}_H2Production.csv'.format(site)) for site in sites] 
# csv_files = [os.path.join('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_V/{}_PowerProduction.csv'.format(site)) for site in sites] 

#Append the files
df_append = pd.DataFrame()

for file in csv_files:
    df_temp = pd.read_csv(file)
    df_append = df_append.append(df_temp, ignore_index=True)
    
df_append.to_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_VI/H2Production.csv', index=False)
# df_append.to_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_VI/PowerProduction.csv', index=False)




#%%

df_h2_with_0 = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_VI/H2Production.csv')
df_pp_with_0 = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_VI/PowerProduction.csv')

df_h2 = df_h2_with_0[df_h2_with_0 != 0]
df_pp = df_pp_with_0[df_pp_with_0 != 0]

dims = (6, 3)
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_pp, x="Wind Energy Generation (kW)", y="Site Name")
plt.xticks(rotation=30)
plt.savefig(os.path.join(output_dir, 'WindEnergyGen_Boxplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_pp, x="Shortfall (kW)", y="Site Name")
plt.xticks(rotation=30)
plt.savefig(os.path.join(output_dir, 'Shortfall_Boxplot.png'), dpi=300, bbox_inches = "tight")


fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_h2, x="Energy to Electrolyzer (kW)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'EnergytoElectrolyzer_Boxplot.png'), dpi=300, bbox_inches = "tight")


fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_h2, x="Hydrogen Production Rate (kg/hr)", y="Site Name")
plt.xticks(rotation=30)
plt.savefig(os.path.join(output_dir, 'H2Production_Boxplot.png'), dpi=300, bbox_inches = "tight")


fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_h2, x="Electrolyzer Total Efficiency (%)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'ElectrolyzerEfficiency_Boxplot.png'), dpi=300, bbox_inches = "tight")


fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(data=df_h2, x="Water Usage (kg/hr)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'WaterUse_Boxplot.png'), dpi=300, bbox_inches = "tight")

#%%

dims = (6, 6)
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_pp, x="Wind Energy Generation (kW)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'WindEnergyGen_Vplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_pp, x="Shortfall (kW)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'Shortfall_Vplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_h2, x="Energy to Electrolyzer (kW)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'EnergytoElectrolyzer_Vplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_h2, x="Hydrogen Production Rate (kg/hr)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'H2Production_Vplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_h2, x="Electrolyzer Total Efficiency (%)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'ElectrolyzerEfficiency_Vplot.png'), dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots(figsize=dims)
ax = sns.violinplot(data=df_h2, x="Water Usage (kg/hr)", y="Site Name")
plt.savefig(os.path.join(output_dir, 'WaterUse_Vplot.png'), dpi=300, bbox_inches = "tight")




# df_x = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_IIII/2_PowerProduction.csv')
# df_y = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_IIII/4_PowerProduction.csv')
# df_z = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_IIII/9_PowerProduction.csv')

# x = df_x['Wind Energy Generation (kW)']
# y = df_y['Wind Energy Generation (kW)']
# z = df_z['Wind Energy Generation (kW)']

# dims = (6, 3)
# plt.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots(figsize=dims)
# plt.hist([x, y, z], label = ['TX', 'GA', 'UT'])
# plt.xticks(rotation=30)
# plt.xlabel('Wind Energy Generation (kW)')
# plt.ylabel('Count')
# plt.legend()
# plt.savefig(os.path.join(output_dir, 'WindEnergyGen_Histplot.png'), dpi=300, bbox_inches = "tight")
