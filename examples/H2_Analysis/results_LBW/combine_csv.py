#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:40:25 2023

@author: cclark2
"""

import pandas as pd

##Combine .csv files for LBW analysis

#List the files

csv_files = ['/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/sites_II/H2_Analysis_LBW_2025_No Policy.csv',\
             '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2025_Base.csv', \
             '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2025_Max.csv', \
             '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2030_No Policy.csv', \
            '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2030_Base.csv', \
            '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2030_Max.csv', \
            '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2035_No Policy.csv', \
            '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2035_Base.csv', \
            '/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_LBW_Feb2023/H2_Analysis_LBW_2035_Max.csv']

#Append the files
df_append = pd.DataFrame()

for file in csv_files:
    df_temp = pd.read_csv(file, names=['Site Name', 'Lat', 'Lon', 'ATB Year', 'Plant life', 'Policy',\
                         'Turbine size (MW)', 'Wind Plant size (MW)', 'Wind Plant Size Adjusted for Turbine Rating(MW)',\
                         'Electrolyzer size (MW)', 'Load Profile (kW)', 'Energy to Electrolyzer (kW)', 'Wind capacity factor (%)',\
                         'Electrolyzer capacity factor (%)', 'LCOH ($/kg)', 'LCOH: Compression & storage ($/kg)',\
                         'LCOH: Electrolyzer CAPEX ($/kg)', 'LCOH: Desalination CAPEX ($/kg)', 'LCOH: Electrolyzer FOM ($/kg)',\
                         'LCOH: Electrolyzer VOM ($/kg)', 'LCOH: Desalination FOM ($/kg)', 'LCOH: Renewable plant ($/kg)',\
                         'LCOH: Renewable FOM ($/kg)', 'LCOH: Taxes ($/kg)', 'LCOH: Water consumption ($/kg)',\
                         'LCOH: Finances ($/kg)', 'LCOH: total ($/kg)'])
    df_append = df_append.append(df_temp, ignore_index=True)
    
df_append.to_csv('H2_Analysis_LBW_All_Feb2023.csv', index=False)