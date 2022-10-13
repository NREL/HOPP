#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 16:40:40 2022

@author: cclark2
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Load_Data/'

df_p = pd.read_csv(os.path.join(data_dir, 'All_Feeder_P.csv'))
df_p['Total_P_95'] = df_p['Total_Feeder_P']*0.95
df_p['Total_P_90'] = df_p['Total_Feeder_P']*0.90
df_p['Total_P_75'] = df_p['Total_Feeder_P']*0.75
df_p['Total_P_50'] = df_p['Total_Feeder_P']*0.50
df_p['Total_P_25'] = df_p['Total_Feeder_P']*0.25

df_p.to_csv(os.path.join(data_dir, 'All_Feeder_P_partial_loads.csv'), index=False)


# df_q = pd.read_csv(os.path.join(data_dir, 'All_Feeder_Q.csv'))
        
# df_p.plot('Timestamp', 'Total_Feeder_P')



