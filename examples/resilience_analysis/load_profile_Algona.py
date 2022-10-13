#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 16:06:26 2022

@author: cclark2
"""

import os
import pandas as pd
from datetime import datetime

data_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Load_Data/'

feeders = ['Feeder1', 'Feeder2', 'Feeder3', 'Feeder4', 'Feeder6', 'Feeder7', 'Feeder8']

df_p_totals = []
df_q_totals = []


for feeder in feeders:

    df_p = pd.read_csv(os.path.join(data_dir, '{}_P.csv'.format(feeder)))
    df_q = pd.read_csv(os.path.join(data_dir, '{}_Q.csv'.format(feeder)))
        
    df_p['{}_P'.format(feeder)] = df_p.sum(axis=1)
    df_q['{}_Q'.format(feeder)] = df_q.sum(axis=1)
    
    df_p_list = df_p['{}_P'.format(feeder)].tolist()
    df_q_list = df_q['{}_Q'.format(feeder)].tolist()    

    df_p_totals.append(df_p_list)
    df_q_totals.append(df_q_list)


df_p_feeder = pd.DataFrame(df_p_totals, index=feeders)
df_q_feeder = pd.DataFrame(df_q_totals, index=feeders)

df_p_feeder = df_p_feeder.transpose()
df_q_feeder = df_q_feeder.transpose()

df_p_feeder['Total_Feeder_P'] = df_p_feeder.sum(axis=1)
df_q_feeder['Total_Feeder_Q'] = df_q_feeder.sum(axis=1)

dates = pd.date_range(start='01/01/2021 00:00:00', end='12/31/2021 23:00:00', freq='H')
# df_p_feeder['Timestamp'] = date_range
# df_q_feeder['Timestamp'] = date_range

df_p_feeder.to_csv(os.path.join(data_dir, 'All_Feeder_P_ts.csv'), index=dates) #False)
# df_q_feeder.to_csv(os.path.join(data_dir, 'All_Feeder_Q.csv'), index=False)
