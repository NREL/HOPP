#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 16:40:40 2022

@author: cclark2
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

#
data_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Load_Data/'
save_outputs_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Outage_Data/'

#Load demand data with various critical load percentages 
load_df = pd.read_csv(os.path.join(data_dir, 'All_Feeder_P_partial_loads.csv'))

#Load generation data
generation_df = pd.read_csv(os.path.j(os.path.join(data_dir, 'All_Feeder_P_partial_loads.csv')))

#Difference between load and generation is met by grid in nominal scenario and load not met in an outage scenario
outage_df = pd.DataFrame(load_df['Timestamp'])
outage_df['Load Not Met'] = load_df['Total_Feeder_P'] - generation_df['Total_P_50']

#Generate outages
# outage_durations = [6, 12, 18, 24]
# for outage_duration in outage_durations:

outage_duration = 6
outage_start_hours = [*range(0,8760-(outage_duration+1))]

load_lost = [0]*8760
for outage_start in outage_start_hours:    
    # s = outage_df['Load Not Met']
    outage = s[outage_start:(outage_start+outage_duration)].sum()
    load_lost[outage_start] = outage
    
    
    
    
# outage_df['Timestamp']

#go to that index start
s = 

#sum the load not met across the outage duration
#save that summation data
    #timestamp
    #outage_start
    #load not met 95%
    #...
    #load not met 25%
    
outage_df.to_csv(os.path.join(data_dir, 'Outage_Data_{}hr.csv'.format(outage_duration)), index=False)
    






