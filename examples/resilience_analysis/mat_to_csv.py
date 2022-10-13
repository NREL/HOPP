#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:39:16 2022

@author: cclark2
"""


# import mat4py
import os
import pandas as pd
from mat4py import loadmat

mat_data_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Matlab_Data/'
data_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/resilience_analysis/Load_Data/'

feeders = ['Feeder1', 'Feeder2', 'Feeder3', 'Feeder4', 'Feeder6', 'Feeder7', 'Feeder8']

for feeder in feeders:
    feeder_header = loadmat(os.path.join(mat_data_dir, '{}_P_Q_Header.mat'.format(feeder)))
    feeder_p = loadmat(os.path.join(mat_data_dir, '{}_P.mat'.format(feeder)))
    feeder_q = loadmat(os.path.join(mat_data_dir, '{}_Q.mat'.format(feeder)))

    
    mdata_header = feeder_header['{}_P_Q_Header'.format(feeder)]
    mdata_p = feeder_p['{}_P'.format(feeder)]
    mdata_q = feeder_q['{}_Q'.format(feeder)]

    df_p = pd.DataFrame(mdata_p, columns = mdata_header)
    df_q = pd.DataFrame(mdata_q, columns = mdata_header)
    
    df_p.to_csv(os.path.join(data_dir, '{}_P.csv'.format(feeder)), index=False)
    df_q.to_csv(os.path.join(data_dir, '{}_Q.csv'.format(feeder)), index=False)

    
            
    
    
    