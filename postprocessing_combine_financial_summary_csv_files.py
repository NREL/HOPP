# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:51:07 2017

@author: jeichman
"""

import fnmatch
import os
import sqlite3
import pandas as pd
import warnings 
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter("ignore",UserWarning)

#Scenario1 = 'Green_steel_ammonia_electrolysis'
Scenario1 = 'Green_steel_ammonia_smr'

#dir0 = 'examples\\H2_Analysis\\Phase1B\\Fin_sum\\' 
#dir0 = 'examples\\H2_Analysis\\Phase1B\\Fin_sum_sens\\' 
#dir0 = 'examples\\H2_Analysis\\Phase1B\\Fin_sum_mid\\' 
#dir0 = 'examples\\H2_Analysis\\Financial_summary_TX_2020_revised_EC_costs_dist_sensitivity\\' 
#dir0 = 'examples\\H2_Analysis\\Financial_summary_distributed_sensitivity\\' 
dir0 = 'examples\\H2_Analysis\\Phase1B\\SMR_fin_summary\\' # Location to put database files
dir1 = dir0                                                                                 # Location of csv files

c0 = [0,0,0]
files2load_input={}
files2load_input_title={}
files2load_input_categories={}
files2load_results={}
files2load_results_title={}
files2load_results_categories={}
files2load_summary={}
files2load_summary_title={}
files2load_summary_categories={}

for files2load in os.listdir(dir1):   
    if Scenario1=='Green_steel_ammonia_electrolysis':

        if fnmatch.fnmatch(files2load, 'Fin_sum_*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[2:]
            #int1[-2]=int1[-2].replace(' ','-')
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        files2load_title_header = ['Site','Year','Turbine Size','Electrolysis case','Electrolysis cost case','Policy Option','Grid case','Renewables case','Wind model','Degradation modeled?','Stack optimized?','NPC string','Num pem clusters','Storage string','Storage multiplier']
        
    if Scenario1=='Green_steel_ammonia_smr':

        if fnmatch.fnmatch(files2load, 'Financial_Summary_*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[2:]
            int1[-2]=int1[-2].replace(' ','-')
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        files2load_title_header = ['Hydrogen model','SMR string','Site','Year','Policy Option','CCS Case','NG price case','string1','string2','Integration']
       # files2load_title_header = ['Steel String','Year','Site String','Site Number','Turbine Year','Turbine Size','Storage Duration','Storage String','Grid Case']


# Connecting to the database file
### conn.close()
sqlite_file = 'Default_summary.db'  # name of the sqlite database file
if os.path.exists(dir0+'/'+sqlite_file):
    os.remove(dir0+'/'+sqlite_file)
conn = sqlite3.connect(dir0+sqlite_file)    # Setup connection with sqlite
c = conn.cursor()

if 1==1:            # This section captures the scenario table from summary files
    # Create Scenarios table and populate   
    if Scenario1=='Green_steel_ammonia_electrolysis':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Site' text,
                                             'Year' text,
                                             'Turbine Size' text,
                                             'Electrolysis case' text,
                                             'Electrolysis cost case' text,
                                             'Policy Option' text,
                                             'Grid case' text,
                                             'Renewables case' text,
                                             'Wind model' text,
                                             'Degradation modeled?' text,
                                             'Stack optimized?' text,
                                             'NPC string' text,
                                             'Num pem clusters' text,
                                             'Storage string' text,
                                             'Storage multiplier' text)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit() 
        
    # Create Scenarios table and populate   
    elif Scenario1=='Green_steel_ammonia_smr':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Hydrogen model' text,
                                             'SMR string' text,
                                             'Site' text,
                                             'Year' text,
                                             'Policy Option' text,
                                             'CCS Case' text,
                                             'NG price case' text,
                                             'string1' text,
                                             'string2' text,
                                             'Integration' text)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit() 
        


        
if 1==1:            # This section captures the summary files         
    # Creating Summary Table
    for i0 in range(len(files2load_summary_title)):
        #i0=0
        # Creating Summary Table header
        #files2load_summary_data = pd.read_csv(dir0+files2load_summary[i0+1],sep=',',header=0,names=['Data'],skiprows=[0,1]).T
        files2load_summary_data = pd.read_csv(dir0+files2load_summary[i0+1],sep=',',header=0,names=['Data']).T
        # if files2load_summary_data['output to input ratio'].values==['         +INF']:
        #     files2load_summary_data['output to input ratio'] = 0
        #     files2load_summary_data = files2load_summary_data.astype('float64', copy=False)
        # files2load_summary_data = files2load_summary_data.drop(['input','output'],axis=1)       # Remove unnecessary rows
        for i1 in range(len(files2load_title_header)):
            files2load_summary_data[files2load_title_header[i1]] = files2load_summary_title[i0+1][i1]
        if i0==0:
            files2load_summary_data_all = files2load_summary_data
        else:
            files2load_summary_data_all = files2load_summary_data_all.append(files2load_summary_data, ignore_index=True)
        print('Combining Data: '+str(i0+1)+' of '+str(len(files2load_summary_title)))

    # Rename duplicate columns (should fix in next version)
    # H1 = {'LSL limit fraction': ['Input LSL limit fraction', 'Output LSL limit fraction']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    # H1 = {'reg up limit fraction': ['Input reg up limit fraction', 'Output reg up limit fraction']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    # H1 = {'reg down limit fraction': ['Input reg down limit fraction', 'Output reg down limit fraction']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    # H1 = {'spining reserve limit fraction': ['Input spining reserve limit fraction', 'Output spining reserve limit fraction']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    # H1 = {'startup cost ($/MW-start)': ['Input startup cost ($/MW-start)', 'Output startup cost ($/MW-start)']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    # H1 = {'minimum run intervals': ['Input minimum run intervals', 'Output minimum run intervals']}
    # files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)

    # if Scenario1=='SCS':
    #     files2load_summary_data_all['Storage Duration'] = files2load_summary_data_all['Storage Duration'].astype(float)
    #     files2load_summary_data_all['Year'] = files2load_summary_data_all['Year'].astype(float)
        
    # Create database table for each column
    files2load_summary_header = files2load_summary_data_all.columns.tolist()
    files2load_summary_data_types = files2load_summary_data_all.dtypes
    execute_text = 'CREATE TABLE Summary (\''
    sql = "INSERT INTO Summary VALUES ("
    for i0 in range(len(files2load_summary_header)):
        if i0==len(files2load_summary_header)-1:
            if files2load_summary_data_types[i0]=='object':
                execute_text = execute_text+files2load_summary_header[i0]+'\' text)'
            elif files2load_summary_data_types[i0]=='float64':
                execute_text = execute_text+files2load_summary_header[i0]+'\' real)'
            sql = sql+'?)'
        else:
            if files2load_summary_data_types[i0]=='object':
                execute_text = execute_text+files2load_summary_header[i0]+'\' text,\''
            elif files2load_summary_data_types[i0]=='float64':
                execute_text = execute_text+files2load_summary_header[i0]+'\' real,\''
            sql = sql+'?,'
    c.execute(execute_text)
    


    # Committing changes and closing the connection to the database file
    params=list()
    for i0 in range(len(files2load_summary_data_all)):
        params.insert(i0,tuple(files2load_summary_data_all.loc[i0,:].tolist()))
        if ((i0+1)%1000)==0:
            print('Creating Output: '+str(i0)+' of '+str(len(files2load_summary_data_all)))
    c.executemany(sql, params)
    conn.commit()
    
    
   
# if 1==1:            # This section captures a subset of the results files 
#     # Creating Results Table
#     for i0 in range(len(files2load_results_title)):
#         # Creating Results Table header
#         files2load_results_data = pd.read_csv(dir0+files2load_results[i0+1],sep=',',header=25,skiprows=[24])
#         for i1 in range(len(files2load_title_header)):
#             files2load_results_data[files2load_title_header[i1]] = files2load_results_title[i0+1][i1]
#         if i0==0:
#             files2load_results_data_all = files2load_results_data
#         else:
#             files2load_results_data_all = files2load_results_data_all.append(files2load_results_data, ignore_index=True)
#         print('Combining Data: '+str(i0+1)+' of '+str(len(files2load_results_title)))
        
#     # Create database table for each column
#     files2load_results_header = files2load_results_data_all.columns.tolist()
#     files2load_results_data_types = files2load_results_data_all.dtypes
#     execute_text = 'CREATE TABLE Results (\''
#     sql = "INSERT INTO results VALUES ("
#     for i0 in range(len(files2load_results_header)):
#         if i0==len(files2load_results_header)-1:
#             if files2load_results_data_types[i0]=='object':
#                 execute_text = execute_text+files2load_results_header[i0]+'\' text)'
#             elif files2load_results_data_types[i0]=='float64':
#                 execute_text = execute_text+files2load_results_header[i0]+'\' real)'
#             sql = sql+'?)'
#         else:
#             if files2load_results_data_types[i0]=='object':
#                 execute_text = execute_text+files2load_results_header[i0]+'\' text,\''
#             elif files2load_results_data_types[i0]=='float64':
#                 execute_text = execute_text+files2load_results_header[i0]+'\' real,\''
#             sql = sql+'?,'
#     c.execute(execute_text)

#     # Committing changes and closing the connection to the database file
#     params=list()
#     for i0 in range(len(files2load_results_data_all)):
#         params.insert(i0,tuple(files2load_results_data_all.loc[i0,:].tolist()))
#         if (i0%10000)==0:
#             print('Creating Output: '+str(i0)+' of '+str(len(files2load_results_data_all)))
#     c.executemany(sql, params)
#     conn.commit()

        
conn.close()


