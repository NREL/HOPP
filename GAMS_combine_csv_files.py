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

Scenario1 = 'Green_steel_ammonia'


dir0 = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\' # Location to put database files
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
    if Scenario1=='Test':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[0] = int1[0].replace('.csv', '')
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[0] = int1[0].replace('.csv', '')
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[0] = int1[0].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        files2load_title_header = ['Name']



    elif Scenario1=='Central_vs_distributed':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[2] = int1[2].replace('CF', '')            
            int1[5] = int1[5].replace('hrs.csv', '')
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[2] = int1[2].replace('CF', '')
            int1[5] = int1[5].replace('hrs.csv', '')
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[2] = int1[2].replace('CF', '')
            int1[5] = int1[5].replace('hrs.csv', '')
            files2load_summary_title[c0[2]] = int1
            
            
    elif Scenario1=='Solar_Hydrogen':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            del int1[1]
            files2load_input_categories[0]='Scenario'
            if int1[1]=='Vaca':
                int1[1:3] = [' '.join(int1[1:3])]
                files2load_input_categories[1]='Location'                    
            for i0 in range(len(int1)):
                if fnmatch.fnmatch(int1[i0], 'itc*'):
                    int1[i0] = int1[i0].replace('itc', '')
                    if c0[1]==1: files2load_input_categories[i0]='ITC'
                if fnmatch.fnmatch(int1[i0], 'NEM*'):
                    int1[i0] = int1[i0].replace('NEM', '')
                    if c0[1]==1: files2load_input_categories[i0]='NEM'
                if fnmatch.fnmatch(int1[i0], 'EY*'):
                    int1[i0] = int1[i0].replace('.csv', '')
                    int1[i0] = int1[i0].replace('EY', '')
                    if c0[1]==1: files2load_input_categories[i0]='EY Size'
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            del int1[1]
            files2load_results_categories[0]='Scenario'
            if int1[1]=='Vaca':
                int1[1:3] = [' '.join(int1[1:3])]
                files2load_results_categories[1]='Location'                    
            for i0 in range(len(int1)):
                if fnmatch.fnmatch(int1[i0], 'itc*'):
                    int1[i0] = int1[i0].replace('itc', '')
                    if c0[1]==1: files2load_results_categories[i0]='ITC'
                if fnmatch.fnmatch(int1[i0], 'NEM*'):
                    int1[i0] = int1[i0].replace('NEM', '')
                    if c0[1]==1: files2load_results_categories[i0]='NEM'
                if fnmatch.fnmatch(int1[i0], 'EY*'):
                    int1[i0] = int1[i0].replace('.csv', '')
                    int1[i0] = int1[i0].replace('EY', '')
                    if c0[1]==1: files2load_results_categories[i0]='EY Size'
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            del int1[1]
            files2load_summary_categories[0]='Scenario'
            if int1[1]=='Vaca':
                int1[1:3] = [' '.join(int1[1:3])]
                files2load_summary_categories[1]='Location'                    
            for i0 in range(len(int1)):
                if fnmatch.fnmatch(int1[i0], 'itc*'):
                    int1[i0] = int1[i0].replace('itc', '')
                    if c0[1]==1: files2load_summary_categories[i0]='ITC'
                if fnmatch.fnmatch(int1[i0], 'NEM*'):
                    int1[i0] = int1[i0].replace('NEM', '')
                    if c0[1]==1: files2load_summary_categories[i0]='NEM'
                if fnmatch.fnmatch(int1[i0], 'EY*'):
                    int1[i0] = int1[i0].replace('.csv', '')
                    int1[i0] = int1[i0].replace('EY', '')
                    if c0[1]==1: files2load_summary_categories[i0]='EY Size'
            files2load_summary_title[c0[2]] = int1
            
            
    elif Scenario1=='VTA_bus_project':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[3] = int1[3].replace('.csv', '')
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results_*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[3] = int1[3].replace('.csv', '')
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary_*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[3] = int1[3].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        files2load_title_header = ['Utility','Block','Services','Renewable']

    elif Scenario1=='SCS':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results_*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary_*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        files2load_title_header = ['Grid Case','EC Case','PV Case','Storage Case','Storage Duration','Demand Case','Year']
        
    elif Scenario1=='Green_steel_ammonia':
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_input*'):
            c0[0]=c0[0]+1
            files2load_input[c0[0]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_input_title[c0[0]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_results_*'):
            c0[1]=c0[1]+1
            files2load_results[c0[1]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_results_title[c0[1]] = int1
        if fnmatch.fnmatch(files2load, 'Storage_dispatch_summary_*'):
            c0[2]=c0[2]+1
            files2load_summary[c0[2]] = files2load
            int1 = files2load.split("_")
            int1 = int1[3:]
            int1[-1] = int1[-1].replace('.csv', '')
            files2load_summary_title[c0[2]] = int1
        #files2load_title_header = ['Steel String','Year','Site String','Site Number','Turbine Year','Turbine Size','Storage Duration','Storage String','Grid Case']
        files2load_title_header = ['Year','Site','Turbine Size','Electrolysis case','Policy Option','Grid Case']


# Connecting to the database file
### conn.close()
sqlite_file = 'Default_summary.db'  # name of the sqlite database file
if os.path.exists(dir0+'/'+sqlite_file):
    os.remove(dir0+'/'+sqlite_file)
conn = sqlite3.connect(dir0+sqlite_file)    # Setup connection with sqlite
c = conn.cursor()

if 1==1:            # This section captures the scenario table from summary files
    # Create Scenarios table and populate
    if Scenario1=='Test':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Name' text)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit()   
        
        
    elif Scenario1=='Central_vs_distributed':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Tariff' text,                                             
                                             'Operating Strategy' text,
                                             'Capacity Factor (%)' real,
                                             'Configuration' text,
                                             'Timeframe' text,
                                             'Storage duration (hours)' real)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit()    
        
        
    elif Scenario1=='Solar_Hydrogen':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Tariff' text,                                             
                                             'Capacity Factor (%)' real,
                                             'Storage duration (hours)' real)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit()    
        
    elif Scenario1=='SCS':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Grid Case' text,
                                             'EC Case' text,
                                             'PV Case' text,
                                             'Storage Case' text,
                                             'Storage duration' text,
                                             'Demand Case' text,
                                             'Year' text)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?,?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit() 
        
    elif Scenario1=='Green_steel_ammonia':
        c.execute('''CREATE TABLE Scenarios ('Scenario Number' real,
                                             'Year' text,
                                             'Site' text,
                                             'Turbine Size' text,
                                             'Electrolysis Case' text,
                                             'Policy Option' text,
                                             'Grid Case' text)''')    
        sql = "INSERT INTO Scenarios VALUES (?,?,?,?,?,?,?)"
        params=list()
        for i0 in range(len(files2load_summary)):    
            params.insert(i0,tuple(list([str(i0+1)])+files2load_summary_title[i0+1]))
            print('Scenario data: '+str(i0+1)+' of '+str(len(files2load_summary)))
        c.executemany(sql, params)
        conn.commit() 
        

        
if 1==1:            # This section captures the summary files         
    # Creating Summary Table
    for i0 in range(len(files2load_summary_title)):
        # Creating Summary Table header
        files2load_summary_data = pd.read_csv(dir0+files2load_summary[i0+1],sep=',',header=0,names=['Data'],skiprows=[0,1]).T
        if files2load_summary_data['output to input ratio'].values==['         +INF']:
            files2load_summary_data['output to input ratio'] = 0
            files2load_summary_data = files2load_summary_data.astype('float64', copy=False)
        files2load_summary_data = files2load_summary_data.drop(['input','output'],axis=1)       # Remove unnecessary rows
        for i1 in range(len(files2load_title_header)):
            files2load_summary_data[files2load_title_header[i1]] = files2load_summary_title[i0+1][i1]
        if i0==0:
            files2load_summary_data_all = files2load_summary_data
        else:
            files2load_summary_data_all = files2load_summary_data_all.append(files2load_summary_data, ignore_index=True)
        print('Combining Data: '+str(i0+1)+' of '+str(len(files2load_summary_title)))

    # Rename duplicate columns (should fix in next version)
    H1 = {'LSL limit fraction': ['Input LSL limit fraction', 'Output LSL limit fraction']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    H1 = {'reg up limit fraction': ['Input reg up limit fraction', 'Output reg up limit fraction']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    H1 = {'reg down limit fraction': ['Input reg down limit fraction', 'Output reg down limit fraction']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    H1 = {'spining reserve limit fraction': ['Input spining reserve limit fraction', 'Output spining reserve limit fraction']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    H1 = {'startup cost ($/MW-start)': ['Input startup cost ($/MW-start)', 'Output startup cost ($/MW-start)']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)
    H1 = {'minimum run intervals': ['Input minimum run intervals', 'Output minimum run intervals']}
    files2load_summary_data_all = files2load_summary_data_all.rename(columns=lambda c: H1[c].pop(0) if c in H1.keys() else c)

    if Scenario1=='SCS':
        files2load_summary_data_all['Storage Duration'] = files2load_summary_data_all['Storage Duration'].astype(float)
        files2load_summary_data_all['Year'] = files2load_summary_data_all['Year'].astype(float)
        
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
    
    
   
if 1==1:            # This section captures a subset of the results files 
    # Creating Results Table
    for i0 in range(len(files2load_results_title)):
        # Creating Results Table header
        files2load_results_data = pd.read_csv(dir0+files2load_results[i0+1],sep=',',header=25,skiprows=[24])
        for i1 in range(len(files2load_title_header)):
            files2load_results_data[files2load_title_header[i1]] = files2load_results_title[i0+1][i1]
        if i0==0:
            files2load_results_data_all = files2load_results_data
        else:
            files2load_results_data_all = files2load_results_data_all.append(files2load_results_data, ignore_index=True)
        print('Combining Data: '+str(i0+1)+' of '+str(len(files2load_results_title)))
        
    # Create database table for each column
    files2load_results_header = files2load_results_data_all.columns.tolist()
    files2load_results_data_types = files2load_results_data_all.dtypes
    execute_text = 'CREATE TABLE Results (\''
    sql = "INSERT INTO results VALUES ("
    for i0 in range(len(files2load_results_header)):
        if i0==len(files2load_results_header)-1:
            if files2load_results_data_types[i0]=='object':
                execute_text = execute_text+files2load_results_header[i0]+'\' text)'
            elif files2load_results_data_types[i0]=='float64':
                execute_text = execute_text+files2load_results_header[i0]+'\' real)'
            sql = sql+'?)'
        else:
            if files2load_results_data_types[i0]=='object':
                execute_text = execute_text+files2load_results_header[i0]+'\' text,\''
            elif files2load_results_data_types[i0]=='float64':
                execute_text = execute_text+files2load_results_header[i0]+'\' real,\''
            sql = sql+'?,'
    c.execute(execute_text)

    # Committing changes and closing the connection to the database file
    params=list()
    for i0 in range(len(files2load_results_data_all)):
        params.insert(i0,tuple(files2load_results_data_all.loc[i0,:].tolist()))
        if (i0%10000)==0:
            print('Creating Output: '+str(i0)+' of '+str(len(files2load_results_data_all)))
    c.executemany(sql, params)
    conn.commit()

        
conn.close()


