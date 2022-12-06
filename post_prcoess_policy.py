# post process policy and integration results

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 

results_dir = 'examples/H2_Analysis/financial_summary_results/'
filenames = os.listdir(results_dir)

# just look at option 1, 2, 3 for Texas 

file1 = results_dir + 'Financial_Summary_PyFAST_TX_2020_6MW_Centralized_option 1.csv'
file2 = results_dir + 'Financial_Summary_PyFAST_TX_2020_6MW_Centralized_option 2.csv'
file3 = results_dir + 'Financial_Summary_PyFAST_TX_2020_6MW_Centralized_option 3.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

df1 = df1.set_index('Unnamed: 0')
df2 = df2.set_index('Unnamed: 0')
df3 = df3.set_index('Unnamed: 0')
print('No policy: ', df1.loc['Policy savings ($/tonne)'].values)
print('Base policy: ', df2.loc['Policy savings ($/tonne)'].values)
print('Max policy: ', df3.loc['Policy savings ($/tonne)'].values)

print('No policy: ', df1.loc['LCOE ($/MWh)'].values)
print('Base policy: ', df2.loc['LCOE ($/MWh)'].values)
print('Max policy: ', df3.loc['LCOE ($/MWh)'].values)

