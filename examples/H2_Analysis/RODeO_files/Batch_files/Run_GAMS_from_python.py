# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:49:43 2020

@author: jeichman

Steps:
0. Initialize values and import data
1. Prepare information to run GAMS
2. Add fixed amount of different storage types and duration and see the resulting RPS
3. Add wind/solar to achieve similar RPS levels as average storage portfolio.
4. Calculate LCOE 
5. Make final comparison between RPS and LCOE for all technologies
6. Repeat steps 2-5.
"""

import pandas as pd
import math
import os

dir0 = 'C:/Users/jeichman/Documents/gamsdir/projdir/RODeO/'
dir1 = dir0+'Projects/CEC_proposal_305/Data_files/TXT_files/'
dir2 = dir0+'Projects/CEC_proposal_305/Output/'

storage_power_increment = 1000      # (MW) Size of storage installed each iteration



"""
0. Load renewable and system electrical load profiles
"""
ElecLoad = pd.read_csv(dir1 + 'Additional_load_CAISO2019_hourly.csv',sep=',')
ElecLoadSUM = sum(ElecLoad['Station1'])
Ren_PV = pd.read_csv(dir1 + 'renewable_profiles_PV_CAISO2019_hourly.csv',sep=',')               # Normalized (12,411.28 MW for 2019)
Ren_PV.drop(Ren_PV.iloc[:, 2:21], inplace = True, axis = 1)
Ren_WIND = pd.read_csv(dir1 + 'renewable_profiles_WIND_CAISO2019_hourly.csv',sep=',')           # Normalized (5,682.233 MW for 2019)
Ren_WIND.drop(Ren_WIND.iloc[:, 2:21], inplace = True, axis = 1)
Ren_OTHERplusHYDRO = pd.read_csv(dir1 + 'renewable_profiles_OTHERplusHYDRO_CAISO2019_hourly.csv',sep=',')
Ren_OTHERplusHYDRO.drop(Ren_OTHERplusHYDRO.iloc[:, 2:21], inplace = True, axis = 1)

Netload = ElecLoad.copy()
Netload.drop(Netload.iloc[:, 1:2], inplace = True, axis = 1)
Netload['Load1'] = ElecLoad['Station1']-Ren_OTHERplusHYDRO['1']-Ren_PV['1']*12411.28-Ren_WIND['1']*5682.233     # Values for PV and Wind represent 2019 max power levels in CAISO data

Netload_overZERO = Netload.copy()
Netload_overZERO['Load1'] = Netload['Load1']-min(Netload['Load1'])     # either makes all values positive or brings to zero
Netload_overZERO.to_csv(dir1+'Netload_1.csv', sep=',', header=True,index=False)   # Output netload with min moved to zero


RTE=[0.5,0.6,0.7,0.8]
stor_dur = [100,70,40,10]
RPS = pd.DataFrame()
#index=range(1,8761), columns=[list(range(1,len(RTE)+1))])

for i1 in range(1,len(RTE)+1):
    """
    1. Prepare information to run GAMS
    """
    # Calculate the hours the netload is less than or equal to zero (this assumption limits operation of storage and is non-optimal for systems with <100% RPS)
    Max_input_cap = Netload.copy()
    Max_input_cap['Load1'] = 0
    int1 = Netload['Load'+str(i1)][Netload['Load'+str(i1)] <= 0].values.tolist()
    int2 = list(map(max, zip(int1, [-storage_power_increment]*len(int1))))
    int3 = [i/-storage_power_increment for i in int2]
    Max_input_cap['Load1'][Netload['Load'+str(i1)] <= 0] = int3
    Max_input_cap=Max_input_cap.rename(columns={"Load1": "1"})
    Max_input_cap=Max_input_cap.assign(**{'2': 0,'3': 0,'4': 0,'5': 0,'6': 0,'7': 0,'8': 0,'9': 0,'10': 0,'11': 0,'12': 0,'13': 0,'14': 0,'15': 0,'16': 0,'17': 0,'18': 0,'19': 0,'20': 0})
    Max_input_cap.to_csv(dir1+'Input_cap/Max_input_cap_'+str(i1)+'.csv', sep=',', header=True,index=False)   
    
    
    
    """
    2. Prepare batch, run RODeO and collect results
    """
    # Loop through storage scenarios
    Scenario1 = 'iter1_RTE'+str(round(RTE[i1-1]*100))+'_'+str(storage_power_increment)+'MW_'+str(stor_dur[i1-1])+'hr'
   
    # Create and save batch file
    txt1 = '"C:\\GAMS\\win64\\27.3\\gams.exe" C:\\Users\\jeichman\\Documents\\gamsdir\\projdir\\RODeO\\Storage_dispatch_v22_1 license=C:\\GAMS\\win64\\27.3\\gamslice.txt'
    scenario_name = ' --file_name_instance='+Scenario1
    load_prof = ' --load_prof_instance=Additional_load_none_hourly'
    ren_prof = ' --ren_prof_instance=renewable_profiles_PV_hourly'
    ren_cap = ' --Renewable_MW_instance=0'
    energy_price = ' --energy_purchase_price_inst=Netload_'+str(i1)+' --energy_sale_price_inst=Netload_'+str(i1)
    max_input_entry = ' --Max_input_prof_inst=Max_input_cap_'+str(i1)
    capacity_values = ' --input_cap_instance='+str(storage_power_increment)+' --output_cap_instance='+str(storage_power_increment)
    efficiency = ' --input_efficiency_inst='+str(round(math.sqrt(RTE[i1-1]),6))+' --output_efficiency_inst='+str(round(math.sqrt(RTE[i1-1]),6))
    storage_cap = ' --storage_cap_instance='+str(stor_dur[i1-1])
    out_dir = ' --outdir=C:\\Users\\jeichman\\Documents\\gamsdir\\projdir\\RODeO\\Projects\\CEC_proposal_305\\Output'
    in_dir = ' --indir=C:\\Users\\jeichman\\Documents\\gamsdir\\projdir\\RODeO\\Projects\\CEC_proposal_305\\Data_files\\TXT_files'
    batch_string = txt1+scenario_name+ren_prof+load_prof+energy_price+max_input_entry+capacity_values+efficiency+storage_cap+ren_cap+out_dir+in_dir
    with open(os.path.join(dir0, 'Output_batch.bat'), 'w') as OPATH:
        OPATH.writelines(batch_string)
    
    # Run batch file
    os.startfile(r'C:\Users\jeichman\Documents\gamsdir\projdir\RODeO\Output_batch.bat')
         
    # Pull storage operation output files
    Results_data = pd.read_csv(dir2+'Storage_dispatch_results_'+Scenario1+'.csv',sep=',',header=25,skiprows=[24])
    Results_data['Power (MW)'] = Results_data['Output Power (MW)']-Results_data['Input Power (MW)'] 
    Results_data.drop(Results_data.iloc[:, 1:18], inplace = True, axis = 1)
    
    Netload_upd = Netload['Load'+str(i1)]-Results_data['Power (MW)']
    RPS_original = 1-sum(Netload['Load'+str(i1)][Netload['Load'+str(i1)]>0])/ElecLoadSUM
    RPS_storage = 1-sum(Netload_upd[Netload_upd>0])/(ElecLoadSUM-sum(Results_data['Power (MW)'][Results_data['Power (MW)']<0]))
    # RPS[str(i1)]
    
    Curtailment_original = -sum(Netload['Load'+str(i1)][Netload['Load'+str(i1)]<0])
    Curtailment_storage = -sum(Netload_upd[Netload_upd<0])
    
    
"""
3. Determine RPS and curtailment for Wind and PV
"""

Netload['Load'+str(i1)]








"""
4. Calculate LCOE for all technologies
"""

# Find least cost option to increase RPS



#Input properties are mostly based on AEO for R&D  (pulled from 2019 ATB)
Inflation_rate = 0.025
Interest_rate_nominal = 0.0373
Rate_of_return_on_equity_nominal = 0.0903
Debt_fraction = 0.6
Combined_tax_rate = 0.2574

# Properties by technology         PV     Wind                              
Capital_recovery_period         = [30    ,30    ]
Project_finance_factor          = [1.046 ,1.045 ]
Construction_finance_factor     = [1.014 ,1.022 ]
Overnight_capital_cost          = [1096  ,1575  ]
Grid_connection_cost            = [0     ,0     ]
FOM_expenses                    = [20    ,44    ]
Annual_capacity_factor          = [0.20  ,0.35  ]   # Caclulated
VOM_expenses                    = [0     ,0     ]
Present_value_of_depreciation   = [0.868 ,0.869 ]

#Nominal to Real
# (1+Property)/(1+Inflation_rate)-1

# Values can be either real or nominal
WACC_nominal = Debt_fraction*Interest_rate_nominal*(1-Combined_tax_rate)+(1-Debt_fraction)*Rate_of_return_on_equity_nominal    
WACC_real = (1+WACC_nominal)/(1+Inflation_rate)-1  
CRF_nominal =  WACC_nominal / (1 - (1 / (1 + WACC_nominal)**Capital_recovery_period))
CRF_real =  WACC_real / (1 - (1 / (1 + WACC_real)**Capital_recovery_period))

LCOE = (CRF_real * Project_finance_factor * Construction_finance_factor * (Overnight_capital_cost * 1 + Grid_connection_cost) + FOM_expenses) * 1000 / (Annual_capacity_factor*8760) + VOM_expenses
LCOE_overnight = (((CRF_real * Overnight_capital_cost * (1-Combined_tax_rate * Present_value_of_depreciation))/(Annual_capacity_factor*8760*(1-Combined_tax_rate)) + FOM_expenses/(Annual_capacity_factor*8760))*1000) + VOM_expenses

# Annualization = (0.07+(0.07/((1+0.07)**20-1)))


        



"""
Scratch
from subprocess import PIPE, run

command = ['echo', 'hello']
result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
print(result.returncode, result.stdout, result.stderr)

subprocess.call([r'C:\\Users\\jeichman\\Documents\\gamsdir\\projdir\\RODeO\\Test.bat'])
subprocess.call([r'C:\Users\jeichman\Documents\gamsdir\projdir\RODeO\Test.bat'])

os.system(batch_string)

subprocess.run(['C:\\Users\\jeichman\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Accessories\\Notepad.exe', dir1+'Test.txt'])

subprocess.Popen(['C:\Program Files\Mozilla Firefox\firefox.exe'])

p1 = subprocess.run('ls', shell=True, capture_output=True, text=True)
print(p1.args)
print(p1.stdout)
print(p1.returncode)
"""