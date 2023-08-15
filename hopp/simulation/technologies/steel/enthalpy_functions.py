## Functions to calculate enthalpy of different streams 

### Hydrogen enthalpy coefficients
# The value of enthalpy returned is the specific enthalpy in kj/g
'''
Author: Charlie Kiefer
Date: 8/14/23

These functions calculate enthalpies of elements relating to green steel

Sources:
Chase, M.W., Jr. 1998. "NIST-JANAF Themochemical Tables, Fourth Edition." J. Phys. Chem. Ref. Data, Monograph 9 1-1951. doi:https://doi.org/10.18434/T4D303.
    Thermochemistry tab of each element

Inputs:
    T: Temperature of the element (K)

Outputs:
    H_t: Enthalpy of element at given temp (kj/g)
'''

def h2_enthalpy_1(T):
    mol_weight_H2=2.01588 #in grams# T1 and T2 should be in the range of 298-1000 K <br>
    t=T/1000 
    A=33.066718 
    B=-11.363417
    C=11.432816 
    D=-2.772874 
    E=-0.158558 
    F=-9.980797 
    G=172.707974 
    H=0  
    H_t=A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H 
    return H_t 
def h2_enthalpy_2(T):
    mol_weight_H2=2.01588#T in range 1000-2500 K 
    t=T/1000 
    A=18.563083 
    B=12.257357 
    C=-2.859786  
    D=0.268238 
    E=1.977990 
    F=-1.147438 
    G=156.288133 
    H=0 
    H_t=(A*t +(B*t**2)/2 +(C*t**3)/3 + (D*t**4)/4-(E/t)+F-H)/mol_weight_H2 
    return H_t 
### Water enthalpy coefficients

def h2o_enthalpy(T):
    mol_weight_H2O=18.0153 
    t=T/1000 
    A=30.09200  
    B=6.832514 
    C=6.793435 
    D=-2.534480 
    E=0.082139 
    F=-250.8810 
    G=223.3967 
    H=-241.8264 
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_H2O 
    return H_t 

def fe_enthalpy_1(T):#298-1809 K
    mol_weight_fe=55.845 #in grams
    t=T/1000
    A=23.97449
    B=8.367750
    C=0.000277
    D=-0.000088
    E=-0.000005
    F=0.268027
    G=62.06336
    H=7.788015    
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_fe
    return H_t
def fe_enthalpy_2(T):#1809 - 3133K
    mol_weight_fe=55.845 #in grams
    t=T/1000
    A=46.02400
    B=-1.88467*10**(-8)
    C=6.094750*10**(-9)
    D=-6.640301*10**(-10)
    E=-0.8246121*10**(-9)
    F=-10.80543
    G=72.54094
    H=12.39052  
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_fe
    return H_t
def feo_enthalpy(T):
    mol_weight_feo=71.844 #in grams
    t=T/1000
    A=45.75120
    B=18.78553
    C=-5.952201
    D=0.852779
    E=-0.081265
    F=-286.7429
    G=110.3120
    H=-272.0441
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_feo
    return H_t
def al2o3_enthalpy(T):
    mol_weight_al2o3=101.9613 #in grams    
    t=T/1000
    A=106.0880
    B=36.33740
    C=-13.86730
    D=2.141221
    E=-3.133231
    F=-1705.970
    G=153.9350
    H=-1662.300  
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_al2o3
    return H_t
def sio2_enthalpy(T):
    mol_weight_Sio2=60.0843 #in grams
    t=T/1000
    A=58.75
    B=10.279
    C=-0.131384
    D=0.025210
    E=0.025601
    F=-929.3292
    G=105.8092
    H=-910.8568   
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_Sio2
    return H_t
def mgo_enthalpy(T):
    mol_weight_mgo=40.3044 #in grams
    t=T/1000
    A=47.25995
    B=5.681621
    C=-0.872665
    D=0.104300
    E=-1.053955
    F=-619.1316
    G=76.46176
    H=-601.2408
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_mgo
    return H_t
def cao_enthalpy(T):
    mol_weight_cao=56.077#in grams
    t=T/1000
    A=49.95403
    B=4.887916
    C=-0.353056
    D=0.046187
    E=-0.825097
    F=-652.9718
    G=92.56096
    H=-635.0894
    H_t=(A*t +(B*t*t)/2 +(C*t*t*t)/3 + (D*t*t*t*t)/4-(E/t)+F-H)/mol_weight_cao
    return H_t
def ch4_enthalpy_1(T):# T in the range 298-1000 K
    mol_weight_CH4=16.04 # in grams
    t=T/1000
    A=-0.703029
    B=108.4773
    C=-42.52157
    D=5.862788
    E=0.678565
    F=-76.84376
    G=158.7163
    H=-74.87310
    H_t=(A*t +(B*t**2)/2 +(C*t**3)/3 + (D*t**4)/4-(E/t)+F-H)/mol_weight_CH4
    return H_t
def ch4_enthalpy_2(T):# T in range 1300-6000 K
    mol_weight_CH4=16.04 # in grams
    t=T/1000
    A=85.81217
    B=11.26467
    C=-2.114146
    D=0.138190
    E=-26.42221
    F=-153.5327
    G=224.4143
    H=-74.87310
    H_t=(A*t +(B*t**2)/2 +(C*t**3)/3 + (D*t**4)/4-(E/t)+F-H)/mol_weight_CH4
    return H_t
def c_enthalpy(T):# T in the range 298-1000 K
    mol_weight_C=12.017 # in grams
    t=T/1000
    A=21.17510
    B=-0.812428
    C=0.448537
    D=-0.043256
    E=-0.013103
    F=710.3470
    G=183.8734
    H=716.6690
    H_t=(A*t +(B*t**2)/2 +(C*t**3)/3 + (D*t**4)/4-(E/t)+F-H)/mol_weight_C
    return H_t# The value of enthalpy returned is the specific enthalpy in kj/g
   