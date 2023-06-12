"""
Author: Jamie Kee
Added to HOPP by: Jared Thomas
Note: ANL costs are in 2018 dollars
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import os

bar2MPa = 0.1
mm2in = 0.0393701

def run_pipe_analysis(L,m_dot,p_inlet,p_outlet,depth, risers=1, data_location=os.path.abspath(os.path.dirname(__file__)+"/data_tables")):
    '''
        This function calculates the cheapest grade, diameter, thickness, subject to ASME B31.12 and .8
    '''
    p_inlet_MPa = p_inlet*bar2MPa
    F = 0.72 # Design option B class 1 - 2011 ASME B31.12 Table  PL-3.7.1.2
    E = 1.0 # Long. Weld Factor: Seamless (Table IX-3B)
    T_derating = 1 #2020 ASME B31.8 Table A841.1.8-1 for T<250F, 121C

    riser = True    #This is a flag for the ASMEB31.8 stress design, if not including risers, then this can be set to false
    total_L = L*(1+0.05) + risers*depth/1000 #km #Assuming 5% extra length and 1 riser. Will need two risers for turbine to central platform

    #   Import mechanical props and pipe thicknesses (remove A,B ,and A25 since no costing data)
    yield_strengths = pd.read_csv(os.path.join(data_location, 'steel_mechanical_props.csv'),index_col = None,header = 0)
    yield_strengths = yield_strengths.loc[~yield_strengths['Grade'].isin(['A','B','A25'])].reset_index()
    schedules_all = pd.read_csv(os.path.join(data_location, 'pipe_dimensions_metric.csv'),index_col = None,header = 0)
    steel_costs_kg = pd.read_csv(os.path.join(data_location, 'steel_costs_per_kg.csv'),index_col = None,header = 0)

    #   First get the minimum diameter required to achieve the outlet pressure for given length and m_dot
    min_diam_mm = get_min_diameter_of_pipe(L,m_dot,p_inlet,p_outlet)
    #   Filter for diameters larger than min diam required
    schedules_spec = schedules_all.loc[schedules_all['DN']>=(min_diam_mm)]

    #   Gather the grades, diameters, and schedules to loop thru
    grades = yield_strengths['Grade'].values
    diams = schedules_spec['Outer diameter [mm]'].values
    schds = schedules_spec.loc[:,~schedules_spec.columns.isin(['DN','Outer diameter [mm]'])].columns
    viable_types = []
 
    #   Loop thru grades
    for grade in grades:
        #   Get SMYS and SMTS for the specific grade
        SMYS = yield_strengths.loc[yield_strengths['Grade']==grade,'SMYS [Mpa]'].iat[0]
        SMTS = yield_strengths.loc[yield_strengths['Grade']==grade,'SMTS [Mpa]'].iat[0]
        #   Loop thru diameters
        for diam in diams:
            diam_row = schedules_spec.loc[schedules_spec['Outer diameter [mm]']==diam]
            #   Loop thru scheudles (which give the thickness)
            for schd in schds:
                thickness = diam_row[schd].iat[0]

                #Check if thickness satisfies ASME B31.12
                mat_perf_factor = get_mat_factor(SMYS,SMTS,p_inlet*bar2MPa)
                t_ASME = p_inlet_MPa*diam/(2*SMYS*F*E*mat_perf_factor)
                if thickness<t_ASME:
                    continue

                #Check if satifies ASME B31.8
                if not checkASMEB318(SMYS,diam,thickness,riser,depth,p_inlet,T_derating):
                    continue

                #Add qualified pipes to saved answers:
                inner_diam = diam-2*thickness
                viable_types.append([grade,diam,inner_diam,schd,thickness])
            
    viable_types_df = pd.DataFrame(viable_types,columns=['Grade','Outer diameter (mm)','Inner Diameter (mm)','Schedule','Thickness (mm)']).dropna()

    #   Calculate material, labor, row, and misc costs
    viable_types_df = get_mat_costs(viable_types_df,total_L,steel_costs_kg)
    viable_types_df = get_anl_costs(viable_types_df,total_L)
    min_row = viable_types_df.sort_values(by='total capital cost [$]').iloc[:1].reset_index()
    return min_row
 
def get_mat_factor(SMYS,SMTS,design_pressure):
    '''
        Determine the material performance factor ASMEB31.12. 
        Dependent on the SMYS and SMTS. 
        Defaulted to 1 if not within parameters - This may not be a good assumption
    '''
    dp_array = np.array([6.8948,13.7895,15.685,16.5474,17.9264,19.3053,20.6843]) # MPa
    if SMYS <= 358.528 or SMTS <= 455.054:
        h_f_array = np.array([1,1,0.954,0.91,0.88,0.84,0.78])
    elif SMYS <=413.686 and (SMTS > 455.054 and SMTS <=517.107):
        h_f_array = np.array([0.874,0.874,0.834,0.796,0.77,0.734,0.682])
    elif SMYS <= 482.633 and (SMTS > 517.107 and SMTS <= 565.370):
        h_f_array = np.array([0.776,0.776,0.742,0.706,0.684,0.652,0.606])
    elif SMYS <= 551.581 and (SMTS >565.370 and SMTS <= 620.528):
        h_f_array = np.array([0.694,0.694,0.662,0.632,0.61,0.584,0.542])   
    else:    
        return 1
    mat_perf_factor = np.interp(design_pressure,dp_array,h_f_array)
    return mat_perf_factor

def checkASMEB318(SMYS,diam,thickness,riser,depth,p_inlet,T_derating):
    '''
        Determine if pipe parameters satisfy hoop and longitudinal stress requirements
    '''
    
    # Hoop Stress - 2020 ASME B31.8 Table A842.2.2-1
    F1 = 0.50 if riser else 0.72 

    #   Hoop stress (MPa) - 2020 ASME B31.8 section A842.2.2.2 eqn (1)
    #   This is the maximum value for S_h
    #   Sh <= F1*SMYS*T_derating
    S_h_check =  F1*SMYS*T_derating

    #   Hoop stress (MPa)
    Pa2bar = 0.00001
    rho_water = 1000 #kg/m3
    p_hydrostatic = rho_water*9.81*depth*Pa2bar # bar
    dP = (p_inlet-p_hydrostatic)*bar2MPa    # MPa
    S_h = dP*(diam-(thickness if diam/thickness>=30 else 0))/(2000*thickness)
    if S_h>=S_h_check:
        return False

    #   Longitudinal stress (MPa)
    S_L_check = 0.8*SMYS #2020 ASME B31.8 Table A842.2.2-1. Same for riser and pipe
    S_L = p_inlet*bar2MPa*(diam-2*thickness)/(4*thickness)
    if S_L>S_L_check:
        return False

    S_combined_check = 0.9*SMYS #2020 ASME B31.8 Table A842.2.2-1. Same for riser and pipe
    #   Torsional stress?? Under what applied torque? Not sure what to do for this.

    return True

def get_anl_costs(costs,total_L):
    labor_coef = [95295,0.53848,0.03070]
    misc_coef = [19211,0.14178,0.04697]
    row_coef = [72634,1.07566,0.05284]
    mat_coef = [5605,0.41642,0.06441] # Don't need this anymore since material cost is coming from Savoy numbers

    L_mi = total_L*0.621371

    # costs['mat cost anl [$]'] = costs.apply(lambda x: (mat_coef[0]*((x['Outer diameter (mm)']*mm2in)**mat_coef[1])/L_mi**mat_coef[2])*(x['Outer diameter (mm)']*mm2in*L_mi),axis=1)
    costs['labor cost [$]'] = costs.apply(lambda x: (labor_coef[0]/((x['Outer diameter (mm)']*mm2in)**labor_coef[1])*L_mi**labor_coef[2])*(x['Outer diameter (mm)']*mm2in*L_mi),axis=1)
    costs['misc cost [$]'] = costs.apply(lambda x: (misc_coef[0]/((x['Outer diameter (mm)']*mm2in)**misc_coef[1])/L_mi**misc_coef[2])*(x['Outer diameter (mm)']*mm2in*L_mi),axis=1)
    costs['ROW cost [$]'] = costs.apply(lambda x: (row_coef[0]/((x['Outer diameter (mm)']*mm2in)**row_coef[1])*L_mi**row_coef[2])*(x['Outer diameter (mm)']*mm2in*L_mi),axis=1)

    costs['total capital cost [$]'] = costs[['mat cost [$]','labor cost [$]','misc cost [$]','ROW cost [$]']].sum(axis=1)

    costs['annual operating cost [$]'] = 0.0117*costs['total capital cost [$]'] # https://doi.org/10.1016/j.esr.2021.100658

    return costs

def get_mat_costs(schedules_spec,total_L,steel_costs_kg):
    '''
        Calculates the material cost based on $/kg from Savoy for each grade
    '''
    rho_steel = 7840 #kg/m3
    mm2m = 0.001
    km2m = 1000
    L_m = total_L*km2m
    schedules_spec['volume [m3]'] = schedules_spec.apply(lambda x: np.pi*(x['Outer diameter (mm)']**2-(x['Outer diameter (mm)']-x['Thickness (mm)']*2)**2)*mm2m**2/4*L_m,axis=1)
    schedules_spec['weight [kg]'] = schedules_spec['volume [m3]']*rho_steel
    schedules_spec['mat cost [$]'] = schedules_spec.apply(lambda x:x['weight [kg]']*steel_costs_kg.loc[steel_costs_kg['Grade']==x['Grade'],'Price [$/kg]'].iat[0],axis=1)

    return schedules_spec

def get_min_diameter_of_pipe(L:float,m_dot:float,p_inlet:float,p_outlet:float):
    '''
    Overview:
    ---------
        This function returns the diameter of a pipe for a given length,flow rate, and pressure boundaries

    Parameters:
    -----------
        L : float - Length of pipeline [km]
        m_dot : float = Mass flow rate [kg/s]
        p_inlet : float = Pressure at inlet of pipe [bar]
        p_outlet : float = Pressure at outlet of pipe [bar]

    Returns:
    --------
        diameter_mm : float - Diameter of pipe [mm]

    '''

    def momentum_bal(x,*params):
        '''
        Overview:
        ---------
            Residual equation for momentum balance

        Parameters:
        -----------
            x : list[float] - Guess values. Index 0 is the diameter. The remaining are the pressures
            params : tuple(float) - Parameters in the order m_dot,p_inlet,p_outlet,L_km,ZRT,f,nodes

        Returns:
        --------
            residual : list[float] - Residual values of momentum equation
        '''
        
        #   Useful conversions
        bar2Pa = 100000     #   Convert bar to Pa 
        km2m = 1000         #   Convert km to m

        # Unload params into local varibles
        m_dot,p_inlet,p_outlet,L_km,ZRT,f,nodes = params 

        #   Extract guess values
        D = x[0]                                                    #   Diameter of pipe [m]
        p_bar = x[1:]                                               #   Pressure inside the pipe [bar]

        #   Geometric parameters
        A = np.pi/4*D**2                                            #   Calculate area of pipe [m2]
        L = L_km * km2m                                             #   Length of pipe [m]
        dz = L/nodes                                                #   Uniform spacing of nodes [m]

        #   Pressure, density, and velocity for each node
        p = np.concatenate(([p_inlet],p_bar,[p_outlet]))*bar2Pa     #   Pressure array with boundary conditions [Pa]
        rho = p/ZRT                                                 #   Density of gas [kg/m3]
        rho_avg = np.mean([rho[0:-1],rho[1:]],axis=0)               #   Average density in each node [kg/m3]
        v = m_dot/rho/A                                             #   Velocity of gas [m/s]
        v_avg = np.mean([v[0:-1],v[1:]],axis=0)                     #   Average density in each node [m/s]

        #   Momentum balance terms
        dpdz = (p[1:]-p[0:-1])/dz                                   #   Pressure drop [Pa/m]
        tau = 0.5*f*rho_avg*v_avg**2                                #   Viscous drag [Pa]
        pududz = rho_avg*v_avg * (v[1:]-v[0:-1])/dz                 #   Velocity change effect [Pa/m]

        #   Momentum balance
        residual = -dpdz - tau/D - pududz 

        return residual

    #   Conversions
    m2mm = 1000

    #   Various inputs
    f = 0.01                # Friction factor
    Z = 0.9                 # Compressibility
    R = 8.314/(2.016/1000)  # J/kg-K For hydrogen
    T = 15+273              # Temperature [K]
    nodes = 50              # Number of nodes to discretize with
    params = (m_dot,p_inlet,p_outlet,L,Z*R*T,f,nodes)

    #   Generate guess values
    d_init = 0.01           # Diameter [m]
    p_init = np.linspace(p_inlet,p_outlet,nodes-1) #[bar]
    x_init = np.concatenate(([d_init],p_init))

    #   Solve for the diameter
    diameter_m = fsolve(momentum_bal,x_init,args = params)[0]
    diameter_mm = diameter_m * m2mm

    return diameter_mm

if __name__ == '__main__':
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]
    costs = run_pipe_analysis(L,m_dot,p_inlet,p_outlet,depth)
    
    for col in costs.columns:
        print(col, costs[col][0])