
from scipy.optimize import fsolve
import numpy as np
import pandas as pd

def get_diameter_of_pipe(L:float,m_dot:float,p_inlet:float,p_outlet:float):
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
        diameter_in : float - Diameter of pipe [inches]

    '''

    #   Conversions
    m2in = 39.3701

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
    diameter_in = diameter_m * m2in

    return diameter_in


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
    m_dot,p_inlet,p_outlet,L,ZRT,f,nodes = params 

    #   Extract guess values
    D = x[0]                                                    #   Diameter of pipe [m]
    p_bar = x[1:]                                               #   Pressure inside the pipe [bar]

    #   Geometric parameters
    A = np.pi/4*D**2                                            #   Calculate area of pipe [m2]
    #L = L_km * km2m                                             #   Length of pipe [m]
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

def get_dn_schedule_of_pipe(pipe_info_dir,grade,design_option,location_class,joint_factor,pipe_inner_diameter,design_pressure_asme):
    # Read in yield strengths for API 5L steels
    yield_strengths = pd.read_csv(pipe_info_dir+'steel_mechanical_props.csv',index_col = None,header = 0)

    # Isolate the yield strength of the grade selecdted
    yield_strength = yield_strengths.loc[yield_strengths['Grade']==grade,'SMYS [Mpa]'].to_list()[0]

    # Assign design factor based on design option
    if design_option == 'a' or design_option == 'A':
        design_factor_list = [0.5,0.5,0.5,0.4]
    elif design_option == 'b' or design_option == 'B':
        design_factor_list = [0.72,0.60,0.50,0.40]
    else:
        design_factor = [0.5,0.5,0.5,0.4]
        print('Error. Invalid design option. Option set to a as default. Please enter a, A, b, or B for design option')

    # Determine design factor based on location class
    design_factor = design_factor_list[location_class - 1]

    # Design pressure array for ASME B31.12 Table for material performance factor
    dp_array = np.array([6.8948,13.7895,15.685,16.5474,17.9264,19.3053,20.6843])
    
    # Read in possible pipe schedules for different nominal diameters in metric units
    pipe_schedules = pd.read_csv(pipe_info_dir+'pipe_dimensions_metric.csv',index_col = None,header = 0)
    
    # Isolate the nominal diameter options
    dn_options = pipe_schedules['DN'].to_list()
    
    for dn in dn_options:
    
        #dn = dn_options[9]
            
        # Clean up table leaving what is actually needed
        pipe_schedules_DN = pipe_schedules.loc[(pipe_schedules['DN'] == dn)].drop(labels = ['DN','Outer diameter [mm]'], axis = 1).transpose().reset_index()
        pipe_schedules_DN = pipe_schedules_DN.rename(columns = {pipe_schedules_DN.columns[0]:'Schedule',pipe_schedules_DN.columns[1]:'Wall thickness [mm]'}).dropna()
        pipe_schedules_DN = pipe_schedules_DN.reset_index().drop(labels = ['index'],axis=1)
        
        pipe_outer_diameter = pipe_schedules.loc[(pipe_schedules['DN']==dn),'Outer diameter [mm]'].values[0]
        
        if pipe_outer_diameter <= pipe_inner_diameter:
            continue
        else:
        
            if design_option == 'b' or design_option == 'B':
                mat_perf_factor = 1
            else: 
                if yield_strength <= 358.528:
                    h_f_array = np.array([1,1,0.954,0.91,0.88,0.84,0.78])
                elif yield_strength > 358.528 and yield_strength <=413.686:
                    h_f_array = np.array([0.874,0.874,0.834,0.796,0.77,0.734,0.682])
                elif yield_strength > 413.686 and yield_strength <= 482.633:
                    h_f_array = np.array([0.776,0.776,0.742,0.706,0.684,0.652,0.606])
                elif yield_strength > 482.633 and yield_strength<= 551.581:
                    h_f_array = np.array([0.694,0.694,0.662,0.632,0.61,0.584,0.542]) 
                mat_perf_factor = np.interp(design_pressure_asme,dp_array,h_f_array)
            #thickness.append(design_pressure_asme*diam_outer[0]/(2*yield_strengths.loc[k,'SMYS [Mpa]']*design_factor*joint_factor*mat_perf_factor))
            thickness = design_pressure_asme*pipe_outer_diameter/(2*yield_strength*design_factor*joint_factor*mat_perf_factor)
            if max(pipe_schedules_DN['Wall thickness [mm]']) >= thickness:
                schedule_minviable=pipe_schedules_DN.loc[pipe_schedules_DN["Wall thickness [mm]"] == min(pipe_schedules_DN.loc[pipe_schedules_DN['Wall thickness [mm]'] >= thickness,'Wall thickness [mm]']),'Schedule'].to_list()[0]
                thickness_of_schedule = pipe_schedules_DN.loc[pipe_schedules_DN['Schedule']==schedule_minviable,'Wall thickness [mm]'].to_list()[0]
            else:
                continue
                
            inner_diameter_min_viable_schedule = pipe_outer_diameter-2*thickness_of_schedule
            
            if inner_diameter_min_viable_schedule > pipe_inner_diameter:
                break
    
    
            
    return(dn,pipe_outer_diameter,schedule_minviable,thickness_of_schedule)


# if __name__ == "__main__":
#     L = 8                   # Length [km]
#     m_dot = 0.12            # Mass flow rate [kg/s]
#     p_inlet = 30            # Inlet pressure [bar]
#     p_outlet = 10           # Outlet pressure [bar]
#     min_diam_in = get_diameter_of_pipe(L,m_dot,p_inlet,p_outlet)
#     print(f'The minimum diameter is {round(min_diam_in,3)} inches')