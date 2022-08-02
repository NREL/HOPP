## Pipeline cost model 
'''
    This code section was references pipeline costs from an H2A Report (NREL and Nexant)
    DE-FG36-05GO15032
            
    Inputs
    =======
       distance to load (miles), pipe diameter (inches)
    Returns
    =======
        CapEx of pipeline($USD)
'''        
import math
import numpy as np

def calcPipelineLength(dist_to_shore, dist_inland, site_depth):
    # Calculate pipeline distance to total length of pipe
    # TODO: Include bends, drops, connections)
    # Inputs: dist_to_shore (km), dist_inland (km), site_depth (m) 
    # assume 90deg bends in pipeline to add all lengths 
    site_depth_km = site_depth / 1000.0 # m to km

    total_pipeline_length = dist_to_shore + site_depth_km + dist_inland

    return total_pipeline_length 

#def pipelineLosses(pipeline_length):
#    h2_leakage = 0.05/100 * pipe_length 
    
#    return h2_leakage

def calcPipelineDiam(flow_rate_kg_per_hr, inlet_pressure_bar, outlet_pressure_bar, length_m):
    """
        Calculate pipe diameter based on a General Flow Equation (Darcy eq.)
        Inputs
        ======
         gas flow rate (kg/hr)
         inlet pressure (bar)
         outlet pressure (bar)
         length (m)
         Output
         ======
         pipe diameter (mm)

    """
    rho = 0.08375 # STP H2 density, kg/m**3
    f = 0.03 # pipe friction factor from a Moody Diagram assume Re ~10^4 and relative roughness ~0.002 
    # TODO: Find a more accurate number for relative roughness and better Reynolds estimate 
    # TODO: Calculate density as a function of temperature (ambient?)

    flow_rate_kg_per_sec = flow_rate_kg_per_hr / 3600.0
    Q = flow_rate_kg_per_sec / rho  # m**3/s  
    L = length_m # km to m
    delP = (inlet_pressure_bar - outlet_pressure_bar) * 100000 # bars to Pa (N/m**2)
    
    # General Flow Equation
    pipe_diam_mm = 1000.0 * ((8 * rho * f * L * Q**2)/(delP * np.pi**2))**(1/5)

    return pipe_diam_mm

def pipeline_cost(pipeline_length_km, pipe_diam_mm, offshore_bool: bool):
    # Calculate pipeline cost based on length and diameter
    # Reference: H2A Hydrogen Delivery Infrastructure Analysis Models and Concentional Pathway Options Analysis Results
    #            DE-FG36-05GO15032 (May 2008) Section 2.2.2.1: Transmission Pipeline Costs
    #            Equation Credit: Nathan parker, Using Natural Gas Transmission Pipeline Cost to Estimate Hydrogen 
    #                             Pipeline Cost. 2004-12-01

    # Natural Gas to H2 cost factor 
    ng_to_h2_factor = 1.1 
    
    # Unit conversions: metric to imperial
    pipeline_length_miles = 0.621371 * pipeline_length_km
    pipe_diam_in = pipe_diam_mm / 25.4 
       
    # CapEx costs based on above reference
    mat_cost   = (330.5 * pipe_diam_in**2 + 687 * pipe_diam_in + 26960) * pipeline_length_miles \
                 + 35000
    
    misc_cost  = (8417 * pipe_diam_in + 7324) * pipeline_length_miles \
                 + 95000
    
    labor_cost = (343 * pipe_diam_in**2 + 2047 * pipe_diam_in + 170013) * pipeline_length_miles \
                 + 185000

    # Least applicable when considering offshore?
    row_cost   = (577 * pipe_diam_in + 29788) * pipeline_length_miles \
                 + 40000 
            
    total_lng_cost = mat_cost + misc_cost + labor_cost + row_cost
    total_h2_cost = total_lng_cost * ng_to_h2_factor 

    # Opex Factor based on on-land or undersea
    # https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub 
    
    # Install Factor based on on-land or undersea 
    # Hydrogen Insite paper: https://hydrogencouncil.com/wp-content/uploads/2021/02/Hydrogen-Insights-2021-Report.pdf 

    # Offshore
    if offshore_bool:
        install_opex_fact = 7.0/100 # 7% of CapEx
        install_capex_fact = 2.3 # offshore cost factor 1.3-2.3 (Hydrogen insight paper)
    # On Land
    else:
        install_opex_fact = 4.0/100 # 4% of CapEx 
        install_capex_fact = 1.0 

    capex_pipe = install_capex_fact * total_h2_cost

    opex_pipe = install_opex_fact * capex_pipe
    
    # Quick sanity check from OSW-H2 draft paper. Table 8. Includes material + installation cost
    oswh2_pipe = 26000.0 * pipeline_length_km + 686000.0 * pipeline_length_km

    return capex_pipe, opex_pipe, oswh2_pipe

# Test sections 
if __name__ == '__main__': 
    print('### Nexant Pipeline Cost Model Test ###')
    plantcap = 997.0 # MW
    life_span = 30.0 # years 
    pipe_length_km = calcPipelineLength(80, 5, 45) # 80km at sea, 5km inland, 45m depth 
    p_in = 1000.0 / 14.5 # psi to bars 
    p_out = 700.0 / 14.5 # psi to bars
    flow_rate = 14000.0 # kg per hr

    pipe_diam_mm = calcPipelineDiam(flow_rate, p_in, p_out, pipe_length_km*1000.0)
    pipe_capex, pipe_opex, oswh2_pipe = pipeline_cost(pipe_length_km, pipe_diam_mm, True)
    print('Wind Plant Design: {} MW for {} years'.format(plantcap,life_span))
    print('Pipeline length (mi): ', pipe_length_km * 0.621371)
    print('Pipeline diameter (in): ', pipe_diam_mm / 25.4)
    print('Pipeline CapEx Cost ($USD/kW):', pipe_capex / (plantcap * 1000.0) )
    print('OSWH2 pipe CapEx ($USD/kW): ', oswh2_pipe / (plantcap * 1000.0))
    print('Pipeline OpEx Cost ($USD/kW*yr):', pipe_opex / (plantcap * 1000.0 * life_span))
  