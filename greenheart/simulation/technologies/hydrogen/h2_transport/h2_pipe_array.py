from greenheart.simulation.technologies.hydrogen.h2_transport.h2_export_pipe import run_pipe_analysis
from numpy import isnan, flip, nansum

"""
Args:
    sections_distance (array[array]): array of arrays where each element of each sub-array holds the horizontal distance in m of a pipe section
    depth (float): depth of the site in m
    p_inlet (float): pipe inlet pressure in bar
    p_outlet (float): pipe outlet pressure in bar
    mass_flow_rate_inlet (float): flow rate at each inlet to the system
Returns:
    capex (float): total capital costs (USD) including labor, materials, and misc
    opex (float): annual operating costs (USD)
"""
def run_pipe_array(sections_distance, depth, p_inlet, p_outlet, mass_flow_rate):
    
    capex = 0
    opex = 0

    # loop over each string
    for i, pipe_string in enumerate(sections_distance):
        
        # initialize values for each string
        m_dot = 0
        p_drop = (p_inlet - p_outlet)/len(pipe_string)
        flow_rates = flip(mass_flow_rate[i])

        # loop over each section
        for j, section_length in enumerate(flip(pipe_string)):

            # nan represents an empty section (no pipe there, but array cannot be ragged)
            if isnan(section_length): continue

            # get mass flow rate for current section
            m_dot += flow_rates[j]

            # get outlet pressure for current section
            p_outlet_section = p_inlet - (j + 1)*p_drop

            # get number of risers for current section
            if j == len(pipe_string)-1:
                risers = 2
            else:
                risers = 1
                
            # get specs and costs for each section
            section_outputs = run_pipe_analysis(section_length, m_dot, p_inlet, p_outlet_section, depth, risers=risers)

            capex += section_outputs["total capital cost [$]"][0]
            opex += section_outputs["annual operating cost [$]"][0]

    return capex, opex

#   Assuming one pipe diameter for the pipeline
def run_pipe_array_const_diam(sections_distance, depth, p_inlet, p_outlet, mass_flow_rate):
    
    capex = 0
    opex = 0

    # loop over each string
    for i, pipe_string in enumerate(sections_distance):
        
        # Calculate maximum flow rate per pipe segment (pipe is sized to largest segment)
        m_dot = max(mass_flow_rate[i])

        #   Add up the length of the segment
        tot_length = nansum(pipe_string)

        #   Assume each full run has 2 risers
        risers = 2
        risers = len(pipe_string)+1 #Wasnt sure on this - is it 1 per turbine + 1 for the storage? - Jamie

        #   get specs and costs for each section
        section_outputs = run_pipe_analysis(tot_length, m_dot, p_inlet, p_outlet, depth, risers=risers)

        capex += section_outputs["total capital cost [$]"][0]
        opex += section_outputs["annual operating cost [$]"][0]

    return capex, opex


if __name__ == "__main__":
    sections_distance = [[2.85105454, 2.016     , 2.016     , 2.016     , 2.016     , 2.016     , 2.016     , 2.016     ],
                        [2.016     , 2.016     , 2.016     , 2.016     , 2.016     ,2.016     , 2.016     , 2.016     ],
                        [2.85105454, 2.016     , 2.016     , 2.016     ,        float("nan"), float("nan"),        float("nan"),        float("nan")]]

    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]

    # capex, opex = run_pipe_array([[L, L], [L, L]], depth, p_inlet, p_outlet, [[m_dot, m_dot], [m_dot, m_dot]])

    # print("CAPEX (USD): ", capex)
    # print("OPEX (USD): ", opex)

    capex, opex = run_pipe_array_const_diam([[L, L], [L, L]], depth, p_inlet, p_outlet, [[m_dot, m_dot], [m_dot, m_dot]])

    print("CAPEX (USD): ", capex)
    print("OPEX (USD): ", opex)
