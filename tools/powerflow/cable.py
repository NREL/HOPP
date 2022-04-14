"""
Properties of underground cables and overhead lines for plant.

M. Sinner 1/13/19
"""

import numpy as np


class Cable():
    """
    Parent class to hold all cable types.
    """
    def __init__(self, R_per_m, L_per_m, Cs_per_m):
        self.resistance = R_per_m
        self.inductance = L_per_m
        self.shunt_capacitance = Cs_per_m
    
    def another_function(self, x):
        # This is a placeholder.
        return x

class CableByProperties(Cable):
    """
    User-defined cable type based on direct knowledge of system
    """
    def __init__(self, R_per_m, L_per_m, Cs_per_m):
        Cable.__init__(self, R_per_m, L_per_m, Cs_per_m)

class CableByGeometry(Cable):
    """
    User-defined cable type.
    """
    def __init__(self, cross_sectional_area,
                       line_type='underground',
                       phase_separation_distance=0.1, # defaulting to 100mm
                       resistivity=1.724e-8, # Copper at 20C
                       skin_correction_factor=1.02,
                       relative_permittivity=2.3,
                       relative_permeability=1):
        """
        Inputs:
            cross_sectional_area - float - representative 
                                           cross-sectional area of
                                           single line phase [m^2]
            line_type - str - underground cables or overhead 
                              transmission lines
                              ('underground', 'overhead')
            phase_separation_distance - float - distance separating 
                                                phase lines [m]
            resistivity - float - electrical resistivity of line 
                                  material [ohm-m]
            relative_permittivity - float - relative permittivity of 
                                            cable insulation [-]
            relative_permeability - float - relative permeability (of
                                            cable insulation?) [-]
        """
    
        epsilon_0 = 8.85*10**(-12) # Permittivity of free space (elec. const.)
        mu_0=4*np.pi*10**(-7) # Permeability of free space (mag. const.)
        radius = np.sqrt(cross_sectional_area/np.pi)
        geometric_mean_radius = np.exp(-1/4)*radius

        R_per_m = (resistivity/cross_sectional_area) * skin_correction_factor

        #TODO: add temperature correction

        L_per_m = (mu_0*relative_permeability) / (2*np.pi) * \
                  np.log(phase_separation_distance/ geometric_mean_radius)
        
        # Capacitance (per meter) due to the presense of other phase cables
        Cpm_3phase = 2*np.pi*(epsilon_0*relative_permittivity)/ \
                     np.log(phase_separation_distance/radius)
        
        if line_type is 'underground': 
            # Capacitance (per meter) due to grounding of the sheath
            # TODO: verify ground capacitance (LARGE CONTRIBUTION (~1.2))
            Cpm_ground = 2*np.pi*(epsilon_0*relative_permittivity)/ \
                         np.log((phase_separation_distance/2)/radius)

        elif line_type is 'overhead':
            Cpm_ground = 0 # Short lines, little ground contribution

        Cs_per_m = Cpm_3phase + Cpm_ground
        
        Cable.__init__(self, R_per_m, L_per_m, Cs_per_m)
    
class CableFromSpec(Cable):
    """
    Cable from manufacturer's specifications.
    """
    
    def __init__(self, name, filename='cable_specifications.json'):

        with open('cable_specifications.json', 'r') as f:
            all_cables = json.load(f)
        
        cable_spec = all_cables['name'==name]

        Cable.__init__(self, cable_spec['R_per_m'], cable_spec['L_per_m'], 
                             cable_spec['Cs_per_m'])