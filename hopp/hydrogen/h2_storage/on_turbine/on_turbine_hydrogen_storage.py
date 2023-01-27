"""
Author: Cory Frontin
Date: 23 Jan 2023
Institution: National Renewable Energy Lab
Description: This file should handle the cost, sizing, and pressure of on-turbine h2 storage
Sources:
    - [1] Kottenstette 2003 (use their chosen favorite design)
Args:
    - year (int): construction year
    - turbine (dict): contains various information about the turbine, including tower_length, section_diameters, and section_heights
    - others may be added as needed
Returns:(can be from separate functions and/or methods as it makes sense):
    - capex (float): the additional CAPEX in USD for including H2 storage in an offshore wind turbine
    - opex (float): the additional OPEX (annual, fixed) in USD for including H2 storage in an offshore wind turbine
    - mass_empty (float): additional mass (approximate) for added components ignoring stored H2
    - capacity (float): maximum amount of H2 that can be stored in kg
    - pressure (float): storage pressure
    - others may be added as needed
"""

import numpy as np

class PressurizedTower():
    def __init__(self,
                 year: int,
                 turbine: dict):
        
        # key inputs
        self.year= year
        self.turbine= turbine
        
        self.tower_length= turbine['tower_length']
        self.section_diameters= turbine['section_diameters']
        self.section_heights= turbine['section_heights']

        # constants/parameters
        self.d_t_ratio= 320. # Kottenstette 2003
        self.thickness_top= 8.7e-3 # m
        self.thickness_bot= 17.4e-3 # m
        self.ultimate_tensile_strength= 636e6 # Pa, Kottenstette 2003
        self.welded_joint_efficiency= 0.85 # double-welded butt joint w/ spot inspection (ASME)
        self.density_steel= 7817. # kg/m^3
        self.gasconstant_H2= 4126. # J/(kg K)
        self.operating_temp= 20. # degC

        self.costrate_steel= 1.50 # $/kg
        self.costrate_endcap= 2.66 # $/kg

        self.costrate_ladder= 32.80 # $/m
        self.cost_door= 2000 # $
        self.cost_mainframe_extension= 6300 # $
        self.cost_nozzles_manway= 16000 # $
        self.costrate_conduit= 35 # $/m

        # set the operating pressure & get resulting constants
        self.operating_pressure= self.get_operating_pressure()
        self.thickness_increment_const= PressurizedTower.get_thickness_increment_const(self.operating_pressure,
                                                                                       self.ultimate_tensile_strength)
    def run(self):

        # set the operating pressure to ensure changes are here
        self.operating_pressure= self.get_operating_pressure()
        self.thickness_increment_const= PressurizedTower.get_thickness_increment_const(self.operating_pressure,
                                                                                       self.ultimate_tensile_strength)
        
        # get the inner volume and traditional material volume, mass, cost
        self.tower_inner_volume= self.get_tower_inner_volume()
        self.wall_material_volume_trad, self.cap_bot_material_volume_trad, self.cap_top_material_volume_trad= \
                self.get_tower_material_volume(pressure= 0.0)
        self.wall_material_mass_trad= self.wall_material_volume_trad*self.density_steel
        self.wall_material_cost_trad= self.wall_material_mass_trad*self.costrate_steel
        self.cap_material_mass_trad= (self.cap_top_material_volume_trad + self.cap_bot_material_volume_trad)*self.density_steel
        self.cap_material_cost_trad= self.cap_material_mass_trad*self.costrate_steel
        self.nonwall_cost_trad= self.get_nonwall_cost(traditional= True)

        self.wall_material_volume, self.cap_bot_material_volume, self.cap_top_material_volume= \
                self.get_tower_material_volume()
        self.wall_material_mass= self.wall_material_volume*self.density_steel
        self.wall_material_cost= self.wall_material_mass*self.costrate_steel
        self.wall_material_mass= self.wall_material_volume*self.density_steel
        self.cap_material_mass= (self.cap_bot_material_volume + self.cap_top_material_volume)*self.density_steel
        self.cap_material_cost= self.cap_material_mass*self.costrate_endcap
        self.nonwall_cost= self.get_nonwall_cost()

        if True:
            # print the inner volume and pressure-free material properties
            print("operating pressure:", self.operating_pressure)
            print("tower inner volume:", self.tower_inner_volume)
            print()
            print("tower wall material volume (non-pressurized):",
                self.wall_material_volume_trad)
            print("tower wall material mass (non-pressurized):",
                self.wall_material_mass_trad)
            print("tower wall material cost (non-pressurized):",
                self.wall_material_cost_trad)
            print("tower cap material volume (non-pressurized):",
                self.cap_top_material_volume_trad + self.cap_bot_material_volume_trad)
            print("tower cap material mass (non-pressurized):",
                self.cap_material_mass_trad)
            print("tower cap material cost (non-pressurized):",
                self.cap_material_cost_trad)
            print("tower total material cost (non-pressurized):",
                self.wall_material_cost_trad + self.cap_material_cost_trad)
            
            # print the changes to the structure
            print()
            print("tower wall material volume (pressurized):", self.wall_material_volume)
            print("tower wall material mass (pressurized):", self.wall_material_mass)
            print("tower wall material cost (pressurized):", self.wall_material_cost)
            print()
            print("tower cap material volume (pressurized):", self.cap_bot_material_volume + self.cap_top_material_volume)
            print("tower cap material mass (pressurized):", self.cap_material_mass)
            print("tower cap material cost (pressurized):", self.cap_material_cost)
            print()
            print("operating mass fraction:", self.get_operational_mass_fraction())
            print("nonwall cost (non-pressurized):", self.nonwall_cost_trad)
            print("nonwall cost (pressurized):", self.nonwall_cost)

    def get_operating_pressure(self):
        """
        get operating pressure, assumed to be single-valued crossover pressure in Pa
        """

        return PressurizedTower.get_crossover_pressure(self.welded_joint_efficiency,
                                                       self.ultimate_tensile_strength,
                                                       self.d_t_ratio)

    def get_tower_inner_volume(self):
        """
        get the inner volume of the tower in m^3

        assume t << d
        """

        Nsection= len(self.section_diameters) - 1
        vol_section= np.zeros((Nsection,))
        for i_section in range(Nsection):
            diameter_bot= self.section_diameters[i_section]
            height_bot= self.section_heights[i_section]
            diameter_top= self.section_diameters[i_section + 1]
            height_top= self.section_heights[i_section + 1]
            dh= np.abs(height_top - height_bot)

            vol_section[i_section]= PressurizedTower.compute_frustum_volume(dh,
                                                                            diameter_bot,
                                                                            diameter_top)

        return np.sum(vol_section)
    
    def get_tower_material_volume(self,
                                  pressure : float = None):
        """
        get the material volume of the tower in m^3

        if pressurized, use pressure to set thickness increment due to pressurization

        assume t << d

        params:
            - pressure: gauge pressure of H2 (defaults to design op. pressure)
        returns:
            - Vmat_wall: material volume of vertical tower
            - Vmat_bot: material volume of bottom cap
            - Vmat_top: material volume of top cap
        """

        # override pressure iff requested
        if pressure is None: pressure= self.operating_pressure

        alpha_dtp= PressurizedTower.get_thickness_increment_const(pressure, self.ultimate_tensile_strength)

        # loop over the sections of the tower
        Nsection= len(self.section_diameters) - 1
        matvol_section= np.zeros((Nsection,))
        for i_section in range(Nsection):
            d1= self.section_diameters[i_section]
            h1= self.section_heights[i_section]
            d2= self.section_diameters[i_section + 1]
            h2= self.section_heights[i_section + 1]
            
            # compute the differential volume of a given section
            handyconst= (d2 - d1)/(h2 - h1)
            handyfun= lambda h: np.pi*(1/self.d_t_ratio + alpha_dtp) \
                    *(d1**2*(h - h1) - d1*handyconst/2.*(h - h1)**3 + handyconst**2/3.*(h - h1)**3)

            matvol_section[i_section]= handyfun(h2) - handyfun(h1)

        # compute wall volume
        Vmat_wall= np.sum(matvol_section)

        # compute caps as well: area by thickness
        Vmat_bot= (np.pi/4*self.section_diameters[0]**2) \
                *(self.thickness_bot + alpha_dtp*self.section_diameters[0]) # assume first is bottom
        Vmat_top= np.pi/4*self.section_diameters[-1]**2 \
                *(self.thickness_top + alpha_dtp*self.section_diameters[-1]) # assume last is top

        # total material volume
        return (Vmat_wall, Vmat_bot, Vmat_top)
    
    def get_tower_material_mass(self,
                                pressure : float = None):
        """
        get the material mass of the tower in m^3

        if pressurized, use pressure to set thickness increment due to pressurization

        assume t << d

        params:
            - pressure: gauge pressure of H2 (defaults to design op. pressure)
        returns:
            - Mmat_wall: material mass of vertical tower
            - Mmat_bot: material mass of bottom cap
            - Mmat_top: material mass of top cap
        """

        # pass through to volume calculator, multiplying by steel density
        return [self.density_steel*x for x in self.get_tower_material_volume]
    
    def get_tower_material_cost(self,
                                pressure : float = None):
        """
        get the material cost of the tower in m^3

        if pressurized, use pressure to set thickness increment due to pressurization

        assume t << d

        params:
            - pressure: gauge pressure of H2 (defaults to design op. pressure)
        returns:
            - Vmat_wall: material cost of vertical tower
            - Vmat_bot: material cost of bottom cap
            - Vmat_top: material cost of top cap
        """

        if pressure:
            Mmat_wall, Mmat_bot, Mmat_top= self.get_tower_material_mass()
            # use adjusted pressure cap cost
            return [self.costrate_steel*Mmat_wall, self.costrate_endcap*Mmat_bot, self.costrate_endcap*Mmat_top]
        else:
            return [self.costrate_steel*x for x in self.get_tower_material_mass()]

    def get_operational_mass_fraction(self):
        """
        get the fraction of stored hydrogen to tower mass
        
        following Kottenstette
        """

        Sut= self.ultimate_tensile_strength
        rho= self.density_steel
        R= self.gasconstant_H2
        T= self.operating_temp + 273.15 # convert to K

        frac= Sut/(rho*R*T)

        return frac

    def get_nonwall_cost(self,
                         traditional : bool = False):
        nonwall_cost= 0
        if traditional:
            nonwall_cost += self.tower_length*self.costrate_ladder # add ladder cost
            nonwall_cost += self.cost_door # add door cost
        else:
            naive= True
            if naive:
                nonwall_cost += self.cost_mainframe_extension
                nonwall_cost += 2*self.cost_door
                nonwall_cost += 2*self.tower_length*self.costrate_ladder
                nonwall_cost += self.cost_nozzles_manway
                nonwall_cost += self.costrate_conduit
            else:
                raise NotImplementedError("not implemented! -cvf")
        return nonwall_cost






    @staticmethod
    def compute_frustum_volume(height, base_diameter, top_diameter):
        """
        return the volume of a frustum (truncated cone)
        """
        return np.pi/12.*height*(base_diameter**2 + base_diameter*top_diameter + top_diameter**2)

    @staticmethod
    def get_crossover_pressure(welded_joint_efficiency : float,
                               ultimate_tensile_strength : float,
                               d_t_ratio : float):
        """
        get burst/fatigue crossover pressure in Pa
        
        following Kottenstette 2003
        """
    
        # convert to nice variables
        E= welded_joint_efficiency
        Sut= ultimate_tensile_strength
        d_over_t= d_t_ratio # assumed fixed in this study

        p_crossover= 4*E*Sut/(7*d_over_t*(1 - E/7.))

        return p_crossover

    @staticmethod
    def get_thickness_increment_const(pressure : float,
                                      ultimate_tensile_strength : float):
        """
        compute Goodman equation-based thickness increment in m

        following Kottenstette 2003
        """

        # convert to text variables
        p= pressure
        # r= diameter/2
        Sut= ultimate_tensile_strength

        alpha_dtp= 0.25*p/Sut

        return alpha_dtp


