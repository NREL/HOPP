"""
Author: Cory Frontin
Date: 23 Jan 2023
Institution: National Renewable Energy Lab
Description: This file handles the cost, sizing, and pressure of on-turbine H2 storage

To use this class, specify a turbine 

Costs are assumed to be in 2003 dollars [1]

Sources:
    - [1] Kottenstette 2003 (use their chosen favorite design)
Args:
    - year (int): construction year
    - turbine (dict): contains various information about the turbine, including tower_length, section_diameters, and section_heights
API member functions:
    - get_capex(): return the total additional capex necessary for H2 production, in 2003 dollars
    - get_opex(): return the result of a simple model for operational expenditures for pressure vessel, in 2003 dollars
    - get_mass_empty(): return the total additional empty mass necessary for H2 production, in kg
    - get_capacity_H2(): return the capacity mass of hydrogen @ operating pressure, ambient temp., in kg
    - get_pressure_H2() return the operating hydrogen pressure, in Pa
"""

import numpy as np

class PressurizedTower():
    def __init__(self,
                 year: int,
                 turbine: dict):
        
        # key inputs
        self.year= year
        self.turbine= turbine
        
        self.tower_length= turbine['tower_length'] # m
        self.section_diameters= turbine['section_diameters'] # m
        self.section_heights= turbine['section_heights'] # m

        # calculation settings
        self.setting_volume_thickness_calc= 'centered' # ['centered', 'outer', 'inner']

        # constants/parameters
        self.d_t_ratio= 320. # Kottenstette 2003
        self.thickness_top= self.section_diameters[-1]/self.d_t_ratio # m
        self.thickness_bot= self.section_diameters[0]/self.d_t_ratio # m
        self.ultimate_tensile_strength= 636e6 # Pa, Kottenstette 2003
        self.yield_strength= 350e6 # Pa, Kottenstette 2003
        self.welded_joint_efficiency= 0.80 # journal edition
        # self.welded_joint_efficiency= 0.85 # double-welded butt joint w/ spot inspection (ASME)
        self.density_steel= 7817. # kg/m^3
        self.gasconstant_H2= 4126. # J/(kg K)
        self.operating_temp= 25. # degC

        self.costrate_steel= 1.50 # $/kg
        self.costrate_endcap= 2.66 # $/kg

        self.costrate_ladder= 32.80 # $/m
        self.cost_door= 2000 # $
        self.cost_mainframe_extension= 6300 # $
        self.cost_nozzles_manway= 16000 # $
        self.costrate_conduit= 35 # $/m

        # based on pressure_vessel maintenance costs
        self.wage= 36 # 2003 dollars (per hour worked)
        self.staff_hours= 60 # hours
        self.maintenance_rate= 0.03 # factor

        # set the operating pressure @ the crossover pressure
        self.operating_pressure= PressurizedTower.get_crossover_pressure(self.welded_joint_efficiency,
                                                                         self.ultimate_tensile_strength,
                                                                         self.d_t_ratio)

    def run(self):

        # get the inner volume and traditional material volume, mass, cost
        self.tower_inner_volume= self.get_volume_tower_inner()
        self.wall_material_volume_trad, self.cap_bot_material_volume_trad, self.cap_top_material_volume_trad= \
                self.get_volume_tower_material(pressure= 0.0)
        self.wall_material_mass_trad= self.wall_material_volume_trad*self.density_steel
        self.wall_material_cost_trad= self.wall_material_mass_trad*self.costrate_steel
        self.cap_material_mass_trad= (self.cap_top_material_volume_trad + self.cap_bot_material_volume_trad)*self.density_steel
        self.cap_material_cost_trad= self.cap_material_mass_trad*self.costrate_steel
        self.nonwall_cost_trad= self.get_cost_nontower(traditional= True)

        self.wall_material_volume, self.cap_bot_material_volume, self.cap_top_material_volume= \
                self.get_volume_tower_material()
        self.wall_material_mass= self.wall_material_volume*self.density_steel
        self.wall_material_cost= self.wall_material_mass*self.costrate_steel
        self.wall_material_mass= self.wall_material_volume*self.density_steel
        self.cap_material_mass= (self.cap_bot_material_volume + self.cap_top_material_volume)*self.density_steel
        self.cap_material_cost= self.cap_material_mass*self.costrate_endcap
        self.nonwall_cost= self.get_cost_nontower()

        if False:
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
            print("tower top cap material cost (pressurized):", self.cap_top_material_volume*self.density_steel*self.costrate_endcap)
            print("tower bot cap material cost (pressurized):", self.cap_bot_material_volume*self.density_steel*self.costrate_endcap)
            print("tower cap material cost (pressurized):", self.cap_material_cost)
            print()
            print("operating mass fraction:", self.get_operational_mass_fraction())
            print("nonwall cost (non-pressurized):", self.nonwall_cost_trad)
            print("nonwall cost (pressurized):", self.nonwall_cost)
            print("tower total material cost (pressurized):",
                self.wall_material_cost + self.cap_material_cost)

            print()
            print("delta tower wall material cost:", self.wall_material_cost - self.wall_material_cost_trad)
            print("empty mass:", self.get_mass_empty())
            print("")
            print("capex:", self.get_capex())
            print("opex:", self.get_opex())
            print("capacity (H2):", self.get_capacity_H2())

    def get_volume_tower_inner(self):
        """
        get the inner volume of the tower in m^3

        assume t << d
        """

        # count the sections, all assumed conic frustum
        Nsection= len(self.section_diameters) - 1

        # loop over sections, calclulating volume of each
        vol_section= np.zeros((Nsection,))
        for i_section in range(Nsection):
            diameter_bot= self.section_diameters[i_section] # m
            height_bot= self.section_heights[i_section] # m
            diameter_top= self.section_diameters[i_section + 1] # m
            height_top= self.section_heights[i_section + 1] # m
            dh= np.abs(height_top - height_bot) # height of section, m

            vol_section[i_section]= PressurizedTower.compute_frustum_volume(dh,
                                                                            diameter_bot,
                                                                            diameter_top)

        # total volume: sum of sections
        return np.sum(vol_section) # m^3
    
    def get_volume_tower_material(self,
                                  pressure : float = None):
        """
        get the material volume of the tower in m^3

        if pressurized, use pressure to set thickness increment due to pressurization

        assume t << d

        params:
            - pressure: gauge pressure of H2 (defaults to design op. pressure)
        returns:
            - Vmat_wall: material volume of vertical tower
            - Vmat_bot: material volume of bottom cap (nonzero only if pressurized)
            - Vmat_top: material volume of top cap (nonzero only if pressurized)
        """

        # override pressure iff requested
        if pressure is None: pressure= self.operating_pressure

        # this is the linear constant s.t. delta t ~ alpha * d
        alpha_dtp= PressurizedTower.get_thickness_increment_const(pressure, self.ultimate_tensile_strength) # 

        # loop over the sections of the tower
        Nsection= len(self.section_diameters) - 1
        matvol_section= np.zeros((Nsection,))
        for i_section in range(Nsection):
            d1= self.section_diameters[i_section]
            h1= self.section_heights[i_section]
            d2= self.section_diameters[i_section + 1]
            h2= self.section_heights[i_section + 1]
            
            if self.setting_volume_thickness_calc == 'centered':
                Vouter= PressurizedTower.compute_frustum_volume(h2 - h1,
                                                                d1*(1 + (1/self.d_t_ratio + alpha_dtp)),
                                                                d2*(1 + (1/self.d_t_ratio + alpha_dtp)))
                Vinner= PressurizedTower.compute_frustum_volume(h2 - h1,
                                                                d1*(1 - (1/self.d_t_ratio + alpha_dtp)),
                                                                d2*(1 - (1/self.d_t_ratio + alpha_dtp)))
            elif self.setting_volume_thickness_calc == 'outer':
                Vouter= PressurizedTower.compute_frustum_volume(h2 - h1,
                                                                d1*(1 + 2*(1/self.d_t_ratio + alpha_dtp)),
                                                                d2*(1 + 2*(1/self.d_t_ratio + alpha_dtp)))
                Vinner= PressurizedTower.compute_frustum_volume(h2 - h1, d1, d2)
            elif self.setting_volume_thickness_calc == 'inner':
                Vouter= PressurizedTower.compute_frustum_volume(h2 - h1, d1, d2)
                Vinner= PressurizedTower.compute_frustum_volume(h2 - h1,
                                                                d1*(1 - 2*(1/self.d_t_ratio + alpha_dtp)),
                                                                d2*(1 - 2*(1/self.d_t_ratio + alpha_dtp)))

            matvol_section[i_section]= Vouter - Vinner

        # compute wall volume by summing sections
        Vmat_wall= np.sum(matvol_section) # m^3

        if pressure == 0:
            Vmat_bot= 0.0
            Vmat_top= 0.0
        else:
            # compute caps as well: area by thickness
            Vmat_bot= (np.pi/4*self.section_diameters[0]**2) \
                    *(PressurizedTower.compute_cap_thickness(pressure,
                                                             self.section_diameters[0], # assume first is bottom
                                                             self.yield_strength,
                                                             efficiency_weld= self.welded_joint_efficiency)) # m^3
            Vmat_top= np.pi/4*self.section_diameters[-1]**2 \
                    *(PressurizedTower.compute_cap_thickness(pressure,
                                                             self.section_diameters[-1], # assume last is top
                                                             self.yield_strength,
                                                             efficiency_weld= self.welded_joint_efficiency)) # m^3

        # total material volume
        return (Vmat_wall, Vmat_bot, Vmat_top) # m^3
    
    def get_mass_tower_material(self,
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
        return [self.density_steel*x for x in self.get_volume_tower_material(pressure)] # kg
    
    def get_cost_tower_material(self,
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

        if pressure == 0:
            return [self.costrate_steel*x for x in self.get_mass_tower_material(pressure= pressure)] # 2003 dollars
        else:
            Mmat_wall, Mmat_bot, Mmat_top= self.get_mass_tower_material(pressure= pressure)
            # use adjusted pressure cap cost
            return [self.costrate_steel*Mmat_wall, self.costrate_endcap*Mmat_bot, self.costrate_endcap*Mmat_top] # 2003 dollars

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

        return frac # nondim.

    def get_cost_nontower(self,
                          traditional : bool = False,
                          naive : bool = True):
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
                nonwall_cost += self.tower_length*self.costrate_conduit
            else:
                # adjust length b.c. conduit, one ladder must ride outside pressure vessel
                adj_length= np.sqrt((self.section_diameters[0] - self.section_diameters[-1])**2 + self.tower_length**2)
                nonwall_cost += self.cost_mainframe_extension
                nonwall_cost += 2*self.cost_door
                nonwall_cost += self.tower_length*self.costrate_ladder
                nonwall_cost += adj_length*self.costrate_ladder
                nonwall_cost += self.cost_nozzles_manway
                nonwall_cost += adj_length*self.costrate_conduit
        return nonwall_cost # 2003 dollars

    ### OFFICIAL OUTPUT INTERFACE

    def get_capex(self):
        """ return the total additional capex necessary for H2 production """
        capex_withH2= self.get_cost_nontower() + np.sum(self.get_cost_tower_material())
        capex_without= self.get_cost_nontower(traditional= True) + np.sum(self.get_cost_tower_material(pressure= 0))
        return capex_withH2 - capex_without # 2003 dollars

    def get_opex(self):
        """
        a simple model for operational expenditures for PV

        maintenance for pressure vessel based on an annual maintenance rate
        against the vessel-specific capital expenditure, plus wages times staff
        hours per year
        """
        
        return self.get_capex()*self.maintenance_rate + self.wage*self.staff_hours # 2003 dollars 

    def get_mass_empty(self):
        """ return the total additional empty mass necessary for H2 production in kg """

        Mtower_withH2= np.sum(self.get_mass_tower_material())
        Mnontower_withH2= 0.0 # not specified

        Mtower_without= np.sum(self.get_mass_tower_material(pressure= 0))
        Mnontower_without= 0.0 # not specified

        Mtotal_withH2= Mtower_withH2 + Mnontower_withH2
        Mtotal_without= Mtower_without + Mnontower_without

        return Mtotal_withH2 - Mtotal_without # kg

    def get_capacity_H2(self):
        """ get the ideal gas H2 capacity in kg """

        Tabs= self.operating_temp + 273.15
        R= self.gasconstant_H2
        p= self.operating_pressure
        V= self.get_volume_tower_inner()

        # ideal gas law
        m_H2= p*V/(R*Tabs)

        return m_H2 # kg

    def get_pressure_H2(self): return self.operating_pressure # Pa, trivial, but for delivery

    ### STATIC FUNCTIONS

    @staticmethod
    def compute_cap_thickness(pressure, diameter, strength_yield,
                              safetyfactor_Sy= 1.5, efficiency_weld= 0.80, constant= 0.10):
        """
        compute the necessary thickness for a pressure vessel cap

        $$
        t= d \\sqrt{\\frac{C P}{S E}}
        $$
        with weld joint efficiency E, allowable stress S, pressure P, diameter
        of pressure action d, edge restraint factor C

        assumed:
            - C= 0.10: Fig-UG-34 of ASME Code S VII, div. 1, via Rao's _Companion
                    Guide to the ASME Boiler and Pressure Vessel Code_ (2009),
                    fig. 21.3. type of sketch (a) assumed
            - E= 0.80: conservatively butt weld, inspected
        
        using the ASME pressure vessel code definitions, and values given in
        Rao _Companion Guide to the ASME Boiler and Pressure Vessel Code_ (2009)
        """

        return diameter*np.sqrt(constant*pressure/(efficiency_weld*strength_yield/safetyfactor_Sy))

    @staticmethod
    def compute_frustum_volume(height, base_diameter, top_diameter):
        """
        return the volume of a frustum (truncated cone)
        """
        return np.pi/12.*height*(base_diameter**2 + base_diameter*top_diameter + top_diameter**2) # volume units

    @staticmethod
    def get_crossover_pressure(welded_joint_efficiency : float,
                               ultimate_tensile_strength : float,
                               d_t_ratio : float):
        """
        get burst/fatigue crossover pressure
        
        following Kottenstette 2003
        """
    
        # convert to nice variables
        E= welded_joint_efficiency # nondim.
        Sut= ultimate_tensile_strength # pressure units
        d_over_t= d_t_ratio # length-per-length; assumed fixed in this study

        p_crossover= 4*E*Sut/(7*d_over_t*(1 - E/7.)) # pressure units

        return p_crossover # pressure units

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

        return alpha_dtp # length per length
