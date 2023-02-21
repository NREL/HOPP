
"""
Author: Cory Frontin
Date: 17 February 2023
Institution: National Renewable Energy Lab
Description: This file handles the cost, sizing, and pressure of on-turbine H2 storage

We assume that, without additional structural considerations, pressure vessels
of a given height can be mounted on the exterior of the monopile, like a mooring
station. Given a monopile diameter, available height, reserved arc angle on the
monopile for accessory emplacement, etc. pressure vessels are sized using
tankinator and given here.

"""

import numpy as np
from ..pressure_vessel.tankinator import TypeITank

def get_capacity_H2(T_celsius, p_bar, V_m3):

    T_kelvin= T_celsius + 273.15
    p_pa= 1e5*p_bar
    gasconstant_H2= 4126. # J/(kg K)

    m_kg= p_pa*V_m3/(gasconstant_H2*T_kelvin)

    return m_kg

class MonopilePressureVesselJacket():
    """
    design the pressure vessel storage possible by wrapping a jacket of pressure
    vessels around the monopile/lower tower of a turbine. the assumption is that
    cylindrical tanks are packed vertically around some portion of the monopile
    exterior.

    arguments:
      - monopile: dict
        - diameter: float- diameter of the monopile, in meters
        - arcangle_reserved: float- angle of arc which is reserved for monopile
            accessorizing (mooring, etc.)
        - height_available: float- vertical monopile region on which tanks can
            be mounted
    """

    def __init__(self,
                 monopile: dict):
        
        # key inputs
        self.diameter_monopile= monopile['diameter'] # m, outer diameter
        self.arcangle_reserved_monopile= monopile['arcangle_reserved'] # radians
        self.height_available= monopile['height_available'] # m

        # empty design, to start
        self.design= {}

    def design_fixed_tank(self,
                          pressure_tank,
                          radius_inner_tank,
                          height_inner_tank,
                          temp_ambient= 15,
                          ratio_mounting_overhead= 0.3,
                          material= 'steel'):
        """
        use a fixed tank for design, put as many on there as we can fit

        arguments:
          - pressure_tank: float- operating pressure in bar
          - radius_inner_tank: float- radius specification for the tank
          - height_inner_tank: float- height specification for the tank
          - temp_ambient: float- ambient temp. for the tank
          - ratio_mounting_overhead: float- mass overhead required for mounting
              materials for the tank
          - material: str- a string to set the material for the tanks, see
              tankinator.py (and dependency material_properties.json) in the
              includes

        returns:
          - design_here: dict
            - type: str- type of design ('fixed-tank')
            - pressure_h2: float- design pressure of stored H2 (bar)
            - tank: dict- specified tank
              - metal: str- metal used for tank
              - radius_inner: float- inner radius of tank (m)
              - height_inner: float- inner height of tank (m)
              - volume_inner: float- inner volume of tank (m)
              - thickness: float- thickness of tank
              - mass_empty: float- empty mass of tank (kg)
            - Nrow: int- number of rows of tanks in jacket
            - Ncolumn: int- number of columns of tanks in jacket
            - mass_tanks_empty: float- empty tank mass of jacket (kg)
            - mass_jacket_installed_empty: float- empty mass of jacket (incl. hardware, kg)
            - mass_capacity_H2: float- capacity of H2 in the tank
        """

        ### use the pythonic tankinator

        tank= TypeITank(material) # assume a steel tank (avoid complexities of composite exposure)
        tank.set_operating_pressure(pressure_tank) # convert to bar from Pa
        tank.set_length_radius(height_inner_tank*100, radius_inner_tank*100)
        tank.set_operating_temperature(temp_ambient) # cool ambient temp.

        # set the thickness using a von mises iteration & get values of interest
        tank.set_thickness_thinwall()
        volume_inner_tank= tank.get_volume_inner()/1e6 # convert to m^3
        radius_outer_tank= tank.get_radius_outer()/100. # convert to m
        height_outer_tank= tank.get_length_outer()/100. # convert to m
        mass_empty_tank= tank.get_mass_metal()
        self.tank= tank # stash the resulting tank

        ### we now have a tank, use it

        # how many layers can be stacked in the available monopile section?
        tank_rows= int(self.height_available/height_outer_tank) # truncates by default
        # how much angle of arc does a tank w/ circular section take mounted on the outside of the monopile?
        arcangle_tank= 2.0*np.arcsin(radius_outer_tank/(0.5*self.diameter_monopile + radius_outer_tank))
        # how many can we pack in?
        tank_cols= int((2*np.pi - self.arcangle_reserved_monopile)/arcangle_tank) # trucates by default

        # how many tanks can we fit
        Ntanks_monopile= tank_rows*tank_cols

        # total volume of capacity
        volume_jacket= Ntanks_monopile*volume_inner_tank

        # find the empty weight of the jacket (w/ & w/o mounting hardware)
        mass_jacket_tanks_empty= Ntanks_monopile*mass_empty_tank
        mass_jacket_installed_empty= (1 + ratio_mounting_overhead)*mass_jacket_tanks_empty

        # capacity mass
        mass_capacity_H2= get_capacity_H2(temp_ambient, pressure_tank, volume_jacket)

        # pack up the design
        design_here= {'type': 'fixed-tank'}
        design_here['pressure_H2']= pressure_tank
        design_here['tank']= {
            'metal': tank.material.metal_type,
            'radius_inner': radius_inner_tank,
            'height_inner': height_inner_tank,
            'volume_inner': volume_inner_tank,
            'thickness': tank.get_thickness()/100, # convert from cm to m
            'mass_empty': mass_empty_tank,
        }
        design_here['Nrow']= tank_rows
        design_here['Ncolumn']= tank_cols
        design_here['mass_tanks_empty']= mass_jacket_tanks_empty
        design_here['mass_jacket_installed_empty']= mass_jacket_installed_empty
        design_here['mass_capacity_H2']= mass_capacity_H2

        # kick the design back up
        self.design= design_here

        # return the design dictionary
        return design_here
