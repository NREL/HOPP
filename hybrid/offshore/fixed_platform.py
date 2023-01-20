"""
Author:
Date:
Institution:
Description: This file should handle the cost and sizing of offshore platforms, but can certainly use WISDEM (fixed_bottomse)
                or ORBIT for much of the modeling effort. 
Sources:
    - [1] ORBIT: https://github.com/WISDEM/ORBIT
    - [2] fixed_bottomse: https://github.com/WISDEM/WISDEM/tree/master/wisdem/fixed_bottomse
Args:
    - year (int): construction year
    - any/all ORBIT inputs are available as needed. Including, but not limited to:
        - depth (float): water depth at desired installation location
        - port_distance (float): distance from port
    - tech_required_area (float): area needed for combination of all tech (m^2), not including buffer or working space
    - tech_combined_mass (float): mass of all tech being placed on the platform (kg or tonnes)
    - lifetime (int): lifetime of the plant in years (may not be needed)
    - others may be added as needed
Returns:(can be from separate functions and/or methods as it makes sense):
    - capex (float): capital expenditures for building the platform, including material costs and installation
    - opex (float): the OPEX (annual, fixed) in USD for the platform
    - others may be added as needed
"""