"""
Author:
Date:
Institution:
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