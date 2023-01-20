"""
Author:
Date:
Institution:
Description: This file should just handle the on-turbine electrolysis costs and sizing, we'll use other models for production 
                unless in the development of this file it becomes apparent that the physics should also be handled separately
                when on-turbine.
Sources:
    - [1] Singlitico 2021 (use this as a jumping off point, I think there may be other good sources available)
    - [2] NREL/Electrolyzer: https://github.com/NREL/electrolyzer - this may be useful and I'd like to eventually switch
            over completely to using the NREL/Electrolyzer model
Args:
    - year (int): construction year
    - turbine (dict): contains various information (rating, height, diameters, etc). Assume what you need and we can adjust.
    - desired_electrolyzer_rating (int)
    - others may be added as needed
Returns (can be from separate functions and/or methods as it makes sense):
    - capex (float): the CAPEX in USD for electrolysis in a wind turbine
    - opex (float): the OPEX (annual, fixed) in USD electrolysis in a wind turbine
    - added_mass (float): additional mass (approximate) for added components for on-turbine electrolysis
    - maximum_rating_possible (int): maximum electrolyzer rating that can be put on a single turbine
    - others may be added as needed
"""