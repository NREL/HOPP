"""
Author:
Date:
Institution:
Description: This file should handle the cost and sizing of compression units for h2 transport 
Sources:
    - [1] Singlitico 2021 (use this as a jumping off point, I think there may be other good sources available)
    - [2] NREL/Electrolyzer: https://github.com/NREL/electrolyzer - this may be useful and I'd like to eventually switch
            over completely to using the NREL/Electrolyzer model
    - [3] HOPP storage compressor model: HOPP/examples/H2_Analysis/compressor.py
Args:
    - year (int): construction year
    - input_pressure (float): pressure of hydrogen input to the compressor (bar)
    - output_pressure (float): pressure of hydrogen output from the compressor (bar)
Returns (can be from separate functions and/or methods as it makes sense):
    - capex (float): the CAPEX in USD for the compressor system
    - opex (float): the OPEX (annual, fixed) in USD for the compressor system
    - energy_used (float): annual energy usage of the compressor
    - mass (float): mass (approximate) for the compressor
    - footprint (int): approximate horizontal space required for the compressor (m^2)
    - others may be added as needed
"""