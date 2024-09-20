import os
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))

def calc_efficiency(operating_ratio, a, b, c, d, min_ratio, max_ratio, min_efficiency, max_efficiency):
    """Calculates efficiency [kwh/kg] given operation ratio with flattened end curves.

    Args:
        operating_ratio (list or np.array): Operation ratios.
        a (float): Coefficient a.
        b (float): Coefficient b.
        c (float): Coefficient c.
        d (float): Coefficient d.
        min_ratio (float): Minimum operating ratio from the CSV.
        max_ratio (float): Maximum operating ratio from the CSV.
        min_efficiency (float): Efficiency at the minimum operating ratio.
        max_efficiency (float): Efficiency at the maximum operating ratio.
    """
    # Ensure operating_ratio is an array for easy vectorized calculations
    operating_ratio = np.array(operating_ratio)

    # Initialize efficiency array
    efficiency = np.zeros_like(operating_ratio)

    # Handle operating ratios less than the min_ratio (flatten at min_efficiency)
    efficiency[operating_ratio <= min_ratio] = min_efficiency

    # Handle operating ratios greater than the max_ratio (flatten at max_efficiency)
    efficiency[operating_ratio >= max_ratio] = max_efficiency

    # For operating ratios within the bounds, calculate efficiency
    within_bounds = (operating_ratio > min_ratio) & (operating_ratio < max_ratio)
    efficiency[within_bounds] = (
        a + b * operating_ratio[within_bounds] 
        + c * operating_ratio[within_bounds]**2 
        + d / operating_ratio[within_bounds]
    )

    # Ensure the calculated efficiency doesn't exceed the bounds
    efficiency[within_bounds] = np.clip(efficiency[within_bounds], min_efficiency, max_efficiency)

    return efficiency

def calc_curve_coefficients():
    """Calculates curve coefficients from BOP_efficiency_BOL.csv
    """
    df = pd.read_csv(os.path.join(file_path, "BOP_efficiency_BOL.csv"))
    operating_ratios = df["operating_ratio"].values
    efficiency = df["efficiency"].values

    # Get min and max operating ratios
    min_ratio_idx = df["operating_ratio"].idxmin()  # Index of minimum operating ratio
    max_ratio_idx = df["operating_ratio"].idxmax()  # Index of maximum operating ratio

    # Get the efficiency at the min and max operating ratios
    min_efficiency = df["efficiency"].iloc[min_ratio_idx]
    max_efficiency = df["efficiency"].iloc[max_ratio_idx]

    # Get the actual min and max ratios
    min_ratio = df["operating_ratio"].iloc[min_ratio_idx]
    max_ratio = df["operating_ratio"].iloc[max_ratio_idx]

    # Fit curve with the modified calc_efficiency that flattens at min/max ratios
    curve_coeff, curve_cov = scipy.optimize.curve_fit(
        lambda or_, a, b, c, d: calc_efficiency(or_, a, b, c, d, min_ratio, max_ratio, min_efficiency, max_efficiency),
        operating_ratios, efficiency, p0=(1.0, 1.0, 1.0, 1.0)
    )
    
    return curve_coeff, min_ratio, max_ratio, min_efficiency, max_efficiency
    
    return curve_coeff

# def calc_efficiency(
#     operating_ratio, a, b, c, d
# ):
#     """Calculates efficiency [kwh/kg] given operation ratio.

#     Args:
#         operating_ratio (_type_): _description_
#         a (_type_): _description_
#         b (_type_): _description_
#         c (_type_): _description_
#         d (_type_): _description_
#     """
#     efficiency = a + b * operating_ratio + c * operating_ratio**2 + d / operating_ratio
#     return efficiency

# def calc_curve_coefficients():
#     """_summary_
#     """
#     df = pd.read_csv(os.path.join(file_path, "BOP_efficiency_BOL.csv"))
#     operating_ratios = df["operating_ratio"].values
#     efficiency = df["efficiency"].values

#     curve_coeff, curve_cov = scipy.optimize.curve_fit(
#     calc_efficiency, operating_ratios, efficiency, p0=(1.0, 1.0, 1.0, 1.0)
# )
#     return curve_coeff
    
def pem_bop(power_profile_to_electrolyzer_kw,
            electrolyzer_rated_mw):
    from greenheart.tools.eco.electrolysis import get_electrolyzer_BOL_efficiency
    """_summary_

    Args:
        power_profile_to_electrolyzer_kw (_type_): _description_
        electrolyzer_rated_mw (_type_): _description_

    Returns:
        _type_: _description_
    """
    operating_ratios = power_profile_to_electrolyzer_kw/(electrolyzer_rated_mw*1e3)

    curve_coeff, min_ratio, max_ratio, min_efficiency, max_efficiency = calc_curve_coefficients()

    efficiencies= calc_efficiency(operating_ratios,*curve_coeff,min_ratio,max_ratio,min_efficiency,max_efficiency) # kwh/kg

    BOL_efficiency = get_electrolyzer_BOL_efficiency() #kwh/kg

    BOL_kg = (electrolyzer_rated_mw*1000) / BOL_efficiency #kg/hr

    energy_consumption_bop = efficiencies*BOL_kg #kwh

    return energy_consumption_bop

