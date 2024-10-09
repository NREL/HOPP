import os
import numpy as np
import pandas as pd
import scipy.optimize
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_tools import get_electrolyzer_BOL_efficiency

file_path = os.path.dirname(os.path.abspath(__file__))


def calc_efficiency_curve(operating_ratio, a, b, c, d):
    """Calculates efficiency [kWh/kg] given operation ratio with flattened end curves.

    Efficiency curve and general equation structure from Wang et. al (2023). See README.md
    in PEM_BOP directory for more details.

    Wang, X.; Star, A.G.; Ahluwalia, R.K. Performance of Polymer Electrolyte Membrane Water Electrolysis Systems: 
    Configuration, Stack Materials, Turndown and Efficiency. Energies 2023, 16, 4964. 
    https://doi.org/10.3390/en16134964

    Args:
        operating_ratio (list or np.array): Operation ratios.
        a (float): Coefficient a.
        b (float): Coefficient b.
        c (float): Coefficient c.
        d (float): Coefficient d.

    Returns:
        efficiency (list or np.array): Efficiency of electrolyzer BOP in kWh/kg.
    """
    efficiency = a + b * operating_ratio + c * operating_ratio**2 + d / operating_ratio

    return efficiency


def calc_efficiency(
    operating_ratio, efficiency, min_ratio, max_ratio, min_efficiency, max_efficiency
):
    """Adjust efficiency list to not go above minimum or maximum operating ratios in BOP_efficiency_BOL.csv

    Args:
        operating_ratio (list or np.array): Operation ratios.
        efficiency (list or np.array): Efficiencies calculated using curve fit.
        min_ratio (float): Minimum operating ratio from the CSV.
        max_ratio (float): Maximum operating ratio from the CSV.
        min_efficiency (float): Efficiency at the minimum operating ratio.
        max_efficiency (float): Efficiency at the maximum operating ratio.

    Returns:
        efficiency (list or np.array): Efficiencies limited with minimum and maximum values in kWh/kg.
    """
    efficiency = np.where(operating_ratio <= min_ratio, min_efficiency, efficiency)

    efficiency = np.where(operating_ratio >= max_ratio, max_efficiency, efficiency)
    return efficiency


def calc_curve_coefficients():
    """Calculates curve coefficients from BOP_efficiency_BOL.csv"""
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

    curve_coeff, curve_cov = scipy.optimize.curve_fit(
        calc_efficiency_curve, operating_ratios, efficiency, p0=(1.0, 1.0, 1.0, 1.0)
    )
    return curve_coeff, min_ratio, max_ratio, min_efficiency, max_efficiency


def pem_bop(
    power_profile_to_electrolyzer_kw,
    electrolyzer_rated_mw,
    electrolyzer_turn_down_ratio,
):
    """
    Calculate PEM balance of plant energy consumption at the beginning-of-life
    based on power provided to the electrolyzer.

    Args:
        power_profile_to_electrolyzer_kw (list or np.array): Power profile to the electrolyzer in kW.
        electrolyzer_rated_mw (float): The rating of the PEM electrolyzer in MW.
        electrolyzer_turn_down_ratio (float): The electrolyzer turndown ratio.

    Returns:
        energy_consumption_bop_kwh (list or np.array): Energy consumed by electrolyzer BOP in kWh.
    """
    operating_ratios = power_profile_to_electrolyzer_kw / (electrolyzer_rated_mw * 1e3)

    curve_coeff, min_ratio, max_ratio, min_efficiency, max_efficiency = (
        calc_curve_coefficients()
    )

    efficiencies = calc_efficiency_curve(
        operating_ratios,
        *curve_coeff,
    )  # kwh/kg

    efficiencies = calc_efficiency(
        operating_ratios,
        efficiencies,
        min_ratio,
        max_ratio,
        min_efficiency,
        max_efficiency,
    )

    BOL_efficiency = get_electrolyzer_BOL_efficiency()  # kwh/kg

    BOL_kg = (electrolyzer_rated_mw * 1000) / BOL_efficiency  # kg/hr

    energy_consumption_bop_kwh = efficiencies * BOL_kg  # kwh

    energy_consumption_bop_kwh = np.where(
        power_profile_to_electrolyzer_kw
        < electrolyzer_turn_down_ratio * electrolyzer_rated_mw * 1000,
        0,
        energy_consumption_bop_kwh,
    )

    return energy_consumption_bop_kwh
