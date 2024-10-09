"""
Author: Jamie Kee
Added to HOPP by: Jared Thomas
Note: ANL costs are in 2018 dollars

07/15/2024: Jamie removed Z=0.9 assumption with linear approx, 
removed f=0.01 assumption with Hofer eqn, added
algebraic solver, and reformatted with black.
08/02/2024: Provide cost overrides
"""

import pandas as pd
import numpy as np
import os

BAR2MPA = 0.1
BAR2PA = 100_000
MM2IN = 0.0393701
M2KM = 1 / 1_000
M2MM = 1_000
KM2MI = 0.621371


def run_pipe_analysis(
    L: float,
    m_dot: float,
    p_inlet: float,
    p_outlet: float,
    depth: float,
    risers: int = 1,
    data_location: str = os.path.abspath(os.path.dirname(__file__) + "/data_tables"),
    labor_in_mi: float = None,
    misc_in_mi: float = None,
    row_in_mi: float = None,
    mat_in_mi: float = None,
    region: str = "SW",
):
    """
    This function calculates the cheapest grade, diameter, thickness, subject to ASME B31.12 and .8

    If $/in/mi values are provided in labor_in_mi, misc_in_mi, row_in_mi, mat_in_mi, those values
    will be used in the cost calculations instead of the defaults
    """
    p_inlet_MPa = p_inlet * BAR2MPA
    F = 0.72  # Design option B class 1 - 2011 ASME B31.12 Table  PL-3.7.1.2
    E = 1.0  # Long. Weld Factor: Seamless (Table IX-3B)
    T_derating = 1  # 2020 ASME B31.8 Table A841.1.8-1 for T<250F, 121C

    # Cost overrides
    anl_cost_overrides = {"labor": labor_in_mi, "misc": misc_in_mi, "ROW": row_in_mi}

    riser = (
        risers > 0
    )  # This is a flag for the ASMEB31.8 stress design, if not including risers, then this can be set to false
    extra_length = 1 + 0.05  # 5% extra
    total_L = (
        L * extra_length + risers * depth * M2KM
    )  # km #Assuming 5% extra length and 1 riser. Will need two risers for turbine to central platform

    #   Import mechanical props and pipe thicknesses (remove A,B ,and A25 since no costing data)
    yield_strengths = pd.read_csv(
        os.path.join(data_location, "steel_mechanical_props.csv"),
        index_col=None,
        header=0,
    )
    yield_strengths = yield_strengths.loc[
        ~yield_strengths["Grade"].isin(["A", "B", "A25"])
    ].reset_index()
    schedules_all = pd.read_csv(
        os.path.join(data_location, "pipe_dimensions_metric.csv"),
        index_col=None,
        header=0,
    )
    steel_costs_kg = pd.read_csv(
        os.path.join(data_location, "steel_costs_per_kg.csv"), index_col=None, header=0
    )

    #   First get the minimum diameter required to achieve the outlet pressure for given length and m_dot
    min_diam_mm = get_min_diameter_of_pipe(
        L=L, m_dot=m_dot, p_inlet=p_inlet, p_outlet=p_outlet
    )
    #   Filter for diameters larger than min diam required
    schedules_spec = schedules_all.loc[schedules_all["DN"] >= (min_diam_mm)]

    #   Gather the grades, diameters, and schedules to loop thru
    grades = yield_strengths["Grade"].values
    diams = schedules_spec["Outer diameter [mm]"].values
    schds = schedules_spec.loc[
        :, ~schedules_spec.columns.isin(["DN", "Outer diameter [mm]"])
    ].columns
    viable_types = []

    #   Loop thru grades
    for grade in grades:
        #   Get SMYS and SMTS for the specific grade
        SMYS = yield_strengths.loc[yield_strengths["Grade"] == grade, "SMYS [Mpa]"].iat[
            0
        ]
        SMTS = yield_strengths.loc[yield_strengths["Grade"] == grade, "SMTS [Mpa]"].iat[
            0
        ]
        #   Loop thru outer diameters
        for diam in diams:
            diam_row = schedules_spec.loc[schedules_spec["Outer diameter [mm]"] == diam]
            dn = diam_row["DN"].iat[0]
            #   Loop thru scheudles (which give the thickness)
            for schd in schds:
                thickness = diam_row[schd].iat[0]

                # Check if thickness satisfies ASME B31.12
                mat_perf_factor = get_mat_factor(
                    SMYS=SMYS, SMTS=SMTS, design_pressure=p_inlet * BAR2MPA
                )
                t_ASME = p_inlet_MPa * dn / (2 * SMYS * F * E * mat_perf_factor)
                if thickness < t_ASME:
                    continue

                # Check if satifies ASME B31.8
                if not checkASMEB318(
                    SMYS=SMYS,
                    diam=diam,
                    thickness=thickness,
                    riser=riser,
                    depth=depth,
                    p_inlet=p_inlet,
                    T_derating=T_derating,
                ):
                    continue

                # Add qualified pipes to saved answers:
                inner_diam = diam - 2 * thickness
                viable_types.append([grade, dn, diam, inner_diam, schd, thickness])

    viable_types_df = pd.DataFrame(
        viable_types,
        columns=[
            "Grade",
            "DN",
            "Outer diameter (mm)",
            "Inner diameter (mm)",
            "Schedule",
            "Thickness (mm)",
        ],
    ).dropna()

    #   Calculate material, labor, row, and misc costs
    viable_types_df = get_mat_costs(
        schedules_spec=viable_types_df,
        total_L=total_L,
        steel_costs_kg=steel_costs_kg,
        mat_cost_override=mat_in_mi,
    )
    viable_types_df = get_anl_costs(
        costs=viable_types_df,
        total_L=total_L,
        anl_cost_overrides=anl_cost_overrides,
        loc=region,
    )
    viable_types_df["total capital cost [$]"] = viable_types_df[
        ["mat cost [$]", "labor cost [$]", "misc cost [$]", "ROW cost [$]"]
    ].sum(axis=1)

    # Annual operating cost assumes 1.17% of total capital
    # https://doi.org/10.1016/j.esr.2021.100658
    viable_types_df["annual operating cost [$]"] = (
        0.0117 * viable_types_df["total capital cost [$]"]
    )

    # Take the option with the lowest total capital cost
    min_row = (
        viable_types_df.sort_values(by="total capital cost [$]").iloc[:1].reset_index()
    )
    return min_row


def get_mat_factor(SMYS: float, SMTS: float, design_pressure: float) -> float:
    """
    Determine the material performance factor ASMEB31.12.
    Dependent on the SMYS and SMTS.
    Defaulted to 1 if not within parameters - This may not be a good assumption
    """
    dp_array = np.array(
        [6.8948, 13.7895, 15.685, 16.5474, 17.9264, 19.3053, 20.6843]
    )  # MPa
    if SMYS <= 358.528 or SMTS <= 455.054:
        h_f_array = np.array([1, 1, 0.954, 0.91, 0.88, 0.84, 0.78])
    elif SMYS <= 413.686 and (SMTS > 455.054 and SMTS <= 517.107):
        h_f_array = np.array([0.874, 0.874, 0.834, 0.796, 0.77, 0.734, 0.682])
    elif SMYS <= 482.633 and (SMTS > 517.107 and SMTS <= 565.370):
        h_f_array = np.array([0.776, 0.776, 0.742, 0.706, 0.684, 0.652, 0.606])
    elif SMYS <= 551.581 and (SMTS > 565.370 and SMTS <= 620.528):
        h_f_array = np.array([0.694, 0.694, 0.662, 0.632, 0.61, 0.584, 0.542])
    else:
        return 1
    mat_perf_factor = np.interp(design_pressure, dp_array, h_f_array)
    return mat_perf_factor


def checkASMEB318(
    SMYS: float,
    diam: float,
    thickness: float,
    riser: bool,
    depth: float,
    p_inlet: float,
    T_derating: float,
) -> bool:
    """
    Determine if pipe parameters satisfy hoop and longitudinal stress requirements
    """

    # Hoop Stress - 2020 ASME B31.8 Table A842.2.2-1
    F1 = 0.50 if riser else 0.72

    #   Hoop stress (MPa) - 2020 ASME B31.8 section A842.2.2.2 eqn (1)
    #   This is the maximum value for S_h
    #   Sh <= F1*SMYS*T_derating
    S_h_check = F1 * SMYS * T_derating

    #   Hoop stress (MPa)
    rho_water = 1_000  # kg/m3
    p_hydrostatic = rho_water * 9.81 * depth / BAR2PA  # bar
    dP = (p_inlet - p_hydrostatic) * BAR2MPA  # MPa
    S_h = (
        dP * (diam - (thickness if diam / thickness >= 30 else 0)) / (2_000 * thickness)
    )
    if S_h >= S_h_check:
        return False

    #   Longitudinal stress (MPa)
    S_L_check = 0.8 * SMYS  # 2020 ASME B31.8 Table A842.2.2-1. Same for riser and pipe
    S_L = p_inlet * BAR2MPA * (diam - 2 * thickness) / (4 * thickness)
    if S_L > S_L_check:
        return False

    S_combined_check = (
        0.9 * SMYS
    )  # 2020 ASME B31.8 Table A842.2.2-1. Same for riser and pipe
    #   Torsional stress?? Under what applied torque? Not sure what to do for this.

    return True


def get_anl_costs(
    costs: pd.DataFrame, total_L: float, anl_cost_overrides: dict, loc: str = "SW"
) -> pd.DataFrame:
    """
    Calculates the labor, right-of-way (ROW), and miscellaneous costs associated with pipe capital cost

    Users can specify a region (GP,NE,MA,GL,RM,SE,PN,SW,CA) that corresponds to grouping of states which
    will apply cost correlations from Brown, D., et al. 2022. “The Development of Natural Gas and Hydrogen Pipeline Capital
    Cost Estimating Equations.” International Journal of Hydrogen Energy https://doi.org/10.1016/j.ijhydene.2022.07.270.

    Alternatively, if a value (not None) is provided in anl_cost_overrides, that value be used as the $/in/mi
    cost correlation for the relevant cost type.
    """

    ANL_COEFS = {
        "GP": {
            "labor": [10406, 0.20953, -0.08419],
            "misc": [4944, 0.17351, -0.07621],
            "ROW": [2751, -0.28294, 0.00731],
            "material": [5813, 0.31599, -0.00376],
        },
        "NE": {
            "labor": [249131, -0.33162, -0.17892],
            "misc": [65990, -0.29673, -0.06856],
            "ROW": [83124, -0.66357, -0.07544],
            "material": [10409, 0.296847, -0.07257],
        },
        "MA": {
            "labor": [43692, 0.05683, -0.10108],
            "misc": [14616, 0.16354, -0.16186],
            "ROW": [1942, 0.17394, -0.01555],
            "material": [9113, 0.279875, -0.00840],
        },
        "GL": {
            "labor": [58154, -0.14821, -0.10596],
            "misc": [41238, -0.34751, -0.11104],
            "ROW": [14259, -0.65318, 0.06865],
            "material": [8971, 0.255012, -0.03138],
        },
        "RM": {
            "labor": [10406, 0.20953, -0.08419],
            "misc": [4944, 0.17351, -0.07621],
            "ROW": [2751, -0.28294, 0.00731],
            "material": [5813, 0.31599, -0.00376],
        },
        "SE": {
            "labor": [32094, 0.06110, -0.14828],
            "misc": [11270, 0.19077, -0.13669],
            "ROW": [9531, -0.37284, 0.02616],
            "material": [6207, 0.38224, -0.05211],
        },
        "PN": {
            "labor": [32094, 0.06110, -0.14828],
            "misc": [11270, 0.19077, -0.13669],
            "ROW": [9531, -0.37284, 0.02616],
            "material": [6207, 0.38224, -0.05211],
        },
        "SW": {
            "labor": [95295, -0.53848, 0.03070],
            "misc": [19211, -0.14178, -0.04697],
            "ROW": [72634, -1.07566, 0.05284],
            "material": [5605, 0.41642, -0.06441],
        },
        "CA": {
            "labor": [95295, -0.53848, 0.03070],
            "misc": [19211, -0.14178, -0.04697],
            "ROW": [72634, -1.07566, 0.05284],
            "material": [5605, 0.41642, -0.06441],
        },
    }

    if loc not in ANL_COEFS.keys():
        raise ValueError(f"Region {loc} was supplied, but is not a valid region")

    L_mi = total_L * KM2MI

    def cost_per_in_mi(coef: list, DN_in: float, L_mi: float) -> float:
        return coef[0] * DN_in ** coef[1] * L_mi ** coef[2]

    diam_col = "DN"
    for cost_type in ["labor", "misc", "ROW"]:
        cost_per_in_mi_val = anl_cost_overrides[cost_type]
        # If no override specified, use defaults
        if cost_per_in_mi_val is None:
            cost_per_in_mi_val = costs.apply(
                lambda x: cost_per_in_mi(
                    ANL_COEFS[loc][cost_type], x[diam_col] * MM2IN, L_mi
                ),
                axis=1,
            )
        costs[f"{cost_type} cost [$]"] = (
            cost_per_in_mi_val * costs[diam_col] * MM2IN * L_mi
        )

    return costs


def get_mat_costs(
    schedules_spec: pd.DataFrame,
    total_L: float,
    steel_costs_kg: pd.DataFrame,
    mat_cost_override: float,
):
    """
    Calculates the material cost based on $/kg from Savoy for each grade
    Inc., S. P. Live Stock List & Current Price. https://www.savoypipinginc.com/blog/live-stock-and-current-price.html. Accessed September 22, 2022.

    Users can alternatively provide a $/in/mi override to calculate material cost
    """
    rho_steel = 7840  # kg/m3
    L_m = total_L / M2KM
    L_mi = total_L * KM2MI

    def get_volume(od_mm: float, id_mm: float, L_m: float) -> float:
        return np.pi / 4 * (od_mm**2 - id_mm**2) / M2MM**2 * L_m

    od_col = "Outer diameter (mm)"
    id_col = "Inner diameter (mm)"
    schedules_spec["volume [m3]"] = schedules_spec.apply(
        lambda x: get_volume(x[od_col], x[id_col], L_m),
        axis=1,
    )
    schedules_spec["weight [kg]"] = schedules_spec["volume [m3]"] * rho_steel

    # If mat cost override is not specified, use $/kg savoy costing
    if mat_cost_override is not None:
        schedules_spec["mat cost [$]"] = (
            mat_cost_override * L_mi * schedules_spec["DN"] * MM2IN
        )
    else:
        schedules_spec["mat cost [$]"] = schedules_spec.apply(
            lambda x: x["weight [kg]"]
            * steel_costs_kg.loc[
                steel_costs_kg["Grade"] == x["Grade"], "Price [$/kg]"
            ].iat[0],
            axis=1,
        )

    return schedules_spec


def get_min_diameter_of_pipe(
    L: float, m_dot: float, p_inlet: float, p_outlet: float
) -> float:
    """
    Overview:
    ---------
        This function returns the diameter of a pipe for a given length,flow rate, and pressure boundaries

    Parameters:
    -----------
        L : float - Length of pipeline [km]
        m_dot : float = Mass flow rate [kg/s]
        p_inlet : float = Pressure at inlet of pipe [bar]
        p_outlet : float = Pressure at outlet of pipe [bar]

    Returns:
    --------
        diameter_mm : float - Diameter of pipe [mm]

    """

    p_in_Pa = p_inlet * BAR2PA
    p_out_Pa = p_outlet * BAR2PA

    p_avg = 2 / 3 * (p_in_Pa + p_out_Pa - p_in_Pa * p_out_Pa / (p_in_Pa + p_out_Pa))
    p_diff = (p_in_Pa**2 - p_out_Pa**2) ** 0.5
    T = 15 + 273.15  # Temperature [K]
    R = 8.314  # J/mol-K
    z_fit_params = (6.5466916131e-9, 9.9941320278e-1)  # Slope fit for 15C
    z = z_fit_params[0] * p_avg + z_fit_params[1]
    zrt = z * R * T
    mw = 2.016 / 1_000  # kg/mol for hydrogen
    RO = 0.012  # mm Roughness
    mu = 8.764167e-6  # viscosity
    L_m = L / M2KM

    f_list = [0.01]

    # Diameter depends on Re and f, but are functions of d. So use initial guess
    # of f-0.01, then iteratively solve until f is no longer changing
    err = np.inf
    max_iter = 50
    while err > 0.001:
        d_m = (
            m_dot / p_diff * 4 / np.pi * (mw / zrt / f_list[-1] / L_m) ** (-0.5)
        ) ** (1 / 2.5)
        d_mm = d_m * M2MM
        Re = 4 * m_dot / (np.pi * d_m * mu)
        f_list.append(
            (-2 * np.log10(4.518 / Re * np.log10(Re / 7) + RO / (3.71 * d_mm))) ** (-2)
        )
        err = abs((f_list[-1] - f_list[-2]) / f_list[-2])

        # Error out if no solution after max iterations
        if len(f_list) > max_iter:
            raise ValueError(f"Could not find pipe diameter in {max_iter} iterations")

    return d_mm


if __name__ == "__main__":
    L = 8  # Length [km]
    m_dot = 1.5  # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30  # Inlet pressure [bar]
    p_outlet = 10  # Outlet pressure [bar]
    depth = 80  # depth of pipe [m]
    costs = run_pipe_analysis(L, m_dot, p_inlet, p_outlet, depth)

    for col in costs.columns:
        print(col, costs[col][0])
