"""
layout.py

Code to calculate system layout properties
"""

import math


def calculate_solar_extent(pv):
    """
    Given a PV dictionary from defaults_data.py, calculate the x, y, and z extent of a solar field
    :param pv: dict, dictionary of PV properties, as in defaults/pv_singleowner.py
    :return: tuple of (x_extent_meters, y_extent_meters, z_extent_meters), assuming an origin of (0, 0, 0)
    """

    gcr = pv['SystemDesign']['subarray1_gcr']  # ground coverage ratio
    tilt_degrees = pv['SystemDesign']['subarray1_tilt']
    modules_per_string = pv['SystemDesign']['subarray1_modules_per_string']
    n_strings = pv['SystemDesign']['subarray1_nstrings']

    nmodx = pv['Layout']['subarray1_nmodx']  # number of modules in x direction
    nmody = pv['Layout']['subarray1_nmody']  # number of modules in y direction (per row)
    orientation = pv['Layout']['subarray1_mod_orient']  # 0 = portrait, 1 = landscape
    module_width = pv['CECPerformanceModelWithModuleDatabase']['cec_module_width']
    module_length = pv['CECPerformanceModelWithModuleDatabase']['cec_module_length']

    # calculate number of rows
    n_modules = n_strings * modules_per_string
    modules_per_row = nmodx * nmody
    n_rows = math.ceil(n_modules / modules_per_row)

    # length of a row depends on the panel orientation
    if orientation == 0:
        width = module_width
        length = module_length
    else:
        width = module_length
        length = module_width

    row_spacing = (nmody * length) / gcr

    x_extent_meters = math.ceil(nmodx * width)
    y_extent_meters = math.ceil(row_spacing * n_rows)
    z_extent_meters = math.ceil(nmody * length * math.sin(math.pi * tilt_degrees / 180))

    extent = (x_extent_meters, y_extent_meters, z_extent_meters)
    return extent
