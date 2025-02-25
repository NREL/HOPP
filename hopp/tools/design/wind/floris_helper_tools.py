import os
import numpy as np
from hopp.utilities.utilities import write_yaml
from floris.turbine_library.turbine_previewer import INTERNAL_LIBRARY
from hopp.utilities import load_yaml
from hopp.tools.design.wind.turbine_library_interface_tools import get_floris_turbine_specs
from hopp.tools.design.wind.turbine_library_tools import check_turbine_name, check_turbine_library_for_turbine

def check_output_formatting(orig_dict):
    """Recursive method to convert arrays to lists and numerical entries to floats. 
    This is primarily used before writing a dictionary to a .yaml file to ensure 
    proper output formatting.

    Args:
        orig_dict (dict): input dictionary

    Returns:
        dict: input dictionary with reformatted values.
    """
    for key, val in orig_dict.items():
        if isinstance(val, dict):
            tmp = check_output_formatting(orig_dict.get(key, { }))
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i,k in enumerate(key):
                    if isinstance(orig_dict[k], str):
                        orig_dict[k] = (orig_dict.get(key, []) + val[i])
                    elif isinstance(orig_dict[k], bool):
                        orig_dict[k] = (orig_dict.get(key, []) + val[i])
                    elif isinstance(orig_dict[k], (list, np.ndarray)):
                        new_val = [float(v) for v in val[i]]
                        orig_dict[k] = new_val
                    else:
                        orig_dict[k] = float(val[i])
            elif isinstance(key,str):
                if not isinstance(orig_dict[key], (str, bool)):
                    if isinstance(orig_dict[key], (list, np.ndarray)):
                        new_val = [float(v) for v in val]
                        orig_dict[key] = new_val
                    else:
                        orig_dict[key] = float(val)
    return orig_dict

def write_floris_layout_to_file(layout_x,layout_y,output_dir,turbine_desc):
    """Export wind farm layout to floris-friendly .yaml file.

    Args:
        layout_x (list[float]): x-coordinates of turbines
        layout_y (list[float]): y-coordinates of turbines
        output_dir (str): output folder to write layout file to.
        turbine_desc (str): turbine name or description.
    """

    layout_x = [float(x) for x in layout_x]
    layout_y = [float(y) for y in layout_y]

    layout = {"layout_x":layout_x,"layout_y":layout_y}
    n_turbs = len(layout_x)
    output_fpath = os.path.join(output_dir,f"layout_{turbine_desc}_{n_turbs}turbs.yaml")
    write_yaml(output_fpath,layout)

def write_turbine_to_floris_file(turbine_dict,output_dir):
    """Export turbine model to floris-friendly .yaml file.

    Args:
        turbine_dict (dict): turbine entry of floris_config file
        output_dir (str): output folder to write turbine model file to.
    """
    turb_name = turbine_dict["turbine_type"]
    output_fpath = os.path.join(output_dir,f"floris_turbine_{turb_name}.yaml")
    new_dict = check_output_formatting(turbine_dict)
    write_yaml(output_fpath,new_dict)

def check_floris_library_for_turbine(turbine_name):
    """Check if a turbine exists in the floris internal library.

    Args:
        turbine_name (str): name of turbine

    Returns:
        bool: whether turbine exists in floris internal library or not.
    """
    floris_library_fpath = INTERNAL_LIBRARY / "{}.yaml".format(turbine_name)
    if os.path.isfile(floris_library_fpath):
        return True
    else:
        return False

def load_turbine_from_floris_library(turbine_name):
    """Load turbine model file from floris internal library.

    Args:
        turbine_name (str): name of turbine

    Raises:
        FileNotFoundError: if file does not exist in floris internal library.

    Returns:
        dict: floris turbine model dictionary
    """
    floris_library_fpath = INTERNAL_LIBRARY / "{}.yaml".format(turbine_name)
    if os.path.isfile(floris_library_fpath):
        turb_dict = load_yaml(floris_library_fpath)
        turb_dict.pop("power_thrust_data_file")
        return turb_dict
    else:
        raise FileNotFoundError("Floris library file for turbine {} does not exist.".format(turbine_name))

def check_libraries_for_turbine_name_floris(turbine_name,floris_model):
    """Check the FLORIS internal turbine library and the turbine-models library for 
    a turbine of ``turbine_name``. Return a FLORIS-compatible dictionary of the turbine 
    parameters if the ``turbine_name`` if valid. If the ``turbine_name`` is invalid, 
    return a warning message string.

    Args:
        turbine_name (str): name of turbine
        floris_model (FlorisModel): _description_

    Returns:
        dict | str: FLORIS-compatible dict of the turbine parameters for a valid ``turbine_name``. 
        If the ``turbine_name`` is invalid, return a warning message string.
    """
    is_floris_lib_turbine = check_floris_library_for_turbine(turbine_name)
    is_turb_lib_turbine = check_turbine_library_for_turbine(turbine_name)

    if is_floris_lib_turbine:
        turb_dict = load_turbine_from_floris_library(turbine_name)
        return turb_dict
    elif is_turb_lib_turbine:
        floris_model.value("turbine_name",turbine_name)
        turb_dict = get_floris_turbine_specs(turbine_name,floris_model)
        return turb_dict
    else:
        best_match_name = check_turbine_name(turbine_name)
        warning_str = f"turbine name {turbine_name} not found in floris or turbine-models library. Did you mean {best_match_name}"
        return warning_str