from turbine_models.parser import Turbines
from difflib import SequenceMatcher

def load_distributed_turbine_options():
    """Load list of distributed turbines from turbine-models library.

    Returns:
        list[str]: list of turbine names in group "distributed"
    """

    t_lib = Turbines()
    distributed_turbines = list(t_lib.turbines(group="distributed").values())
    return distributed_turbines

def load_land_based_turbine_options():
    """Load list of onshore turbines from turbine-models library.

    Returns:
        list[str]: list of turbine names in group "onshore"
    """

    t_lib = Turbines()
    lbw_turbines = list(t_lib.turbines(group="onshore").values())
    return lbw_turbines

def load_offshore_turbine_options():
    """Load list of offshore turbines from turbine-models library.

    Returns:
        list[str]: list of turbine names in group "offshore"
    """

    t_lib = Turbines()
    osw_turbines = list(t_lib.turbines(group="offshore").values())
    return osw_turbines

def check_turbine_name(turbine_name:str):
    """Check turbine-models library for turbine named ``turbine_name`` and return a valid turbine name.

    Args:
        turbine_name (str): name of turbine in turbine-models library

    Returns:
        str: turbine name that most closely matches the input ``turbine_name`` of the turbines 
            in turbine-models library.
    """

    t_lib = Turbines()
    valid_name = False
    best_match = ""
    max_match_ratio = 0.0
    for turb_group in t_lib.groups:
        turbines_in_group = t_lib.turbines(group = turb_group)
        if any(turb.lower()==turbine_name.lower() for turb in turbines_in_group.values()):
            valid_name = True
            return turbine_name
        elif any(turbine_name.lower() in turb.lower() for turb in turbines_in_group.values()):
            turbine_options = [turb for turb in turbines_in_group.values() if turbine_name.lower() in turb.lower()]
            if len(turbine_options)==1:
                best_match = turbine_options[0]
                return best_match
            else:
                for turb in turbine_options:
                    match_ratio = SequenceMatcher(None,turbine_name.lower(), turb.lower()).ratio()
                    if match_ratio>max_match_ratio:
                        best_match = str(turb)
                        max_match_ratio = max(match_ratio,max_match_ratio)

        else:
            for turb in turbines_in_group.values():
                match_ratio = SequenceMatcher(None,turbine_name.lower(), turb.lower()).ratio()
                if match_ratio>max_match_ratio:
                    best_match = str(turb)
                    max_match_ratio = max(match_ratio,max_match_ratio)
    if valid_name:
        return turbine_name
    else:
        return best_match

def check_turbine_library_for_turbine(turbine_name:str):
    """Check turbine-models library for turbine named ``turbine_name``.

    Args:
        turbine_name (str): name of turbine in turbine-models library

    Returns:
        bool: whether the input turbine name matches a turbine available in the turbine-models library.
    """

    t_lib = Turbines()
    valid_name = False
    for turb_group in t_lib.groups:
        turbines_in_group = t_lib.turbines(group = turb_group)
        if any(turb.lower()==turbine_name.lower() for turb in turbines_in_group.values()):
            valid_name = True
    return valid_name

def print_turbine_name_list():
    """Print the turbine names for each group of turbines in turbine-models library.
    """
    
    osw_turbs = load_offshore_turbine_options()
    
    print("-".join("" for i in range(25)))
    print("Offshore Turbine Names:")
    print("-".join("" for i in range(25)))
    osw_msg = "\n " + "\n ".join(t for t in osw_turbs)
    print(osw_msg)

    lbw_turbs = load_land_based_turbine_options()
    print("-".join("" for i in range(25)))
    print("Onshore Turbine Names:")
    print("-".join("" for i in range(25)))
    lbw_msg = "\n " + "\n ".join(t for t in lbw_turbs)
    print(lbw_msg)

    dw_turbs = load_distributed_turbine_options()
    print("-".join("" for i in range(25)))
    print("Distributed Turbine Names:")
    print("-".join("" for i in range(25)))
    dw_msg = "\n " + "\n ".join(t for t in dw_turbs)
    print(dw_msg)