import numpy as np

import PySAM.Windpower as windpower
from turbine_models.parser import Turbines
import hopp.tools.design.wind.power_curve_tools as curve_tools
from hopp.utilities.log import hybrid_logger as logger

def extract_power_curve(turbine_specs: dict, model_name: str):
    """Creates power-curve for turbine based on available data and formats it for the corresponding simulation model.

    Args:
        turbine_specs (dict): turbine specs loaded from turbine-models library.
        model_name (str): wind simulation model, either "pysam" or "floris".

    Raises:
        ValueError: if turbine data doesn't have the minimum required power-curve information.
        ValueError: if model name is not either 'pysam' or 'floris'

    Returns:
        dict: power-curve dictionary formatted for the corresponding ``model_name``.
    """

    if model_name not in ("floris", "pysam"):
        raise ValueError(f"model_name {model_name} is invalid, options are either 'floris' or 'pysam'.")
    turbine_specs["power_curve"] = turbine_specs["power_curve"].dropna()
    wind_speeds = np.nan_to_num(turbine_specs["power_curve"]["wind_speed_ms"].to_list())
    turbine_curve_cols = turbine_specs["power_curve"].columns.to_list()
    
    has_cp_curve = "cp" in turbine_curve_cols
    has_power_curve = "power_kw" in turbine_curve_cols
    has_ct_curve = "ct" in turbine_curve_cols
    
    if not has_cp_curve and not has_power_curve:
        turbine_name = turbine_specs["name"]
        msg = (
            f"Turbine {turbine_name} does not have the minimum required power curve data. "
            "Either power_kw or cp are required."
            )
        raise ValueError(msg)
    
    if has_cp_curve:
        cp_curve = np.array(turbine_specs["power_curve"]["cp"].to_list())
        cp_curve = np.nan_to_num(cp_curve)
        cp_curve = np.where(cp_curve<0,0,cp_curve).tolist()

    if has_power_curve:
        power_curve_kw = np.array(turbine_specs["power_curve"]["power_kw"].to_list())
        power_curve_kw = np.nan_to_num(power_curve_kw)
        power_curve_kw = np.where(power_curve_kw<0,0,power_curve_kw)
        power_curve_kw = np.where(power_curve_kw>turbine_specs["rated_power"],turbine_specs["rated_power"],power_curve_kw).tolist()

    if has_cp_curve and not has_power_curve:
        power_curve_kw = curve_tools.calculate_power_from_cp(wind_speeds,cp_curve,turbine_specs["rotor_diameter"],turbine_specs["rated_power"])
    
    if has_power_curve and not has_cp_curve:
        cp_curve = curve_tools.calculate_cp_from_power(wind_speeds,power_curve_kw)
        
    if has_ct_curve:
        ct = turbine_specs["power_curve"]["ct"].to_list()
    else:
        ct = curve_tools.estimate_thrust_coefficient(wind_speeds,cp_curve)
    
    _, cp_curve = curve_tools.pad_power_curve(wind_speeds,cp_curve)
    _, ct = curve_tools.pad_power_curve(wind_speeds,ct)
    wind_speeds, power_curve_kw = curve_tools.pad_power_curve(wind_speeds,power_curve_kw)
    
    if model_name == "floris":
        power_thrust_table = {
            "wind_speed":wind_speeds,
            "power":power_curve_kw,
            "thrust_coefficient":ct,
            }
        return power_thrust_table
    
    # if model_name is "pysam"
    power_thrust_table = {
        "wind_turbine_max_cp": max(cp_curve),
        "wind_turbine_ct_curve":ct,
        "wind_turbine_powercurve_windspeeds":wind_speeds,
        "wind_turbine_powercurve_powerout":power_curve_kw,
        }
    return power_thrust_table


def check_hub_height(turbine_specs, wind_plant):
    """Check the hub-height from the turbine-library specs against the other possible hub-height entries. 
    If multiple hub-height options are available from the turbine_specs, this method will choose 
    one based on other user-input parameters within wind_plant. The other variables checked are:
    
    1) wind_plant.config.hub_height
    2) wind_plant.site.hub_height
    3) wind_plant._system_model.Turbine.wind_turbine_hub_ht (for PySAM simulations only)
    4) if none of the heights from 1-3 match a possible hub-height option, the hub-height is chosen
    as the median hub-height from the list of options from turbine-library.

    Args:
        turbine_specs (dict): turbine specs loaded from turbine-models library.
        wind_plant (None | :obj:`hopp.simulation.technologies.wind.floris.Floris` | :obj:`hopp.simulation.technologies.wind.wind_plant.WindPlant`): wind 
            plant object for either PySAM or FLORIS wind simulation model.

    Returns:
        float: hub-height to use in meters.
    """
    turbine_name = turbine_specs["name"]
    # if multiple hub height options are available
    if isinstance(turbine_specs["hub_height"],list):
        # check for hub height in wind_plant
        is_pysam = isinstance(wind_plant,windpower.Windpower)
            
        # check if hub_height was put in WindConfig
        if (hub_height := wind_plant.config.hub_height) is not None:
            if any(float(k) == float(hub_height) for k in turbine_specs["hub_height"]):
                msg = (
                    f"Multiple hub height options available for {turbine_name} turbine. " 
                    f"Setting hub height to WindConfig hub_height: {hub_height}"
                )
                logger.info(msg)
                return hub_height
        
        # check the hub_height used for wind resource
        if (hub_height := wind_plant.site.hub_height) is not None:
            if any(float(k) == float(hub_height) for k in turbine_specs["hub_height"]):
                msg = (
                    f"Multiple hub height options available for {turbine_name} turbine. " 
                    f"Setting hub height to WindConfig hub_height: {hub_height}"
                )
                logger.info(msg)
                return hub_height
        
        # check the hub-height of PySAM wind turbine object
        if is_pysam:
            if any(float(k) == float(wind_plant._system_model.Turbine.wind_turbine_hub_ht) for k in turbine_specs["hub_height"]):
                hub_height = wind_plant._system_model.Turbine.wind_turbine_hub_ht
                msg = (
                    f"Multiple hub height options available for {turbine_name} turbine. "
                    f"Setting hub height to WindPower.WindPower.Turbine.wind_turbine_hub_ht: {hub_height}"
                )
                logger.info(msg)
                return hub_height
        
        # set hub height as median from options
        else:
            hub_height = np.median(turbine_specs["hub_height"])
            msg = (
                f"Multiple hub height options available for {turbine_name} turbine. "
                f"Setting hub height to median available height: {hub_height}"
            )
            logger.info(msg)
            return hub_height
                
    else:
        hub_height = turbine_specs["hub_height"]
        if wind_plant is not None:
            if wind_plant.config.hub_height is not None:
                if hub_height != wind_plant.config.hub_height:
                    msg = (
                        f"Turbine hub height ({hub_height}) does not equal "
                        f"wind_plant.config.hub_height ({wind_plant.config.hub_height})"
                    )
                    logger.warning(msg)
            if hub_height != wind_plant.site.hub_height:
                msg = (
                    f"Turbine hub height ({hub_height}) does not equal "
                    f"site_info.hub_height ({wind_plant.site.hub_height})"
                )
                logger.warning(msg)

    return hub_height


def get_pysam_turbine_specs(turbine_name, wind_plant):
    """Load turbine data from turbine-models library to use with PySAM wind simulation.

    Args:
        turbine_name (str): name of turbine in turbine-models library
        wind_plant (:obj:`hopp.simulation.technologies.wind.wind_plant.WindPlant`): wind plant object.

    Raises:
        ValueError: if turbine is missing data.

    Returns:
        dict: turbine model dictionary formatted for PySAM.
    """
    t_lib = Turbines()
    turbine_specs = t_lib.specs(turbine_name)
    if isinstance(turbine_specs,dict):
        turbine_dict = extract_power_curve(turbine_specs, model_name = "pysam")

        hub_height = check_hub_height(turbine_specs,wind_plant)
        
        turbine_dict.update({
            "wind_turbine_rotor_diameter":turbine_specs["rotor_diameter"],
            "wind_turbine_hub_ht":hub_height,
            })
        return turbine_dict

    raise ValueError(f"Turbine {turbine_name} is missing some data, please try another turbine.")
    

def get_floris_turbine_specs(turbine_name, wind_plant):
    """Load turbine data from turbine-models library to use with FLORIS wind simulation. 
    
    Sets turbine's rated tip speed ratio (TSR) to 8.0 if not included in turbine data.
    Sets default values in the power thrust table as:
    
    - ``ref_air_density``: 1.225
    - ``ref_tilt``: 5.0
    - ``cosine_loss_exponent_yaw``: 1.88
    - ``cosine_loss_exponent_tilt``: 1.88

    Args:
        turbine_name (str): name of turbine in turbine-models library
        wind_plant (:obj:`hopp.simulation.technologies.wind.floris.Floris`): FLORIS wrapper object.

    Raises:
        ValueError: if turbine is missing data.

    Returns:
        dict: turbine model dictionary formatted for FLORIS.
    """
    t_lib = Turbines()
    turb_group = t_lib.find_group_for_turbine(turbine_name)
    turbine_specs = t_lib.specs(turbine_name,group = turb_group)
    if isinstance(turbine_specs,dict):
        
        hub_height = check_hub_height(turbine_specs,wind_plant)
        power_thrust_table = extract_power_curve(turbine_specs, model_name = "floris")
        
        turbine_specs.setdefault("rated_tsr", 8.0)
        if turbine_specs["rated_tsr"] is None:
            turbine_specs["rated_tsr"] = 8.0

        power_thrust_table.update({
            "ref_air_density": 1.225,
            "ref_tilt": turbine_specs.setdefault("rotor_tilt_angle", 5.0),
            "cosine_loss_exponent_yaw": 1.88,
            "cosine_loss_exponent_tilt": 1.88,
            })
        turbine_dict = {
            "turbine_type":turbine_name,
            "hub_height":hub_height,
            "TSR": turbine_specs["rated_tsr"],
            "rotor_diameter":turbine_specs["rotor_diameter"],
            "power_thrust_table": power_thrust_table,
        }
        return turbine_dict
    
    msg = (
        f"Turbine {turbine_name} is missing some data, "
        "please try another turbine."
    )
    raise ValueError(msg)
    
