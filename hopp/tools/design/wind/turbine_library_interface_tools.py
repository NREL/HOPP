from turbine_models.parser import Turbines
import hopp.tools.design.wind.power_curve_tools as curve_tools
from hopp.utilities.log import hybrid_logger as logger
import numpy as np
from floris.turbine_library import build_cosine_loss_turbine_dict
import PySAM.Windpower as windpower

def extract_power_curve(turbine_specs: dict, model_name: str):
    """Creates power-curve for turbine based on available data and formats it for the corresponding simulation model.

    Args:
        turbine_specs (dict): turbine specs loaded from turbine-models library.
        model_name (str): wind simulation model, either "pysam" or "floris".

    Raises:
        UserWarning: if turbine data doesnt have the minimum required power-curve information.

    Returns:
        dict: power-curve dictionary formatted for the corresponding `model_name`.
    """
    turbine_name = turbine_specs["name"]
    wind_speeds = turbine_specs["power_curve"]["wind_speed_ms"].to_list()
    turbine_curve_cols = turbine_specs["power_curve"].columns.to_list()
    if "power_kw" in turbine_curve_cols and "cp" in turbine_curve_cols:
        power_curve_kw = np.array(turbine_specs["power_curve"]["power_kw"].to_list())
        cp_curve = np.array(turbine_specs["power_curve"]["cp"].to_list())
        power_curve_kw = np.where(power_curve_kw<0,0,power_curve_kw)
        power_curve_kw = np.where(power_curve_kw>turbine_specs["rated_power"],turbine_specs["rated_power"],power_curve_kw)
        cp_curve = np.where(cp_curve<0,0,cp_curve)
        power_curve_kw = list(power_curve_kw)
        cp_curve = list(cp_curve)
    elif "power_kw" not in turbine_curve_cols and "cp" in turbine_curve_cols:
        cp_curve = np.array(turbine_specs["power_curve"]["cp"].to_list())
        cp_curve = np.where(cp_curve<0,0,cp_curve)
        cp_curve = list(cp_curve)

        power_curve_kw = curve_tools.calculate_power_from_cp(wind_speeds,cp_curve,turbine_specs["rotor_diameter"],turbine_specs["rated_power"])
    elif "power_kw" in turbine_curve_cols and "cp" not in turbine_curve_cols:
        power_curve_kw = np.array(turbine_specs["power_curve"]["power_kw"].to_list())
        power_curve_kw = np.where(power_curve_kw<0,0,power_curve_kw)
        power_curve_kw = np.where(power_curve_kw>turbine_specs["rated_power"],turbine_specs["rated_power"],power_curve_kw)
        power_curve_kw = list(power_curve_kw)
        cp_curve = curve_tools.calculate_cp_from_power(wind_speeds,power_curve_kw)
    else:
        raise UserWarning(f"turbine {turbine_name} does not have minimum required power curve data (needs either power_kw or cp)")
    if "ct" in turbine_specs["power_curve"].columns.to_list():
        ct = turbine_specs["power_curve"]["ct"].to_list()
    else:
        ct = curve_tools.estimate_thrust_coefficient(wind_speeds,cp_curve)
    
    _, cp_curve = curve_tools.pad_power_curve(wind_speeds,cp_curve)
    _, ct = curve_tools.pad_power_curve(wind_speeds,ct)
    wind_speeds, power_curve_kw = curve_tools.pad_power_curve(wind_speeds,power_curve_kw)
    if model_name == "floris":
        power_thrust_table = {
            # "ref_air_density": 1.225,
            # "ref_tilt": 5.0,
            # "cosine_loss_exponent_yaw": 1.88,
            # "cosine_loss_exponent_tilt": 1.88,
            "wind_speed":wind_speeds,
            "power":power_curve_kw,
            "thrust_coefficient":ct,
            }
    elif model_name == "pysam":
        power_thrust_table = {
            "wind_turbine_max_cp": max(cp_curve),
            "wind_turbine_ct_curve":ct,
            "wind_turbine_powercurve_windspeeds":wind_speeds,
            "wind_turbine_powercurve_powerout":power_curve_kw,
            }
    return power_thrust_table


def check_hub_height(turbine_specs,wind_plant):
    """Check the hub-height from the turbine-library specs against the other possible hub-height entries.
        If multiple hub-height options are available from the turbine_specs, it will choose one based on user-input information.

    Args:
        turbine_specs (dict): turbine specs loaded from turbine-models library.
        wind_plant (:obj:`hopp.simulation.technologies.wind.floris.Floris` | :obj:`hopp.simulation.technologies.wind.wind_plant.WindPlant`): wind 
            plant object for either PySAM or FLORIS wind simulation model.

    Returns:
        float: hub-height to use in meters.
    """
    turbine_name = turbine_specs["name"]
    # if multiple hub height options are available
    if isinstance(turbine_specs["hub_height"],list):
        # check for hub height in wind_plant
        if isinstance(wind_plant,object) and wind_plant is not None:
            
            # check if hub_height was put in WindConfig
            if wind_plant.config.hub_height is not None:
                if any(float(k) == float(wind_plant.config.hub_height) for k in turbine_specs["hub_height"]):
                    hub_height = wind_plant.config.hub_height
                    logger.info(f"multiple hub height options available for {turbine_name} turbine. Setting hub height to WindConfig hub_height: {hub_height}")
                    return hub_height
                    
            
            # check the hub_height used for wind resource
            elif wind_plant.site.hub_height is not None:
                if any(float(k) == float(wind_plant.site.hub_height) for k in turbine_specs["hub_height"]):
                    hub_height = wind_plant.site.hub_height
                    logger.info(f"multiple hub height options available for {turbine_name} turbine. Setting hub height to SiteInfo hub_height: {hub_height}")
                    return hub_height
            
            # check the hub-height of PySAM wind turbine object
            elif isinstance(wind_plant._system_model,windpower.Windpower):
                if any(float(k) == float(wind_plant._system_model.Turbine.wind_turbine_hub_ht) for k in turbine_specs["hub_height"]):
                    hub_height = wind_plant._system_model.Turbine.wind_turbine_hub_ht
                    logger.info(f"multiple hub height options available for {turbine_name} turbine. Setting hub height to WindPower.WindPower.Turbine.wind_turbine_hub_ht: {hub_height}")
                    return hub_height
            
            # set hub height as median from options
            else:
                hub_height = np.median(turbine_specs["hub_height"])
                logger.info(f"multiple hub height options available for {turbine_name} turbine. Setting hub height to median available height: {hub_height}")
                return hub_height
                
    else:
        hub_height = turbine_specs["hub_height"]
        if wind_plant.config.hub_height is not None:
            if hub_height != wind_plant.config.hub_height:
                logger.warning(f"turbine hub height ({hub_height}) does not equal wind_plant.config.hub_height ({wind_plant.config.hub_height})")
        if hub_height != wind_plant.site.hub_height:
            logger.warning(f"turbine hub height ({hub_height}) does not equal site_info.hub_height ({wind_plant.site.hub_height})")

    return hub_height


def get_pysam_turbine_specs(turbine_name,wind_plant):
    """Load and set turbine data from turbine-models library to use with PySAM wind simulation.

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
        
        wind_plant._system_model.Turbine.assign(turbine_dict)
        wind_plant.rotor_diameter = turbine_specs["rotor_diameter"]

    else:
        raise ValueError(f"turbine {turbine_name} is missing some data, please try another turbines")
    return turbine_dict

def get_floris_turbine_specs(turbine_name,wind_plant):
    """Load and set turbine data from turbine-models library to use with FLORIS wind simulation.

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

        power_thrust_table.update({
            "ref_air_density": 1.225,
            "ref_tilt": 5.0,
            "cosine_loss_exponent_yaw": 1.88,
            "cosine_loss_exponent_tilt": 1.88,
            # "TSR": 8.0,
            })
        turbine_dict = {
            "turbine_type":turbine_name,
            "turbine_rating":turbine_specs["rated_power"],
            "hub_height":hub_height,
            "TSR": 8.0,
            "rotor_diameter":turbine_specs["rotor_diameter"],
            # "operation_model": "cosine-loss",
            "power_thrust_table": power_thrust_table,
        }
        
    else:
        raise ValueError(f"turbine {turbine_name} is missing some data, please try another turbines")
    return turbine_dict
