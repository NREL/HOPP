import PySAM.Windpower as Windpower
import PySAM.Singleowner as Singleowner
from hopp.simulation.technologies.layout.wind_layout_tools import create_grid
#https://nrel-pysam.readthedocs.io/en/main/modules/Windpower.html
def set_wind_farm_layout(system_model,n_turbs,):
    farm_dict = {
        "wind_farm_wake_model": wake_model_num,
        "system_capacity": wind_capacity_kW,
        "wind_farm_xCoordinates": x_pos,
        "wind_farm_yCoordinates": y_pos}
    system_model.Farm.assign(farm_dict)

    return system_model

def set_turbine_params(system_model,params):
    
    turb_dict = {"wind_turbine_hub_ht":hub_ht,
    "wind_turbine_max_cp":cp_max,
    "wind_turbine_powercurve_powerout": power_kW,
    "wind_turbine_powercurve_windspeeds": v,
    "wind_turbine_rotor_diameter": rotor_diam,
    }
    system_model.Turbine.assign(turb_dict)
    return system_model
