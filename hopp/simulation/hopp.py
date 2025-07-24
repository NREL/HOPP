from __future__ import annotations
import yaml
import warnings
from attrs import define, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
from hopp.simulation.base import BaseClass
from hopp.simulation.hybrid_simulation import HybridSimulation, TechnologiesConfig
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities import load_yaml


@define
class Hopp(BaseClass):
    site: Union[dict, SiteInfo] = field()
    technologies: dict = field()
    name: Optional[str] = field(converter=str, default="HOPP Simulation")
    config: Optional[dict] = field(default=None)

    system: HybridSimulation = field(init=False)

    def __attrs_post_init__(self) -> None:
        # self.interconnection_size_mw = self.config['grid_config']['interconnection_size_mw']
        self.config = self.config or {}
        
        if isinstance(self.site, dict):
            site = SiteInfo.from_dict(self.site)
        else:
            site = self.site

        tech_config = TechnologiesConfig.from_dict(self.technologies)

        self.system = HybridSimulation(
            site,
            tech_config,
            self.config.get("dispatch_options") or {},
            self.config.get("cost_info") or {},
            self.config.get("simulation_options") or {},
        )

        # self.system.ppa_price = self.config['grid_config']['ppa_price']

    def simulate(self, project_life: int = 25, lifetime_sim: bool = False):
        self.system.simulate(project_life, lifetime_sim)
    
    def simulate_power(self, project_life: int = 25, lifetime_sim: bool = False):
        self.system.simulate_power(project_life, lifetime_sim)

    # I/O

    @classmethod
    def from_file(cls, input_file_path: Union[str, Path], filetype: Optional[str] = None):
        """Creates an `Hopp` instance from an input file. Must be filetype YAML.

        Args:
            input_file_path (str): The relative or absolute file path and name to the
                input file.
            filetype (str): The type to export: [YAML]

        Returns:
            Floris: The class object instance.
        """
        input_file_path = Path(input_file_path).resolve()
        if filetype is None:
            filetype = input_file_path.suffix.strip(".")

        # with open(input_file_path) as input_file:
        if filetype.lower() in ("yml", "yaml"):
            input_dict = load_yaml(input_file_path)
        else:
            raise ValueError("Supported import filetype is YAML")
        
        input_dict = overwrite_fin_values(input_dict)
        return Hopp.from_dict(input_dict)

    def to_file(self, output_file_path: str, filetype: str="YAML") -> None:
        """Converts the `Floris` object to an input-ready JSON or YAML file at `output_file_path`.

        Args:
            output_file_path (str): The full path and filename for where to save the file.
            filetype (str): The type to export: [YAML]
        """
        with open(output_file_path, "w+") as f:
            if filetype.lower() == "yaml":
                yaml.dump(self.as_dict(), f, default_flow_style=False)
            else:
                raise ValueError("Supported export filetype is YAML")



def overwrite_fin_values(hopp_config):
    """
    Overrides specific financial model values in the HOPP configuration with values from the `cost_info` section.

    This function ensures that the financial model values for technologies (e.g., wind, PV, battery) are updated 
    with the corresponding values provided in the `cost_info` section of the HOPP configuration. If discrepancies 
    are found between the values in the financial model and the `cost_info`, the financial model values are 
    overwritten, and a warning is issued to notify the user.

    Args:
        hopp_config (dict): The HOPP configuration dictionary containing information about technologies, 
            financial models, and cost information.

            Expected structure:
            - `technologies`: Contains technology-specific financial models (e.g., wind, PV, battery).
            - `config`: Contains the `cost_info` section with updated cost values.

    Returns:
        dict: The updated HOPP configuration dictionary with overwritten financial model values.

    Raises:
        UserWarning: If a financial model value is overwritten due to a mismatch with the `cost_info` value.

    Notes:
        - This function supports the following technologies: wind, PV (solar), and battery.
        - The following financial model values can be overwritten in individual technology financial models:
            - `om_capacity`: Fixed O&M costs per unit capacity [$/kW].
            - `om_production`: Variable O&M costs per unit production [$/MWh].

    Example:
        If the `cost_info` section specifies a new value for `wind_om_per_kw`, and this value differs from the 
        existing `om_capacity` value in the wind financial model, the `om_capacity` value will be updated, and 
        a warning will be issued.
    """
    # override individual fin_model values with cost_info values
    if ("config" in hopp_config.keys()) and ("cost_info" in hopp_config["config"]) and (hopp_config["config"]["cost_info"] is not None):
        if "wind" in hopp_config["technologies"]:
            if ("wind_om_per_kw" in hopp_config["config"]["cost_info"]) and (
                np.any(hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_capacity"]
                != hopp_config["config"]["cost_info"]["wind_om_per_kw"])
            ):
                for i in range(
                    len(hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_capacity"])
                ):
                    hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_capacity"][
                        i
                    ] = hopp_config["config"]["cost_info"]["wind_om_per_kw"]

                    om_fixed_wind_fin_model = hopp_config["technologies"]["wind"]["fin_model"][
                        "system_costs"
                    ]["om_capacity"][i]
                    wind_om_per_kw = hopp_config["config"]["cost_info"]["wind_om_per_kw"]
                    msg = (
                        f"'om_capacity[{i}]' in the wind 'fin_model' was {om_fixed_wind_fin_model},"
                        f" but 'wind_om_per_kw' in 'cost_info' was {wind_om_per_kw}. The 'om_capacity'"
                        " value in the wind 'fin_model' is being overwritten with the value from the"
                        " 'cost_info'"
                    )
                    warnings.warn(msg, UserWarning)
            if ("wind_om_per_mwh" in hopp_config["config"]["cost_info"]) and (
                hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_production"][0]
                != hopp_config["config"]["cost_info"]["wind_om_per_mwh"]
            ):
                # Use this to set the Production-based O&M amount [$/MWh]
                for i in range(
                    len(
                        hopp_config["technologies"]["wind"]["fin_model"]["system_costs"][
                            "om_production"
                        ]
                    )
                ):
                    hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_production"][
                        i
                    ] = hopp_config["config"]["cost_info"]["wind_om_per_mwh"]
                om_wind_variable_cost = hopp_config["technologies"]["wind"]["fin_model"][
                    "system_costs"
                ]["om_production"][i]
                wind_om_per_mwh = hopp_config["config"]["cost_info"]["wind_om_per_mwh"]
                msg = (
                    f"'om_production' in the wind 'fin_model' was {om_wind_variable_cost}, but"
                    f" 'wind_om_per_mwh' in 'cost_info' was {wind_om_per_mwh}. The 'om_production'"
                    " value in the wind 'fin_model' is being overwritten with the value from the"
                    " 'cost_info'"
                )
                warnings.warn(msg, UserWarning)

        if "pv" in hopp_config["technologies"]:
            if ("pv_om_per_kw" in hopp_config["config"]["cost_info"]) and (
                hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_capacity"][0]
                != hopp_config["config"]["cost_info"]["pv_om_per_kw"]
            ):
                for i in range(
                    len(hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_capacity"])
                ):
                    hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_capacity"][i] = (
                        hopp_config["config"]["cost_info"]["pv_om_per_kw"]
                    )

                    om_fixed_pv_fin_model = hopp_config["technologies"]["pv"]["fin_model"][
                        "system_costs"
                    ]["om_capacity"][i]
                    pv_om_per_kw = hopp_config["config"]["cost_info"]["pv_om_per_kw"]
                    msg = (
                        f"'om_capacity[{i}]' in the pv 'fin_model' was {om_fixed_pv_fin_model}, but"
                        f" 'pv_om_per_kw' in 'cost_info' was {pv_om_per_kw}. The 'om_capacity' value"
                        " in the pv 'fin_model' is being overwritten with the value from the"
                        " 'cost_info'"
                    )
                    warnings.warn(msg, UserWarning)
            if ("pv_om_per_mwh" in hopp_config["config"]["cost_info"]) and (
                hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_production"][0]
                != hopp_config["config"]["cost_info"]["pv_om_per_mwh"]
            ):
                # Use this to set the Production-based O&M amount [$/MWh]
                for i in range(
                    len(hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_production"])
                ):
                    hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_production"][
                        i
                    ] = hopp_config["config"]["cost_info"]["pv_om_per_mwh"]
                om_pv_variable_cost = hopp_config["technologies"]["pv"]["fin_model"]["system_costs"][
                    "om_production"
                ][i]
                pv_om_per_mwh = hopp_config["config"]["cost_info"]["pv_om_per_mwh"]
                msg = (
                    f"'om_production' in the pv 'fin_model' was {om_pv_variable_cost}, but"
                    f" 'pv_om_per_mwh' in 'cost_info' was {pv_om_per_mwh}. The 'om_production' value"
                    " in the pv 'fin_model' is being overwritten with the value from the 'cost_info'"
                )
                warnings.warn(msg, UserWarning)

        if "battery" in hopp_config["technologies"]:
            if ("battery_om_per_kw" in hopp_config["config"]["cost_info"]) \
                and (hopp_config["technologies"]["battery"]["fin_model"]["system_costs"]["om_capacity"][0]
                        != hopp_config["config"]["cost_info"]["battery_om_per_kw"]
            ):
                for i in range(
                    len(
                        hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                            "om_capacity"
                        ]
                    )
                ):
                    hopp_config["technologies"]["battery"]["fin_model"]["system_costs"]["om_capacity"][
                        i
                    ] = hopp_config["config"]["cost_info"]["battery_om_per_kw"]

                om_batt_fixed_cost = hopp_config["technologies"]["battery"]["fin_model"][
                    "system_costs"
                ]["om_capacity"][i]
                battery_om_per_kw = hopp_config["config"]["cost_info"]["battery_om_per_kw"]
                msg = (
                    f"'om_capacity' in the battery 'fin_model' was {om_batt_fixed_cost}, but"
                    f" 'battery_om_per_kw' in 'cost_info' was {battery_om_per_kw}. The"
                    " 'om_capacity' value in the battery 'fin_model' is being overwritten with the"
                    " value from the 'cost_info'"
                )
                warnings.warn(msg, UserWarning)
            if ("battery_om_per_mwh" in hopp_config["config"]["cost_info"]) and (
                hopp_config["technologies"]["battery"]["fin_model"]["system_costs"]["om_production"][0]
                != hopp_config["config"]["cost_info"]["battery_om_per_mwh"]
            ):
                # Use this to set the Production-based O&M amount [$/MWh]
                for i in range(
                    len(
                        hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                            "om_production"
                        ]
                    )
                ):
                    hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                        "om_production"
                    ][i] = hopp_config["config"]["cost_info"]["battery_om_per_mwh"]
                om_batt_variable_cost = hopp_config["technologies"]["battery"]["fin_model"][
                    "system_costs"
                ]["om_production"][i]
                battery_om_per_mwh = hopp_config["config"]["cost_info"]["battery_om_per_mwh"]
                msg = (
                    f"'om_production' in the battery 'fin_model' was {om_batt_variable_cost}, but"
                    f" 'battery_om_per_mwh' in 'cost_info' was {battery_om_per_mwh}. The"
                    " 'om_production' value in the battery 'fin_model' is being overwritten with the"
                    " value from the 'cost_info'",
                )
                warnings.warn(msg, UserWarning)

    return hopp_config
