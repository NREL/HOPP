from __future__ import annotations
import yaml
from attrs import define, field
from pathlib import Path
from typing import Optional, Union

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
