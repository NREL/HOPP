from __future__ import annotations
from pathlib import Path
from typing import Union

from hopp.simulation.hopp import Hopp


class HoppInterface():
    def __init__(self, configuration: Union[dict, str, Path]):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            self.hopp = Hopp.from_file(self.configuration)

        elif isinstance(self.configuration, dict):
            self.hopp = Hopp.from_dict(self.configuration)

    def reinitialize(self):
        pass

    def simulate(self, project_life):
        self.hopp.simulate(project_life)

    def parse_output(self):
        self.annual_energies = self.hopp.system.annual_energies
        self.wind_plus_solar_npv = self.hopp.system.net_present_values.wind + self.hopp.system.net_present_values.pv
        self.npvs = self.hopp.system.net_present_values
        self.wind_installed_cost = self.hopp.system.wind.total_installed_cost
        self.solar_installed_cost = self.hopp.system.pv.total_installed_cost
        self.hybrid_installed_cost = self.hopp.system.grid.total_installed_cost

    def print_output(self):
        print("Wind Installed Cost: {}".format(self.wind_installed_cost))
        print("Solar Installed Cost: {}".format(self.solar_installed_cost))
        print("Hybrid Installed Cost: {}".format(self.hybrid_installed_cost))
        print("Wind NPV: {}".format(self.hopp.system.net_present_values.wind))
        print("Solar NPV: {}".format(self.hopp.system.net_present_values.pv))
        print("Hybrid NPV: {}".format(self.hopp.system.net_present_values.hybrid))
        print("Wind + Solar Expected NPV: {}".format(self.wind_plus_solar_npv))
