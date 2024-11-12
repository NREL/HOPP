from __future__ import annotations
from pathlib import Path
from typing import Union, TYPE_CHECKING

from hopp.simulation.hopp import Hopp, SiteInfo

# avoid potential circular dep
if TYPE_CHECKING:
    from hopp.simulation.hybrid_simulation import HybridSimulation


class HoppInterface:
    """
    Main interface for HOPP simulations.

    Args:
        configuration: Top level configuration for a HOPP simulation. Can be either
            a string/Path to a YAML configuration file, or a dict with the same 
            structure. The structure is:

                - **name**: Optional name for the simulation

                - **site**: Site information. See :class:`hopp.simulation.technologies.sites.SiteInfo`

                - **technologies**: Technology information. See :class:`hopp.simulation.hybrid_simulation.TechnologiesConfig`

                - **config**: Additional config options

                    - **dispatch_options**: Dispatch optimization options. See :class:`hopp.simulation.technologies.dispatch.hybrid_dispatch_options.HybridDispatchOptions`

                    - **cost_info**: Cost info. See :class:`hopp.tools.analysis.bos.cost_calculator.CostCalculator`

                    - **simulation_options**: Nested ``dict``, i.e., ``{'pv': {'skip_financial': bool}}`` (optional) nested dictionary of simulation options. First level key is technology consistent with ``technologies``

    """
    def __init__(self, configuration: Union[dict, str, Path]):
        self.reinitialize(configuration=configuration)

    def reinitialize(self, configuration: Union[dict, str, Path]):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            self.hopp = Hopp.from_file(self.configuration)

        elif isinstance(self.configuration, dict):
            self.hopp = Hopp.from_dict(self.configuration)

    def simulate(self, project_life: int = 25, lifetime_sim: bool = False):
        self.hopp.simulate(project_life, lifetime_sim)

    @property
    def system(self) -> "HybridSimulation":
        """Returns the configured simulation instance."""
        return self.hopp.system

    def parse_input(self):
        pass

    def parse_output(self):
        self.annual_energies = self.hopp.system.annual_energies
        self.wind_plus_solar_npv = self.hopp.system.net_present_values.wind + self.hopp.system.net_present_values.pv
        self.npvs = self.hopp.system.net_present_values
        self.wind_installed_cost = self.hopp.system.wind.total_installed_cost
        self.solar_installed_cost = self.hopp.system.pv.total_installed_cost
        self.hybrid_installed_cost = self.hopp.system.grid.total_installed_cost

    def print_output(self):
        print("Wind Installed Cost: {}".format(self.system.wind.total_installed_cost))
        print("Solar Installed Cost: {}".format(self.system.pv.total_installed_cost))
        print("Wave Installed Cost: {}".format(self.system.wave.total_installed_cost))
        print("Battery Installed Cost: {}".format(self.system.battery.total_installed_cost))
        print("Hybrid Installed Cost: {}".format(self.system.grid.total_installed_cost))
        print("Wind NPV: {}".format(self.hopp.system.net_present_values.wind))
        print("Solar NPV: {}".format(self.hopp.system.net_present_values.pv))
        print("Wave NPV: {}".format(self.hopp.system.net_present_values.wave))
        print("Battery NPV: {}".format(self.hopp.system.net_present_values.battery))
        print("Wave NPV: {}".format(self.hopp.system.net_present_values.hybrid*1E-2))
        print("Wind LCOE (USD/kWh): {}".format(self.hopp.system.lcoe_nom.wind*1E-2))
        print("Solar LCOE (USD/kWh): {}".format(self.hopp.system.lcoe_nom.pv*1E-2))
        print("Wave LCOE (USD/kWh): {}".format(self.hopp.system.lcoe_nom.wave*1E-2))
        print("Battery LCOE (USD/kWh): {}".format(self.hopp.system.lcoe_nom.battery*1E-2))
        print("Hybrid LCOE (USD/kWh): {}".format(self.hopp.system.lcoe_nom.hybrid*1E-2))
        # print("Wind + Solar Expected NPV: {}".format(self.wind_plus_solar_npv))
