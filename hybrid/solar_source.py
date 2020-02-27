from typing import Union, Sequence

import PySAM.Pvsamv1 as Pvsam
import PySAM.Pvwattsv7 as Pvwatts

from hybrid.site_info import SiteInfo
from hybrid.power_source import *


class SolarPlant(PowerSource):
    system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv7]
    financial_model: Singleowner.Singleowner

    def __init__(self, site: SiteInfo, system_capacity_kw: float, detailed_not_simple: bool = False):
        """

        :param system_capacity_kw:
        :param detailed_not_simple:
            Detailed model uses Pvsamv1, simple uses PVWatts
        """
        super().__init__(site)

        self.detailed_not_simple: bool = detailed_not_simple

        if not detailed_not_simple:
            self.system_model = Pvwatts.default("PVWattsLeveragedPartnershipFlip")
            self.financial_model = Singleowner.from_existing(self.system_model, "PVWattsSingleOwner")
        else:
            self.system_model = Pvsam.default("FlatPlatePVSingleOwner")
            self.financial_model = Singleowner.from_existing(self.system_model, "FlatPlatePVSingleOwner")

        self.system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.total_installed_cost_dollars = 0
        self._construction_financing_cost_per_kw = self.financial_model.FinancialParameters.construction_financing_cost\
                                                   / self.financial_model.FinancialParameters.system_capacity
        self.financial_model.Revenue.ppa_soln_mode = 1

        self.system_capacity_kw: float = system_capacity_kw

    @property
    def system_capacity_kw(self) -> float:
        return self.system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        if self.detailed_not_simple:
            raise NotImplementedError("SolarPlant error: system_capacity setter for detailed pv")
        self.system_model.SystemDesign.system_capacity = size_kw
        logger.info("SolarPlant set system_capacity to {} kW".format(size_kw))

    @property
    def construction_financing_cost_per_kw(self):
        return self._construction_financing_cost_per_kw

    @property
    def total_installed_cost_dollars(self) -> float:
        return self.financial_model.SystemCosts.total_installed_cost

    @total_installed_cost_dollars.setter
    def total_installed_cost_dollars(self, total_installed_cost_dollars: float):
        self.financial_model.SystemCosts.total_installed_cost = total_installed_cost_dollars
        logger.info("SolarPlant set total_installed_cost to ${}".format(self.total_installed_cost_dollars))

    def simulate(self):
        """
        Runs the system, cost and financial models
        :return:
        """
        self.system_model.execute(0)
        if self.system_capacity_kw > 0:
            self.financial_model.execute(0)
        logger.info("SolarPlant simulation executed")

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kw > 0:
            if self.detailed_not_simple:
                return self.system_model.Outputs.gen
            else:
                return self.system_model.Outputs.ac
        else:
            return [0] * self.site.n_timesteps

    def annual_energy_kw(self) -> float:
        if self.system_capacity_kw > 0:
            return self.system_model.Outputs.annual_energy
        else:
            return 0

    def copy(self):
        raise NotImplementedError
