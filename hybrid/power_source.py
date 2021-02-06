import logging
from typing import Iterable, Sequence
from hybrid.sites import SiteInfo
import PySAM.Singleowner as Singleowner

from hybrid.log import hybrid_logger as logger


class PowerSource:
    def __init__(self, name, site: SiteInfo, system_model, financial_model):
        """
        Abstract class for a renewable energy power plant simulation.
        """
        self.name = name
        self.site = site
        self._system_model = system_model
        self._financial_model = financial_model
        self._layout = None
        self.set_construction_financing_cost_per_kw(financial_model.FinancialParameters.construction_financing_cost \
                                                    / financial_model.FinancialParameters.system_capacity)

    @property
    def system_capacity_kw(self) -> float:
        raise NotImplementedError

    @property
    def ppa_price(self):
        if self._financial_model:
            return self._financial_model.value("ppa_price_input")

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        if not isinstance(ppa_price, Iterable):
            ppa_price = (ppa_price,)
        if self._financial_model:
            self._financial_model.value("ppa_price_input", ppa_price)

    @property
    def dispatch_factors(self):
        if self._financial_model:
            return self._financial_model.value("dispatch_factors_ts")

    @dispatch_factors.setter
    def dispatch_factors(self, dispatch_factors):
        if not isinstance(dispatch_factors, Iterable):
            dispatch_factors = (dispatch_factors,)
        if self._financial_model:
            self._financial_model.value("ppa_multiplier_model", 1)
            self._financial_model.value("dispatch_factors_ts", dispatch_factors)

    def get_total_installed_cost_dollars(self) -> float:
        return self._financial_model.SystemCosts.total_installed_cost

    def set_total_installed_cost_dollars(self, total_installed_cost_dollars: float):
        self._financial_model.SystemCosts.total_installed_cost = total_installed_cost_dollars
        logger.info("{} set total_installed_cost to ${}".format(self.name, total_installed_cost_dollars))

    def set_construction_financing_cost_per_kw(self, construction_financing_cost_per_kw):
        self._construction_financing_cost_per_kw = construction_financing_cost_per_kw

    def get_construction_financing_cost(self) -> float:
        return self._construction_financing_cost_per_kw * self.system_capacity_kw

    def simulate(self, project_life: int = 25):
        """
        Run the system and financial model
        """
        if not self._system_model:
            return
        self._system_model.execute(0)

        if not self._financial_model:
            return

        self._financial_model.value("construction_financing_cost", self.get_construction_financing_cost())

        self._financial_model.Revenue.ppa_soln_mode = 1

        self._financial_model.Lifetime.system_use_lifetime_output = 1
        self._financial_model.FinancialParameters.analysis_period = project_life
        single_year_gen = self._financial_model.SystemOutput.gen
        self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self._financial_model.SystemOutput.system_pre_curtailment_kwac = self._system_model.Outputs.gen * project_life
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self._system_model.Outputs.annual_energy

        self._financial_model.execute(0)
        logger.info("{} simulation executed".format(self.name))

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kw:
            return self._system_model.Outputs.gen
        else:
            return [0] * self.site.n_timesteps

    def copy(self):
        """
        :return: new instance
        """
        raise NotImplementedError

    def plot(self,
             figure=None,
             axes=None,
             color='b',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        self._layout.plot(figure, axes, color, site_border_color, site_alpha, linewidth)
