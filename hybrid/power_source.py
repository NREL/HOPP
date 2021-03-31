import logging
from typing import Iterable, Sequence
from hybrid.sites import SiteInfo
import PySAM.Singleowner as Singleowner

from hybrid.log import hybrid_logger as logger
from hybrid.dispatch.power_source_dispatch import PowerSourceDispatch


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
        self._dispatch = PowerSourceDispatch
        self.set_construction_financing_cost_per_kw(financial_model.FinancialParameters.construction_financing_cost \
                                                    / financial_model.FinancialParameters.system_capacity)

    def value(self, var_name, var_value=None):
        if var_value is None:
            val = None
            try:
                val = self._system_model.value(var_name)
            except:
                try:
                    val = self._financial_model.value(var_name)
                except:
                    raise ValueError("Variable {} not found in technology or financial model".format(var_name))
            return val
        else:
            try:
                val = self._system_model.value(var_name, var_value)
            except:
                try:
                    val = self._financial_model.value(var_name, var_value)
                except:
                    raise ValueError("Variable {} not found in technology or financial model".format(var_name))

    #
    # Inputs
    #
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

    @property
    def total_installed_cost(self) -> float:
        return self._financial_model.value("total_installed_cost")

    @total_installed_cost.setter
    def total_installed_cost(self, total_installed_cost_dollars: float):
        self._financial_model.value("total_installed_cost", total_installed_cost_dollars)
        logger.info("{} set total_installed_cost to ${}".format(self.name, total_installed_cost_dollars))

    @property
    def om_capacity(self):
        return self._financial_model.value("om_capacity")

    @om_capacity.setter
    def om_capacity(self, om_dollar_per_kw: float):
        self._financial_model.value("om_capacity", om_dollar_per_kw)

    def initialize_dispatch_model_parameters(self):
        self.dispatch.time_weighting_factor = 1.0
        self.dispatch.generation_cost = self.om_capacity[0]*1e3/8760. # shouldn't this be 8760 * self.site.interval / 60.?

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        n_horizon = len(self.dispatch.blocks.index_set())
        horizon_generation = self.generation_profile[start_time:start_time+n_horizon]
        self.dispatch.available_generation = [gen_kwh/1e3 for gen_kwh in horizon_generation]

        self.update_dispatch_time_steps_and_prices(start_time)

    def update_dispatch_time_steps_and_prices(self, start_time: int):
        n_horizon = len(self.dispatch.blocks.index_set())
        self.dispatch.time_duration = [self.site.interval / 60.] * n_horizon

        if start_time + n_horizon > len(self.dispatch_factors):
            prices = list(self.dispatch_factors[start_time:])
            prices.extend(list(self.dispatch_factors[0:n_horizon-len(prices)]))
        else:
            prices = self.dispatch_factors[start_time:start_time+n_horizon]
        self.dispatch.electricity_sell_price = [norm_price*self.ppa_price[0]*1e3
                                                for norm_price
                                                in prices]

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

        if self.system_capacity_kw <= 0:
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

    #
    # Outputs
    #
    @property
    def dispatch(self):
        return self._dispatch

    @property
    def annual_energy_kw(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("annual_energy")
        else:
            return 0

    @property
    def generation_profile(self) -> list:
        if self.system_capacity_kw:
            return list(self._system_model.value("gen"))
        else:
            return [0] * self.site.n_timesteps

    @property
    def capacity_factor(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor")
        else:
            return 0

    @property
    def net_present_value(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("project_return_aftertax_npv")
        else:
            return 0

    @property
    def internal_rate_of_return(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("project_return_aftertax_irr")
        else:
            return 0

    @property
    def levelized_cost_of_energy_real(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("lcoe_real")
        else:
            return 0

    @property
    def levelized_cost_of_energy_nominal(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("lcoe_nom")
        else:
            return 0

    @property
    def total_revenue(self) -> list:
        if self.system_capacity_kw > 0 and self._financial_model:
            return list(self._financial_model.value("cf_total_revenue"))
        else:
            return [0]

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
