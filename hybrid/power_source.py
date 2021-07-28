from typing import Iterable
import numpy as np
from hybrid.sites import SiteInfo

from hybrid.log import hybrid_logger as logger
from hybrid.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch


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
                    raise ValueError("Variable {} not found in technology or financial model {}".format(
                        var_name, self.__class__.__name__))
            return val
        else:
            try:
                val = self._system_model.value(var_name, var_value)
            except:
                try:
                    val = self._financial_model.value(var_name, var_value)
                except:
                    raise ValueError("Variable {} not found in technology or financial model {}".format(
                        var_name, self.__class__.__name__))

    #
    # Inputs
    #
    @property
    def system_capacity_kw(self) -> float:
        raise NotImplementedError

    @property
    def degradation(self) -> float:
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
    def capacity_credit_percent(self):
        return self._financial_model.value("cp_capacity_credit_percent")

    @capacity_credit_percent.setter
    def capacity_credit_percent(self, cap_credit_percent):
        if not isinstance(cap_credit_percent, Iterable):
            cap_credit_percent = (cap_credit_percent,)
        if self._financial_model:
            self._financial_model.value("cp_capacity_credit_percent", cap_credit_percent)

    @property
    def capacity_price(self):
        return self._financial_model.value("cp_capacity_payment_amount") * 1000

    @capacity_price.setter
    def capacity_price(self, cap_price_per_kw_year):
        if not isinstance(cap_price_per_kw_year, Iterable):
            cap_price_per_kw_year = (cap_price_per_kw_year,)
        if self._financial_model:
            cap_price_per_kw_year = [i * 1e3 for i in cap_price_per_kw_year]
            self._financial_model.value("cp_capacity_payment_amount", cap_price_per_kw_year)

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

        if project_life > 1:
            self._financial_model.Lifetime.system_use_lifetime_output = 1
        else:
            self._financial_model.Lifetime.system_use_lifetime_output = 0
        self._financial_model.FinancialParameters.analysis_period = project_life

        self._system_model.execute(0)

        self._financial_model.SystemOutput.gen = self._system_model.value("gen")
        self._financial_model.value("construction_financing_cost", self.get_construction_financing_cost())
        self._financial_model.Revenue.ppa_soln_mode = 1
        if len(self._financial_model.SystemOutput.gen) == self.site.n_timesteps:
            single_year_gen = self._financial_model.SystemOutput.gen
            self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self._financial_model.SystemOutput.system_pre_curtailment_kwac = self._system_model.value("gen") * project_life
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self._system_model.value("annual_energy")

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
    def cost_installed(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cost_installed")
        else:
            return 0

    @property
    def internal_rate_of_return(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("project_return_aftertax_irr")
        else:
            return 0

    @property
    def energy_value(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_energy_value")
        else:
            return (0, )

    @property
    def federal_depreciation_total(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_feddepr_total")
        else:
            return (0, )

    @property
    def federal_taxes(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_fedtax")
        else:
            return (0, )

    @property
    def insurance_expense(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_insurance_expense")
        else:
            return (0, )

    @property
    def om_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            om_exp = np.array(0)
            om_types = ("capacity1", "capacity2", "capacity",
                        "fixed1", "fixed2", "fixed",
                        "production1", "production2", "production")
            for om in om_types:
                om_exp += np.array(self._financial_model.value("cf_" + om + "_expense"))
            return om_exp.tolist()
        else:
            return [0, ]

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

    @property
    def capacity_payment(self) -> list:
        if self.system_capacity_kw > 0 and self._financial_model:
            return list(self._financial_model.value("cf_capacity_payment"))
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
