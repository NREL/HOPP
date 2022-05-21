from typing import Iterable, Sequence
from attr import has
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
        self.initialize_financial_values()

    def initialize_financial_values(self):
        """
        These values are provided as default values from PySAM but should be customized by user

        Debt, Reserve Account and Construction Financing Costs are initialized to 0
        Federal Bonus Depreciation also initialized to 0
        """
        self._financial_model.value("debt_option", 1)
        self._financial_model.value("dscr", 0)
        self._financial_model.value("debt_percent", 0)
        self._financial_model.value("cost_debt_closing", 0)
        self._financial_model.value("cost_debt_fee", 0)
        self._financial_model.value("term_int_rate", 0)
        self._financial_model.value("term_tenor", 0)
        self._financial_model.value("dscr_reserve_months", 0)
        self._financial_model.value("equip1_reserve_cost", 0)
        self._financial_model.value("months_working_reserve", 0)
        self._financial_model.value("insurance_rate", 0)
        self._financial_model.value("construction_financing_cost", 0)
        self._financial_model.value("om_land_lease", (0,))
        self._financial_model.unassign("battery_total_cost_lcos")
        self._financial_model.value("cp_battery_nameplate", 0)

    def value(self, var_name, var_value=None):
        attr_obj = None
        if var_name in self.__dir__():
            attr_obj = self
        if not attr_obj:
            for a in self._system_model.__dir__():
                group_obj = getattr(self._system_model, a)
                try:
                    if var_name in group_obj.__dir__():
                        attr_obj = group_obj
                        break
                except:
                    pass
        if not attr_obj:
            for a in self._financial_model.__dir__():
                group_obj = getattr(self._financial_model, a)
                try:
                    if var_name in group_obj.__dir__():
                        attr_obj = group_obj
                        break
                except:
                    pass
        if not attr_obj:
            raise ValueError("Variable {} not found in technology or financial model {}".format(
                var_name, self.__class__.__name__))

        if var_value is None:
            try:
                return getattr(attr_obj, var_name)
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} error: {e}")
        else:
            try:
                setattr(attr_obj, var_name, var_value)
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}")
    #
    # Inputs
    #

    @property
    def system_capacity_kw(self) -> float:
        raise NotImplementedError

    @property
    def degradation(self) -> float:
        if self._financial_model:
            return self._financial_model.value("degradation")

    @degradation.setter
    def degradation(self, deg_percent):
        if self._financial_model:
            if not isinstance(deg_percent, Iterable):
                deg_percent = (deg_percent,)
            self._financial_model.value("degradation", deg_percent)

    @property
    def ppa_price(self):
        if self._financial_model:
            return self._financial_model.value("ppa_price_input")

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        if self._financial_model:
            if not isinstance(ppa_price, Iterable):
                ppa_price = (ppa_price,)
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
        if self.name != "Battery":
            return self._financial_model.value("om_capacity")
        return self._financial_model.value("om_batt_capacity_cost")

    @om_capacity.setter
    def om_capacity(self, om_capacity_per_kw: Sequence):
        if not isinstance(om_capacity_per_kw, Sequence):
            om_capacity_per_kw = (om_capacity_per_kw,)
        if self.name != "Battery":
            self._financial_model.value("om_capacity", om_capacity_per_kw)
        else:
            self._financial_model.value("om_batt_capacity_cost", om_capacity_per_kw)

    @property
    def om_fixed(self):
        if self.name != "Battery":
            return self._financial_model.value("om_fixed")
        return self._financial_model.value("om_batt_fixed_cost")

    @om_fixed.setter
    def om_fixed(self, om_fixed_per_year: Sequence):
        if not isinstance(om_fixed_per_year, Sequence):
            om_fixed_per_year = (om_fixed_per_year,)
        if self.name != "Battery":
            self._financial_model.value("om_fixed", om_fixed_per_year)
        else:
            self._financial_model.value("om_batt_fixed_cost", om_fixed_per_year)

    @property
    def om_variable(self):
        """
        Variable cost per kW of production
        """
        if self.name != "Battery":
            return self._financial_model.value("om_production")
        else:
            return [i * 1e3 for i in self._financial_model.value("om_batt_variable_cost")]

    @om_variable.setter
    def om_variable(self, om_variable_per_kw: Sequence):
        if not isinstance(om_variable_per_kw, Sequence):
            om_variable_per_kw = (om_variable_per_kw,)
        if self.name != "Battery":
            self._financial_model.value("om_production", om_variable_per_kw)
        else:
            self._financial_model.value("om_batt_variable_cost", [i * 1e-3 for i in om_variable_per_kw])

    @property
    def construction_financing_cost(self) -> float:
        return self._financial_model.value("construction_financing_cost")

    @construction_financing_cost.setter
    def construction_financing_cost(self, construction_financing_cost):
        self._financial_model.value("construction_financing_cost", construction_financing_cost)

    def simulate_power(self, project_life, lifetime_sim=False):
        if not self._system_model:
            return
        if self.system_capacity_kw <= 0:
            return

        if hasattr(self._system_model, "Lifetime"):
            self._system_model.Lifetime.system_use_lifetime_output = 1 if lifetime_sim else 0
            self._system_model.Lifetime.analysis_period = project_life if lifetime_sim else 1

        self._system_model.execute(0)
        
    def simulate_financials(self, project_life):
        if not self._financial_model:
            return
        if self.system_capacity_kw <= 0:
            return

        self._financial_model.FinancialParameters.analysis_period = project_life
        self._financial_model.Lifetime.system_use_lifetime_output = 1 if project_life > 1 else 0
        self._financial_model.Revenue.ppa_soln_mode = 1

        # try to copy over system_model's generation_profile to the financial_model
        if len(self._financial_model.SystemOutput.gen) == 1:
            if len(self.generation_profile) == self.site.n_timesteps:
                self._financial_model.SystemOutput.gen = self.generation_profile
            else:
                raise RuntimeError(f"simulate_financials error: generation profile of len {self.site.n_timesteps} required")

        if len(self._financial_model.SystemOutput.gen) == self.site.n_timesteps:
            single_year_gen = self._financial_model.SystemOutput.gen
            self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self._financial_model.SystemOutput.system_pre_curtailment_kwac = self._system_model.value("gen") * project_life
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self._system_model.value("annual_energy")
            self._financial_model.CapacityPayments.cp_system_nameplate = self.system_capacity_kw

        self._financial_model.execute(0)

    def simulate(self, project_life: int = 25, lifetime_sim=False):
        """
        Run the system and financial model
        """

        self.simulate_power(project_life, lifetime_sim)
        self.simulate_financials(project_life)
        
        logger.info(f"{self.name} simulation executed with AEP {self.annual_energy_kw}")

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
    def energy_sales_value(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_energy_sales_value")
        else:
            return (0, )

    @property
    def energy_purchases_value(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_utility_bill")
        else:
            return (0, )

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
    def debt_payment(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_debt_payment_total")
        else:
            return (0, )

    @property
    def insurance_expense(self) -> tuple:
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_insurance_expense")
        else:
            return (0, )

    @property
    def tax_incentives(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            tc = np.array(self._financial_model.value("cf_ptc_fed"))
            tc += np.array(self._financial_model.value("cf_ptc_sta"))
            try:
                tc[1] += self._financial_model.value("itc_total")
            except:
                pass
            return tc.tolist()
        else:
            return (0,)

    @property
    def om_capacity_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            if self.name == "Battery":
                return self._financial_model.value("cf_om_capacity1_expense")
            return self._financial_model.value("cf_om_capacity_expense")
        else:
            return [0, ]

    @property
    def om_fixed_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            if self.name == "Battery":
                return self._financial_model.value("cf_om_fixed1_expense")
            return self._financial_model.value("cf_om_fixed_expense")
        else:
            return [0, ]

    @property
    def om_variable_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            if self.name == "Battery":
                return self._financial_model.value("cf_om_production1_expense")
            elif self.name == "Grid":
                return [self._financial_model.value("cf_om_production_expense")[i] +
                        self._financial_model.value("cf_om_production1_expense")[i] for i in
                        range(len(self._financial_model.value("cf_om_production_expense")))]
            return self._financial_model.value("cf_om_production_expense")
        else:
            return [0, ]

    @property
    def om_total_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            op_exp = self._financial_model.value("cf_operating_expenses")
            if self.name != "Battery" and self.name != "Grid":
                return op_exp
            # Battery's operating costs include electricity purchased to charge the battery
            return [op_exp[i] - self._financial_model.value("cf_utility_bill")[i] for i in range(len(op_exp))]
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

    @property
    def benefit_cost_ratio(self) -> float:
        if self.system_capacity_kw > 0 and self._financial_model:
            benefit_names = ("npv_ppa_revenue", "npv_capacity_revenue", "npv_curtailment_revenue",
                             "npv_fed_pbi_income", "npv_oth_pbi_income", "npv_salvage_value", "npv_sta_pbi_income",
                             "npv_uti_pbi_income")
            benefits = 0
            for b in benefit_names:
                benefits += self._financial_model.value(b)
            return benefits / self._financial_model.value("npv_annual_costs")

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
