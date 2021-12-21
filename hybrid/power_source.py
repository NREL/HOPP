from typing import Iterable
import numpy as np
from hybrid.sites import SiteInfo
import pandas as pd

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
        self.gen_max_feasible = [0.] * self.site.n_timesteps

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
        # turn off LCOS calculation
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
            return getattr(attr_obj, var_name)
        else:
            try:
                setattr(attr_obj, var_name, var_value)
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}")

    def simulate(self, project_life: int = 25, skip_fin=False):
        """
        Run the system model
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

        if skip_fin:
            return

        self._financial_model.SystemOutput.gen = self._system_model.value("gen")
        self._financial_model.Revenue.ppa_soln_mode = 1
        if len(self._financial_model.SystemOutput.gen) == self.site.n_timesteps:
            single_year_gen = self._financial_model.SystemOutput.gen
            self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self._financial_model.SystemOutput.system_pre_curtailment_kwac = self._system_model.value("gen")
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self._system_model.value(
                "annual_energy")
            self.gen_max_feasible = self.calc_gen_max_feasible_kwh()  # need to store for later grid aggregation

        self.capacity_credit_percent = self.calc_capacity_credit_percent(  # for wind, PV and grid
            self.site.capacity_hours,
            self.gen_max_feasible)

        if type(self).__name__ == 'PVPlant':
            # [kW] (AC output)
            self._financial_model.CapacityPayments.cp_system_nameplate = self.system_capacity_kw / \
                                                                         self._system_model.SystemDesign.dc_ac_ratio
        elif type(self).__name__ == 'Grid':
            self._financial_model.CapacityPayments.cp_system_nameplate = self.interconnect_kw
        else:
            self._financial_model.CapacityPayments.cp_system_nameplate = self.system_capacity_kw


        self._financial_model.execute(0)
        logger.info(f"{self.name} simulation executed with AEP {self.annual_energy_kwh}")

    def calc_gen_max_feasible_kwh(self) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred (year 1)

        :return: maximum feasible generation [kWh]: list of floats
        """
        t_step = self.site.interval / 60                                                # hr
        E_net_max_feasible = [x * t_step for x in self.generation_profile[0:self.site.n_timesteps]]              # [kWh]
        return E_net_max_feasible

    def calc_capacity_credit_percent(self, cap_credit_hours: list, gen_max_feasible: list) -> float:
        """
        Calculates the capacity credit (value) using the last simulated year's max feasible generation profile.

        :param cap_credit_hours: list of bool
            hourly boolean if the hour counts towards capacity credit (True), o.w. the hour doesn't count (False) [-]

        :param gen_max_feasible: list of floats
            hourly maximum feasible generation profile [kWh]

        :return: capacity value [%]
        """
        TIMESTEPS_YEAR = 8760

        t_step = self.site.interval / 60  # [hr]
        if t_step != 1 or len(cap_credit_hours) != TIMESTEPS_YEAR or len(gen_max_feasible) != TIMESTEPS_YEAR:
            print("WARNING: Capacity credit could not be calculated. Therefore, it was set to zero for "
                  + type(self).__name__)
            return 0
        else:
            df = pd.DataFrame()
            df['cap_hours'] = cap_credit_hours
            df['E_net_max_feasible'] = gen_max_feasible                            # [kWh]

            sel_df = df[df['cap_hours'] == True]

            # df.sort_values(by=['cap_hours'], ascending=False, inplace=True)

            if type(self).__name__ == 'PVPlant':
                W_ac_nom = self.system_capacity_kw / self._system_model.SystemDesign.dc_ac_ratio  # [kW] (AC output)
            elif type(self).__name__ == 'Grid':
                W_ac_nom = self.interconnect_kw
            else:
                W_ac_nom = self.system_capacity_kw              # [kW]

            capacity_value = 0
            if len(sel_df.index) > 0:
                capacity_value = sel_df['E_net_max_feasible'].sum() / (W_ac_nom * len(sel_df.index)) * 100
            capacity_value = min(100, capacity_value)       # [%]
            return capacity_value
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
        return self._financial_model.value("cp_capacity_credit_percent")[0]

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

    @property
    def construction_financing_cost(self) -> float:
        return self._financial_model.value("construction_financing_cost")

    @construction_financing_cost.setter
    def construction_financing_cost(self, construction_financing_cost):
        self._financial_model.value("construction_financing_cost", construction_financing_cost)

    #
    # Outputs
    #
    @property
    def dispatch(self):
        return self._dispatch

    @property
    def annual_energy_kwh(self) -> float:
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
            return self._financial_model.value("cf_energy_purchases_value")
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
    def om_expense(self):
        if self.system_capacity_kw > 0 and self._financial_model:
            om_exp = np.array(0.)
            om_types = ("batt_capacity", "batt_fixed", "batt_production", "capacity1", "capacity2", "capacity",
                        "fixed1", "fixed2", "fixed", "fuel", "opt_fuel_1", "opt_fuel_2"
                        "production1", "production2", "production")
            for om in om_types:
                try:
                    om_exp = om_exp + np.array(self._financial_model.value("cf_om_" + om + "_expense"))
                except:
                    pass
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

    @property
    def gen_max_feasible(self) -> list:
        """Returns maximum feasible generation profile that could have occurred"""
        return self._gen_max_feasible

    @gen_max_feasible.setter
    def gen_max_feasible(self, gen_max_feas: list):
        """Maximum feasible generation profile that could have occurred"""
        self._gen_max_feasible = gen_max_feas

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
