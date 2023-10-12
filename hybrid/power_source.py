from typing import Iterable, Sequence
import numpy as np
from hybrid.sites import SiteInfo
import PySAM.Singleowner as Singleowner
import pandas as pd
from tools.utils import array_not_scalar, equal
from hybrid.log import hybrid_logger as logger
from hybrid.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch


class PowerSource:
    """
    Abstract class for a renewable energy power plant simulation.
    
    Attributes
    ----------
    name : string
        Name used to identify technology
    site : :class:`hybrid.sites.SiteInfo`
        Power source site information
    """

    def __init__(self, name, site: SiteInfo, system_model, financial_model):
        """
        Abstract class for a renewable energy power plant simulation.

        Financial model parameters are linked to the technology model when either: the
        model is native to PySAM and linked using `from_existing`, a `set_financial_inputs`
        method is defined in a user-defined financial model, or the financial and
        technology parameters are named the same when the model is native to PySAM but not
        linked using `from_existing`.

        :param name: Name used to identify technology
        :param site: Power source site information (SiteInfo object)
        :param system_model: Technology performance model
        :param financial_model: Financial model for the specific technology
        """
        self.name = name
        self.site = site
        self._system_model = system_model
        self._financial_model = financial_model
        self._layout = None
        self._dispatch = PowerSourceDispatch

        if isinstance(self._financial_model, Singleowner.Singleowner):
            self.initialize_financial_values()
        else:
            self._financial_model.assign(self._system_model.export(), ignore_missing_vals=True)       # copy system parameter values having same name
            self._financial_model.set_financial_inputs(system_model=self._system_model)               # for custom financial models

        self.capacity_factor_mode = "cap_hours"                                    # to calculate via "cap_hours" method or None to use external value
        self.gen_max_feasible = [0.] * self.site.n_timesteps
        
    @staticmethod
    def import_financial_model(financial_model, system_model, config_name): 
        if isinstance(financial_model, Singleowner.Singleowner):
            financial_model_new = Singleowner.from_existing(system_model, config_name)      # make a linked model instead
            financial_model_new.assign(financial_model.export())                            # transfer parameter values
        else:
            def check_if_callable(obj, func_name):
                if not hasattr(obj, func_name) or not callable(getattr(obj, func_name)):
                    raise ValueError(f"{obj.__class__.__name__} must have a callable function {func_name}() defined")
            check_if_callable(financial_model, "set_financial_inputs")
            check_if_callable(financial_model, "value")
            check_if_callable(financial_model, "assign")
            check_if_callable(financial_model, "unassign")
            check_if_callable(financial_model, "execute")
            financial_model_new = financial_model
        return financial_model_new

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
        # turn off LCOS calculation
        self._financial_model.unassign("battery_total_cost_lcos")
        self._financial_model.value("cp_battery_nameplate", 0)

    def value(self, var_name: str, var_value=None):
        """
        Gets or Sets a variable value within either the system or financial PySAM models. Method looks in system
        model first. If unsuccessful, then it looks in the financial model.

        .. note::

            If system and financial models contain a variable with the same name, only the system model variable will
            be set.

        ``value(var_name)`` Gets variable value

        ``value(var_name, var_value)`` Sets variable value

        :param var_name: PySAM variable name
        :param var_value: (optional) PySAM variable value

        :returns: Variable value (when getter)
        """
        var_name = var_name.replace('adjust:', '')
        attr_obj = None
        if var_name in self.__dir__():
            attr_obj = self
        if not attr_obj:
            for a in self._system_model.__dir__():
                try:
                    group_obj = getattr(self._system_model, a)
                    if var_name in group_obj.__dir__():
                        attr_obj = group_obj
                        break
                except:
                    pass
        if not attr_obj:
            for a in self._financial_model.__dir__():
                try:
                    group_obj = getattr(self._financial_model, a)
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
                if self._financial_model is not None and not isinstance(self._financial_model, Singleowner.Singleowner):
                    try:
                        # update custom financial model if it has the same named attribute
                        # avoid infinite loops if same functionality is implemented in financial model
                        if not equal(self._financial_model.value(var_name), var_value):
                            self._financial_model.value(var_name, var_value)
                    except:
                        pass
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}")

    def assign(self, input_dict: dict):
        """
        Sets input variables in the PowerSource class or any of its subclasses (system or financial models)
        """
        for k, v in input_dict.items():
            self.value(k, v)

    def calc_nominal_capacity(self, interconnect_kw: float):
        """
        Calculates the nominal AC net system capacity based on specific technology.

        :param interconnect_kw: Interconnection limit [kW]

        :returns: system's nominal AC net capacity [kW]
        """
        # TODO: overload function for different systems
        if type(self).__name__ in ['PVPlant', 'DetailedPVPlant']:
            W_ac_nom = min(self.system_capacity_kw / self.value('dc_ac_ratio'), interconnect_kw)
            # [kW] (AC output)
        elif type(self).__name__ == 'Grid':
            W_ac_nom = self.interconnect_kw
        elif type(self).__name__ in ['TowerPlant', 'TroughPlant']:
            W_ac_nom = min(self.system_capacity_kw * self.value('gross_net_conversion_factor'), interconnect_kw)
            # Note: Need to limit to interconnect size. Actual generation is limited by dispatch, but max feasible
            # generation (including storage) is not
        else:
            W_ac_nom = min(self.system_capacity_kw, interconnect_kw)
            # [kW]
        return W_ac_nom

    def calc_gen_max_feasible_kwh(self, interconnect_kw: float) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred (year 1)

        :param interconnect_kw: Interconnection limit [kW]

        :return: maximum feasible generation [kWh]
        """
        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        t_step = self.site.interval / 60                                                # hr
        E_net_max_feasible = [min(x,W_ac_nom) * t_step for x in self.generation_profile[0:self.site.n_timesteps]]      # [kWh]
        return E_net_max_feasible

    def calc_capacity_credit_percent(self, interconnect_kw: float) -> float:
        """
        Calculates the capacity credit (value) using the last simulated year's max feasible generation profile.

        :param interconnect_kw: Interconnection limit [kW]

        :return: capacity value [%]
        """
        if self.capacity_factor_mode == "cap_hours":
            TIMESTEPS_YEAR = 8760

            t_step = self.site.interval / 60  # [hr]
            if t_step != 1 or len(self.site.capacity_hours) != TIMESTEPS_YEAR or len(self.gen_max_feasible) != TIMESTEPS_YEAR:
                print("WARNING: Capacity credit could not be calculated. Therefore, it was set to zero for "
                    + type(self).__name__)
                return 0
            else:
                df = pd.DataFrame()
                df['cap_hours'] = self.site.capacity_hours
                df['E_net_max_feasible'] = self.gen_max_feasible  # [kWh]

                sel_df = df[df['cap_hours'] == True]

                if type(self).__name__ != 'Grid':
                    W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
                else:
                    W_ac_nom = np.min((self.hybrid_nominal_capacity, interconnect_kw))

                if len(sel_df.index) > 0 and W_ac_nom > 0:
                    capacity_value = sum(np.minimum(sel_df['E_net_max_feasible'].values/(W_ac_nom*t_step), 1.0)) / len(sel_df.index) * 100
                    capacity_value = np.min((100, capacity_value))       # [%]
                else:
                    capacity_value = 0

                return capacity_value
        else:
            return self.capacity_credit_percent

    def setup_performance_model(self):
        """
        Sets up performance model to before simulating power production. Required by specific technologies 
        """
        pass

    def simulate_power(self, project_life, lifetime_sim=False):
        """
        Runs the system models for individual sub-systems

        :param project_life: ``int``,
            Number of year in the analysis period (execepted project lifetime) [years]
        :param lifetime_sim: ``bool``,
            For simulation modules which support simulating each year of the project_life, whether or not to do so; otherwise the first year data is repeated
        :return:
        """
        if not self._system_model:
            return
        if self.system_capacity_kw <= 0:
            return

        if hasattr(self._system_model, "Lifetime"):
            self._system_model.Lifetime.system_use_lifetime_output = 1 if lifetime_sim else 0
            self._system_model.Lifetime.analysis_period = project_life if lifetime_sim else 1

        self._system_model.execute(0)
        logger.info(f"{self.name} simulation executed with AEP {self.annual_energy_kwh}")
        
    def simulate_financials(self, interconnect_kw: float, project_life: int):
        """
        Runs the finanical model for individual sub-systems
        
        :param interconnect_kw: ``float``,
            Hybrid interconnect limit [kW]
        :param project_life: ``int``,
            Number of year in the analysis period (execepted project lifetime) [years]
        :return:
        """   
        if not self._financial_model:
            return
        if self.system_capacity_kw <= 0:
            return

        if not isinstance(self._financial_model, Singleowner.Singleowner):
            self._financial_model.assign(self._system_model.export(), ignore_missing_vals=True)       # copy system parameter values having same name
        else:
            self._financial_model.value('ppa_soln_mode', 1)
        self._financial_model.value('system_capacity', self.system_capacity_kw) # [kW] needed for custom financial models
        self._financial_model.value('analysis_period', project_life)
        self._financial_model.value('system_use_lifetime_output', 1 if project_life > 1 else 0)

        # try to copy over system_model's generation_profile to the financial_model
        if len(self._financial_model.value('gen')) == 1:
            if len(self.generation_profile) == self.site.n_timesteps or \
              len(self.generation_profile) == self.site.n_timesteps * project_life:
                self._financial_model.value('gen', self.generation_profile)
            else:
                raise RuntimeError(f"simulate_financials error: generation profile of len {self.site.n_timesteps} required")

        if len(self._financial_model.value('gen')) == self.site.n_timesteps:
            self._financial_model.value('gen', self._financial_model.value('gen') * project_life)
        self._financial_model.value('system_pre_curtailment_kwac', self._financial_model.value('gen'))
        self._financial_model.value('annual_energy_pre_curtailment_ac', self._system_model.value("annual_energy"))
        # TODO: Should we use the nominal capacity function here?
        self.gen_max_feasible = self.calc_gen_max_feasible_kwh(interconnect_kw)
        self.capacity_credit_percent = self.calc_capacity_credit_percent(interconnect_kw)

        self._financial_model.execute(0)

    def simulate(self, interconnect_kw: float, project_life: int = 25, lifetime_sim=False):
        """
        Run the system and financial model

        :param project_life: ``int``,
            Number of year in the analysis period (execepted project lifetime) [years]
        :param lifetime_sim: ``bool``,
            For simulation modules which support simulating each year of the project_life, whether or not to do so; otherwise the first year data is repeated
        """
        self.setup_performance_model()
        self.simulate_power(project_life, lifetime_sim)
        self.simulate_financials(interconnect_kw, project_life)
        
        logger.info(f"{self.name} simulation executed with AEP {self.annual_energy_kwh}")

    #
    # Inputs
    #

    @property
    def system_capacity_kw(self) -> float:
        """System's nameplate capacity [kW]"""
        raise NotImplementedError

    @property
    def degradation(self) -> tuple:
        """Annual energy degradation [%/year]"""
        if self._financial_model:
            return self._financial_model.value("degradation")

    @degradation.setter
    def degradation(self, deg_percent):
        """
        :param deg_percent: float or list, degradation rate [%/year] If a float is provided, then it is applied to
        every year during analysis period, o.w. list is required to be the length of analysis period.
        """
        if self._financial_model:
            if not isinstance(deg_percent, Iterable):
                deg_percent = (deg_percent,)
            self._financial_model.value("degradation", deg_percent)

    @property
    def ppa_price(self) -> tuple:
        """PPA price [$/kWh]"""
        if self._financial_model:
            return self._financial_model.value("ppa_price_input")

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        """PPA price [$/kWh] used in the financial model.

        :param ppa_price: float or list, PPA price [$/kWh] If a float is provided, then it is applied to
        every year during analysis period, o.w. list is required to be the length of analysis period."""
        if self._financial_model:
            if not isinstance(ppa_price, Iterable):
                ppa_price = (ppa_price,)
            self._financial_model.value("ppa_price_input", ppa_price)

    @property
    def system_nameplate_mw(self) -> float:
        """System nameplate [MW]"""
        return self._financial_model.value("cp_system_nameplate")

    @property
    def capacity_credit_percent(self) -> float:
        """Capacity credit (eligible portion of nameplate) [%]"""
        # TODO: should we remove the indexing to be consistent with other properties
        return self._financial_model.value("cp_capacity_credit_percent")[0]

    @capacity_credit_percent.setter
    def capacity_credit_percent(self, cap_credit_percent):
        """Sets capacity credit (eligible portion of nameplate)

        :param cap_credit_percent: float or list, capacity credit [%] If a float is provided, then it is applied to
        every year during analysis period, o.w. list is required to be the length of analysis period."""
        if not isinstance(cap_credit_percent, Iterable):
            cap_credit_percent = (cap_credit_percent,)
        if self._financial_model:
            self._financial_model.value("cp_capacity_credit_percent", cap_credit_percent)

    @property
    def capacity_price(self) -> list:
        """Capacity payment price [$/MW]"""
        return [x for x in self._financial_model.value("cp_capacity_payment_amount")]

    @capacity_price.setter
    def capacity_price(self, cap_price_per_kw_year):
        if not isinstance(cap_price_per_kw_year, Iterable):
            cap_price_per_kw_year = (cap_price_per_kw_year,)
        if self._financial_model:
            cap_price_per_kw_year = [i * 1e3 for i in cap_price_per_kw_year]
            self._financial_model.value("cp_capacity_payment_amount", cap_price_per_kw_year)

    @property
    def dispatch_factors(self) -> tuple:
        """Time-series dispatch factors normalized by PPA price [-]"""
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
        """Installed cost [$]"""
        return self._financial_model.value("total_installed_cost")

    @total_installed_cost.setter
    def total_installed_cost(self, total_installed_cost_dollars: float):
        self._financial_model.value("total_installed_cost", total_installed_cost_dollars)
        logger.info("{} set total_installed_cost to ${}".format(self.name, total_installed_cost_dollars))

    @property
    def om_capacity(self):
        """Capacity-based O&M amount [$/kWcap]"""
        if self.name != "Battery":
            return self._financial_model.value("om_capacity")
        return self._financial_model.value("om_batt_capacity_cost")

    @om_capacity.setter
    def om_capacity(self, om_capacity_per_kw: Sequence):
        if not array_not_scalar(om_capacity_per_kw):
            om_capacity_per_kw = (om_capacity_per_kw,)
        if self.name != "Battery":
            self._financial_model.value("om_capacity", om_capacity_per_kw)
        else:
            self._financial_model.value("om_batt_capacity_cost", om_capacity_per_kw)

    @property
    def om_fixed(self):
        """Fixed O&M annual amount [$/year]"""
        if self.name != "Battery":
            return self._financial_model.value("om_fixed")
        return self._financial_model.value("om_batt_fixed_cost")

    @om_fixed.setter
    def om_fixed(self, om_fixed_per_year: Sequence):
        if not array_not_scalar(om_fixed_per_year):
            om_fixed_per_year = (om_fixed_per_year,)
        if self.name != "Battery":
            self._financial_model.value("om_fixed", om_fixed_per_year)
        else:
            self._financial_model.value("om_batt_fixed_cost", om_fixed_per_year)

    @property
    def om_variable(self):
        """
        For non-battery technologies: Production-based O&M amount [$/kWh]
        For battery: production-based System Costs amount [$/kWh-discharged]
        """
        if self.name != "Battery":
            return [i * 1e3 for i in self._financial_model.value("om_production")]
        else:
            return [i * 1e3 for i in self._financial_model.value("om_batt_variable_cost")]

    @om_variable.setter
    def om_variable(self, om_variable_per_kwh: Sequence):
        if not array_not_scalar(om_variable_per_kwh):
            om_variable_per_kwh = (om_variable_per_kwh,)
        if self.name != "Battery":
            self._financial_model.value("om_production", [i * 1e-3 for i in om_variable_per_kwh])
        else:
            self._financial_model.value("om_batt_variable_cost", [i * 1e-3 for i in om_variable_per_kwh])
   
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
        """Dispatch object"""
        return self._dispatch

    @property
    def annual_energy_kwh(self) -> float:
        """Annual energy [kWh]"""
        if self.system_capacity_kw > 0:
            return self._system_model.value("annual_energy")
        else:
            return 0

    @property
    def generation_profile(self) -> list:
        """System power generated [kW]"""
        if self.system_capacity_kw:
            return list(self._system_model.value("gen"))
        else:
            return [0] * self.site.n_timesteps

    @property
    def capacity_factor(self) -> float:
        """System capacity factor [%]"""
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor")
        else:
            return 0

    @property
    def net_present_value(self) -> float:
        """After-tax cumulative NPV [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("project_return_aftertax_npv")
        else:
            return 0

    @property
    def cost_installed(self) -> float:
        """Net capital cost [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cost_installed")
        else:
            return 0

    @property
    def internal_rate_of_return(self) -> float:
        """Internal rate of return (after-tax) [%]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("project_return_aftertax_irr")
        else:
            return 0

    @property
    def energy_sales_value(self) -> tuple:
        """PPA revenue gross [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_energy_sales_value")
        else:
            return (0, )

    @property
    def energy_purchases_value(self) -> tuple:
        """Energy purchases from grid [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_utility_bill")
        else:
            return (0, )

    @property
    def energy_value(self) -> tuple:
        """PPA revenue net [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_energy_value")
        else:
            return (0, )

    @property
    def federal_depreciation_total(self) -> tuple:
        """Total federal tax depreciation [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_feddepr_total")
        else:
            return (0, )

    @property
    def federal_taxes(self) -> tuple:
        """Federal tax benefit (liability) [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_fedtax")
        else:
            return (0, )

    @property
    def debt_payment(self) -> tuple:
        """Debt total payment [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_debt_payment_total")
        else:
            return (0, )

    @property
    def insurance_expense(self) -> tuple:
        """Insurance expense [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("cf_insurance_expense")
        else:
            return (0, )

    @property
    def tax_incentives(self) -> list:
        """The sum of Federal and State PTC and ITC tax incentives [$]"""
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
        """O&M capacity-based expense [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            if self.name == "Battery":
                return self._financial_model.value("cf_om_capacity1_expense")
            return self._financial_model.value("cf_om_capacity_expense")
        else:
            return [0, ]

    @property
    def om_fixed_expense(self):
        """O&M fixed expense [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            if self.name == "Battery":
                return self._financial_model.value("cf_om_fixed1_expense")
            return self._financial_model.value("cf_om_fixed_expense")
        else:
            return [0, ]

    @property
    def om_variable_expense(self):
        """O&M production-based expense [$]"""
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
        """Total operating expenses [$]"""
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
        """Levelized cost (real) [cents/kWh]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("lcoe_real")
        else:
            return 0

    @property
    def levelized_cost_of_energy_nominal(self) -> float:
        """Levelized cost (nominal) [cents/kWh]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return self._financial_model.value("lcoe_nom")
        else:
            return 0

    @property
    def total_revenue(self) -> list:
        """Total revenue [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return list(self._financial_model.value("cf_total_revenue"))
        else:
            return [0]

    @property
    def capacity_payment(self) -> list:
        """Capacity payment revenue [$]"""
        if self.system_capacity_kw > 0 and self._financial_model:
            return list(self._financial_model.value("cf_capacity_payment"))
        else:
            return [0]

    @property
    def benefit_cost_ratio(self) -> float:
        """
        Benefit cost ratio [-] = Benefits / Costs

        Benefits include (using present values):

        #. PPA, capacity payment, and curtailment revenues
        #. Federal, state, utility, and other production-based incentive income
        #. Salvage value

        Costs: uses the present value of annual costs
        """
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
        """Maximum feasible generation profile that could have occurred (year 1)"""
        return self._gen_max_feasible

    @gen_max_feasible.setter
    def gen_max_feasible(self, gen_max_feas: list):
        self._gen_max_feasible = gen_max_feas

    def copy(self):
        """
        :return: new instance
        """
        raise NotImplementedError
    
    def export(self):
        """
        :return: dictionary of variables for system and financial
        """
        export_dict = {"system": self._system_model.export()}
        if self._financial_model:
            export_dict['financial'] = self._financial_model.export()
        return export_dict

    def plot(self,
             figure=None,
             axes=None,
             color='b',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        self._layout.plot(figure, axes, color, site_border_color, site_alpha, linewidth)
