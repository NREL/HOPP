from attrs import define, field, validators
from dataclasses import dataclass, asdict
import inspect
from typing import Sequence, List
import numpy as np
from hopp.tools.utils import flatten_dict, equal
from hopp.simulation.base import BaseClass
import ProFAST

@dataclass
class FinancialData(BaseClass):
    """
    Groups similar variables together into logical organization and replicates some of
    PySAM.Singleowner's subclass structure. HybridSimulation has some financial-model-required
    properties that are stored within subclasses. These are accessed in the following ways: 
        ```
        hybrid_simulation_model.VariableGroup.variable_name
        hybrid_simulation_model.VariableGroup.export()
        ```
    This dataclass duplicates that structure for a custom financial model so that it can be
    interoperable within HybridSimulation.
    """
    def items(self):
        return asdict(self).items()

    def __getitem__(self, item):
        return getattr(self, item)

    def export(self):
        return asdict(self)

    def assign(self, input_dict):
        for k, v in input_dict.items:
            if hasattr(self, k):
                setattr(self[k], v)
            else:
                raise IOError(f"{self.__class__}'s attribute {v} does not exist.")


@define
class BatterySystem(FinancialData):
    """
    These financial inputs are used in simulate_financials in `battery.py`
    The names correspond to PySAM.Singleowner variables.
    To add any additional system cost, first see if the variable exists in Singleowner, and re-use name.
    This will simplify interoperability
    """
    batt_bank_replacement: tuple = field(default=[0])
    batt_computed_bank_capacity: tuple = field(default=0)
    batt_meter_position: tuple = field(default=0)
    batt_replacement_option: float = field(default=0)
    batt_replacement_schedule_percent: tuple = field(default=[0])


@define
class SystemCosts(FinancialData):
    om_fixed: tuple = field(default=[1])
    om_production: tuple = field(default=[2])
    om_capacity: tuple = field(default=(0,))
    om_batt_fixed_cost: float = field(default=0)
    om_batt_variable_cost: float = field(default=[0])
    om_batt_capacity_cost: float = field(default=0)
    om_batt_replacement_cost: float = field(default=0)
    om_replacement_cost_escal: float = field(default=0)
    total_installed_cost: float = field(default=None)


@define
class Revenue(FinancialData):
    """
    Represents the revenue parameters.

    Attributes:
        ppa_price_input (list): List of PPA prices in cents per kWh. Can be hourly or a single value to set flat rate for a year.
        ppa_escalation (float): Annual escalation rate of the PPA price in percentage (default is 1%).
        ppa_multiplier_model (float): Multiplier model applied to adjust the PPA revenue.
        dispatch_factors_ts (Sequence): 

    Args:
        FinancialData (class): Parent class representing base financial data.
    """
    ppa_price_input: list = field(default=None) # cents/kWh
    ppa_escalation: float = field(default=1) # percent (%)
    ppa_multiplier_model: float = field(default=None)
    dispatch_factors_ts: Sequence = field(default=(0,))


@define
class FinancialParameters(FinancialData):
    """
    Represents a set of financial parameters.
    
    This class extends the `FinancialData` class and is designed to support financial analyses with various parameters
    related to costs, tax rates, debt, and other financial metrics, specifically for use in ProFAST and the CustomFinancialModel 
    (not all parameters correspond directly to pySAM).

    Attributes:
        construction_financing_cost (float): Cost of financing for construction.
        analysis_period (float): Number of years the financial analysis is to be conducted.
        inflation_rate (float): Annual inflation rate (%).
        real_discount_rate (float): Discount rate adjusted for inflation (%).
        federal_tax_rate (float): Federal income tax rate (%).
        state_tax_rate (float): State income tax rate (%).
        property_tax_rate (float): Annual property tax rate (%).
        insurance_rate (float): Annual insurance rate (%).
        debt_percent (float): Percentage of the project financed by debt.
        term_int_rate (float): Interest rate on the debt (%).
        months_working_reserve (float): Number of months of working capital reserve.
        analysis_start_year (int): Start year of the financial analysis (ProFAST only, not in pySAM).
        installation_months (int): Duration in months for the installation process (ProFAST only, not in pySAM).
        sales_tax_rate_state (float): State sales tax rate (%) (ProFAST only, not in pySAM).
        admin_expense_percent_of_sales (float): Administrative expenses as a percentage of sales (%) (ProFAST only, not in pySAM).
        capital_gains_tax_rate (float): Capital gains tax rate (%) (ProFAST only, not in pySAM).
        debt_type (str): Type of debt financing used; options are "Revolving debt" or "One time loan" (ProFAST only, not in pySAM).
        depreciation_method (str): Depreciation method used for tax purposes; options are "MACRS" or "Straight line" (ProFAST only, not in pySAM because pySAM handles depreciation options differently).
        depreciation_period (int): Duration, in years, over which assets are depreciated (ProFAST only, not in pySAM - handled differently).
    """
    construction_financing_cost: float = field(default=None)
    analysis_period: float = field(default=None)
    inflation_rate: float = field(default=None)
    real_discount_rate: float = field(default=None)
    federal_tax_rate: float = field(default=None)
    state_tax_rate: float = field(default=None)
    property_tax_rate: float = field(default=None)
    insurance_rate: float = field(default=None)
    debt_percent: float = field(default=None)
    term_int_rate: float = field(default=None)
    months_working_reserve: float = field(default=None)
    analysis_start_year: int = field(default=None)  # ProFAST only, no corresponding parameter in pySAM 
    installation_months: int = field(default=None) # ProFAST only, no corresponding parameter in pySAM 
    sales_tax_rate_state: float = field(default=None) # ProFAST only, no corresponding parameter in pySAM 
    admin_expense_percent_of_sales: float = field(default=None) # ProFAST only, no corresponding parameter in pySAM 
    capital_gains_tax_rate: float = field(default=None) # ProFAST only, no corresponding parameter in pySAM 
    debt_type: str = field(default=None, validator=validators.in_(["Revolving debt", "One time loan"])) # ProFAST only, no corresponding parameter in pySAM
    depreciation_method: str = field(default=None, validator=validators.in_(["MACRS", "Straight line"])) # ProFAST only, no corresponding parameter in pySAM - handled differently
    depreciation_period: int = field(default=None) # ProFAST only, no corresponding parameter in pySAM - handled differently

@define
class Outputs(FinancialData):
    """
    These financial outputs are all matched with PySAM.Singleowner outputs, but most have different names.
    For example, `net_present_value` is `Singleowner.project_return_aftertax_npv`.
    To see the PySAM variable referenced by each name below, see power_source.py's Output section.
    Any additional financial outputs should be added here. 
    The names can be different from the PySAM.Singleowner names.
    To enable programmatic access via the HybridSimulation class, getter and setters can be added
    """
    cp_capacity_payment_amount: Sequence=(0,)
    capacity_factor: float=None
    net_present_value: float=None
    cost_installed: float=None
    internal_rate_of_return: float=None
    energy_sales_value: float=None
    energy_purchases_value: float=None
    energy_value: float=None
    federal_depreciation_total: float=None
    federal_taxes: float=None
    debt_payment: float=None
    insurance_expense: float=None
    tax_incentives: float=None
    om_capacity_expense: float=None
    om_fixed_expense: float=None
    om_variable_expense: float=None
    om_total_expense: Sequence=None
    levelized_cost_of_energy_real: float=None
    levelized_cost_of_energy_nominal: float=None
    total_revenue: float=None
    capacity_payment: float=None
    benefit_cost_ratio: float=None
    project_return_aftertax_npv: float=None
    cf_project_return_aftertax: Sequence=(0,)

@define
class SystemOutput(FinancialData):
    gen: Sequence = field(default=(0,))
    system_capacity: float= field(default=None)
    annual_energy_kwh: float= field(default=None)
    degradation: Sequence= field(default=(0,))
    system_pre_curtailment_kwac: float= field(default=None)
    annual_energy_pre_curtailment_ac: float= field(default=None)


class CustomFinancialModel():
    """
    This custom financial model slots into the PowerSource's financial model that is originally a
    PySAM.Singleowner model. PowerSource and the overlaying classes that call on PowerSource expect
    properties and functions from the financial model. The minimum expectations are listed here as
    the class interface.
    
    The financial model is constructed with financial configuration inputs. During simulation, the
    financial model needs to update all its design inputs from changes made to the system
    performance models, such as changing capacities, total_installed_cost, benefits, etc. Part of
    this is done in HybridSimulation::calculate_financials, which uses many more of
    PySAM.Singleowner inputs than are included here. Any of those variables can be added here.
    
    This class can be expanded with completely new variables too, which can be added to the class
    itself or within a dataclass. Any financial variable's dependence on system design needs to be
    accounted for.

    :param fin_config: dictionary of financial parameters
    """
    def __init__(self,
                 fin_config: dict, name: str) -> None:
        # super().__init__(fname, lname)

        # Input parameters
        self._system_model = None
        self.batt_annual_discharge_energy = None
        self.batt_annual_charge_energy = None
        self.batt_annual_charge_from_system = None
        self.battery_total_cost_lcos = None            # not currently used but referenced
        self.system_use_lifetime_output = 0            # Lifetime
        self.cp_capacity_credit_percent = [0]          # CapacityPayments
        self.name = "UnnamedFinancailModel"

        # Input parameters within dataclasses
        if 'battery_system' in fin_config:
            self.BatterySystem: BatterySystem = BatterySystem.from_dict(
                fin_config['battery_system']
            )
        else:
            self.BatterySystem: BatterySystem = BatterySystem()

        if 'system_costs' in fin_config:
            self.SystemCosts: SystemCosts = SystemCosts.from_dict(fin_config['system_costs'])
        else:
            self.SystemCosts: SystemCosts = SystemCosts()

        if 'revenue' in fin_config:
            self.Revenue: Revenue = Revenue.from_dict(fin_config['revenue'])
        else:
            self.Revenue: Revenue = Revenue()

        if 'financial_parameters' in fin_config:
            self.FinancialParameters: FinancialParameters = FinancialParameters.from_dict(
                fin_config['financial_parameters']
            )
        else:
            self.FinancialParameters: FinancialParameters = FinancialParameters()

        self.SystemOutput: SystemOutput = SystemOutput()
        self.Outputs: Outputs = Outputs()
        self.subclasses = [
            self.BatterySystem,
            self.SystemCosts,
            self.Revenue,
            self.FinancialParameters,
            self.SystemOutput,
            self.Outputs,
        ]
        self.assign(fin_config)


    def set_financial_inputs(self, system_model=None):
        """
        Set financial inputs from PowerSource (e.g., PVPlant)

        This custom financial model needs to be able to update its inputs from the system model, as
        parameters are not linked like they are when a PySAM.Singleowner model is created using
        from_existing(). The inputs that need to be updated will depend on the financial model
        implementation, and these are specified here. The system model reference is also updated
        here, as the system model is not always available during __init__.
        """
        if system_model is not None:
            self._system_model = system_model
        elif self._system_model is None:
            raise ValueError('System model not set in custom financial model')

        if inspect.ismethod(getattr(self._system_model, 'export', None)):
            power_source_dict = flatten_dict(self._system_model.export())
        else:
            power_source_dict = {}
        if 'system_capacity' in power_source_dict:
            self.value('system_capacity', power_source_dict['system_capacity'])

    def execute(self, n=0):
        self.set_financial_inputs()         # update inputs from system model

        
        npv = self.npv(
                rate=self.nominal_discount_rate(
                    inflation_rate=self.value('inflation_rate'),
                    real_discount_rate=self.value('real_discount_rate')
                    ) / 100,
                net_cash_flow=self.net_cash_flow(self.value('analysis_period'))
                )
        self.value('project_return_aftertax_npv', npv)

        # TODO since we are using ProFAST for LCOE, I think it would make sense to use ProFAST for all other metrics as well

        pf_real = self.setup_profast(gen_inflation=0.0)
        sol_real = pf_real.solve_price()
        lcoe_real = sol_real['price']

        pf_nominal = self.setup_profast(gen_inflation=self.value('inflation_rate')/100.0)
        sol_nominal = pf_nominal.solve_price()
        lcoe_nominal = sol_nominal['price']

        usd_per_kwh_to_cents_per_kwh = 100
        self.value('levelized_cost_of_energy_real', lcoe_real*usd_per_kwh_to_cents_per_kwh)
        self.value('levelized_cost_of_energy_nominal', lcoe_nominal*usd_per_kwh_to_cents_per_kwh)

        return
    
    def setup_profast(self, gen_inflation) -> ProFAST.ProFAST:

        """This method sets up a cash-flow financial model based on the input financial parameters.

        Args:
            gen_inflation (float): inflation is left as an input and should be a float between 0 and 1, not a percentage. 
            Leaving inflation as an input to the method allows for easily setting up a ProFAST model for either real or nominal calculations.

        Returns:
            ProFAST.ProFAST: an instance of the ProFAST class set up according to the input financial parameters
        """

        nominal_discount_rate = self.nominal_discount_rate(
            inflation_rate=self.value('inflation_rate'),
            real_discount_rate=self.value('real_discount_rate')
        ) / 100

        pf = ProFAST.ProFAST()
        pf.set_params(
            "commodity",
            {
                "name": "Electricity",
                "unit": "kWh",
                "initial price": 10,
                "escalation": gen_inflation,
            }, 
        )
        
        if "Battery" in self.name:
            pf.set_params(
                "capacity",
                max([1E-6, self.value("batt_annual_discharge_energy")[0]/365.0]),
            )  # kWh/day
        else:
            pf.set_params(
                "capacity",
                max([1E-6, self.value("annual_energy_kwh")/365.0]),
            )  # kWh/day

        pf.set_params("maintenance", {"value": self.o_and_m_cost(), "escalation": gen_inflation})

        pf.set_params(
            "analysis start year", self.value('analysis_start_year'), # no explicit year in single owner,  # Add financial analysis start year
        )
        pf.set_params(
            "operating life", self.value('analysis_period')
        )
        pf.set_params(
            "installation months",
            self.value('installation_months'),  # Add installation time to yaml default=0
        )
        pf.set_params(
            "installation cost",
            {
                "value": 0,
                "depr type": "Straight line",
                "depr period": 4,
                "depreciable": False,
            },
        )
        pf.set_params("demand rampup", 0)
        pf.set_params("long term utilization", 1)  # TODO should use utilization
        pf.set_params("credit card fees", 0)
        pf.set_params(
            "sales tax", self.value('sales_tax_rate_state')/100.0
        )
        pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
        pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
        # TODO how to handle property tax and insurance for fully offshore?
        pf.set_params(
            "property tax and insurance",
            self.value('property_tax_rate')/100.0 + self.value('insurance_rate')/100.0,
        )
        pf.set_params(
            "admin expense",
            self.value("admin_expense_percent_of_sales")/100.0,
        )
        pf.set_params(
            "total income tax rate",
            self.value('federal_tax_rate')/100.0 + self.value('state_tax_rate')/100.0,
        )
        pf.set_params(
            "capital gains tax rate",
            self.value('capital_gains_tax_rate')/100.0,
        )
        pf.set_params("sell undepreciated cap", True)
        pf.set_params("tax losses monetized", True)
        pf.set_params("general inflation rate", gen_inflation)
        pf.set_params(
            "leverage after tax nominal discount rate",
            nominal_discount_rate,
        )
        
        pf.set_params(
            "debt equity ratio of initial financing",
            (
                self.value('debt_percent')
                / (100 - self.value('debt_percent'))
            ),
        )
        
        pf.set_params("debt type", self.value('debt_type'))
        pf.set_params(
            "debt interest rate",
            self.value('term_int_rate')/100.0,
        )
        pf.set_params(
            "cash onhand", self.value('months_working_reserve')
        )

        # ----------------------------------- Add capital and fixed items to ProFAST ----------------
        pf.add_capital_item(
                name="Total installed cost",
                cost=self.value('total_installed_cost'),
                depr_type=self.value('depreciation_method'),
                depr_period=self.value('depreciation_period'),
                refurb=[0],
            )

        return pf

    @staticmethod
    def npv(rate: float, net_cash_flow: List[float]):
        """
        Returns the NPV (Net Present Value) of a cash flow series.

        borrowed from the numpy-financial package
        :param rate: rate [-]
        :param net_cash_flow: net cash flow timeseries
        """
        values = np.atleast_2d(net_cash_flow)
        timestep_array = np.arange(0, values.shape[1])
        npv = (values / (1 + rate) ** timestep_array).sum(axis=1)
        try:
            # If size of array is one, return scalar
            return npv.item()
        except ValueError:
            # Otherwise, return entire array
            return npv


    @staticmethod
    def nominal_discount_rate(inflation_rate: float, real_discount_rate: float):
        """
        Computes the nominal discount rate [%]

        :param inflation_rate: inflation rate [%]
        :param real_discount_rate: real discount rate [%]
        """
        if inflation_rate is None:
            raise Exception(
                "'inflation_rate' must be a number. Make sure that `inflation_rate` is defined in "
                "your `fin_model` configuration under `financial_parameters` for each of your "
                "technologies."
            )
        if real_discount_rate is None:
            raise Exception(
                "'real_discount_rate' must be a number. Make sure that `real_discount_rate` is "
                "defined in your `fin_model` configuration under `financial_parameters` for each "
                "of your technologies."
            )

        return ( (1 + real_discount_rate / 100) * (1 + inflation_rate / 100) - 1 ) * 100


    def net_cash_flow(self, project_life=25):
        """
        Computes the net cash flow timeseries of annual values over lifetime
        """
        degradation = self.value('degradation')
        if isinstance(degradation, float) or isinstance(degradation, int):
            degradation = [degradation] * project_life
        elif len(degradation) == 1:
            degradation = [degradation[0]] * project_life
        else:
            degradation = list(degradation)

        ncf = list()
        ncf.append(-self.value('total_installed_cost'))
        degrad_fraction = 1                         # fraction of annual energy after degradation
        om_costs = self.o_and_m_cost()
        self.cf_operating_expenses = np.asarray([om_costs*(1 + self.value('inflation_rate') / 100)**(year - 1) for year in range(1, project_life+1)])
        self.cf_utility_bill = np.zeros_like(self.cf_operating_expenses) #TODO make it possible for this to be non-zero
        for i, year in enumerate(range(1, project_life + 1)):
            degrad_fraction *= (1 - degradation[year - 1])
            ncf.append(
                        (
                        - self.cf_operating_expenses[i]
                        - self.cf_utility_bill[i]
                        + self.value('annual_energy_kwh')
                        * degrad_fraction
                        * self.value('ppa_price_input')[0]
                        * (1 + self.value('ppa_escalation') / 100)**(year - 1)
                        )
                      )
        return ncf


    def o_and_m_cost(self):
        """
        Computes the annual O&M cost from the fixed, per capacity and per production costs
        """
        
        return self.value('om_fixed')[0] \
               + self.value('om_capacity')[0] * self.value('system_capacity') \
               + self.value('om_production')[0] * self.value('annual_energy_kwh') * 1e-3

    def value(self, var_name, var_value=None):
        attr_obj = None
        if var_name in self.__dir__():
            attr_obj = self
        if not attr_obj:
            for sc in self.subclasses:
                if var_name in sc.__dir__():
                    attr_obj = sc
                    break
        if not attr_obj:
            raise ValueError("Variable {} not found in CustomFinancialModel".format(var_name))

        if var_value is None:
            try:
                return getattr(attr_obj, var_name)
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} error: {e}")
        else:
            try:
                setattr(attr_obj, var_name, var_value)
                try:
                    # update system model if it has the same named attribute
                    # avoid infinite loops if same functionality is implemented in system model
                    if (
                        (not equal(self._system_model.value(var_name), var_value)) and
                        (var_name != 'gen')
                    ):
                        self._system_model.value(var_name, var_value)
                except:
                    pass
            except Exception as e:
                raise IOError(
                    f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}"
                )

    
    def assign(self, input_dict, ignore_missing_vals=False):
        """
        Assign attribues from nested dictionary, except for Outputs

        :param input_dict: nested dictionary of values
        :param ignore_missing_vals: if True, do not throw exception if value not in self
        """
        for k, v in input_dict.items():
            if not isinstance(v, dict):
                try:
                    self.value(k, v)
                except Exception as e:
                    if not ignore_missing_vals:
                        raise IOError(
                            f"{self.__class__}'s attribute {k} could not be set to {v}: {e}"
                        )
            elif k == 'Outputs':
                continue    # do not assign from Outputs category
            else:
                self.assign(input_dict[k], ignore_missing_vals)

    
    def unassign(self, var_name):
        self.value(var_name, None)


    def export_battery_values(self):
        return {
            'batt_bank_replacement': self.BatterySystem.batt_bank_replacement,
            'batt_computed_bank_capacity': self.BatterySystem.batt_computed_bank_capacity,
            'batt_meter_position': self.BatterySystem.batt_meter_position,
            'batt_replacement_option': self.BatterySystem.batt_replacement_option,
            'batt_replacement_schedule_percent': 
                self.BatterySystem.batt_replacement_schedule_percent,
        }

    @property
    def annual_energy_kwh(self) -> float:
        return self.value('annual_energy_pre_curtailment_ac')
    
    @property
    def om_total_expense(self) -> float:
        return self.value('om_total_expense')
    
    # for compatibility with calls to SingleOwner
    @property
    def lcoe_real(self) -> float:
        return self.value('levelized_cost_of_energy_real')
    
    # for compatibility with calls to SingleOwner
    @property
    def lcoe_nom(self) -> float:
        return self.value('levelized_cost_of_energy_nominal')

    