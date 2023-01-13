from dataclasses import dataclass, _is_classvar, asdict
from typing import Sequence
# other required properties...


@dataclass
class FinancialData:
    """
    Groupes similar variables together into logical organization and replicates some of PySAM.Singleowner's subclass structure

    HybridSimulation has some financial-model-required properties that are stored within subclasses
    These are accessed in the following ways: 
        ```
        hybrid_simulation_model.VariableGroup.variable_name
        hybrid_simulation_model.VariableGroup.export()
        ```

    This dataclass duplicates that structure for a custom financial model so that it can be interoperable
    within HybridSimulation

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

    @classmethod
    def from_dict(cls, input_dict):
        fields = cls.__dataclass_fields__
        # parse the input dict, but only include keys that are:        
        return cls(**{
            k: v for k, v in input_dict.items()                  
            if k in fields                                # - part of the fields
            and fields[k].init                            # - not a post_init field
        })


@dataclass
class BatterySystem(FinancialData):
    """
    These financial inputs are used in simulate_financials in `battery.py`
    The names correspond to PySAM.Singleowner variables.
    To add any additional system cost, first see if the variable exists in Singleowner, and re-use name.
    This will simplify interoperability
    """
    batt_bank_replacement: tuple                # same format as Singleowner.BatterySystem.batt_bank_replacement
    batt_replacement_schedule_percent: tuple    # same format as Singleowner.BatterySystem.batt_replacement_schedule_percent
    batt_replacement_option: float              # same enumeration as Singleowner.BatterySystem.batt_replacement_option


@dataclass
class SystemCosts(FinancialData):
    """
    These financial inputs are used in simulate_financials in various files.
    The names correspond to PySAM.Singleowner variables.
    To add any additional system cost, first see if the variable exists in Singleowner, and re-use name.
    This will simplify interoperability
    """
    om_fixed: float
    om_production: float
    om_capacity: float
    om_batt_fixed_cost: float
    om_batt_variable_cost: float
    om_batt_capacity_cost: float
    om_batt_replacement_cost: float


@dataclass
class Depreciation(FinancialData):
    """
    These financial inputs are used in simulate_financials in various files.
    The names correspond to PySAM.Singleowner variables.
    To add any additional system cost, first see if the variable exists in Singleowner, and re-use name.
    This will simplify interoperability
    """
    depr_alloc_macrs_5_percent: float
    depr_alloc_macrs_15_percent: float
    depr_alloc_sl_5_percent: float
    depr_alloc_sl_15_percent: float
    depr_alloc_sl_20_percent: float
    depr_alloc_sl_39_percent: float
    depr_alloc_custom_percent: float
    depr_bonus_fed_macrs_5: float
    depr_bonus_sta_macrs_5: float
    depr_itc_fed_macrs_5: float
    depr_itc_sta_macrs_5: float
    depr_bonus_fed_macrs_15: float
    depr_bonus_sta_macrs_15: float
    depr_itc_fed_macrs_15: float
    depr_itc_sta_macrs_15: float
    depr_bonus_fed_sl_5: float
    depr_bonus_sta_sl_5: float
    depr_itc_fed_sl_5: float
    depr_itc_sta_sl_5: float
    depr_bonus_fed_sl_15: float
    depr_bonus_sta_sl_15: float
    depr_itc_fed_sl_15: float
    depr_itc_sta_sl_15: float
    depr_bonus_fed_sl_20: float
    depr_bonus_sta_sl_20: float
    depr_itc_fed_sl_20: float
    depr_itc_sta_sl_20: float
    depr_bonus_fed_sl_39: float
    depr_bonus_sta_sl_39: float
    depr_itc_fed_sl_39: float
    depr_itc_sta_sl_39: float
    depr_bonus_fed_custom: float
    depr_bonus_sta_custom: float
    depr_itc_fed_custom: float
    depr_itc_sta_custom: float


@dataclass
class TaxCreditIncentives(FinancialData):
    """
    These financial inputs are used in simulate_financials in various files.
    The names correspond to PySAM.Singleowner variables.
    To add any additional system cost, first see if the variable exists in Singleowner, and re-use name.
    This will simplify interoperability
    """
    ptc_fed_amount: float
    ptc_fed_escal: float
    itc_fed_amount: float
    itc_fed_percent: float
    

@dataclass
class Outputs(FinancialData):
    """
    These financial outputs are all matched with PySAM.Singleowner outputs, but most have different names.
    For example, `net_present_value` is `Singleowner.project_return_aftertax_npv`.
    To see the PySAM variable referenced by each name below, see power_source.py's Output section.

    Any additional financial outputs should be added here. 
    The names can be different from the PySAM.Singleowner names.
    To enable programmatic access via the HybridSimulation class, getter and setters can be added
    """
    cp_capacity_payment_amount: float
    capacity_factor: float
    net_present_value: float
    cost_installed: float
    internal_rate_of_return: float
    energy_sales_value: float
    energy_purchases_value: float
    energy_value: float
    federal_depreciation_total: float
    federal_taxes: float
    debt_payment: float
    insurance_expense: float
    tax_incentives: float
    om_capacity_expense: float
    om_fixed_expense: float
    om_variable_expense: float
    om_total_expense: float
    levelized_cost_of_energy_real: float
    levelized_cost_of_energy_nominal: float
    total_revenue: float
    capacity_payment: float
    benefit_cost_ratio: float


class CustomFinancialModel:
    """
    This custom financial model slots into the PowerSource's financial model that is originally a PySAM.Singleowner model
    PowerSource and the overlaying classes that call on PowerSource expect properties and functions from the financial model
    The mininum expectations are listed here as the class interface.

    The financial model is constructed with financial configuration inputs.
    During simulation, the financial model needs to update all its design inputs from changes made to
    the system performance models, such as changing capacities, total_installed_cost, benefits, etc.
    Part of this is done in HybridSimulation::calculate_financials, which uses many more of PySAM.Singleowner
    inputs than are included here. Any of those variables can be added here.

    This class can be expanded with completely new variables too, which can be added to the class itself or within a dataclass.
    Any financial variable's dependence on system design needs to be accounted for.
    """
    def __init__(self, system_model, fin_config) -> None:
        self._system_model = system_model

        # input parameters
        self.system_use_lifetime_output = None
        self.analysis_period = None
        self.ppa_price_input = None
        self.dispatch_factors_ts = None
        self.cp_system_nameplate = None                  # mw, unlike many other variables
        self.cp_capacity_credit_percent = None

        # input parameters within dataclasses
        self.BatterySystem: BatterySystem = BatterySystem.from_dict(fin_config)
        self.SystemCosts: SystemCosts = SystemCosts.from_dict(fin_config)
        self.Depreciation: Depreciation = Depreciation.from_dict(fin_config)
        self.TaxCreditIncentives: TaxCreditIncentives = TaxCreditIncentives.from_dict(fin_config)
        self.subclasses = [self.BatterySystem, self.SystemCosts, self.Depreciation, self.TaxCreditIncentives]
        self.assign(fin_config)

        # system-performance dependent inputs
        self.system_pre_curtailment_kwac = None
        self.annual_energy_pre_curtailment_ac = None
        self.total_installed_cost = None
        self.construction_financing_cost = None

        # outputs
        self.Outputs = None

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
            raise ValueError("Variable {} not found in SimpleFinancialModel".format(var_name))

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

    def assign(self, input_dict):
        for k, v in input_dict.items():
            if not isinstance(v, dict):
                self.value(k, v)
            else:
                getattr(self, k).assign(v)

    def export(self):
        """
        Export all properties to a dictionary
        """
        pass


    @property
    def system_capacity(self) -> float:
        return self._system_model.value("system_capacity")

    @property
    def gen(self) -> Sequence:
        return self._system_model.value("gen")

    @gen.setter
    def gen(self, gen_kw: Sequence):
        self._system_model.value("gen", gen_kw)

    @property
    def degradation(self) -> float:
        return self._system_model.value("dc_degradation")

    def simulate_financials(self, interconnect_kw: float, project_life: int):
        raise NotImplementedError
