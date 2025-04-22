from attrs import define, field, validators, fields_dict
import numpy as np
from typing import Optional, Union
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType

from hopp.simulation.technologies.sites.site_info import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.utilities.validators import range_val
from hopp.simulation.base import BaseClass

@define 
class LDESConfig(BaseClass):
    """
    Configuration class for `LDES`.

    Args:
        tracking: default True -> `LDES`
        system_capacity_kwh: LDES energy capacity [kWh]
        system_capacity_kw: LDES rated power capacity [kW]
        chemistry: LDES chemistry option
            - "LDES" generic long-duration energy storage
        minimum_SOC: Minimum state of charge [%]
        maximum_SOC: Maximum state of charge [%]
        initial_SOC: Initial state of charge [%]
        fin_model: Financial model. Can be any of the following:
            - a dict representing a `CustomFinancialModel`
            - an object representing a `CustomFinancialModel` or a `Singleowner.Singleowner` instance
    """
    system_capacity_kwh: float = field(validator=validators.gt(0.0))
    system_capacity_kw: float = field(validator=validators.gt(0.0))
    system_model_source: str = field(default="pysam", validator=validators.in_(["pysam", "hopp"]))
    chemistry: str = field(default="LDES", validator=validators.in_(["LDES"]))
    tracking: bool = field(default=True)
    minimum_SOC: float = field(default=10, validator=range_val(0, 100))
    maximum_SOC: float = field(default=90, validator=range_val(0, 100))
    initial_SOC: float = field(default=10, validator=range_val(0, 100))
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)
    name: str = field(default="LDES")

@define
class Params:
    # cp: float # specific heat capacity [J/KgK]
    # h: float # Heat transfer between battery and environment [W/m2K]

    nominal_energy: float # nominal installed energy [kWh]
    nominal_voltage: float # nominal DC voltage [V] - > not used for dispatch
    duration: float
    valid_control_modes = [0.0, 1.0] # control mode 1 is power in kW, control mode 0 is current in A
    control_mode: float = field(default=1.0, validator=validators.in_(valid_control_modes)) # TODO how set?
    dt_hr: float = field(default=1.0, validator=validators.gt(0.0))

@define
class State:
    I: float = field(default=None)
    P: float = field(default=None)
    Q: float = field(default=None)
    SOC: float = field(default=None)
    T_batt: float = field(default=None)
    gen: float = field(default=0)
    n_cycles: float = field(default=0)
    input_power: float = field(default=None)
    input_current: float = field(default=None)

@define
class LDES(PowerSource):
    config: LDESConfig = field()
    site: SiteInfo = field()
    
    def __attrs_post_init__(self):
        if self.config.fin_model is None:
            raise AttributeError("Financial model must be set in `config.fin_model`")

        if isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model

        self.financial_model = self.import_financial_model(financial_model, self, self.config.name)

        self._system_capacity_kw = self.config.system_capacity_kw
        self._system_capacity_kwh = self.config.system_capacity_kwh
        self.initial_SOC = self.config.initial_SOC

        self.state = State()
        self.state.SOC = self.initial_SOC
        self.state.P = self.SOC*self.system_capacity_kw

        self.params = Params(nominal_energy=self.config.system_capacity_kwh, 
                            nominal_voltage=None, 
                            duration=self.config.system_capacity_kw/self.config.system_capacity_kwh,
                           )
        
        self.params.nominal_energy = self.config.system_capacity_kwh
        self.params.duration = self.config.system_capacity_kwh / self.config.system_capacity_kw
        

        super().__init__(self.config.name, self.site, self, financial_model)

    def setup(self):
        pass

    def export(self):
        """Export class data as a dictionary. Method generated using ChatGPT"""
        # Create a dictionary of the instance's attributes
        data = {name: getattr(self, name) for name in fields_dict(self.__class__) if not name.startswith("_")}
        return data
    
    def model_type(self):
        if self.chemistry == "LDES":
            return 0
        elif self.chemistry == "AES":
            return 1
        
    def calc_degradation_rate_eff_per_hour(lifetime_yrs: float, eol_efficiency: float) -> float:
        """Calculate the degradation rate per hour of operation

        Args:
            lifetime_yrs (float): number of year the battery is expected to operate
            eol_efficiency (float): end of life efficiency. Should be between 0 and 1

        Returns:
            float: efficiency loss expected per hour
        """
        
        days_pr_yr = 365.25
        hours_pr_day = 24.0
        hour_pr_yr = days_pr_yr*hours_pr_day
        hour_pr_life = hour_pr_yr*lifetime_yrs
        eff_loss_pr_hour = (1.0 - eol_efficiency)/hour_pr_life

        return eff_loss_pr_hour
    
    def calc_degradation_rate_per_cycle(lifetime_cycles: float, eol_efficiency: float):
        """Calculate degradation rate per cycle

        Args:
            lifetime_cycles (float): number of cycles for a lifetime
            eol_efficiency (float): efficiency at end of life, should be between 0 and 1

        Returns:
            float: efficiency loss expected per cycle
        """
        eff_loss_pr_cycle = (1.0 - eol_efficiency)/lifetime_cycles

        return eff_loss_pr_cycle
    
    def execute(self, verbosity=0):
        """Execute battery simulation with the specified level of verbosity. This
        mimics the PySAM battery model execute function.

        Args:
            verbosity (int, optional): Verbosity level (0, or 1). 
                0 means no extra printing, 1 means more printing. Defaults to 0.
        """

        dt_hr = self.value('dt_hr')
        max_soc_dec = self.maximum_SOC/100.0
        min_soc_dec = self.minimum_SOC/100.0
        prev_soc_dec = self.state.SOC/100.0
        control_power = self.input_power
        
        if self.control_mode == 0.0:
            raise(ValueError(f"control_mode {self.control_mode} has not been implemented. Must be one of [1.0]."))
        elif self.control_mode == 1.0:

            # check power capacity constraint
            if abs(control_power) > self.system_capacity_kw:
                control_power = np.sign(control_power)*self.system_capacity_kw

            # check energy capacity constraint
            if -(control_power*dt_hr) + prev_soc_dec*self.params.nominal_energy > self.params.nominal_energy*max_soc_dec:
                control_power = -(max_soc_dec - prev_soc_dec)*self.params.nominal_energy/dt_hr

            elif -(control_power*dt_hr) + prev_soc_dec*self.params.nominal_energy < self.params.nominal_energy*min_soc_dec:
                control_power = (prev_soc_dec - min_soc_dec)*self.params.nominal_energy/dt_hr
                
        else:
            raise(ValueError(f"control_mode {self.control_mode} has not been implemented. Must be one of [1.0]."))

        # update state
        self.state.P = control_power
        self.state.gen = self.state.P
        self.state.SOC += (-control_power*dt_hr/self.system_capacity_kwh ) * 100

    @property
    def control_mode(self):
        return self.params.control_mode
    
    @control_mode.setter
    def control_mode(self, control_mode_in):
        self.params.control_mode = control_mode_in

    @property
    def input_current(self):
        return self.params.input_current
    
    @input_current.setter
    def input_current(self, input_current_in):
        self.state.input_current = input_current_in

    @property
    def input_power(self):
        return self.state.input_power
    
    @input_power.setter
    def input_power(self, input_power_in):
        self.state.input_power = input_power_in

    @property
    def dt_hr(self):
        return self.params.dt_hr
    
    @dt_hr.setter
    def dt_hr(self, dt_hr_in):
        self.params.control_mode = dt_hr_in

    @property
    def minimum_SOC(self):
        return self.config.minimum_SOC
    
    @minimum_SOC.setter
    def minimum_SOC(self, minimum_SOC_in):
        self.config.minimum_SOC = minimum_SOC_in

    @property
    def maximum_SOC(self):
        return self.config.maximum_SOC
    
    @maximum_SOC.setter
    def maximum_SOC(self, maximum_SOC_in):
        self.config.maximum_SOC = maximum_SOC_in

    @property
    def initial_SOC(self):
        return self.config.initial_SOC
    
    @initial_SOC.setter
    def initial_SOC(self, initial_SOC):
        self.config.initial_SOC = initial_SOC

    @property
    def chemistry(self):
        return self.config.chemistry
    
    @chemistry.setter
    def chemistry(self, chemistry_in):
        self.config.chemistry = chemistry_in

    @property
    def SOC(self) -> float:
        return self.state.SOC
    
    @SOC.setter
    def SOC(self, SOC):
        self.state.SOC = SOC

    @property
    def nominal_voltage() -> float:
        return None
    
    @property
    def nominal_energy(self) -> float:
        return self.system_capacity_kwh

    @property
    def system_capacity_kw(self) -> float:
        """Battery power rating [kW]"""
        return self._system_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self.financial_model.value("system_capacity", size_kw)
        self.system_capacity_kw = size_kw

    @property
    def system_capacity_kwh(self) -> float:
        """Battery energy capacity [kWh]"""
        return self._system_capacity_kwh

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        self.financial_model.value("batt_computed_bank_capacity", size_kwh)
        self.system_capacity_kwh = size_kwh

    @property
    def n_cycles(self) -> float:
        """ Battery cycle count """
        return self._n_cycles

    @n_cycles.setter
    def n_cycles(self, lifecycles: float):
        self.dispatch.value("lifecycles", lifecycles)
        self.n_cycles = lifecycles
