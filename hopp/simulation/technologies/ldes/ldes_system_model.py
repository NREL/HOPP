from attrs import define, field, validators, fields_dict, setters

from typing import Optional, Tuple, Union, Sequence
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType

from hopp.simulation.technologies.sites.site_info import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.utilities.validators import range_val
from hopp.simulation.base import BaseClass


# from hopp.simulation.technologies.battery import BatteryConfig


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
            - "AEF" Aqueous electrolyte flow battery
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
    chemistry: str = field(default="LDES", validator=validators.in_(["LDES", "AEF"]))
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
    valid_control_modes = [0.0, 1.0] # control mode 0 is power in kW, control mode 1 is current in A
    control_mode: float = field(default=0.0, validator=validators.in_(valid_control_modes)) # TODO how set?
    # charge_rate: float
    # discharge_rate: float
    # mass: float 
    # surface_area: float

@define
class State:
    SOC: float = field(default=None)
    # ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen', 'n_cycles']

@define
class LDES(PowerSource):
    """self.name = name
        self.site = site
        self._system_model = system_model
        self._financial_model = financial_model
        self._layout = None
        self._dispatch = PowerSourceDispatch"""
    
    config: LDESConfig = field()
    site: SiteInfo = field()
    
    def __attrs_post_init__(self):
        

        if self.config.fin_model is None:
            raise AttributeError("Financial model must be set in `config.fin_model`")

        if isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model
        system_model = self
        financial_model = self.import_financial_model(financial_model, system_model, self.config.name)
        self.sizing(self.config.system_capacity_kw, rating_kwh=self.config.system_capacity_kwh)
        self.initial_SOC = self.config.initial_SOC
        self.state.SOC = self.initial_SOC

        self.state = State(SOC=self.config.initial_SOC)

        self.params = Params(nominal_energy=self.config.system_capacity_kwh, 
                            nominal_voltage=None, 
                            duration=self.config.system_capacity_kw/self.config.system_capacity_kwh,
                           )
        

        super().__init__(self.config.name, self.site, None, financial_model)
        # self.sizing(self.config.system_capacity_kw, rating_kwh=self.config.system_capacity_kwh)

    # def __attrs_post_init__(self):
    #     """Auto-populate _parameters with class attributes on initialization. Method generated using ChatGPT"""
    #     self._parameters = {name: getattr(self, name) for name in fields_dict(self.__class__) if not name.startswith("_")}

    # def value(self, key: str, val=None):
    #     """Getter and setter for attributes using _parameters. Method generated using ChatGPT"""
    #     if val is None:
    #         return getattr(self, key)  # Get the attribute directly
    #     setattr(self, key, val)  # Set attribute, which auto-updates _parameters

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
        
    def sizing(self, rating_kw, rating_kwh):
        self.system_capacity_kw = rating_kw
        self.system_capacity_kwh = rating_kwh
        self.params.nominal_energy = rating_kwh
        self.params.duration = rating_kwh/rating_kw

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
    
    def execute(verbosity=0):
        """Execute battery simulation with the specified level of verbosity. This
        mimics the PySAM battery model execute function.

        Args:
            verbosity (int, optional): Verbosity level (0, or 1). 
                0 means no extra printing, 1 means more printing. Defaults to 0.
        """

        import pdb; pdb.set_trace()
        # need to set
        # ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen', 'n_cycles']

        #
        # - must have
        #     [x] power capacity
        #     [x] duration
        #     - dicharge penalty

        # - team agrees that
        #     - including degradation on time and degradation on cycles
        #         - should be able to do just time
        #         - should be able to do just cycles (rainflow - already in hopp)
        #         - should be able to combine time and cycles
        #     - need charge and discharge rates

        # stateful_attributes = ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen', 'n_cycles']
        # what I get from dispatch


        # needed attributes set here
        # P (power/gen)


        # trying to figure out what to pull from for the battery model
        # following is from PySAM
        # energy_hourly_kW
        # Power output of array [kW]

        # Info: Lifetime system generation

        # INOUT: This variable is both an input and an output to the compute module.

        # Required: Required if en_wave_batt=1

        # Type
        # :
        # sequence
        # gen
        # System power generated [kW]

        # INOUT: This variable is both an input and an output to the compute module.

        # Type
        # :
        # sequence


    @property
    def control_mode(self):
        return self.params.control_mode
    
    @control_mode.setter
    def control_mode(self, control_mode):
        self.params.control_mode = control_mode
 
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

    # @ property
    # def footprint_area() -> float:
    #     return None
    
    # @property
    # def system_mas() -> float:
    #     return None

class LDESTools:

    dummy: 0