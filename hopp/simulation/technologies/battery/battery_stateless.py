from typing import Sequence, List, Optional, Union
from dataclasses import dataclass, asdict

from attrs import define, field

from hopp.simulation.technologies.financial import CustomFinancialModel
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.utilities.log import hybrid_logger as logger
from hopp.utilities.validators import gt_zero, range_val
from hopp.simulation.base import BaseClass


@dataclass
class BatteryStatelessOutputs:
    I: List[float]
    P: List[float]
    SOC: List[float]
    lifecycles_per_day: List[Optional[int]]
    """
    The following outputs are from the HOPP dispatch model, an entry per timestep:
        I: current [A], only applicable to battery dispatch models with current modeled
        P: power [kW]
        SOC: state-of-charge [%]

    This output has a different length, one entry per day:
        lifecycles_per_day: number of cycles per day
    """
    def __init__(self, n_timesteps, n_periods_per_day):
        """Class for storing battery.outputs."""
        self.I = [0.0] * n_timesteps
        self.P = [0.0] * n_timesteps
        self.SOC = [0.0] * n_timesteps
        self.lifecycles_per_day = [None] * int(n_timesteps / n_periods_per_day)

    def export(self):
        return asdict(self)


@define
class BatteryStatelessConfig(BaseClass):
    """
    Configuration class for `BatteryStateless`.

    Converts nested dicts into relevant financial configurations.

    Args:
        tracking: default False -> `BatteryStateless`
        system_capacity_kwh: Battery energy capacity [kWh]
        system_capacity_kw: Battery rated power capacity [kW]
        minimum_SOC: Minimum state of charge [%]
        maximum_SOC: Maximum state of charge [%]
        initial_SOC: Initial state of charge [%]
        fin_model: Financial model. Can be any of the following:

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` instance
    """
    system_capacity_kwh: float = field(validator=gt_zero)
    system_capacity_kw: float = field(validator=gt_zero)
    tracking: bool = field(default=False)
    minimum_SOC: float = field(default=10, validator=range_val(0, 100))
    maximum_SOC: float = field(default=90, validator=range_val(0, 100))
    initial_SOC: float = field(default=10, validator=range_val(0, 100))
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)
    name: str = field(default="BatteryStateless")


@define
class BatteryStateless(PowerSource):
    """
    Battery Storage class with no system model for tracking the state of the battery
    The state variables are pulled directly from the BatteryDispatch pyomo model.
    Therefore, this battery model is compatible only with dispatch methods that use pyomo
    such as:             

    - 'simple': SimpleBatteryDispatch
    - 'convex_LV': ConvexLinearVoltageBatteryDispatch
    - 'non_convex_LV': NonConvexLinearVoltageBatteryDispatch

    Args:
        site: Site information
        config: Battery configuration

    """
    site: SiteInfo
    config: BatteryStatelessConfig

    # initialized from config
    minimum_SOC: float = field(init=False)
    maximum_SOC: float = field(init=False)
    initial_SOC: float = field(init=False)

    def __attrs_post_init__(self):
        system_model = self

        if isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model

        self.financial_model = self.import_financial_model(financial_model, system_model, None)

        self._system_capacity_kw = self.config.system_capacity_kw
        self._system_capacity_kwh = self.config.system_capacity_kwh

        # Minimum set of parameters to set to get statefulBattery to work
        self.minimum_SOC = self.config.minimum_SOC
        self.maximum_SOC = self.config.maximum_SOC
        self.initial_SOC = self.config.initial_SOC

        self._dispatch = None
        self.outputs = BatteryStatelessOutputs(
            n_timesteps=self.site.n_timesteps, 
            n_periods_per_day=self.site.n_periods_per_day
        )

        super().__init__("Battery", self.site, system_model, self.financial_model)

        logger.info("Initialized battery with parameters")

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: Optional[int] = None):
        """
        Step through dispatch solution for battery to collect outputs

        Args:
            n_periods: Number of hours to simulate [hrs]
            sim_start_time: Start hour of simulation horizon
        """
        # Store Dispatch model values, converting to kW from mW
        if sim_start_time is not None:
            time_slice = slice(sim_start_time, sim_start_time + n_periods)
            self.outputs.SOC[time_slice] = [i for i in self.dispatch.soc[0:n_periods]]
            self.outputs.P[time_slice] = [i * 1e3 for i in self.dispatch.power[0:n_periods]]
            self.outputs.I[time_slice] = [i * 1e3 for i in self.dispatch.current[0:n_periods]]
            if self.dispatch.options.include_lifecycle_count:
                days_in_period = n_periods // (self.site.n_periods_per_day)
                start_day = sim_start_time // self.site.n_periods_per_day
                for d in range(days_in_period):
                    self.outputs.lifecycles_per_day[start_day + d] = self.dispatch.lifecycles[d]

        # logger.info("battery.outputs at start time {}".format(sim_start_time, self.outputs))

    def simulate_power(self, time_step=None):
        """
        Runs battery simulate and stores values if time step is provided

        Args:
            time_step: (optional) if provided outputs are stored, o.w. they are not stored.
        """
        pass

    def validate_replacement_inputs(self, project_life):
        """
        Checks that the battery replacement part of the model has the required inputs and that they are formatted correctly.

        `batt_bank_replacement` is a required array of length (project_life + 1), where year 0 is "financial year 0" and is prior to system operation
        If the battery replacements are to follow a schedule (`batt_replacement_option` == 2), the `batt_replacement_schedule_percent` is required.
        This array is of length (project_life), where year 0 is the first year of system operation.
        """
        pass

    def export(self):
        """
        Return all the battery system configuration in a dictionary for the financial model
        """
        config = {
            'system_capacity': self.system_capacity_kw,
            'batt_computed_bank_capacity': self.system_capacity_kwh,
            'minimum_SOC': self.minimum_SOC,
            'maximum_SOC': self.maximum_SOC,
            'initial_SOC': self.initial_SOC,
            'outputs': self.outputs.export()
        }
        return config

    def simulate_financials(self, interconnect_kw: float, project_life: int):
        """
        Sets-up and simulates financial model for the battery

        Args:
            interconnect_kw: Interconnection limit [kW]
            project_life: Analysis period [years]
        """
        self.financial_model.assign(self._system_model.export())       # copy system parameter values having same name
        
        if project_life > 1:
            self.financial_model.value('system_use_lifetime_output', 1)
        else:
            self.financial_model.value('system_use_lifetime_output', 0)
        self.financial_model.value('analysis_period', project_life)

        if len(self.outputs.P) == self.site.n_timesteps:
            single_year_gen = self.outputs.P
            self.financial_model.value('gen', list(single_year_gen) * project_life)

            self.financial_model.value('system_pre_curtailment_kwac', list(single_year_gen) * project_life)
            self.financial_model.value('annual_energy_pre_curtailment_ac', sum(single_year_gen))
            self.financial_model.value('batt_annual_discharge_energy', [sum(i for i in single_year_gen if i > 0)] * project_life)
            self.financial_model.value('batt_annual_charge_energy', [sum(i for i in single_year_gen if i < 0)] * project_life)
            self.financial_model.value('batt_annual_charge_from_system', (0,))
        else:
            raise RuntimeError

        self.financial_model.execute(0)
        logger.info("{} simulation executed".format('battery'))

    @property
    def system_capacity_kwh(self) -> float:
        """Battery energy capacity [kWh]"""
        return self._system_capacity_kwh

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        self.financial_model.value("batt_computed_bank_capacity", size_kwh)
        self.system_capacity_kwh = size_kwh

    @property
    def system_capacity_kw(self) -> float:
        """Battery power rating [kW]"""
        return self._system_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self.financial_model.value("system_capacity", size_kw)
        self.system_capacity_kw = size_kw

    @property
    def system_nameplate_mw(self) -> float:
        """System nameplate [MW]"""
        return self._system_capacity_kw * 1e-3
    
    @property
    def nominal_energy(self) -> float:
        """Battery energy capacity [kWh]"""
        return self._system_capacity_kwh

    @property
    def capacity_factor(self):
        """System capacity factor [%]"""
        return None

    @property
    def generation_profile(self) -> Sequence:
        if self.system_capacity_kwh:
            return self.outputs.P
        else:
            return [0] * self.site.n_timesteps

    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return sum(self.outputs.P)
        else:
            return 0
        
    @property
    def SOC(self) -> float:
        if len(self.outputs.SOC):
            return self.outputs.SOC[0]
        else:
            return self.initial_SOC

    @property
    def lifecycles(self) -> List[Optional[int]]:
        return self.outputs.lifecycles_per_day