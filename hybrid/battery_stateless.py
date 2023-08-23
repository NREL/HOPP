from typing import Sequence
from dataclasses import dataclass, asdict

from hybrid.financial.custom_financial_model import CustomFinancialModel
from hybrid.power_source import *


@dataclass
class BatteryStatelessOutputs:
    I: Sequence
    P: Sequence 
    SOC: Sequence
    lifecycles_per_day: Sequence
    """
    The following outputs are from the HOPP dispatch model, an entry per timestep:
        I: current [A], only applicable to battery dispatch models with current modeled
        P: power [kW]
        SOC: state-of-charge [%]

    This output has a different length, one entry per day:
        lifecycles_per_day: number of cycles per day
    """
    def __init__(self, n_timesteps, n_periods_per_day):
        """Class for storing battery outputs."""
        self.I = [0.0] * n_timesteps
        self.P = [0.0] * n_timesteps
        self.SOC = [0.0] * n_timesteps
        self.lifecycles_per_day = [None] * int(n_timesteps / n_periods_per_day)

    def export(self):
        return asdict(self)


class BatteryStateless(PowerSource):
    _financial_model: CustomFinancialModel

    def __init__(self,
                 site: SiteInfo,
                 battery_config: dict):
        """
        Battery Storage class with no system model for tracking the state of the battery
        The state variables are pulled directly from the BatteryDispatch pyomo model.
        Therefore, this battery model is compatible only with dispatch methods that use pyomo
        such as             
            'simple': SimpleBatteryDispatch,
            'convex_LV': ConvexLinearVoltageBatteryDispatch}
            'non_convex_LV': NonConvexLinearVoltageBatteryDispatch,

        :param site: Power source site information (SiteInfo object)
        :param battery_config: Battery configuration with the following keys:

            #. ``tracking``: bool, must be True, otherwise Battery will be used instead
            #. ``system_capacity_kwh``: float, Battery energy capacity [kWh]
            #. ``minimum_SOC``: float, (default=10) Minimum state of charge [%]
            #. ``maximum_SOC``: float, (default=90) Maximum state of charge [%]
            #. ``initial_SOC``: float, (default=50) Initial state of charge [%]
            #. ``fin_model``: CustomFinancialModel, instance of financial model

        """
        for key in ('system_capacity_kwh', 'system_capacity_kw'):
            if key not in battery_config.keys():
                raise ValueError

        system_model = self

        if 'fin_model' in battery_config.keys():
            financial_model = self.import_financial_model(battery_config['fin_model'], system_model, None)
        else:
            raise ValueError("When using 'BatteryStateless', an instantiated CustomFinancialModel must be provided as the 'fin_model' in the battery_config")

        self._system_capacity_kw: float = battery_config['system_capacity_kw']
        self._system_capacity_kwh: float = battery_config['system_capacity_kwh']

        # Minimum set of parameters to set to get statefulBattery to work
        self.minimum_SOC = battery_config['minimum_SOC'] if 'minimum_SOC' in battery_config.keys() else 10.0
        self.maximum_SOC = battery_config['maximum_SOC'] if 'maximum_SOC' in battery_config.keys() else 90.0
        self.initial_SOC = battery_config['initial_SOC'] if 'initial_SOC' in battery_config.keys() else 10.0

        self._dispatch = None
        self.Outputs = BatteryStatelessOutputs(n_timesteps=site.n_timesteps, n_periods_per_day=site.n_periods_per_day)

        super().__init__("Battery", site, system_model, financial_model)

        logger.info("Initialized battery with parameters")

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None):
        """
        Step through dispatch solution for battery to collect outputs

        :param n_periods: Number of hours to simulate [hrs]
        :param sim_start_time: Start hour of simulation horizon
        """
        # Store Dispatch model values, converting to kW from mW
        if sim_start_time is not None:
            time_slice = slice(sim_start_time, sim_start_time + n_periods)
            self.Outputs.SOC[time_slice] = [i for i in self.dispatch.soc[0:n_periods]]
            self.Outputs.P[time_slice] = [i * 1e3 for i in self.dispatch.power[0:n_periods]]
            self.Outputs.I[time_slice] = [i * 1e3 for i in self.dispatch.current[0:n_periods]]
            if self.dispatch.options.include_lifecycle_count:
                days_in_period = n_periods // (self.site.n_periods_per_day)
                start_day = sim_start_time // self.site.n_periods_per_day
                for d in range(days_in_period):
                    self.Outputs.lifecycles_per_day[start_day + d] = self.dispatch.lifecycles[d]

        # logger.info("Battery Outputs at start time {}".format(sim_start_time, self.Outputs))

    def simulate_power(self, time_step=None):
        """
        Runs battery simulate and stores values if time step is provided

        :param time_step: (optional) if provided outputs are stored, o.w. they are not stored.
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
            'Outputs': self.Outputs.export()
        }
        return config

    def simulate_financials(self, interconnect_kw: float, project_life: int):
        """
        Sets-up and simulates financial model for the battery

        :param interconnect_kw: Interconnection limit [kW]
        :param project_life: Analysis period [years]
        """
        self._financial_model.assign(self._system_model.export(), ignore_missing_vals=True)       # copy system parameter values having same name
        
        if project_life > 1:
            self._financial_model.value('system_use_lifetime_output', 1)
        else:
            self._financial_model.value('system_use_lifetime_output', 0)
        self._financial_model.value('analysis_period', project_life)

        if len(self.Outputs.P) == self.site.n_timesteps:
            single_year_gen = self.Outputs.P
            self._financial_model.value('gen', list(single_year_gen) * project_life)

            self._financial_model.value('system_pre_curtailment_kwac', list(single_year_gen) * project_life)
            self._financial_model.value('annual_energy_pre_curtailment_ac', sum(single_year_gen))
            self._financial_model.value('batt_annual_discharge_energy', [sum(i for i in single_year_gen if i > 0)] * project_life)
            self._financial_model.value('batt_annual_charge_energy', [sum(i for i in single_year_gen if i < 0)] * project_life)
            self._financial_model.value('batt_annual_charge_from_system', (0,))
        else:
            raise RuntimeError

        self._financial_model.execute(0)
        logger.info("{} simulation executed".format('battery'))

    @property
    def system_capacity_kwh(self) -> float:
        """Battery energy capacity [kWh]"""
        return self._system_capacity_kwh

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        self._financial_model.value("batt_computed_bank_capacity", size_kwh)
        self.system_capacity_kwh = size_kwh

    @property
    def system_capacity_kw(self) -> float:
        """Battery power rating [kW]"""
        return self._system_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._financial_model.value("system_capacity", size_kw)
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
    def capacity_factor(self) -> float:
        """System capacity factor [%]"""
        return None

    @property
    def generation_profile(self) -> Sequence:
        if self.system_capacity_kwh:
            return self.Outputs.P
        else:
            return [0] * self.site.n_timesteps

    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return sum(self.Outputs.P)
        else:
            return 0
        
    @property
    def SOC(self) -> float:
        if len(self.Outputs.SOC):
            return self.Outputs.SOC[0]
        else:
            return self.initial_SOC

    @property
    def lifecycles(self) -> float:
        return self.Outputs.lifecycles_per_day