from dataclasses import dataclass, asdict
from typing import Sequence

import PySAM.BatteryStateful as BatteryModel
import PySAM.BatteryTools as BatteryTools
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *


@dataclass
class BatteryOutputs:
    I: Sequence
    P: Sequence
    Q: Sequence
    SOC: Sequence
    T_batt: Sequence
    gen: Sequence
    n_cycles: Sequence
    dispatch_I: Sequence
    dispatch_P: Sequence
    dispatch_SOC: Sequence
    dispatch_lifecycles_per_day: Sequence
    """
    The following outputs are simulated from the BatteryStateful model, an entry per timestep:
        I: current [A]
        P: power [kW]
        Q: capacity [Ah]
        SOC: state-of-charge [%]
        T_batt: temperature [C]
        gen: same as P
        n_cycles: number of rainflow cycles elapsed since start of simulation [1]

    The next outputs, an entry per timestep, are from the HOPP dispatch model, which are then passed to the simulation:
        dispatch_I: current [A], only applicable to battery dispatch models with current modeled
        dispatch_P: power [mW]
        dispatch_SOC: state-of-charge [%]
    
    This output has a different length, one entry per day:
        dispatch_lifecycles_per_day: number of cycles per day
    """

    def __init__(self, n_timesteps, n_periods_per_day):
        """Class for storing stateful battery and dispatch outputs."""
        self.stateful_attributes = ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen', 'n_cycles']
        for attr in self.stateful_attributes:
            setattr(self, attr, [0.0] * n_timesteps)

        dispatch_attributes = ['I', 'P', 'SOC']
        for attr in dispatch_attributes:
            setattr(self, 'dispatch_'+attr, [0.0] * n_timesteps)

        self.dispatch_lifecycles_per_day = [None] * int(n_timesteps / n_periods_per_day)

    def export(self):
        return asdict(self)


class Battery(PowerSource):
    _system_model: BatteryModel.BatteryStateful
    _financial_model: Singleowner.Singleowner

    module_specs = {'capacity': 400, 'surface_area': 30} # 400 [kWh] -> 30 [m^2]

    def __init__(self,
                 site: SiteInfo,
                 battery_config: dict,
                 chemistry: str = 'lfpgraphite',
                 system_voltage_volts: float = 500):
        """
        Battery Storage class based on PySAM's BatteryStateful Model

        :param site: Power source site information (SiteInfo object)
        :param battery_config: Battery configuration with the following keys:

            #. ``tracking``: bool, must be False, otherwise BatteryStateless will be used instead
            #. ``system_capacity_kwh``: float, Battery energy capacity [kWh]
            #. ``system_capacity_kw``: float, Battery rated power capacity [kW]
            #. ``minimum_SOC``: float, (default=10) Minimum state of charge [%]
            #. ``maximum_SOC``: float, (default=90) Maximum state of charge [%]
            #. ``initial_SOC``: float, (default=10) Initial state of charge [%]

        :param chemistry: Battery storage chemistry, options include:

            #. ``LFPGraphite``: Lithium Iron Phosphate (Lithium Ion)
            #. ``LMOLTO``: LMO/Lithium Titanate (Lithium Ion)
            #. ``LeadAcid``: Lead Acid
            #. ``NMCGraphite``: Nickel Manganese Cobalt Oxide (Lithium Ion)

        :param system_voltage_volts: Battery system voltage [VDC]
        """
        for key in ('system_capacity_kwh', 'system_capacity_kw'):
            if key not in battery_config.keys():
                raise ValueError

        self.config_name = "StandaloneBatterySingleOwner"
        system_model = BatteryModel.default(chemistry)

        if 'fin_model' in battery_config.keys():
            financial_model = self.import_financial_model(battery_config['fin_model'], system_model, self.config_name)
        else:
            financial_model = Singleowner.from_existing(system_model, self.config_name)

        super().__init__("Battery", site, system_model, financial_model)

        self.Outputs = BatteryOutputs(n_timesteps=site.n_timesteps, n_periods_per_day=site.n_periods_per_day)
        self.system_capacity_kw: float = battery_config['system_capacity_kw']
        self.chemistry = chemistry
        BatteryTools.battery_model_sizing(self._system_model,
                                          battery_config['system_capacity_kw'],
                                          battery_config['system_capacity_kwh'],
                                          system_voltage_volts,
                                          module_specs=Battery.module_specs)
        self._system_model.ParamsPack.h = 20
        self._system_model.ParamsPack.Cp = 900
        self._system_model.ParamsCell.resistance = 0.001
        self._system_model.ParamsCell.C_rate = battery_config['system_capacity_kw'] / battery_config['system_capacity_kwh']

        # Minimum set of parameters to set to get statefulBattery to work
        self._system_model.value("control_mode", 0.0)
        self._system_model.value("input_current", 0.0)
        self._system_model.value("dt_hr", 1.0)
        self._system_model.value("minimum_SOC", battery_config['minimum_SOC'] if 'minimum_SOC' in battery_config.keys() else 10.0)
        self._system_model.value("maximum_SOC", battery_config['maximum_SOC'] if 'maximum_SOC' in battery_config.keys() else 90.0)
        self._system_model.value("initial_SOC", battery_config['initial_SOC'] if 'initial_SOC' in battery_config.keys() else 10.0)

        self._dispatch = None

        logger.info("Initialized battery with parameters and state {}".format(self._system_model.export()))

    def setup_system_model(self):
        """Executes Stateful Battery setup"""
        self._system_model.setup()

    @property
    def system_capacity_voltage(self) -> tuple:
        """Battery energy capacity [kWh] and voltage [VDC]"""
        return self._system_model.ParamsPack.nominal_energy, self._system_model.ParamsPack.nominal_voltage

    @system_capacity_voltage.setter
    def system_capacity_voltage(self, capacity_voltage: tuple):
        size_kwh = capacity_voltage[0]
        voltage_volts = capacity_voltage[1]

        # sizing function may run into future issues if size_kwh == 0 is allowed
        if size_kwh == 0:
            size_kwh = 1e-7

        BatteryTools.battery_model_sizing(self._system_model,
                                          0.,
                                          size_kwh,
                                          voltage_volts,
                                          module_specs=Battery.module_specs)
        logger.info("Battery set system_capacity to {} kWh".format(size_kwh))
        logger.info("Battery set system_voltage to {} volts".format(voltage_volts))

    @property
    def system_capacity_kwh(self) -> float:
        """Battery energy capacity [kWh]"""
        return self._system_model.ParamsPack.nominal_energy

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        self.system_capacity_voltage = (size_kwh, self.system_voltage_volts)

    @property
    def system_capacity_kw(self) -> float:
        """Battery power rating [kW]"""
        return self._system_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._financial_model.value("system_capacity", size_kw)
        self._system_capacity_kw = size_kw

    @property
    def system_voltage_volts(self) -> float:
        """Battery bank voltage [VDC]"""
        return self._system_model.ParamsPack.nominal_voltage

    @system_voltage_volts.setter
    def system_voltage_volts(self, voltage_volts: float):
        self.system_capacity_voltage = (self.system_capacity_kwh, voltage_volts)

    @property
    def chemistry(self) -> str:
        """Battery chemistry type"""
        model_type = self._system_model.ParamsCell.chem
        if model_type == 0 or model_type == 1:
            return self._chemistry
        else:
            raise ValueError("chemistry model type unrecognized")

    @chemistry.setter
    def chemistry(self, battery_chemistry: str):
        BatteryTools.battery_model_change_chemistry(self._system_model, battery_chemistry)
        self._chemistry = battery_chemistry
        logger.info("Battery chemistry set to {}".format(battery_chemistry))

    def setup_performance_model(self):
        """Executes Stateful Battery setup"""
        self._system_model.setup()

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None):
        """
        Step through dispatch solution for battery and simulate battery

        :param n_periods: Number of hours to simulate [hrs]
        :param sim_start_time: Start hour of simulation horizon
        """
        # Set stateful control value [Discharging (+) + Charging (-)]
        if self.value("control_mode") == 1.0:
            control = [pow_MW*1e3 for pow_MW in self.dispatch.power]    # MW -> kW
        elif self.value("control_mode") == 0.0:
            control = [cur_MA * 1e6 for cur_MA in self.dispatch.current]    # MA -> A
        else:
            raise ValueError("Stateful battery module 'control_mode' invalid value.")

        time_step_duration = self.dispatch.time_duration
        for t in range(n_periods):
            self.value('dt_hr', time_step_duration[t])
            self.value(self.dispatch.control_variable, control[t])

            # Only store information if passed the previous day simulations (used in clustering)
            try:
                index_time_step = sim_start_time + t  # Store information
            except TypeError:
                index_time_step = None  # Don't store information
            self.simulate_power(time_step=index_time_step)

        # Store Dispatch model values
        if sim_start_time is not None:
            time_slice = slice(sim_start_time, sim_start_time + n_periods)
            self.Outputs.dispatch_SOC[time_slice] = self.dispatch.soc[0:n_periods]
            self.Outputs.dispatch_P[time_slice] = self.dispatch.power[0:n_periods]
            self.Outputs.dispatch_I[time_slice] = self.dispatch.current[0:n_periods]
            if self.dispatch.options.include_lifecycle_count:
                days_in_period = n_periods // (self.site.n_periods_per_day)
                start_day = sim_start_time // self.site.n_periods_per_day
                for d in range(days_in_period):
                    self.Outputs.dispatch_lifecycles_per_day[start_day + d] = self.dispatch.lifecycles[d]

        # logger.info("Battery Outputs at start time {}".format(sim_start_time, self.Outputs))

    def simulate_power(self, time_step=None):
        """
        Runs battery simulate and stores values if time step is provided

        :param time_step: (optional) if provided outputs are stored, o.w. they are not stored.
        """
        if not self._system_model:
            return
        self._system_model.execute(0)

        if time_step is not None:
            self.update_battery_stored_values(time_step)

    def update_battery_stored_values(self, time_step):
        """
        Stores Stateful battery outputs at time step provided.

        :param time_step: time step where outputs will be stored.
        """
        for attr in self.Outputs.stateful_attributes:
            if hasattr(self._system_model.StatePack, attr) or hasattr(self._system_model.StateCell, attr):
                getattr(self.Outputs, attr)[time_step] = self.value(attr)
            else:
                if attr == 'gen':
                    getattr(self.Outputs, attr)[time_step] = self.value('P')

    def validate_replacement_inputs(self, project_life):
        """
        Checks that the battery replacement part of the model has the required inputs and that they are formatted correctly.

        `batt_bank_replacement` is a required array of length (project_life + 1), where year 0 is "financial year 0" and is prior to system operation
        If the battery replacements are to follow a schedule (`batt_replacement_option` == 2), the `batt_replacement_schedule_percent` is required.
        This array is of length (project_life), where year 0 is the first year of system operation.
        """
        try:
            self._financial_model.value('batt_bank_replacement')
        except:
            self._financial_model.value('batt_bank_replacement', [0] * (project_life + 1))

        if self._financial_model.value('batt_replacement_option') == 2:
            if len(self._financial_model.value('batt_replacement_schedule_percent')) != project_life:
                raise ValueError(f"Error in Battery model: `batt_replacement_schedule_percent` should be length of project_life {project_life} but is instead {len(self._financial_model.value('batt_replacement_schedule_percent'))}")
            if len(self._financial_model.value('batt_bank_replacement')) != project_life + 1:
                if len(self._financial_model.value('batt_bank_replacement')) == project_life:
                    # likely an input mistake: add a zero for financial year 0 
                    self._financial_model.value('batt_bank_replacement', [0] + list(self._financial_model.value('batt_bank_replacement')))
                else:
                    raise ValueError(f"Error in Battery model: `batt_bank_replacement` should be length of project_life {project_life} but is instead {len(self._financial_model.value('batt_bank_replacement'))}")

    def simulate_financials(self, interconnect_kw: float, project_life: int, cap_cred_avail_storage: bool = True):
        """
        Sets-up and simulates financial model for the battery

        :param interconnect_kw: Interconnection limit [kW]
        :param project_life: Analysis period [years]
        :param cap_cred_avail_storage: Base capacity credit on available storage (True),
                                            otherwise use only dispatched generation (False)
        """
        if not isinstance(self._financial_model, Singleowner.Singleowner):
            self._financial_model.assign(self._system_model.export(), ignore_missing_vals=True)       # copy system parameter values having same name
        else:
            self._financial_model.value('om_batt_nameplate', self.system_capacity_kw)
            self._financial_model.value('ppa_soln_mode', 1)
        
        self._financial_model.value('batt_computed_bank_capacity', self.system_capacity_kwh)

        self.validate_replacement_inputs(project_life)

        if project_life > 1:
            self._financial_model.value('system_use_lifetime_output', 1)
        else:
            self._financial_model.value('system_use_lifetime_output', 0)
        self._financial_model.value('analysis_period', project_life)
        try:
            if self._financial_model.value('om_production') != 0:
                raise ValueError("Battery's 'om_production' must be 0. For variable O&M cost based on battery discharge, "
                                 "use `om_batt_variable_cost`, which is in $/MWh.")
        except:
            # om_production not set, so ok
            pass

        if len(self.Outputs.gen) == self.site.n_timesteps:
            single_year_gen = self.Outputs.gen
            self._financial_model.value('gen', list(single_year_gen) * project_life)

            self._financial_model.value('system_pre_curtailment_kwac', list(single_year_gen) * project_life)
            self._financial_model.value('annual_energy_pre_curtailment_ac', sum(single_year_gen))
            self._financial_model.value('batt_annual_discharge_energy', [sum(i for i in single_year_gen if i > 0)] * project_life)
            self._financial_model.value('batt_annual_charge_energy', [sum(i for i in single_year_gen if i < 0)] * project_life)
            # Do not calculate LCOS, so skip these inputs for now by unassigning or setting to 0
            self._financial_model.unassign("battery_total_cost_lcos")
            self._financial_model.value('batt_annual_charge_from_system', (0,))
        else:
            raise NotImplementedError

        # need to store for later grid aggregation
        self.gen_max_feasible = self.calc_gen_max_feasible_kwh(interconnect_kw, cap_cred_avail_storage)
        self.capacity_credit_percent = self.calc_capacity_credit_percent(interconnect_kw)

        self._financial_model.execute(0)
        logger.info("{} simulation executed".format('battery'))

    def calc_gen_max_feasible_kwh(self, interconnect_kw, use_avail_storage: bool = True) -> list:
        """
        Calculates the maximum feasible capacity (generation profile) that could have occurred.

        :param interconnect_kw: Interconnection limit [kW]
        :param use_avail_storage: Base capacity credit on available storage (True),
                                            otherwise use only dispatched generation (False)

        :return: maximum feasible capacity [kWh]: list of floats
        """
        t_step = self.site.interval / 60                                                # hr
        df = pd.DataFrame()
        df['E_delivered'] = [max(0, x * t_step) for x in self.Outputs.P]                # [kWh]
        df['SOC_perc'] = self.Outputs.SOC                                               # [%]
        df['E_stored'] = df.SOC_perc / 100 * self.system_capacity_kwh                   # [kWh]

        def max_feasible_kwh(row):
            return min(self.system_capacity_kw * t_step, row.E_delivered + row.E_stored)

        if use_avail_storage:
            E_max_feasible = df.apply(max_feasible_kwh, axis=1)                             # [kWh]
        else:
            E_max_feasible = df['E_delivered']

        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        E_max_feasible = np.minimum(E_max_feasible, W_ac_nom*t_step) 
        
        return list(E_max_feasible)

    @property
    def generation_profile(self) -> Sequence:
        if self.system_capacity_kwh:
            return self.Outputs.gen
        else:
            return [0] * self.site.n_timesteps

    @property
    def replacement_costs(self) -> Sequence:
        """Battery replacement cost [$]"""
        if self.system_capacity_kw:
            return self._financial_model.value('cf_battery_replacement_cost')
        else:
            return [0] * self.site.n_timesteps

    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return sum(self.Outputs.gen)
        else:
            return 0
