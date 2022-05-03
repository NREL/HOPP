from typing import Sequence

import PySAM.BatteryStateful as BatteryModel
import PySAM.BatteryTools as BatteryTools
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *


class Battery_Outputs:
    def __init__(self, n_timesteps):
        """ Class of stateful battery outputs

        """
        self.stateful_attributes = ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen']
        for attr in self.stateful_attributes:
            setattr(self, attr, [0.0]*n_timesteps)

        # dispatch output storage
        dispatch_attributes = ['I', 'P', 'SOC']
        for attr in dispatch_attributes:
            setattr(self, 'dispatch_'+attr, [0.0]*n_timesteps)


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

        :param battery_config: dict, with keys ('system_capacity_kwh', 'system_capacity_kw')
        :param chemistry:
        :param system_voltage_volts:
        """
        for key in ('system_capacity_kwh', 'system_capacity_kw'):
            if key not in battery_config.keys():
                raise ValueError

        system_model = BatteryModel.default(chemistry)
        financial_model = Singleowner.from_existing(system_model, "StandaloneBatterySingleOwner")
        super().__init__("Battery", site, system_model, financial_model)

        self.Outputs = Battery_Outputs(n_timesteps=site.n_timesteps)
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
        self._system_model.value("minimum_SOC", 10.0)
        self._system_model.value("maximum_SOC", 90.0)
        self._system_model.value("initial_SOC", 10.0)

        self._dispatch = None   # TODO: this could be the union of the models

        logger.info("Initialized battery with parameters and state {}".format(self._system_model.export()))

    @property
    def system_capacity_voltage(self) -> tuple:
        return self._system_model.ParamsPack.nominal_energy, self._system_model.ParamsPack.nominal_voltage

    @system_capacity_voltage.setter
    def system_capacity_voltage(self, capacity_voltage: tuple):
        """
        Sets the system capacity and voltage, and updates the system, cost and financial model
        :param capacity_voltage:
        :return:
        """
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
        return self._system_model.ParamsPack.nominal_energy

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kwh:
        """
        self.system_capacity_voltage = (size_kwh, self.system_voltage_volts)

    @property
    def system_capacity_kw(self) -> float:
        return self._system_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        """
        self._financial_model.value("system_capacity", size_kw)
        self._system_capacity_kw = size_kw

    @property
    def system_voltage_volts(self) -> float:
        return self._system_model.ParamsPack.nominal_voltage

    @system_voltage_volts.setter
    def system_voltage_volts(self, voltage_volts: float):
        """
        Sets the system voltage and updates the system, cost and financial model
        :param voltage_volts:
        :return:
        """
        self.system_capacity_voltage = (self.system_capacity_kwh, voltage_volts)

    @property
    def chemistry(self) -> str:
        model_type = self._system_model.ParamsCell.chem
        if model_type == 0 or model_type == 1:
            return self._chemistry
        else:
            raise ValueError("chemistry model type unrecognized")

    @chemistry.setter
    def chemistry(self, battery_chemistry: str):
        """
        Sets the system chemistry and updates the system, cost and financial model
        :param battery_chemistry:
        :return:
        """
        BatteryTools.battery_model_change_chemistry(self._system_model, battery_chemistry)
        self._chemistry = battery_chemistry
        logger.info("Battery chemistry set to {}".format(battery_chemistry))

    def _simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None):
        """
        Step through dispatch solution for battery and simulate battery
        """
        # TODO: This is specific to the Stateful battery model
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

        # logger.info("Battery Outputs at start time {}".format(sim_start_time, self.Outputs))

    def simulate_power(self, time_step=None):
        """
        Runs battery simulate stores values if time step is provided
        """
        if not self._system_model:
            return
        self._system_model.execute(0)

        if time_step is not None:
            self.update_battery_stored_values(time_step)

        # TODO: Do we need to update financial model after battery simulation is complete?

    def update_battery_stored_values(self, time_step):
        # Physical model values
        for attr in self.Outputs.stateful_attributes:
            if hasattr(self._system_model.StatePack, attr):
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
            self._financial_model.BatterySystem.batt_bank_replacement
        except:
            self._financial_model.BatterySystem.batt_bank_replacement = [0] * (project_life + 1)

        if self._financial_model.BatterySystem.batt_replacement_option == 2:
            if len(self._financial_model.BatterySystem.batt_replacement_schedule_percent) != project_life:
                raise ValueError(f"Error in Battery model: `batt_replacement_schedule_percent` should be length of project_life {project_life} but is instead {len(self._financial_model.BatterySystem.batt_replacement_schedule_percent)}")
            if len(self._financial_model.BatterySystem.batt_bank_replacement) != project_life + 1:
                if len(self._financial_model.BatterySystem.batt_bank_replacement) == project_life:
                    # likely an input mistake: add a zero for financial year 0 
                    self._financial_model.BatterySystem.batt_bank_replacement = [0] + list(self._financial_model.BatterySystem.batt_bank_replacement)
                else:
                    raise ValueError(f"Error in Battery model: `batt_bank_replacement` should be length of project_life {project_life} but is instead {len(self._financial_model.BatterySystem.batt_bank_replacement)}")

    def simulate_financials(self, project_life):
        self._financial_model.BatterySystem.batt_computed_bank_capacity = self.system_capacity_kwh

        self.validate_replacement_inputs(project_life)

        if project_life > 1:
            self._financial_model.Lifetime.system_use_lifetime_output = 1
        else:
            self._financial_model.Lifetime.system_use_lifetime_output = 0
        self._financial_model.FinancialParameters.analysis_period = project_life
        self._financial_model.CapacityPayments.cp_system_nameplate = self.system_capacity_kw
        self._financial_model.SystemCosts.om_batt_nameplate = self.system_capacity_kw
        try:
            if self._financial_model.SystemCosts.om_production != 0:
                raise ValueError("Battery's 'om_production' must be 0. For variable O&M cost based on battery discharge, "
                                 "use `om_batt_variable_cost`, which is in $/MWh.")
        except:
            # om_production not set, so ok
            pass
        self._financial_model.Revenue.ppa_soln_mode = 1
        # TODO: out to get SystemOutput.gen to populate?
        # if len(self._financial_model.SystemOutput.gen) == self.site.n_timesteps:
        if len(self.Outputs.gen) == self.site.n_timesteps:
            single_year_gen = self.Outputs.gen
            self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

            self._financial_model.SystemOutput.system_pre_curtailment_kwac = list(single_year_gen) * project_life
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = sum(single_year_gen)
        else:
            raise NotImplementedError

        self._financial_model.LCOS.batt_annual_discharge_energy = [sum(i for i in single_year_gen if i > 0)] * project_life
        self._financial_model.LCOS.batt_annual_charge_energy = [sum(i for i in single_year_gen if i < 0)] * project_life
        # Do not calculate LCOS, so skip these inputs for now by unassigning or setting to 0
        self._financial_model.unassign("battery_total_cost_lcos")
        self._financial_model.LCOS.batt_annual_charge_from_system = (0,)

        self._financial_model.execute(0)
        logger.info("{} simulation executed".format('battery'))

    @property
    def generation_profile(self) -> Sequence:
        if self.system_capacity_kwh:
            return self.Outputs.gen
        else:
            return [0] * self.site.n_timesteps

    @property
    def replacement_costs(self) -> Sequence:
        if self.system_capacity_kw:
            return self._financial_model.Outputs.cf_battery_replacement_cost
        else:
            return [0] * self.site.n_timesteps

    @property
    def annual_energy_kw(self) -> float:
        if self.system_capacity_kw > 0:
            return sum(self.Outputs.gen)
        else:
            return 0
