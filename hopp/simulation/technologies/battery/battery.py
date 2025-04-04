from dataclasses import dataclass, asdict
from typing import Optional, Sequence, List, Union
import numpy as np
import pandas as pd
import math


from attrs import define, field
import PySAM.BatteryStateful as PySAMBatteryModel
import PySAM.BatteryTools as BatteryTools
import PySAM.Singleowner as Singleowner
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import FinancialModelType, CustomFinancialModel
from hopp.simulation.technologies.ldes.ldes_system_model import LDES

from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.sites.site_info import SiteInfo

from hopp.utilities.log import hybrid_logger as logger
from hopp.utilities.validators import contains, gt_zero, range_val

@dataclass
class BatteryOutputs:
    I: Sequence
    P: Sequence
    Q: Sequence
    SOC: Sequence
    T_batt: Sequence
    gen: Sequence
    n_cycles: Sequence
    dispatch_I: List[float]
    dispatch_P: List[float]
    dispatch_SOC: List[float]
    dispatch_lifecycles_per_day: List[Optional[int]]
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


@define
class BatteryConfig(BaseClass):
    """
    Configuration class for `Battery`.

    Args:
        tracking: default True -> `Battery`
        system_capacity_kwh: Battery energy capacity [kWh]
        system_capacity_kw: Battery rated power capacity [kW]
        system_model_source: software source for the system model, can by 'pysam' or 'hopp'
        chemistry: Battery chemistry option

            PySAM options:
                - "LFPGraphite" (default)
                - "LMOLTO"
                - "LeadAcid" 
                - "NMCGraphite"
            HOPP options:
                - "LDES" generic long-duration energy storage
        minimum_SOC: Minimum state of charge [%]
        maximum_SOC: Maximum state of charge [%]
        initial_SOC: Initial state of charge [%]
        fin_model: Financial model. Can be any of the following:
            - a dict representing a `CustomFinancialModel`
            - an object representing a `CustomFinancialModel` or a `Singleowner.Singleowner` instance
    """
    system_capacity_kwh: float = field(validator=gt_zero)
    system_capacity_kw: float = field(validator=gt_zero)
    system_model_source: str = field(default="pysam", validator=contains(["pysam", "hopp"]))
    chemistry: str = field(default="LFPGraphite", validator=contains(["LFPGraphite", "LMOLTO", "LeadAcid", "NMCGraphite", "LDES"]))
    tracking: bool = field(default=True)
    minimum_SOC: float = field(default=10, validator=range_val(0, 100))
    maximum_SOC: float = field(default=90, validator=range_val(0, 100))
    initial_SOC: float = field(default=10, validator=range_val(0, 100))
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)
    name: str = field(default="Battery")

@define
class Battery(PowerSource):
    """
    Battery storage class based on PySAM's BatteryStateful Model. 

    Args:
        site: Site information
        config: Battery configuration
    """
    site: SiteInfo
    config: BatteryConfig
    config_name: str = field(default="StandaloneBatterySingleOwner")

    outputs: BatteryOutputs = field(init=False)

    # TODO: should this be made configurable by users?
    module_specs = {'capacity': 400, 'surface_area': 30} # 400 [kWh] -> 30 [m^2]

    def __attrs_post_init__(self):
        """
        """
        if self.config.system_model_source == "pysam":
            system_model = PySAMBatteryModel.default(self.config.chemistry)
        elif self.config.system_model_source == "hopp":
            system_model = LDES(self.config, self.site)
            self.cycle_count = 0.0
        else:
            raise(ValueError("Invalid value for battery system_model_source, must be one of ['pysam', 'hopp']"))

        if isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model

        if financial_model is None:
            if self.config.system_model_source == "hopp":
                raise(ValueError("Must specify an instance of or input dict for the custom financial model when using 'hopp' as the `system_model_source`"))
            # default
            else:
                financial_model = Singleowner.from_existing(system_model, self.config_name)
        else:
            financial_model = self.import_financial_model(financial_model, system_model, self.config_name)

        super().__init__("Battery", self.site, system_model, financial_model)

        self.outputs = BatteryOutputs(n_timesteps=self.site.n_timesteps, n_periods_per_day=self.site.n_periods_per_day)
        
        self.system_capacity_kw = self.config.system_capacity_kw ## failing here on the set

        self.chemistry = self.config.chemistry
        if self.config.system_model_source == "pysam":
            BatteryTools.battery_model_sizing(self._system_model,
                                          self.config.system_capacity_kw,
                                          self.config.system_capacity_kwh,
                                          self.system_voltage_volts,
                                          module_specs=Battery.module_specs)
            self._system_model.ParamsPack.h = 20
            self._system_model.ParamsPack.Cp = 900
            self._system_model.ParamsCell.resistance = 0.001
            self._system_model.ParamsCell.C_rate = self.config.system_capacity_kw / self.config.system_capacity_kwh


        # Minimum set of parameters to set to get statefulBattery to work
        self._system_model.value("control_mode", 0.0)
        self._system_model.value("input_current", 0.0)

        self._system_model.value("dt_hr", 1.0)
        self._system_model.value("minimum_SOC", self.config.minimum_SOC)
        self._system_model.value("maximum_SOC", self.config.maximum_SOC)
        self._system_model.value("initial_SOC", self.config.initial_SOC)

        self._dispatch = None

        logger.info("Initialized battery with parameters and state {}".format(self._system_model.export()))



    def setup_system_model(self):
        """Executes Stateful Battery setup"""
        if self.config.system_model_source == "pysam":
            self._system_model.setup()
        else:
            pass

    @property
    def system_capacity_voltage(self) -> tuple:
        """Battery energy capacity [kWh] and voltage [VDC]"""

        if self.config.system_model_source == "pysam":
            return self._system_model.ParamsPack.nominal_energy, self._system_model.ParamsPack.nominal_voltage
        elif self.config.system_model_source == "hopp":
            return self._system_model.params.nominal_energy, self._system_model.params.nominal_voltage

    @system_capacity_voltage.setter
    def system_capacity_voltage(self, capacity_voltage: tuple):
        size_kwh = capacity_voltage[0]
        voltage_volts = capacity_voltage[1]

        # sizing function may run into future issues if size_kwh == 0 is allowed
        if size_kwh == 0:
            size_kwh = 1e-7

        if self.config.system_model_source == "pysam":
            BatteryTools.battery_model_sizing(self._system_model,
                                            0.,
                                            size_kwh,
                                            voltage_volts,
                                            module_specs=Battery.module_specs)
        else:
            self._system_model.params.nominal_voltage = voltage_volts
            self._system_model.params.nominal_energy = size_kwh
            
        logger.info("Battery set system_capacity to {} kWh".format(size_kwh))
        logger.info("Battery set system_voltage to {} volts".format(voltage_volts))

    @property
    def system_capacity_kwh(self) -> float:
        """Battery energy capacity [kWh]"""

        if self.config.system_model_source == "pysam":
            return self._system_model.ParamsPack.nominal_energy
        else:
            return self._system_model.params.nominal_energy

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

        if self.config.system_model_source == "pysam":
            return self._system_model.ParamsPack.nominal_voltage
        # not in LDES/HOPP yet, but could add? May need more specifics about the physics/chemistry

    @system_voltage_volts.setter
    def system_voltage_volts(self, voltage_volts: float):
        self.system_capacity_voltage = (self.system_capacity_kwh, voltage_volts)

    @property
    def chemistry(self) -> str:
        """Battery chemistry type"""

        if self.config.system_model_source == "pysam":
            model_type = self._system_model.ParamsCell.chem
        elif self.config.system_model_source == "hopp":
            model_type = self._system_model.model_type() #TODO revisit this
        
        if model_type == 0 or model_type == 1:
            return self._chemistry
        else:
            raise ValueError("chemistry model type unrecognized")

    @property    
    def system_mass(self) -> float:
        """Battery bank mass [kg]"""

        if self.config.system_model_source == "pysam":
            return self._system_model.ParamsPack.mass

    @property
    def footprint_area(self) -> float:
        """Battery bank footprint area [m^2]"""

        if self.config.system_model_source == "pysam":
            #Battery thermal model assumes a cube for heat exchange
            cube_surface_area = self._system_model.ParamsPack.surface_area 
            footprint = cube_surface_area / 6 # Single side of cube
            return footprint
        # TODO should be added for individual LDES storage types

    @property
    def energy_capital_cost(self) -> Union[float, int]:
        """The capital cost per unit of energy capacity [$/kWh] for battery storage technology """
        return self.config.energy_capital_cost
    
    @property
    def power_capital_cost(self) -> Union[float, int]:
        """ The capital cost per unit of power capacity [$/kW] for battery storage technology """
        return self.config.power_capital_cost
    
    @chemistry.setter
    def chemistry(self, battery_chemistry: str):
        """
        Set battery chemistry.

        Battery storage chemistry options include:
            - `LFPGraphite`: Lithium Iron Phosphate (Lithium Ion)
            - `LMOLTO`: LMO/Lithium Titanate (Lithium Ion)
            - `LeadAcid`: Lead Acid
            - `NMCGraphite`: Nickel Manganese Cobalt Oxide (Lithium Ion)
        """

        if self.config.system_model_source == "pysam":
            BatteryTools.battery_model_change_chemistry(self._system_model, battery_chemistry)
        # TODO update LDES in this setter   
        self._chemistry = battery_chemistry
        logger.info("Battery chemistry set to {}".format(battery_chemistry))

    def setup_performance_model(self):
        """Executes Stateful Battery setup"""

        if self.config.system_model_source == "pysam":
            self._system_model.setup()

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: Optional[int] = None):
        """
        Step through dispatch solution for battery and simulate battery

        Args:
            n_periods: Number of hours to simulate [hrs]
            sim_start_time: Start hour of simulation horizon

        """
        if self.dispatch is None:
            raise ValueError("No dispatch set for this battery.")

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
            if sim_start_time is not None:
                index_time_step = sim_start_time + t  # Store information
            else:
                index_time_step = None

            self.simulate_power(time_step=index_time_step) #, cycles=self.dispatch.lifecycles[t])

        if self.config.system_model_source == "hopp" and self.dispatch.options.include_lifecycle_count:
            self.cycle_count += self.dispatch.lifecycles[0]

        # Store Dispatch model values
        if sim_start_time is not None:
            time_slice = slice(sim_start_time, sim_start_time + n_periods)
            self.outputs.dispatch_SOC[time_slice] = self.dispatch.soc[0:n_periods]
            self.outputs.dispatch_P[time_slice] = self.dispatch.power[0:n_periods]
            self.outputs.dispatch_I[time_slice] = self.dispatch.current[0:n_periods]
            if self.config.system_model_source == "hopp" and self.dispatch.options.include_lifecycle_count:
                self.outputs.n_cycles[time_slice] = [math.floor(self.cycle_count)]*int(n_periods)
            if self.dispatch.options.include_lifecycle_count:
                days_in_period = n_periods // (self.site.n_periods_per_day)
                start_day = sim_start_time // self.site.n_periods_per_day
                for d in range(days_in_period):
                    self.outputs.dispatch_lifecycles_per_day[start_day + d] = self.dispatch.lifecycles[d]

        # logger.info("battery.outputs at start time {}".format(sim_start_time, self.outputs))

    def simulate_power(self, time_step=None):#, cycles=0):
        """
        Runs battery simulate and stores values if time step is provided

        Args:
            time_step: (optional) if provided outputs are stored, o.w. they are not stored.
        """
        if not self._system_model:
            return
        if self.config.system_model_source == "pysam":
            self._system_model.execute(0)   # TODO mimic this function in LDES model - needs to update SOC? - needs 
        else:
            self._system_model.execute(0)

        if time_step is not None:
            self.update_battery_stored_values(time_step)

    def update_battery_stored_values(self, time_step):
        """
        Stores Stateful battery.outputs at time step provided.

        Args:
            time_step: time step where outputs will be stored.
        """
        for attr in self.outputs.stateful_attributes: # TODO include State in LDES model
            if self.config.system_model_source == "pysam":
                if hasattr(self._system_model.StatePack, attr) or hasattr(self._system_model.StateCell, attr):
                    getattr(self.outputs, attr)[time_step] = self.value(attr)
                elif attr == 'gen':
                    getattr(self.outputs, attr)[time_step] = self.value('P')
            else:
                if hasattr(self._system_model.state, attr):
                    getattr(self.outputs, attr)[time_step] = self.value(attr)
                    if attr == 'n_cycles' and self.dispatch.options.include_lifecycle_count:
                        getattr(self.outputs, attr)[time_step] = math.floor(self.dispatch.lifecycles[0])
                elif attr == 'gen':
                    getattr(self.outputs, attr)[time_step] = self.value('P')

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
            if (len(self._financial_model.value('batt_replacement_schedule_percent')) != project_life):
                raise ValueError(f"Error in Battery model: `batt_replacement_schedule_percent` should be length of project_life {project_life} but is instead {len(self._financial_model.value('batt_replacement_schedule_percent'))}")
            if (len(self._financial_model.value('batt_bank_replacement')) != project_life + 1) and (type(self._financial_model) is not CustomFinancialModel):
                if len(self._financial_model.value('batt_bank_replacement')) == project_life:
                    # likely an input mistake: add a zero for financial year 0 
                    self._financial_model.value('batt_bank_replacement', [0] + list(self._financial_model.value('batt_bank_replacement')))
                else:
                    raise ValueError(f"Error in Battery model: `batt_bank_replacement` should be length of project_life {project_life} but is instead {len(self._financial_model.value('batt_bank_replacement'))}")
    
    def set_overnight_capital_cost(self, energy_capital_cost, power_capital_cost):
        """Set overnight capital costs [$/kW].

        This method calculates and sets the overnight capital cost based on the given energy and power capital costs.
        
        Args:
            energy_capital_cost (float): The capital cost per unit of energy capacity [$/kWh].
            power_capital_cost (float): The capital cost per unit of power capacity [$/kW].
        
        Returns:
            None: This method does not return any value. The calculated overnight capital cost is stored internally.

        Note:
            The overnight capital cost is calculated using the formula:
            `overnight_capital_cost = (energy_capital_cost * hours) + power_capital_cost`
            where `hours` is the ratio of energy capacity to power capacity.

        Example:
            >>> set_overnight_capital_cost(1500, 500)
        """
        hours = self.system_capacity_kwh/self.system_capacity_kw
        self._overnight_capital_cost = (energy_capital_cost * hours) + power_capital_cost
    
    def calculate_total_installed_cost(self, energy_capital_cost: float, power_capital_cost: float) -> float:
        hours = self.system_capacity_kwh/self.system_capacity_kw
        if isinstance(self._financial_model, Singleowner.Singleowner):
            return self.system_capacity_kw * (power_capital_cost +
                                            (energy_capital_cost * hours))
        else:
            self.set_overnight_capital_cost(energy_capital_cost, power_capital_cost)
            total_installed_cost = self.system_capacity_kw * self._overnight_capital_cost
            return self._financial_model.value("total_installed_cost", total_installed_cost)
    
    def simulate_financials(
        self,
        interconnect_kw: float,
        project_life: int, 
        cap_cred_avail_storage: bool = True
    ):
        """
        Sets-up and simulates financial model for the battery

        Args:
            interconnect_kw: Interconnection limit [kW]
            project_life: Analysis period [years]
            cap_cred_avail_storage: Base capacity credit on available storage (True),
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

        if len(self.outputs.gen) == self.site.n_timesteps:
            single_year_gen = self.outputs.gen
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

    def calc_gen_max_feasible_kwh(self, interconnect_kw, use_avail_storage: bool = True) -> List[float]:
        """
        Calculates the maximum feasible capacity (generation profile) that could have occurred.

        Args:
            interconnect_kw: Interconnection limit [kW]
            use_avail_storage: Base capacity credit on available storage (True),
                otherwise use only dispatched generation (False)

        Returns:
            Maximum feasible capacity [kWh]
        """
        t_step = self.site.interval / 60                                                # hr
        df = pd.DataFrame()
        df['E_delivered'] = [max(0, x * t_step) for x in self.outputs.P]                # [kWh]
        df['SOC_perc'] = self.outputs.SOC                                               # [%]
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
            return self.outputs.gen
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
            return sum(self.outputs.gen)
        else:
            return 0
