import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.simple_battery_dispatch import SimpleBatteryDispatch


class NonConvexLinearVoltageBatteryDispatch(SimpleBatteryDispatch):
    """

    """
    # TODO: add a reference to original paper

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'LV_battery',
                 dispatch_options: dict = None,
                 use_exp_voltage_point: bool = False):
        u.load_definitions_from_strings(['amp_hour = amp * hour = Ah = amphour'])
        if dispatch_options is None:
            dispatch_options = {}
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         dispatch_options=dispatch_options)
        self.use_exp_voltage_point = use_exp_voltage_point

    def dispatch_block_rule(self, battery):
        # Parameters
        self._create_lv_battery_parameters(battery)
        # Variables
        self._create_lv_battery_variables(battery)
        # Base storage dispatch (parameters, variables, and constraints)
        super().dispatch_block_rule(battery)
        # Constraints
        self._create_lv_battery_constraints(battery)
        self._create_lv_battery_power_equation_constraints(battery)

    def _create_efficiency_parameters(self, battery):
        # Not defined in this formulation
        pass

    def _create_capacity_parameter(self, battery):
        battery.capacity = pyomo.Param(
            doc=self.block_set_name + " capacity [MAh]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MAh)

    def _create_lv_battery_parameters(self, battery):
        battery.voltage_slope = pyomo.Param(
            doc=self.block_set_name + " linear voltage model slope coefficient [V]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.V)
        battery.voltage_intercept = pyomo.Param(
            doc=self.block_set_name + " linear voltage model intercept coefficient [V]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.V)
        # TODO: Add this if wanted
        # self.alphaP = Param(None)  # [kW_DC]    Bi-directional intercept for charge
        # self.betaP = Param(None)  # [-]         Bi-directional slope for charge
        # self.alphaN = Param(None)  # [kW_DC]    Bi-directional intercept for discharge
        # self.betaN = Param(None)  # [-]         Bi-directional slope for discharge
        battery.average_current = pyomo.Param(
            doc="Typical cell current for both charge and discharge [A]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.A)
        battery.internal_resistance = pyomo.Param(
            doc=self.block_set_name + " internal resistance [Ohm]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.ohm)
        battery.minimum_charge_current = pyomo.Param(
            doc=self.block_set_name + " minimum charge current [MA]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MA)
        battery.maximum_charge_current = pyomo.Param(
            doc=self.block_set_name + " maximum charge current [MA]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MA)
        battery.minimum_discharge_current = pyomo.Param(
            doc=self.block_set_name + " minimum discharge current [MA]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MA)
        battery.maximum_discharge_current = pyomo.Param(
            doc=self.block_set_name + " maximum discharge current [MA]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MA)

    @staticmethod
    def _create_lv_battery_variables(battery):
        battery.charge_current = pyomo.Var(
            doc="Current into the battery [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)
        battery.discharge_current = pyomo.Var(
            doc="Current out of the battery [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)

    def _create_soc_inventory_constraint(self, storage):
        def soc_inventory_rule(m):
            # TODO: add alpha and beta terms
            return m.soc == (m.soc0 + m.time_duration * (m.charge_current - m.discharge_current) / m.capacity)
        # Storage State-of-charge balance
        storage.soc_inventory = pyomo.Constraint(
            doc=self.block_set_name + " state-of-charge inventory balance",
            rule=soc_inventory_rule)

    @staticmethod
    def _create_lv_battery_constraints(battery):
        # Charge current bounds
        battery.charge_current_lb = pyomo.Constraint(
            doc="Battery Charging current lower bound",
            expr=battery.charge_current >= battery.minimum_charge_current * battery.is_charging)
        battery.charge_current_ub = pyomo.Constraint(
            doc="Battery Charging current upper bound",
            expr=battery.charge_current <= battery.maximum_charge_current * battery.is_charging)
        battery.charge_current_ub_soc = pyomo.Constraint(
            doc="Battery Charging current upper bound state-of-charge dependence",
            expr=battery.charge_current <= battery.capacity * (1.0 - battery.soc0) / battery.time_duration)
        # Discharge current bounds
        battery.discharge_current_lb = pyomo.Constraint(
            doc="Battery Discharging current lower bound",
            expr=battery.discharge_current >= battery.minimum_discharge_current * battery.is_discharging)
        battery.discharge_current_ub = pyomo.Constraint(
            doc="Battery Discharging current upper bound",
            expr=battery.discharge_current <= battery.maximum_discharge_current * battery.is_discharging)
        battery.discharge_current_ub_soc = pyomo.Constraint(
            doc="Battery Discharging current upper bound state-of-charge dependence",
            expr=battery.discharge_current <= battery.maximum_discharge_current * battery.soc0)

    @staticmethod
    def _create_lv_battery_power_equation_constraints(battery):
        battery.charge_power_equation = pyomo.Constraint(
            doc="Battery charge power equation equal to the product of current and voltage",
            expr=battery.charge_power == battery.charge_current * (battery.voltage_slope * battery.soc0
                                                                   + (battery.voltage_intercept
                                                                      + battery.average_current
                                                                      * battery.internal_resistance)))
        battery.discharge_power_equation = pyomo.Constraint(
            doc="Battery discharge power equation equal to the product of current and voltage",
            expr=battery.discharge_power == battery.discharge_current * (battery.voltage_slope * battery.soc0
                                                                         + (battery.voltage_intercept
                                                                            - battery.average_current
                                                                            * battery.internal_resistance)))

    def _lifecycle_count_rule(self, m, i):
        # current accounting
        start = int(i * self.timesteps_per_day)
        end = int((i + 1) * self.timesteps_per_day)
        return self.model.lifecycles[i] == sum(self.blocks[t].time_duration
                                            * (0.8 * self.blocks[t].discharge_current
                                               - 0.8 * self.blocks[t].discharge_current * self.blocks[t].soc0)
                                            / self.blocks[t].capacity for t in range(start, end))

    def _set_control_mode(self):
        self._system_model.value("control_mode", 0.0)  # Current control
        self.control_variable = "input_current"

    def _set_model_specific_parameters(self):
        # Getting information from system_model
        nominal_voltage = self._system_model.value('nominal_voltage')
        nominal_energy = self._system_model.value('nominal_energy')
        Vnom_default = self._system_model.value('Vnom_default')
        C_rate = self._system_model.value('C_rate')
        resistance = self._system_model.value('resistance')

        Qfull = self._system_model.value('Qfull')
        Qnom = self._system_model.value('Qnom')
        Qexp = self._system_model.value('Qexp')

        Vfull = self._system_model.value('Vfull')
        Vnom = self._system_model.value('Vnom')
        Vexp = self._system_model.value('Vexp')

        # Using the Ceiling for both these -> Ceil(a/b) = -(-a//b)
        cells_in_series = - (- nominal_voltage // Vnom_default)
        strings_in_parallel = - (- nominal_energy * 1e3 // (Qfull * cells_in_series * Vnom_default))

        self.capacity = Qfull * strings_in_parallel / 1e6  # [MAh]

        # Calculating linear approximation for Voltage as a function of state-of-charge
        soc_nom = (Qfull - Qnom) / Qfull
        if self.use_exp_voltage_point:
            # Using cell exp and nom voltage points
            #       Using this method makes the problem more difficult for the solver.
            #       TODO: This behavior is not fully understood and
            #        there could be a better way to create the linear approximation
            soc_exp = (Qfull - Qexp) / Qfull
            a = (Vexp - Vnom) / (soc_exp - soc_nom)
            b = Vexp - a * soc_exp
        else:
            # Using Cell full and nom voltage points
            a = (Vfull - Vnom) / (1.0 - soc_nom)
            b = Vfull - a

        self.voltage_slope = cells_in_series * a
        self.voltage_intercept = cells_in_series * b
        self.average_current = (Qfull * strings_in_parallel * C_rate / 2.)
        self.internal_resistance = resistance * cells_in_series / strings_in_parallel
        # TODO: These parameters might need updating
        self.minimum_charge_current = 0.0
        self.maximum_charge_current = (Qfull * strings_in_parallel * C_rate) / 1e6
        self.minimum_discharge_current = 0.0
        self.maximum_discharge_current = (Qfull * strings_in_parallel * C_rate) / 1e6

    # Inputs
    @property
    def voltage_slope(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].voltage_slope.value

    @voltage_slope.setter
    def voltage_slope(self, voltage_slope: float):
        for t in self.blocks.index_set():
            self.blocks[t].voltage_slope = round(voltage_slope, self.round_digits)

    @property
    def voltage_intercept(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].voltage_intercept.value

    @voltage_intercept.setter
    def voltage_intercept(self, voltage_intercept: float):
        for t in self.blocks.index_set():
            self.blocks[t].voltage_intercept = round(voltage_intercept, self.round_digits)

    # # TODO: Add this if wanted
    # # self.alphaP = Param(None)  # [kW_DC]    Bi-directional intercept for charge
    # # self.betaP = Param(None)  # [-]         Bi-directional slope for charge
    # # self.alphaN = Param(None)  # [kW_DC]    Bi-directional intercept for discharge
    # # self.betaN = Param(None)  # [-]         Bi-directional slope for discharge

    @property
    def average_current(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].average_current.value

    @average_current.setter
    def average_current(self, average_current: float):
        for t in self.blocks.index_set():
            self.blocks[t].average_current = round(average_current, self.round_digits)

    @property
    def internal_resistance(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].internal_resistance.value

    @internal_resistance.setter
    def internal_resistance(self, internal_resistance: float):
        for t in self.blocks.index_set():
            self.blocks[t].internal_resistance = round(internal_resistance, self.round_digits)

    @property
    def minimum_charge_current(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_charge_current.value

    @minimum_charge_current.setter
    def minimum_charge_current(self, minimum_charge_current: float):
        for t in self.blocks.index_set():
            self.blocks[t].minimum_charge_current = round(minimum_charge_current, self.round_digits)

    @property
    def maximum_charge_current(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_charge_current.value

    @maximum_charge_current.setter
    def maximum_charge_current(self, maximum_charge_current: float):
        for t in self.blocks.index_set():
            self.blocks[t].maximum_charge_current = round(maximum_charge_current, self.round_digits)

    @property
    def minimum_discharge_current(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_discharge_current.value

    @minimum_discharge_current.setter
    def minimum_discharge_current(self, minimum_discharge_current: float):
        for t in self.blocks.index_set():
            self.blocks[t].minimum_discharge_current = round(minimum_discharge_current, self.round_digits)

    @property
    def maximum_discharge_current(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_discharge_current.value

    @maximum_discharge_current.setter
    def maximum_discharge_current(self, maximum_discharge_current: float):
        for t in self.blocks.index_set():
            self.blocks[t].maximum_discharge_current = round(maximum_discharge_current, self.round_digits)

    # Outputs
    @property
    def charge_current(self) -> list:
        return [self.blocks[t].charge_current.value for t in self.blocks.index_set()]

    @property
    def discharge_current(self) -> list:
        return [self.blocks[t].discharge_current.value for t in self.blocks.index_set()]

    @property
    def current(self) -> list:
        return [self.blocks[t].discharge_current.value - self.blocks[t].charge_current.value
                for t in self.blocks.index_set()]
