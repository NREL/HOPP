import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.linear_voltage_nonconvex_battery_dispatch import NonConvexLinearVoltageBatteryDispatch


class ConvexLinearVoltageBatteryDispatch(NonConvexLinearVoltageBatteryDispatch):
    """
    
    """
    # TODO: add a reference to original paper

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'convex_LV_battery',
                 dispatch_options: dict = None,
                 use_exp_voltage_point: bool = False):
        if dispatch_options is None:
            dispatch_options = {}
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         dispatch_options=dispatch_options,
                         use_exp_voltage_point=use_exp_voltage_point)

    def dispatch_block_rule(self, battery):
        # Additional formulation
        # Variables
        self._create_lv_battery_auxiliary_variables(battery)
        super().dispatch_block_rule(battery)

    @staticmethod
    def _create_lv_battery_auxiliary_variables(battery):
        # Auxiliary Variables
        battery.aux_charge_current_soc = pyomo.Var(
            doc="Auxiliary bi-linear term equal to the product of charge current and previous state-of-charge [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)  # = charge_current[t] * soc[t-1]
        battery.aux_charge_current_is_charging = pyomo.Var(
            doc="Auxiliary bi-linear term equal to the product of charge current and charging binary [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)  # = charge_current[t] * is_charging[t]
        battery.aux_discharge_current_soc = pyomo.Var(
            doc="Auxiliary bi-linear equal to the product of discharge current and previous state-of-charge [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)  # = discharge_current[t] * soc[t-1]
        battery.aux_discharge_current_is_discharging = pyomo.Var(
            doc="Auxiliary bi-linear term equal to the product of discharge current and discharging binary [MA]",
            domain=pyomo.NonNegativeReals,
            units=u.MA)  # = discharge_current[t] * is_discharging[t]

    @staticmethod
    def _create_lv_battery_power_equation_constraints(battery):
        battery.charge_power_equation = pyomo.Constraint(
            doc="Battery charge power equation equal to the product of current and voltage",
            expr=battery.charge_power == (battery.voltage_slope * battery.aux_charge_current_soc
                                          + (battery.voltage_intercept
                                             + battery.average_current * battery.internal_resistance
                                             ) * battery.aux_charge_current_is_charging))
        battery.discharge_power_equation = pyomo.Constraint(
            doc="Battery discharge power equation equal to the product of current and voltage",
            expr=battery.discharge_power == (battery.voltage_slope * battery.aux_discharge_current_soc
                                             + (battery.voltage_intercept
                                                - battery.average_current * battery.internal_resistance
                                                ) * battery.aux_discharge_current_is_discharging))
        # Auxiliary Variable bounds (binary*continuous exact linearization)
        # Charge current * charging binary
        battery.aux_charge_lb = pyomo.Constraint(
            doc="Charge current * charge binary lower bound",
            expr=battery.aux_charge_current_is_charging >= battery.minimum_charge_current * battery.is_charging)
        battery.aux_charge_ub = pyomo.Constraint(
            doc="Charge current * charge binary upper bound",
            expr=battery.aux_charge_current_is_charging <= battery.maximum_charge_current * battery.is_charging)
        battery.aux_charge_diff_lb = pyomo.Constraint(
            doc="Charge current and auxiliary difference lower bound",
            expr=(battery.charge_current - battery.aux_charge_current_is_charging
                  >= - battery.maximum_charge_current * (1 - battery.is_charging)))
        battery.aux_charge_diff_ub = pyomo.Constraint(
            doc="Charge current and auxiliary difference upper bound",
            expr=(battery.charge_current - battery.aux_charge_current_is_charging
                  <= battery.maximum_charge_current * (1 - battery.is_charging)))
        # Discharge current * discharging binary
        battery.aux_discharge_lb = pyomo.Constraint(
            doc="discharge current * discharge binary lower bound",
            expr=(battery.aux_discharge_current_is_discharging
                  >= battery.minimum_discharge_current * battery.is_discharging))
        battery.aux_discharge_ub = pyomo.Constraint(
            doc="discharge current * discharge binary upper bound",
            expr=(battery.aux_discharge_current_is_discharging
                  <= battery.maximum_discharge_current * battery.is_discharging))
        battery.aux_discharge_diff_lb = pyomo.Constraint(
            doc="discharge current and auxiliary difference lower bound",
            expr=(battery.discharge_current - battery.aux_discharge_current_is_discharging
                  >= - battery.maximum_discharge_current * (1 - battery.is_discharging)))
        battery.aux_discharge_diff_ub = pyomo.Constraint(
            doc="discharge current and auxiliary difference upper bound",
            expr=(battery.discharge_current - battery.aux_discharge_current_is_discharging
                  <= battery.maximum_discharge_current * (1 - battery.is_discharging)))
        # Auxiliary Variable bounds (continuous*continuous approx. linearization)
        # TODO: The error in these constraints should be quantified
        # TODO: scaling the problem to between [0,1] might help
        battery.aux_charge_soc_lower1 = pyomo.Constraint(
            doc="McCormick envelope underestimate 1",
            expr=battery.aux_charge_current_soc >= (battery.maximum_charge_current * battery.soc0
                                                    + battery.maximum_soc * battery.charge_current
                                                    - battery.maximum_soc * battery.maximum_charge_current))
        battery.aux_charge_soc_lower2 = pyomo.Constraint(
            doc="McCormick envelope underestimate 2",
            expr=battery.aux_charge_current_soc >= (battery.minimum_charge_current * battery.soc0
                                                    + battery.minimum_soc * battery.charge_current
                                                    - battery.minimum_soc * battery.minimum_charge_current))
        battery.aux_charge_soc_upper1 = pyomo.Constraint(
            doc="McCormick envelope overestimate 1",
            expr=battery.aux_charge_current_soc <= (battery.maximum_charge_current * battery.soc0
                                                    + battery.minimum_soc * battery.charge_current
                                                    - battery.minimum_soc * battery.maximum_charge_current))
        battery.aux_charge_soc_upper2 = pyomo.Constraint(
            doc="McCormick envelope overestimate 2",
            expr=battery.aux_charge_current_soc <= (battery.minimum_charge_current * battery.soc0
                                                    + battery.maximum_soc * battery.charge_current
                                                    - battery.maximum_soc * battery.minimum_charge_current))

        battery.aux_discharge_soc_lower1 = pyomo.Constraint(
            doc="McCormick envelope underestimate 1",
            expr=battery.aux_discharge_current_soc >= (battery.maximum_discharge_current * battery.soc0
                                                       + battery.maximum_soc * battery.discharge_current
                                                       - battery.maximum_soc * battery.maximum_discharge_current))
        battery.aux_discharge_soc_lower2 = pyomo.Constraint(
            doc="McCormick envelope underestimate 2",
            expr=battery.aux_discharge_current_soc >= (battery.minimum_discharge_current * battery.soc0
                                                       + battery.minimum_soc * battery.discharge_current
                                                       - battery.minimum_soc * battery.minimum_discharge_current))
        battery.aux_discharge_soc_upper1 = pyomo.Constraint(
            doc="McCormick envelope overestimate 1",
            expr=battery.aux_discharge_current_soc <= (battery.maximum_discharge_current * battery.soc0
                                                       + battery.minimum_soc * battery.discharge_current
                                                       - battery.minimum_soc * battery.maximum_discharge_current))
        battery.aux_discharge_soc_upper2 = pyomo.Constraint(
            doc="McCormick envelope overestimate 2",
            expr=battery.aux_discharge_current_soc <= (battery.minimum_discharge_current * battery.soc0
                                                       + battery.maximum_soc * battery.discharge_current
                                                       - battery.maximum_soc * battery.minimum_discharge_current))

    def _lifecycle_count_rule(self, m, i):
        # current accounting
        # TODO: Check for cheating -> there seems to be a lot of error
        start = int(i * self.timesteps_per_day)
        end = int((i + 1) * self.timesteps_per_day)
        return self.model.lifecycles[i] == sum(self.blocks[t].time_duration
                                            * (0.8 * self.blocks[t].discharge_current
                                               - 0.8 * self.blocks[t].aux_discharge_current_soc)
                                            / self.blocks[t].capacity for t in range(start, end))

    # Auxiliary Variables
    @property
    def aux_charge_current_soc(self) -> list:
        return [self.blocks[t].aux_charge_current_soc.value for t in self.blocks.index_set()]

    @property
    def real_charge_current_soc(self) -> list:
        return [self.blocks[t].charge_current.value * self.blocks[t].soc0.value for t in self.blocks.index_set()]

    @property
    def aux_charge_current_is_charging(self) -> list:
        return [self.blocks[t].aux_charge_current_is_charging.value for t in self.blocks.index_set()]

    @property
    def aux_discharge_current_soc(self) -> list:
        return [self.blocks[t].aux_discharge_current_soc.value for t in self.blocks.index_set()]

    @property
    def real_discharge_current_soc(self) -> list:
        return [self.blocks[t].discharge_current.value * self.blocks[t].soc0.value for t in self.blocks.index_set()]

    @property
    def aux_discharge_current_is_discharging(self) -> list:
        return [self.blocks[t].aux_discharge_current_is_discharging.value for t in self.blocks.index_set()]

