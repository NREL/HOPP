from typing import Tuple
import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.one_cycle_battery_dispatch_heuristic import OneCycleBatteryDispatchHeuristic


class OneCycleBaseloadBatteryDispatchHeuristic(OneCycleBatteryDispatchHeuristic):
    """

    fixed_dispatch: list of normalized values [-1, 1] (Charging (-), Discharging (+))
    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'one_cycle_baseload_heuristic_battery',
                 include_lifecycle_count: bool = False):
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         include_lifecycle_count=False)
        self.baseload_difference = list([0.0] * len(self.blocks.index_set()))

    def _heuristic_method(self, gen):
        """This sets battery dispatch using a 1 cycle per day assumption.
           This battery is based on meeting a firm power goal

        Method:
         1. Sort the differences between the firm power and the power generation profile
         2. Determine the duration required to fully discharge and charge the battery
         3. Set discharge and charge operations based on sorted power differences
         3. Check SOC feasibility
         4. If infeasible, find infeasibility, shift operation to the next sorted power differences periods
         5. Repeat step 4 until SOC feasible
                NOTE: If operation is tried on half of time periods, then operation defaults to 'do nothing'
        """
        if sum(self.baseload_difference) == 0.0 and max(self.baseload_difference) == 0.0:
            raise ValueError("baseload differences must be set before calling heuristic method.")

        discharge_time, charge_time = self._get_duration_battery_full_cycle()
        fixed_dispatch = [0.0] * len(self.baseload_difference)
        diff_and_gen = zip(range(0, len(self.baseload_difference)), self.baseload_difference, gen)
        sorted_diff = sorted(diff_and_gen, key=lambda i: i[2], reverse=True)
        sorted_diff = sorted(sorted_diff, key=lambda i: i[1])

        # Set initial fixed dispatch
        fixed_dispatch, next_charge_idx = self._charge_battery(charge_time, 0,
                                                               sorted_diff,
                                                               fixed_dispatch)

        fixed_dispatch, next_discharge_idx = self._discharge_battery(discharge_time, 0,
                                                                     sorted_diff,
                                                                     fixed_dispatch)

        # test feasibility and find infeasibility
        
        feasible = self.test_soc_feasibility(fixed_dispatch)
        while not feasible[0]:
            # TODO: Improve algorithm
            idx_infeasible = feasible[1]
            infeasible_value = fixed_dispatch[idx_infeasible]
            if infeasible_value > 0:  # Discharging
                discharge_remaining = fixed_dispatch[idx_infeasible] * self.time_duration[idx_infeasible]
                fixed_dispatch[idx_infeasible] = 0
                if next_discharge_idx < len(sorted_diff)/2:
                    fixed_dispatch, next_discharge_idx = self._discharge_battery(discharge_remaining,
                                                                                 next_discharge_idx,
                                                                                 sorted_diff,
                                                                                 fixed_dispatch)
            elif infeasible_value < 0:    # Charging
                charge_remaining = -fixed_dispatch[idx_infeasible] * self.time_duration[idx_infeasible]
                fixed_dispatch[idx_infeasible] = 0
                if next_charge_idx < len(sorted_diff)/2:  # TODO: maybe too restrictive
                    fixed_dispatch, next_charge_idx = self._charge_battery(charge_remaining,
                                                                           next_charge_idx,
                                                                           sorted_diff,
                                                                           fixed_dispatch)
            feasible = self.test_soc_feasibility(fixed_dispatch)

        self._fixed_dispatch = fixed_dispatch

    @property
    def baseload_difference(self) -> list:
        return self._baseload_difference

    @baseload_difference.setter
    def baseload_difference(self, baseload_difference: list):
        if len(baseload_difference) != len(self.blocks.index_set()):
            raise ValueError("prices must be the same length as dispatch index set.")
        self._baseload_difference = baseload_difference
