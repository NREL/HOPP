from typing import Optional, List

import pyomo.environ as pyomo
from pyomo.environ import units as u
import PySAM.BatteryStateful as PySAMBatteryModel
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch_heuristic import (
    SimpleBatteryDispatchHeuristic,
)


class HeuristicLoadFollowingDispatch(SimpleBatteryDispatchHeuristic):
    """Operates the battery based on heuristic rules to meet the demand profile based power available from power generation profiles and
        power demand profile.

    Currently, enforces available generation and grid limit assuming no battery charging from grid
    
    """

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model: PySAMBatteryModel.BatteryStateful,
        financial_model: Singleowner.Singleowner,
        fixed_dispatch: Optional[List] = None,
        block_set_name: str = "heuristic_load_following_battery",
        dispatch_options: Optional[dict] = None,
    ):
        """Initialize HeuristicLoadFollowingDispatch.

        Args:
            pyomo_model (pyomo.ConcreteModel): Pyomo concrete model.
            index_set (pyomo.Set): Indexed set.
            system_model (PySAMBatteryModel.BatteryStateful): System model.
            financial_model (Singleowner.Singleowner): Financial model.
            fixed_dispatch (Optional[List], optional): List of normalized values [-1, 1] (Charging (-), Discharging (+)). Defaults to None.
            block_set_name (str, optional): Name of the block set. Defaults to 'heuristic_load_following_battery'.
            dispatch_options (Optional[dict], optional): Dispatch options. Defaults to None.

        """
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            fixed_dispatch,
            block_set_name,
            dispatch_options,
        )

    def set_fixed_dispatch(self, gen: list, grid_limit: list, goal_power: list):
        """Sets charge and discharge power of battery dispatch using fixed_dispatch attribute
            and enforces available generation and grid limits.

        Args:
            gen (list): List of power generation.
            grid_limit (list): List of grid limits.
            goal_power (list): List of goal power.

        """

        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._heuristic_method(gen, goal_power)
        self._fix_dispatch_model_variables()

    def _heuristic_method(self, gen, goal_power):
        """Enforces battery power fraction limits and sets _fixed_dispatch attribute.
        Sets the _fixed_dispatch based on goal_power and gen (power generation profile).

        Args:
            gen: Power generation profile.
            goal_power: Goal power.

        """
        for t in self.blocks.index_set():
            fd = (goal_power[t] - gen[t]) / self.maximum_power
            if fd > 0.0:  # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if -fd > self.max_charge_fraction[t]:
                    fd = -self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd
