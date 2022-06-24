import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.simple_battery_dispatch_heuristic import SimpleBatteryDispatchHeuristic


class BatteryBaseloadDispatchHeuristic(SimpleBatteryDispatchHeuristic):
    """Fixes battery dispatch operations based power available from power generation profiles and 
        power demand profile.

    Currently, enforces available generation and grid limit assuming no battery charging from grid
    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 fixed_dispatch: list = None,
                 block_set_name: str = 'heuristic_baseload_battery',
                 include_lifecycle_count: bool = False):
        """

        :param fixed_dispatch: list of normalized values [-1, 1] (Charging (-), Discharging (+))
        """
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         fixed_dispatch,
                         block_set_name=block_set_name,
                         include_lifecycle_count=False)

    def set_fixed_dispatch(self, gen: list, grid_limit: list, goal_power: list):
        """Sets charge and discharge power of battery dispatch using fixed_dispatch attribute and enforces available
        generation and grid limits.

        """
        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._heuristic_method(gen, goal_power)
        self._fix_dispatch_model_variables()

    def _heuristic_method(self, gen, goal_power):
        """ Enforces battery power fraction limits and sets _fixed_dispatch attribute
            Sets the _fixed_dispatch based on goal_power and gen (power genration profile)
        """
        for t in self.blocks.index_set():
            fd = (goal_power[t] - gen[t]) / self.maximum_power
            if fd > 0.0:    # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if - fd > self.max_charge_fraction[t]:
                    fd = - self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd        
