import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.simple_battery_dispatch import SimpleBatteryDispatch


class SimpleBatteryDispatchHeuristic(SimpleBatteryDispatch):
    """

    fixed_dispatch: list of normalized values [-1, 1] (Charging (-), Discharging (+))
    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 fixed_dispatch: list = None,
                 block_set_name: str = 'heuristic_battery',
                 include_lifecycle_count: bool = False):
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         include_lifecycle_count=False)

        self.fixed_dispatch = list()
        if fixed_dispatch is not None:
            self.set_fixed_dispatch(fixed_dispatch)

    def set_fixed_dispatch(self, fixed_dispatch):
        if len(fixed_dispatch) != len(self.blocks.index_set()):
            raise ValueError("fixed_dispatch must be the same length as dispatch index set.")
            # TODO: this seems restrictive...
        elif max(fixed_dispatch) > 1.0 or min(fixed_dispatch) < -1.0:
            raise ValueError("fixed_dispatch must be normalized values between -1 and 1.")
        else:
            self.fixed_dispatch = fixed_dispatch

    def fix_dispatch(self, gen, grid_limit):
        if len(gen) != len(self.fixed_dispatch):
            raise ValueError("gen must be the same length as fixed_dispatch.")
        elif len(grid_limit) != len(self.fixed_dispatch):
            raise ValueError("grid_limit must be the same length as fixed_dispatch.")

        for t in self.blocks.index_set():
            dispatch_factor = self.fixed_dispatch[t]
            self.blocks[t].soc.fix(0.0)
            if dispatch_factor == 0.0:
                # Do nothing
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(0.0)
            elif dispatch_factor > 0.0:
                # Discharging
                self.blocks[t].charge_power.fix(0.0)
                power = dispatch_factor * self.maximum_power
                if power + gen[t] <= grid_limit[t]:
                    # discharge and generation below grid limit
                    self.blocks[t].discharge_power.fix(power)
                else:
                    # set discharge to available transmission
                    self.blocks[t].discharge_power.fix(grid_limit[t] - gen[t])
            elif dispatch_factor < 0.0:
                # Charging
                self.blocks[t].discharge_power.fix(0.0)
                power = - dispatch_factor * self.maximum_power
                if power <= gen[t]:
                    # available generation can support desired charging
                    self.blocks[t].charge_power.fix(power)
                else:
                    # do the best you can
                    self.blocks[t].charge_power.fix(gen[t])
