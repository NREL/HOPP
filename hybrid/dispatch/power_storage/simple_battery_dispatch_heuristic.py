import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_storage.simple_battery_dispatch import SimpleBatteryDispatch


class SimpleBatteryDispatchHeuristic(SimpleBatteryDispatch):
    """Fixes battery dispatch operations based on user input.

    Currently, enforces available generation and grid limit assuming no battery charging from grid
    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 fixed_dispatch: list = None,
                 block_set_name: str = 'heuristic_battery',
                 dispatch_options: dict = None):
        """

        :param fixed_dispatch: list of normalized values [-1, 1] (Charging (-), Discharging (+))
        """
        if dispatch_options is None:
            dispatch_options = {}
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         dispatch_options=dispatch_options)

        self.max_charge_fraction = list([0.0]*len(self.blocks.index_set()))
        self.max_discharge_fraction = list([0.0]*len(self.blocks.index_set()))
        self.user_fixed_dispatch = list([0.0]*len(self.blocks.index_set()))
        # TODO: should I enforce either a day schedule or a year schedule year and save it as user input.
        #  Additionally, Should I drop it as input in the init function?
        if fixed_dispatch is not None:
            self.user_fixed_dispatch = fixed_dispatch

        self._fixed_dispatch = list([0.0] * len(self.blocks.index_set()))

    def set_fixed_dispatch(self, gen: list, grid_limit: list):
        """Sets charge and discharge power of battery dispatch using fixed_dispatch attribute and enforces available
        generation and grid limits.

        """
        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._heuristic_method(gen)
        self._fix_dispatch_model_variables()

    def check_gen_grid_limit(self, gen: list, grid_limit: list):
        if len(gen) != len(self.fixed_dispatch):
            raise ValueError("gen must be the same length as fixed_dispatch.")
        elif len(grid_limit) != len(self.fixed_dispatch):
            raise ValueError("grid_limit must be the same length as fixed_dispatch.")

    def _set_power_fraction_limits(self, gen: list, grid_limit: list):
        """Set battery charge and discharge power fraction limits based on available generation and grid capacity,
        respectively.

        NOTE: This method assumes that battery cannot be charged by the grid.
        """
        for t in self.blocks.index_set():
            self.max_charge_fraction[t] = self.enforce_power_fraction_simple_bounds(gen[t] / self.maximum_power)
            self.max_discharge_fraction[t] = self.enforce_power_fraction_simple_bounds((grid_limit[t] - gen[t])
                                                                                       / self.maximum_power)

    @staticmethod
    def enforce_power_fraction_simple_bounds(power_fraction) -> float:
        """ Enforces simple bounds (0,1) for battery power fractions."""
        if power_fraction > 1.0:
            power_fraction = 1.0
        elif power_fraction < 0.0:
            power_fraction = 0.0
        return power_fraction

    def update_soc(self, power_fraction, soc0) -> float:
        if power_fraction > 0.0:
            discharge_power = power_fraction * self.maximum_power
            soc = soc0 - self.time_duration[0] * (1/(self.discharge_efficiency/100.) * discharge_power) / self.capacity
        elif power_fraction < 0.0:
            charge_power = - power_fraction * self.maximum_power
            soc = soc0 + self.time_duration[0] * (self.charge_efficiency / 100. * charge_power) / self.capacity
        else:
            soc = soc0
        soc = max(0, min(1, soc))
        return soc

    def _heuristic_method(self, _):
        """ Does specific heuristic method to fix battery dispatch."""
        self._enforce_power_fraction_limits()

    def _enforce_power_fraction_limits(self):
        """ Enforces battery power fraction limits and sets _fixed_dispatch attribute"""
        for t in self.blocks.index_set():
            fd = self.user_fixed_dispatch[t]
            if fd > 0.0:    # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if - fd > self.max_charge_fraction[t]:
                    fd = - self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd

    def _fix_dispatch_model_variables(self):
        soc0 = self.model.initial_soc.value
        for t in self.blocks.index_set():
            dispatch_factor = self._fixed_dispatch[t]
            self.blocks[t].soc.fix(self.update_soc(dispatch_factor, soc0))
            soc0 = self.blocks[t].soc.value

            if dispatch_factor == 0.0:
                # Do nothing
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(0.0)
            elif dispatch_factor > 0.0:
                # Discharging
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(dispatch_factor * self.maximum_power)
            elif dispatch_factor < 0.0:
                # Charging
                self.blocks[t].discharge_power.fix(0.0)
                self.blocks[t].charge_power.fix(- dispatch_factor * self.maximum_power)

    @property
    def fixed_dispatch(self) -> list:
        return self._fixed_dispatch

    @property
    def user_fixed_dispatch(self) -> list:
        return self._user_fixed_dispatch

    @user_fixed_dispatch.setter
    def user_fixed_dispatch(self, fixed_dispatch: list):
        # TODO: Annual dispatch array...
        if len(fixed_dispatch) != len(self.blocks.index_set()):
            raise ValueError("fixed_dispatch must be the same length as dispatch index set.")
        elif max(fixed_dispatch) > 1.0 or min(fixed_dispatch) < -1.0:
            raise ValueError("fixed_dispatch must be normalized values between -1 and 1.")
        else:
            self._user_fixed_dispatch = fixed_dispatch

