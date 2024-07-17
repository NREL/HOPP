from typing import Optional, List, Dict

import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch import (
    SimpleBatteryDispatch,
)


class SimpleBatteryDispatchHeuristic(SimpleBatteryDispatch):
    """Fixes battery dispatch operations based on user input.

    Currently, enforces available generation and grid limit assuming no battery charging from grid.

    """

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model: BatteryModel.BatteryStateful,
        financial_model: Singleowner.Singleowner,
        fixed_dispatch: Optional[List] = None,
        block_set_name: str = "heuristic_battery",
        dispatch_options: Optional[Dict] = None,
    ):
        """Initialize SimpleBatteryDispatchHeuristic.

        Args:
            pyomo_model (pyomo.ConcreteModel): Pyomo concrete model.
            index_set (pyomo.Set): Indexed set.
            system_model (BatteryModel.BatteryStateful): Battery system model.
            financial_model (Singleowner.Singleowner): Financial model.
            fixed_dispatch (Optional[List], optional): List of normalized values [-1, 1] (Charging (-), Discharging (+)). Defaults to None.
            block_set_name (str, optional): Name of block set. Defaults to 'heuristic_battery'.
            dispatch_options (dict, optional): Dispatch options. Defaults to None.

        """
        if dispatch_options is None:
            dispatch_options = {}
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
            dispatch_options=dispatch_options,
        )

        self.max_charge_fraction = list([0.0] * len(self.blocks.index_set()))
        self.max_discharge_fraction = list([0.0] * len(self.blocks.index_set()))
        self.user_fixed_dispatch = list([0.0] * len(self.blocks.index_set()))
        # TODO: should I enforce either a day schedule or a year schedule year and save it as user input.
        #  Additionally, Should I drop it as input in the init function?
        if fixed_dispatch is not None:
            self.user_fixed_dispatch = fixed_dispatch

        self._fixed_dispatch = list([0.0] * len(self.blocks.index_set()))

    def set_fixed_dispatch(self, gen: list, grid_limit: list):
        """Sets charge and discharge power of battery dispatch using fixed_dispatch attribute and enforces available
        generation and grid limits.

        Args:
            gen (list): Generation blocks.
            grid_limit (list): Grid capacity.

        Raises:
            ValueError: If gen or grid_limit length does not match fixed_dispatch length.

        """
        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._heuristic_method(gen)
        self._fix_dispatch_model_variables()

    def check_gen_grid_limit(self, gen: list, grid_limit: list):
        """Checks if generation and grid limit lengths match fixed_dispatch length.

        Args:
            gen (list): Generation blocks.
            grid_limit (list): Grid capacity.

        Raises:
            ValueError: If gen or grid_limit length does not match fixed_dispatch length.

        """
        if len(gen) != len(self.fixed_dispatch):
            raise ValueError("gen must be the same length as fixed_dispatch.")
        elif len(grid_limit) != len(self.fixed_dispatch):
            raise ValueError("grid_limit must be the same length as fixed_dispatch.")

    def _set_power_fraction_limits(self, gen: list, grid_limit: list):
        """Set battery charge and discharge power fraction limits based on
        available generation and grid capacity, respectively.

        Args:
            gen (list): Generation blocks.
            grid_limit (list): Grid capacity.

        NOTE: This method assumes that battery cannot be charged by the grid.

        """
        for t in self.blocks.index_set():
            self.max_charge_fraction[t] = self.enforce_power_fraction_simple_bounds(
                gen[t] / self.maximum_power
            )
            self.max_discharge_fraction[t] = self.enforce_power_fraction_simple_bounds(
                (grid_limit[t] - gen[t]) / self.maximum_power
            )

    @staticmethod
    def enforce_power_fraction_simple_bounds(power_fraction: float) -> float:
        """Enforces simple bounds (0, .9) for battery power fractions.

        Args:
            power_fraction (float): Power fraction from heuristic method.

        Returns:
            power_fraction (float): Bounded power fraction.

        """
        if power_fraction > 0.9:
            power_fraction = 0.9
        elif power_fraction < 0.0:
            power_fraction = 0.0
        return power_fraction

    def update_soc(self, power_fraction: float, soc0: float) -> float:
        """Updates SOC based on power fraction threshold (0.1).

        Args:
            power_fraction (float): Power fraction from heuristic method. Below threshold
                is charging, above is discharging.
            soc0 (float): Initial SOC.

        Returns:
            soc (float): Updated SOC.
            
        """
        if power_fraction > 0.0:
            discharge_power = power_fraction * self.maximum_power
            soc = (
                soc0
                - self.time_duration[0]
                * (1 / (self.discharge_efficiency / 100.0) * discharge_power)
                / self.capacity
            )
        elif power_fraction < 0.0:
            charge_power = -power_fraction * self.maximum_power
            soc = (
                soc0
                + self.time_duration[0]
                * (self.charge_efficiency / 100.0 * charge_power)
                / self.capacity
            )
        else:
            soc = soc0

        min_soc = self._system_model.value("minimum_SOC") / 100
        max_soc = self._system_model.value("maximum_SOC") / 100

        soc = max(min_soc, min(max_soc, soc))

        return soc

    def _heuristic_method(self, _):
        """Executes specific heuristic method to fix battery dispatch."""
        self._enforce_power_fraction_limits()

    def _enforce_power_fraction_limits(self):
        """Enforces battery power fraction limits and sets _fixed_dispatch attribute."""
        for t in self.blocks.index_set():
            fd = self.user_fixed_dispatch[t]
            if fd > 0.0:  # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if -fd > self.max_charge_fraction[t]:
                    fd = -self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd

    def _fix_dispatch_model_variables(self):
        """Fixes dispatch model variables based on the fixed dispatch values."""
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
                self.blocks[t].charge_power.fix(-dispatch_factor * self.maximum_power)

    @property
    def fixed_dispatch(self) -> list:
        """list: List of fixed dispatch."""
        return self._fixed_dispatch

    @property
    def user_fixed_dispatch(self) -> list:
        """list: List of user fixed dispatch."""
        return self._user_fixed_dispatch

    @user_fixed_dispatch.setter
    def user_fixed_dispatch(self, fixed_dispatch: list):
        # TODO: Annual dispatch array...
        if len(fixed_dispatch) != len(self.blocks.index_set()):
            raise ValueError(
                "fixed_dispatch must be the same length as dispatch index set."
            )
        elif max(fixed_dispatch) > 1.0 or min(fixed_dispatch) < -1.0:
            raise ValueError(
                "fixed_dispatch must be normalized values between -1 and 1."
            )
        else:
            self._user_fixed_dispatch = fixed_dispatch
