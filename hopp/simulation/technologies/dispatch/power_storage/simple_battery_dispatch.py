import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as PySAMBatteryModel

import hopp.simulation.technologies.ldes.ldes_system_model as ldes
from hopp.simulation.technologies.dispatch.power_storage.power_storage_dispatch import (
    PowerStorageDispatch,
)
from hopp.simulation.technologies.financial import FinancialModelType

from typing import Union

class SimpleBatteryDispatch(PowerStorageDispatch):
    """A dispatch class for simple battery operations."""

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model: PySAMBatteryModel.BatteryStateful,
        financial_model: FinancialModelType,
        block_set_name: str,
        dispatch_options,
    ):
        """Initializes SimpleBatteryDispatch.

        Args:
            pyomo_model (pyomo.ConcreteModel): The Pyomo model instance.
            index_set (pyomo.Set): The Pyomo index set.
            system_model (PySAMBatteryModel.BatteryStateful): The battery stateful model.
            financial_model (FinancialModelType): The financial model type.
            block_set_name (str): Name of the block set.
            dispatch_options: Dispatch options.
            
        """
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
            dispatch_options=dispatch_options,
        )

    def initialize_parameters(self):
        """Initializes parameters."""
        if self.options.include_lifecycle_count:
            self.lifecycle_cost = (
                self.options.lifecycle_cost_per_kWh_cycle
                * self._system_model.value("nominal_energy")
            )

        self.cost_per_charge = self._financial_model.value("om_batt_variable_cost")[
            0
        ]  # [$/MWh]
        self.cost_per_discharge = self._financial_model.value("om_batt_variable_cost")[
            0
        ]  # [$/MWh]
        self.minimum_power = 0.0
        # FIXME: Change C_rate call to user set system_capacity_kw
        # self.maximum_power = self._system_model.value('nominal_energy') * self._system_model.value('C_rate') / 1e3
        self.maximum_power = self._financial_model.value("system_capacity") / 1e3
        self.minimum_soc = self._system_model.value("minimum_SOC")
        self.maximum_soc = self._system_model.value("maximum_SOC")
        self.initial_soc = self._system_model.value("initial_SOC")

        self._set_control_mode()
        self._set_model_specific_parameters()

    def _set_control_mode(self):
        """Sets control mode."""
        if isinstance(self._system_model, Union[PySAMBatteryModel.BatteryStateful, ldes.LDES]):
            self._system_model.value("control_mode", 1.0)  # Power control
            self._system_model.value("input_power", 0.0)
            self.control_variable = "input_power"

    def _set_model_specific_parameters(self, round_trip_efficiency=88.0):
        """Sets model-specific parameters.

        Args:
            round_trip_efficiency (float, optional): The round-trip efficiency including converter efficiency.
                Defaults to 88.0, which includes converter efficiency.

        """
        self.round_trip_efficiency = (
            round_trip_efficiency  # Including converter efficiency
        )
        self.capacity = self._system_model.value("nominal_energy") / 1e3  # [MWh]

    def update_time_series_parameters(self, start_time: int):
        """Updates time series parameters.

        Args:
            start_time (int): The start time.

        """
        # TODO: provide more control
        self.time_duration = [1.0] * len(self.blocks.index_set())

    def update_dispatch_initial_soc(self, initial_soc: float = None):
        """Updates dispatch initial state of charge (SOC).

        Args:
            initial_soc (float, optional): Initial state of charge. Defaults to None.

        """
        if initial_soc is not None:
            self._system_model.value("initial_SOC", initial_soc)
            self._system_model.setup()  # TODO: Do I need to re-setup stateful battery?
        self.initial_soc = self._system_model.value("SOC")
