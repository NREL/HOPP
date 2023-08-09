import pyomo.environ as pyomo
from pyomo.environ import units as u

import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner

from hopp.dispatch.power_storage.power_storage_dispatch import PowerStorageDispatch


class SimpleBatteryDispatch(PowerStorageDispatch):
    _system_model: BatteryModel.BatteryStateful
    _financial_model: Singleowner.Singleowner
    """

    """

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model: BatteryModel.BatteryStateful,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'battery',
                 include_lifecycle_count: bool = True):
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name,
                         include_lifecycle_count=include_lifecycle_count)

    def initialize_parameters(self):
        if self.include_lifecycle_count:
            self.lifecycle_cost = self.lifecycle_cost_per_kWh_cycle * self._system_model.value('nominal_energy')

        self.cost_per_charge = 0.75  # [$/MWh]
        self.cost_per_discharge = 0.75  # [$/MWh]
        self.minimum_power = 0.0
        # FIXME: Change C_rate call to user set system_capacity_kw
        # self.maximum_power = self._system_model.value('nominal_energy') * self._system_model.value('C_rate') / 1e3
        self.maximum_power = self._financial_model.value("system_capacity") / 1e3
        self.minimum_soc = self._system_model.value('minimum_SOC')
        self.maximum_soc = self._system_model.value('maximum_SOC')
        self.initial_soc = self._system_model.value('initial_SOC')

        self._set_control_mode()
        self._set_model_specific_parameters()

    def _set_control_mode(self):
        self._system_model.value("control_mode", 1.0)  # Power control
        self._system_model.value("input_power", 0.)
        self.control_variable = "input_power"

    def _set_model_specific_parameters(self):
        self.round_trip_efficiency = 88.0  # Including converter efficiency
        self.capacity = self._system_model.value('nominal_energy') / 1e3  # [MWh]

    def update_time_series_parameters(self, start_time: int):
        # TODO: provide more control
        self.time_duration = [1.0] * len(self.blocks.index_set())

    def update_dispatch_initial_soc(self, initial_soc: float = None):
        if initial_soc is not None:
            self._system_model.value("initial_SOC", initial_soc)
            self._system_model.setup()  # TODO: Do I need to re-setup stateful battery?
        self.initial_soc = self._system_model.value('SOC')
