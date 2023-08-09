import datetime
from pyomo.environ import ConcreteModel, Set

import PySAM.Singleowner as Singleowner

from hopp.dispatch.power_sources.csp_dispatch import CspDispatch


class TowerDispatch(CspDispatch):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: None,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'tower'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

    def update_initial_conditions(self):
        super().update_initial_conditions()
        csp = self._system_model
        # Note, SS receiver model in ssc assumes full available power is used for startup
        # (even if, time requirement is binding)
        rec_accumulate_time = max(0.0, csp.value('rec_su_delay') - csp.plant_state['rec_startup_time_remain_init'])
        rec_accumulate_energy = max(0.0, self.receiver_required_startup_energy
                                    - csp.plant_state['rec_startup_energy_remain_init'] / 1e6)
        self.initial_receiver_startup_inventory = min(rec_accumulate_energy,
                                                      rec_accumulate_time * self.allowable_receiver_startup_power)
        if self.initial_receiver_startup_inventory > (1.0 - 1.e-6) * self.receiver_required_startup_energy:
            self.initial_receiver_startup_inventory = self.receiver_required_startup_energy

