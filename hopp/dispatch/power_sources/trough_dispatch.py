from pyomo.environ import ConcreteModel, Set
import datetime

import PySAM.Singleowner as Singleowner

from hopp.dispatch.power_sources.csp_dispatch import CspDispatch


class TroughDispatch(CspDispatch):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: None,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'trough'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

    def update_initial_conditions(self):
        super().update_initial_conditions()
        self.initial_receiver_startup_inventory = 0.0  # FIXME:
        if self.is_field_starting_initial:
            print('Warning: Solar field is starting at the initial time step of the dispatch horizon, but initial '
                  'startup energy inventory is assumed to be zero. This may result in persistent receiver start-up')

