from pyomo.environ import ConcreteModel, Set

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


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

