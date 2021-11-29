from pyomo.environ import ConcreteModel, Set

import PySAM.Windpower as Windpower
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch


class WindDispatch(PowerSourceDispatch):
    _system_model: Windpower.Windpower
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: Windpower.Windpower,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'wind'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

