from pyomo.environ import ConcreteModel, Set

import PySAM.MhkWave as MhkWave
import PySAM.Singleowner as Singleowner

from hopp.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch


class WaveDispatch(PowerSourceDispatch):
    _system_model: MhkWave.MhkWave
    _financial_model: None #Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: MhkWave.MhkWave,
                 financial_model: None,             #Singleowner.Singleowner
                 block_set_name: str = 'wind'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

