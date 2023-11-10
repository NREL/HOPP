from pyomo.environ import ConcreteModel, Set

import PySAM.MhkWave as MhkWave

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch



class WaveDispatch(PowerSourceDispatch):
    _system_model: MhkWave.MhkWave
    _financial_model: FinancialModelType
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: MhkWave.MhkWave,
                 financial_model: FinancialModelType,
                 block_set_name: str = 'wave'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

