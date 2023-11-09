from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Set

import PySAM.Windpower as Windpower

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import PowerSourceDispatch

if TYPE_CHECKING:
    from hopp.simulation.technologies.wind.floris import Floris

class WindDispatch(PowerSourceDispatch):
    _system_model: Union[Windpower.Windpower,"Floris"]
    _financial_model: FinancialModelType
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: Union[Windpower.Windpower,"Floris"],
                 financial_model: FinancialModelType,
                 block_set_name: str = 'wind'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

