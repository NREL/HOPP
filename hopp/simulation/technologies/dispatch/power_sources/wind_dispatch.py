from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

import PySAM.Windpower as Windpower

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import (
    PowerSourceDispatch
)

if TYPE_CHECKING:
    from hopp.simulation.technologies.wind.floris import Floris


class WindDispatch(PowerSourceDispatch):
    wind_obj: Union[Expression, float]
    _system_model: Union[Windpower.Windpower,"Floris"]
    _financial_model: FinancialModelType
    """

    """
    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: Union[Windpower.Windpower,"Floris"],
        financial_model: FinancialModelType,
        block_set_name: str = 'wind',
    ):
        super().__init__(
            pyomo_model,
            indexed_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
        )

    def max_gross_profit_objective(self, blocks):
        self.obj = Expression(
                expr=sum(
                    - (1/blocks[t].time_weighting_factor)
                    * self.blocks[t].time_duration
                    * self.blocks[t].cost_per_generation
                    * blocks[t].wind_generation
                    for t in blocks.index_set()
                )
            )

    def min_operating_cost_objective(self, blocks):
        self.obj = sum(
            blocks[t].time_weighting_factor 
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * blocks[t].wind_generation
            for t in blocks.index_set()
        )

    def _create_variables(self, hybrid):
        hybrid.wind_generation = Var(
            doc="Power generation of wind turbines [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.wind_generation, 0

    def _create_port(self, hybrid):
        hybrid.wind_port = Port(initialize={'generation': hybrid.wind_generation})
        return hybrid.wind_port
