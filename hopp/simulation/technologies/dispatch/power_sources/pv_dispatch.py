from typing import Union
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

import PySAM.Pvsamv1 as Pvsam
import PySAM.Pvwattsv8 as Pvwatts

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources import PowerSourceDispatch


class PvDispatch(PowerSourceDispatch):
    pv_obj: Union[Expression, float]
    _system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv8]
    _financial_model: FinancialModelType
    """

    """
    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv8],
        financial_model: FinancialModelType,
        block_set_name: str = 'pv',
    ):
        super().__init__(
            pyomo_model,
            indexed_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
        )

    def update_time_series_parameters(self, start_time: int):
        super().update_time_series_parameters(start_time)

        # zero out any negative load
        self.available_generation = [max(0, i) for i in self.available_generation]

    def max_gross_profit_objective(self, blocks):
        self.obj = Expression(
                expr=sum(
                    - (1/blocks[t].time_weighting_factor)
                    * self.blocks[t].time_duration
                    * self.blocks[t].cost_per_generation
                    * blocks[t].pv_generation
                    for t in blocks.index_set()
                )
            )

    def min_operating_cost_objective(self, blocks):
        self.obj = sum(
            blocks[t].time_weighting_factor 
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * blocks[t].pv_generation
            for t in blocks.index_set()
        )

    def _create_variables(self, hybrid) -> Var:
        hybrid.pv_generation = Var(
            doc="Power generation of photovoltaics [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.pv_generation

    def _create_port(self, hybrid) -> Port:
        hybrid.pv_port = Port(initialize={'generation': hybrid.pv_generation})
        return hybrid.pv_port
