from typing import Union
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

import PySAM.MhkTidal as MhkTidal

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import (
    PowerSourceDispatch,
)


class TidalDispatch(PowerSourceDispatch):
    tidal_obj: Union[Expression, float]
    _system_model: MhkTidal.MhkTidal
    _financial_model: FinancialModelType
    """Dispatch optimization model for mhk tidal power source."""

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: MhkTidal.MhkTidal,
        financial_model: FinancialModelType,
        block_set_name: str = "tidal",
    ):
        """Initialize TidalDispatch.

        Args:
            pyomo_model (ConcreteModel): Pyomo concrete model.
            indexed_set (Set): Indexed set.
            system_model (MhkTidal.MhkTidal): System model.
            financial_model (FinancialModelType): Financial model.
            block_set_name (str): Name of the block set.
            
        """
        super().__init__(
            pyomo_model,
            indexed_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
        )

    def max_gross_profit_objective(self, hybrid_blocks):
        """MHK tidal instance of maximum gross profit objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = Expression(
            expr=sum(
                -(1 / hybrid_blocks[t].time_weighting_factor)
                * self.blocks[t].time_duration
                * self.blocks[t].cost_per_generation
                * hybrid_blocks[t].tidal_generation
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """MHK tidal instance of minimum operating cost objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * hybrid_blocks[t].tidal_generation
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        """Create MHK tidal variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.tidal_generation = Var(
            doc="Power generation of tidal devices [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.tidal_generation, 0

    def _create_port(self, hybrid):
        """Create mhk tidal port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: MHK tidal Port object.

        """
        hybrid.tidal_port = Port(initialize={"generation": hybrid.tidal_generation})
        return hybrid.tidal_port
