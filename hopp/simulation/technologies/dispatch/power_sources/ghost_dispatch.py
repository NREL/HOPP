from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

if TYPE_CHECKING:
    from hopp.simulation.technologies.ghost.ghost_plant import GhostSystem
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import (
    PowerSourceDispatch,
)


class GhostDispatch(PowerSourceDispatch):
    ghost_obj: Union[Expression, float]
    _system_model: "GhostSystem"
    _financial_model: FinancialModelType
    """Dispatch optimization model for ghost power source."""

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: "GhostSystem",
        financial_model: FinancialModelType,
        block_set_name: str = "ghost",
    ):
        """Initialize GhostDispatch.

        Args:
            pyomo_model (ConcreteModel): Pyomo concrete model.
            indexed_set (Set): Indexed set.
            system_model (GhostSystem): System model.
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
        """Ghost instance of maximum gross profit objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = Expression(
            expr=sum(
                -(1 / hybrid_blocks[t].time_weighting_factor)
                * self.blocks[t].time_duration
                * self.blocks[t].cost_per_generation
                * hybrid_blocks[t].ghost_generation
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """Ghost instance of minimum operating cost objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * hybrid_blocks[t].ghost_generation
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        """Create Ghost variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.ghost_generation = Var(
            doc="Power generation of ghost devices [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.ghost_generation, 0

    def _create_port(self, hybrid):
        """Create ghost port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: Ghost Port object.

        """
        hybrid.ghost_port = Port(initialize={"generation": hybrid.ghost_generation})
        return hybrid.ghost_port
