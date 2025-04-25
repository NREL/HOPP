from typing import Union, TYPE_CHECKING
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

if TYPE_CHECKING:
    from hopp.simulation.technologies.generic.generic_plant import GenericSystem
    from hopp.simulation.technologies.generic.generic_multi import GenericMultiSystem
from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import (
    PowerSourceDispatch,
)


class GenericDispatch(PowerSourceDispatch):
    """Dispatch optimization model for generic power source.
    Adapted from tidal_dispatch with minor changes.
    """
    
    generic_obj: Union[Expression, float]
    _system_model: Union["GenericSystem","GenericMultiSystem"]
    _financial_model: FinancialModelType

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: Union["GenericSystem","GenericMultiSystem"],
        financial_model: FinancialModelType,
        block_set_name: str = "generic",
    ):
        """Initialize GenericDispatch.

        Args:
            pyomo_model (ConcreteModel): Pyomo concrete model.
            indexed_set (Set): Indexed set.
            system_model (GenericSystem): System model.
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
        """Generic instance of maximum gross profit objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = Expression(
            expr=sum(
                -(1 / hybrid_blocks[t].time_weighting_factor)
                * self.blocks[t].time_duration
                * self.blocks[t].cost_per_generation
                * hybrid_blocks[t].generic_generation
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """Generic instance of minimum operating cost objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * hybrid_blocks[t].generic_generation
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        """Create Generic variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.generic_generation = Var(
            doc="Power generation of generic devices [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.generic_generation, 0

    def _create_port(self, hybrid):
        """Create generic port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: Generic Port object.

        """
        hybrid.generic_port = Port(initialize={"generation": hybrid.generic_generation})
        return hybrid.generic_port
