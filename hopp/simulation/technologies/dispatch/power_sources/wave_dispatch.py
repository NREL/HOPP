from typing import Union
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

import PySAM.MhkWave as MhkWave

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.power_source_dispatch import (
    PowerSourceDispatch,
)


class WaveDispatch(PowerSourceDispatch):
    wave_obj: Union[Expression, float]
    _system_model: MhkWave.MhkWave
    _financial_model: FinancialModelType
    """Dispatch optimization model for mhk wave power source."""

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: MhkWave.MhkWave,
        financial_model: FinancialModelType,
        block_set_name: str = "wave",
    ):
        """Initialize WaveDispatch.

        Args:
            pyomo_model (ConcreteModel): Pyomo concrete model.
            indexed_set (Set): Indexed set.
            system_model (MhkWave.MhkWave): System model.
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
        """MHK wave instance of maximum gross profit objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = Expression(
            expr=sum(
                -(1 / hybrid_blocks[t].time_weighting_factor)
                * self.blocks[t].time_duration
                * self.blocks[t].cost_per_generation
                * hybrid_blocks[t].wave_generation
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """MHK wave instance of minimum operating cost objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * self.blocks[t].time_duration
            * self.blocks[t].cost_per_generation
            * hybrid_blocks[t].wave_generation
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        """Create MHK wave variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.wave_generation = Var(
            doc="Power generation of wave devices [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.wave_generation, 0

    def _create_port(self, hybrid):
        """Create mhk wave port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: MHK wave Port object.

        """
        hybrid.wave_port = Port(initialize={"generation": hybrid.wave_generation})
        return hybrid.wave_port
