from typing import Union
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.csp_dispatch import CspDispatch


class TroughDispatch(CspDispatch):
    trough_obj: Union[Expression, float]
    _system_model: None
    _financial_model: FinancialModelType
    """Dispatch optimization model for CSP trough systems."""

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: None,
        financial_model: FinancialModelType,
        block_set_name: str = "trough",
    ):
        """Initialize TroughDispatch.

        Args:
            pyomo_model (ConcreteModel): Pyomo concrete model.
            indexed_set (Set): Indexed set.
            system_model (None): System model.
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

    def update_initial_conditions(self):
        """Update initial conditions method."""
        super().update_initial_conditions()
        self.initial_receiver_startup_inventory = 0.0  # FIXME:
        if self.is_field_starting_initial:
            print(
                "Warning: Solar field is starting at the initial time step of the dispatch "
                "horizon, but initial startup energy inventory is assumed to be zero. This may "
                "result in persistent receiver start-up."
            )

    def max_gross_profit_objective(self, hybrid_blocks):
        """Trough CSP instance of maximum gross profit objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = Expression(
            expr=sum(
                -(1 / hybrid_blocks[t].time_weighting_factor)
                * (
                    (
                        self.blocks[t].cost_per_field_generation
                        * self.blocks[t].receiver_thermal_power
                        * self.blocks[t].time_duration
                    )
                    + (
                        self.blocks[t].cost_per_field_start
                        * self.blocks[t].incur_field_start
                    )
                    + (
                        self.blocks[t].cost_per_cycle_generation
                        * self.blocks[t].cycle_generation
                        * self.blocks[t].time_duration
                    )
                    + (
                        self.blocks[t].cost_per_cycle_start
                        * self.blocks[t].incur_cycle_start
                    )
                    + (
                        self.blocks[t].cost_per_change_thermal_input
                        * self.blocks[t].cycle_thermal_ramp
                    )
                )
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """Trough CSP instance of minimum operating cost objective.

        Args:
            hybrid_blocks (Pyomo.block): A generalized container for defining hierarchical
                models by adding modeling components as attributes.

        """
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * (
                self.blocks[t].cost_per_field_start * self.blocks[t].incur_field_start
                - (
                    self.blocks[t].cost_per_field_generation
                    * self.blocks[t].receiver_thermal_power
                    * self.blocks[t].time_duration
                )  # Trying to incentivize TES generation
                + (
                    self.blocks[t].cost_per_cycle_generation
                    * self.blocks[t].cycle_generation
                    * self.blocks[t].time_duration
                )
                + self.blocks[t].cost_per_cycle_start * self.blocks[t].incur_cycle_start
                + self.blocks[t].cost_per_change_thermal_input
                * self.blocks[t].cycle_thermal_ramp
            )
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        """Create Trough CSP variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.trough_generation = Var(
            doc="Power generation of CSP trough [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        hybrid.trough_load = Var(
            doc="Load of CSP trough [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.trough_generation, hybrid.trough_load

    def _create_port(self, hybrid):
        """Create CSP trough port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: CSP Trough Port object.

        """
        hybrid.trough_port = Port(
            initialize={
                "cycle_generation": hybrid.trough_generation,
                "system_load": hybrid.trough_load,
            }
        )
        return hybrid.trough_port
