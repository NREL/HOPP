from typing import Union
from pyomo.environ import ConcreteModel, Expression, NonNegativeReals, Set, units, Var
from pyomo.network import Port

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.csp_dispatch import CspDispatch


class TowerDispatch(CspDispatch):
    tower_obj: Union[Expression, float]
    _system_model: None
    _financial_model: FinancialModelType
    """Dispatch optimization model for CSP tower systems."""

    def __init__(
        self,
        pyomo_model: ConcreteModel,
        indexed_set: Set,
        system_model: None,
        financial_model: FinancialModelType,
        block_set_name: str = "tower",
    ):
        """Initialize TowerDispatch.

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
        """Update initial conditions."""
        super().update_initial_conditions()
        csp = self._system_model
        # Note, SS receiver model in ssc assumes full available power is used for startup
        # (even if, time requirement is binding)
        rec_accumulate_time = max(
            0.0,
            csp.value("rec_su_delay") - csp.plant_state["rec_startup_time_remain_init"],
        )
        rec_accumulate_energy = max(
            0.0,
            (
                self.receiver_required_startup_energy
                - csp.plant_state["rec_startup_energy_remain_init"] / 1e6
            ),
        )
        self.initial_receiver_startup_inventory = min(
            rec_accumulate_energy,
            rec_accumulate_time * self.allowable_receiver_startup_power,
        )
        if (
            self.initial_receiver_startup_inventory
            > (1.0 - 1.0e-6) * self.receiver_required_startup_energy
        ):
            self.initial_receiver_startup_inventory = (
                self.receiver_required_startup_energy
            )

    def max_gross_profit_objective(self, hybrid_blocks):
        """Tower CSP instance of maximum gross profit objective.

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
        """Tower CSP instance of minimum operating cost objective.

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
        """Create Tower CSP variables to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            tuple: Tuple containing created variables.
                - generation: Generation from given technology.
                - load: Load from given technology.

        """
        hybrid.tower_generation = Var(
            doc="Power generation of CSP tower [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        hybrid.tower_load = Var(
            doc="Load of CSP tower [MW]",
            domain=NonNegativeReals,
            units=units.MW,
            initialize=0.0,
        )
        return hybrid.tower_generation, hybrid.tower_load

    def _create_port(self, hybrid):
        """Create CSP tower port to add to hybrid plant instance.

        Args:
            hybrid: Hybrid plant instance.

        Returns:
            Port: CSP Tower Port object.

        """
        hybrid.tower_port = Port(
            initialize={
                "cycle_generation": hybrid.tower_generation,
                "system_load": hybrid.tower_load,
            }
        )
        return hybrid.tower_port
