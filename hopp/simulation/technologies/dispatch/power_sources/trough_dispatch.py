from typing import Union
from pyomo.environ import ConcreteModel, Expression, Set

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.dispatch.power_sources.csp_dispatch import CspDispatch


class TroughDispatch(CspDispatch):
    trough_obj: Union[Expression, float]
    _system_model: None
    _financial_model: FinancialModelType
    """

    """
    def __init__(
            self,
            pyomo_model: ConcreteModel,
            indexed_set: Set,
            system_model: None,
            financial_model: FinancialModelType,
            block_set_name: str = 'trough',
        ):
        super().__init__(
            pyomo_model,
            indexed_set,
            system_model,
            financial_model,
            block_set_name=block_set_name
        )

    def update_initial_conditions(self):
        super().update_initial_conditions()
        self.initial_receiver_startup_inventory = 0.0  # FIXME:
        if self.is_field_starting_initial:
            print(
                "Warning: Solar field is starting at the initial time step of the dispatch "
                "horizon, but initial startup energy inventory is assumed to be zero. This may "
                "result in persistent receiver start-up."
            )

    def max_gross_profit_objective(self, blocks):
        self.trough_obj = Expression(
            expr=sum(
                - (1/blocks[t].time_weighting_factor)
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
                for t in blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, blocks):
        self.trough_obj = sum(
            blocks[t].time_weighting_factor
            * (
                self.blocks[t].cost_per_field_start
                * self.blocks[t].incur_field_start
                - (
                    self.blocks[t].cost_per_field_generation
                    * self.blocks[t].receiver_thermal_power
                    * self.blocks[t].time_duration
                )   # Trying to incentivize TES generation
                + (
                    self.blocks[t].cost_per_cycle_generation
                    * self.blocks[t].cycle_generation
                    * self.blocks[t].time_duration
                )
                + self.blocks[t].cost_per_cycle_start
                * self.blocks[t].incur_cycle_start
                + self.blocks[t].cost_per_change_thermal_input
                * self.blocks[t].cycle_thermal_ramp
            )
            for t in blocks.index_set()
        )
