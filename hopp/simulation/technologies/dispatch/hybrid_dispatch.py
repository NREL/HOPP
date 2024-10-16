import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hopp.simulation.technologies.dispatch.dispatch import Dispatch
from hopp.simulation.technologies.dispatch.hybrid_dispatch_options import (
    HybridDispatchOptions,
)


class HybridDispatch(Dispatch):
    """ """

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        power_sources: dict,
        dispatch_options: HybridDispatchOptions = None,
        block_set_name: str = "hybrid",
    ):
        """

        Parameters
        ----------
        dispatch_options :
            Contains attribute key, value pairs to change default dispatch options.
            For details see HybridDispatchOptions in hybrid_dispatch_options.py

        """
        self.power_sources = power_sources
        self.options = dispatch_options
        self.power_source_gen_vars = {key: [] for key in index_set}
        self.load_vars = {key: [] for key in index_set}
        self.ports = {key: [] for key in index_set}
        self.arcs = []

        super().__init__(
            pyomo_model,
            index_set,
            None,
            None,
            block_set_name=block_set_name,
        )

    def dispatch_block_rule(self, hybrid, t):
        ##################################
        # Parameters                     #
        ##################################
        self._create_parameters(hybrid)
        ##################################
        # Variables / Ports              #
        ##################################
        self._create_variables_and_ports(hybrid, t)
        ##################################
        # Constraints                    #
        ##################################
        self._create_hybrid_constraints(hybrid, t)

    @staticmethod
    def _create_parameters(hybrid):
        hybrid.time_weighting_factor = pyomo.Param(
            doc="Exponential time weighting factor [-]",
            initialize=1.0,
            within=pyomo.PercentFraction,
            mutable=True,
            units=u.dimensionless,
        )

    def _create_variables_and_ports(self, hybrid, t):
        for tech in self.power_sources.keys():
            try:
                gen_var, load_var = self.power_sources[
                    tech
                ]._dispatch._create_variables(hybrid)
                self.power_source_gen_vars[t].append(gen_var)
                self.load_vars[t].append(load_var)
                self.ports[t].append(
                    self.power_sources[tech]._dispatch._create_port(hybrid)
                )
            except AttributeError:
                raise ValueError(
                    "'{}' is not supported in the hybrid dispatch model.".format(tech)
                )
            except Exception as e:
                raise RuntimeError(
                    "Error in setting up dispatch for {}: {}".format(tech, e)
                )

    def _create_hybrid_constraints(self, hybrid, t):
        hybrid.generation_total = pyomo.Constraint(
            doc="hybrid system generation total",
            rule=hybrid.system_generation == sum(self.power_source_gen_vars[t]),
        )

        hybrid.load_total = pyomo.Constraint(
            doc="hybrid system load total",
            rule=hybrid.system_load == sum(self.load_vars[t]),
        )

        if "battery" in self.power_sources.keys():
            if self.options.pv_charging_only:
                self._create_pv_battery_limitation(hybrid)
            elif not self.options.grid_charging:
                self._create_grid_battery_limitation(hybrid)

    @staticmethod
    def _create_grid_battery_limitation(hybrid):
        hybrid.no_grid_battery_charge = pyomo.Constraint(
            doc="Battery storage cannot charge via the grid",
            expr=hybrid.system_generation >= hybrid.battery_charge,
        )

    @staticmethod
    def _create_pv_battery_limitation(hybrid):
        hybrid.only_pv_battery_charge = pyomo.Constraint(
            doc="Battery storage can only charge from pv",
            expr=hybrid.pv_generation >= hybrid.battery_charge,
        )

    def create_arcs(self):
        ##################################
        # Arcs                           #
        ##################################
        for tech in self.power_sources.keys():

            def arc_rule(m, t):
                source_port = self.power_sources[tech].dispatch.blocks[t].port
                destination_port = getattr(self.blocks[t], tech + "_port")
                return {"source": source_port, "destination": destination_port}

            setattr(
                self.model,
                tech + "_hybrid_arc",
                Arc(self.blocks.index_set(), rule=arc_rule),
            )
            self.arcs.append(getattr(self.model, tech + "_hybrid_arc"))

        pyomo.TransformationFactory("network.expand_arcs").apply_to(self.model)

    def initialize_parameters(self):
        self.time_weighting_factor = (
            self.options.time_weighting_factor
        )  # Discount factor
        for tech in self.power_sources.values():
            tech.dispatch.initialize_parameters()

    def update_time_series_parameters(self, start_time: int):
        for tech in self.power_sources.values():
            tech.dispatch.update_time_series_parameters(start_time)

    def _delete_objective(self):
        if hasattr(self.model, "objective"):
            self.model.del_component(self.model.objective)

    def create_max_gross_profit_objective(self):
        self._delete_objective()

        def gross_profit_objective_rule(m) -> float:
            obj = 0.0
            for tech in self.power_sources.keys():
                # Create the max_gross_profit_objective within each of the technology
                # dispatch classes.
                self.power_sources[tech]._dispatch.max_gross_profit_objective(
                    self.blocks
                )
                # Copy the technology objective to the pyomo model.
                setattr(m, tech + "_obj", self.power_sources[tech]._dispatch.obj)
                # TODO: Does the objective really need to be stored on the self.model object?
                # Trying to grab the attribute 'obj' from the dispatch classes
                # themselves doesn't seem to work within pyomo, e.g.:
                # `getattr(self.power_sources[tech]._dispatch, "obj")`. If we could avoid
                # this, then the above `setattr` would not be needed.

                # Assemble the objective as a linear summation.
                obj += getattr(m, tech + "_obj")
            return obj

        self.model.objective = pyomo.Objective(
            expr=gross_profit_objective_rule, sense=pyomo.maximize
        )

    def create_min_operating_cost_objective(self):
        self._delete_objective()

        def operating_cost_objective_rule(m) -> float:
            obj = 0.0
            for tech in self.power_sources.keys():
                # Create the min_operating_cost_objective within each of the technology
                # dispatch classes.
                self.power_sources[tech]._dispatch.min_operating_cost_objective(
                    self.blocks
                )

                # Assemble the objective as a linear summation.
                obj += self.power_sources[tech]._dispatch.obj

            return obj

        self.model.objective = pyomo.Objective(
            rule=operating_cost_objective_rule, sense=pyomo.minimize
        )

    @property
    def time_weighting_factor(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t + 1].time_weighting_factor.value

    @time_weighting_factor.setter
    def time_weighting_factor(self, weighting: float):
        for t in self.blocks.index_set():
            self.blocks[t].time_weighting_factor = round(
                weighting**t, self.round_digits
            )

    @property
    def time_weighting_factor_list(self) -> list:
        return [
            self.blocks[t].time_weighting_factor.value for t in self.blocks.index_set()
        ]

    # Outputs
    @property
    def objective_value(self):
        return pyomo.value(self.model.objective)

    @property
    def pv_generation(self) -> list:
        return [self.blocks[t].pv_generation.value for t in self.blocks.index_set()]

    @property
    def wind_generation(self) -> list:
        return [self.blocks[t].wind_generation.value for t in self.blocks.index_set()]

    @property
    def wave_generation(self) -> list:
        return [self.blocks[t].wave_generation.value for t in self.blocks.index_set()]

    @property
    def tower_generation(self) -> list:
        return [self.blocks[t].tower_generation.value for t in self.blocks.index_set()]

    @property
    def tower_load(self) -> list:
        return [self.blocks[t].tower_load.value for t in self.blocks.index_set()]

    @property
    def trough_generation(self) -> list:
        return [self.blocks[t].trough_generation.value for t in self.blocks.index_set()]

    @property
    def trough_load(self) -> list:
        return [self.blocks[t].trough_load.value for t in self.blocks.index_set()]

    @property
    def battery_charge(self) -> list:
        return [self.blocks[t].battery_charge.value for t in self.blocks.index_set()]

    @property
    def battery_discharge(self) -> list:
        return [self.blocks[t].battery_discharge.value for t in self.blocks.index_set()]

    @property
    def system_generation(self) -> list:
        return [self.blocks[t].system_generation.value for t in self.blocks.index_set()]

    @property
    def system_load(self) -> list:
        return [self.blocks[t].system_load.value for t in self.blocks.index_set()]

    @property
    def electricity_sales(self) -> list:
        if "grid" in self.power_sources:
            tb = self.power_sources["grid"].dispatch.blocks
            return [
                tb[t].time_duration.value
                * tb[t].electricity_sell_price.value
                * self.blocks[t].electricity_sold.value
                for t in self.blocks.index_set()
            ]

    @property
    def electricity_purchases(self) -> list:
        if "grid" in self.power_sources:
            tb = self.power_sources["grid"].dispatch.blocks
            return [
                tb[t].time_duration.value
                * tb[t].electricity_purchase_price.value
                * self.blocks[t].electricity_purchased.value
                for t in self.blocks.index_set()
            ]
