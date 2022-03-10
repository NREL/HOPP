import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hybrid.dispatch.dispatch import Dispatch
from hybrid.dispatch.hybrid_dispatch_options import HybridDispatchOptions


class HybridDispatch(Dispatch):
    """

    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 power_sources: dict,
                 dispatch_options: HybridDispatchOptions = None,
                 block_set_name: str = 'hybrid'):
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

        super().__init__(pyomo_model,
                         index_set,
                         None,
                         None,
                         block_set_name=block_set_name)

    def dispatch_block_rule(self, hybrid, t):
        ##################################
        # Parameters                     #
        ##################################
        self._create_parameters(hybrid)
        ##################################
        # Variables / Ports              #
        ##################################
        for tech in self.power_sources.keys():
            try:
                getattr(self, "_create_" + tech + "_variables")(hybrid, t)
                getattr(self, "_create_" + tech + "_port")(hybrid, t)
            except AttributeError:
                raise ValueError("'{}' is not supported in the hybrid dispatch model.".format(tech))
            except Exception as e:
                raise RuntimeError("Error in setting up dispatch for {}: {}".format(tech, e))
        ##################################
        # Constraints                    #
        ##################################
        self._create_grid_constraints(hybrid, t)
        if 'battery' in self.power_sources.keys():
            if self.options.pv_charging_only:
                self._create_pv_battery_limitation(hybrid)
            elif not self.options.grid_charging:
                self._create_grid_battery_limitation(hybrid)

    @staticmethod
    def _create_parameters(hybrid):
        hybrid.time_weighting_factor = pyomo.Param(
            doc="Exponential time weighting factor [-]",
            initialize=1.0,
            within=pyomo.PercentFraction,
            mutable=True,
            units=u.dimensionless)

    def _create_pv_variables(self, hybrid, t):
        hybrid.pv_generation_cost = pyomo.Var(
            doc="Generation cost of photovoltaics [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD,
            initialize=0.0)
        hybrid.pv_generation = pyomo.Var(
            doc="Power generation of photovoltaics [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.pv_generation)

    def _create_pv_port(self, hybrid, t):
        hybrid.pv_port = Port(initialize={'generation_cost': hybrid.pv_generation_cost,
                                          'generation': hybrid.pv_generation})
        self.ports[t].append(hybrid.pv_port)

    def _create_wind_variables(self, hybrid, t):
        hybrid.wind_generation_cost = pyomo.Var(
            doc="Generation cost of wind turbines [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD,
            initialize=0.0)
        hybrid.wind_generation = pyomo.Var(
            doc="Power generation of wind turbines [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.wind_generation)

    def _create_wind_port(self, hybrid, t):
        hybrid.wind_port = Port(initialize={'generation_cost': hybrid.wind_generation_cost,
                                            'generation': hybrid.wind_generation})
        self.ports[t].append(hybrid.wind_port)

    def _create_battery_variables(self, hybrid, t):
        hybrid.battery_charge = pyomo.Var(
            doc="Power charging the electric battery [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        hybrid.battery_discharge = pyomo.Var(
            doc="Power discharging the electric battery [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        hybrid.battery_charge_cost = pyomo.Var(
            doc="Cost to charge the electric battery [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD,
            initialize=0.0)
        hybrid.battery_discharge_cost = pyomo.Var(
            doc="Cost to discharge the electric battery [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.battery_discharge)
        self.load_vars[t].append(hybrid.battery_charge)

        # TODO: add lifecycle cost port variables...

    def _create_battery_port(self, hybrid, t):
        hybrid.battery_port = Port(initialize={'charge_power': hybrid.battery_charge,
                                               'discharge_power': hybrid.battery_discharge,
                                               'charge_cost': hybrid.battery_charge_cost,
                                               'discharge_cost': hybrid.battery_discharge_cost})
        self.ports[t].append(hybrid.battery_port)

    @staticmethod
    def _create_grid_variables(hybrid, _):
        hybrid.system_generation = pyomo.Var(
            doc="System generation [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        hybrid.system_load = pyomo.Var(
            doc="System load [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        hybrid.electricity_sales = pyomo.Var(
            doc="Electricity sells value [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD)
        hybrid.electricity_purchases = pyomo.Var(
            doc="Electricity purchases value [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD)

    def _create_grid_port(self, hybrid, t):
        hybrid.grid_port = Port(initialize={'system_generation': hybrid.system_generation,
                                            'system_load': hybrid.system_load,
                                            'electricity_sales': hybrid.electricity_sales,
                                            'electricity_purchases': hybrid.electricity_purchases})
        self.ports[t].append(hybrid.grid_port)

    def _create_grid_constraints(self, hybrid, t):
        hybrid.generation_total = pyomo.Constraint(
            doc="hybrid system generation total",
            rule=hybrid.system_generation == sum(self.power_source_gen_vars[t]))

        hybrid.load_total = pyomo.Constraint(
            doc="hybrid system load total",
            rule=hybrid.system_load == sum(self.load_vars[t]))

    @staticmethod
    def _create_grid_battery_limitation(hybrid):
        hybrid.no_grid_battery_charge = pyomo.Constraint(
            doc="Battery storage cannot charge via the grid",
            expr=hybrid.system_generation >= hybrid.battery_charge)

    @staticmethod
    def _create_pv_battery_limitation(hybrid):
        hybrid.only_pv_battery_charge = pyomo.Constraint(
            doc="Battery storage can only charge from pv",
            expr=hybrid.pv_generation >= hybrid.battery_charge)

    def create_arcs(self):
        ##################################
        # Arcs                           #
        ##################################
        for tech in self.power_sources.keys():
            def arc_rule(m, t):
                source_port = self.power_sources[tech].dispatch.blocks[t].port
                destination_port = getattr(self.blocks[t], tech + "_port")
                return {'source': source_port, 'destination': destination_port}

            setattr(self.model, tech + "_hybrid_arc", Arc(self.blocks.index_set(), rule=arc_rule))
            self.arcs.append(getattr(self.model, tech + "_hybrid_arc"))

        pyomo.TransformationFactory("network.expand_arcs").apply_to(self.model)

    def initialize_dispatch_model_parameters(self):
        self.time_weighting_factor = 1.0
        for tech in self.power_sources.values():
            tech.dispatch.initialize_dispatch_model_parameters()

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        for tech in self.power_sources.values():
            tech.dispatch.update_time_series_dispatch_model_parameters(start_time)

    def _delete_objective(self):
        if hasattr(self.model, "objective"):
            self.model.del_component(self.model.objective)

    def create_gross_profit_objective(self):
        self._delete_objective()

        def gross_profit_objective_rule(m):
            objective = 0.0
            for tech in self.power_sources.keys():
                if tech == 'grid':
                    objective += sum(self.blocks[t].time_weighting_factor * self.blocks[t].electricity_sales
                                     - (1/self.blocks[t].time_weighting_factor) * self.blocks[t].electricity_purchases
                                     for t in self.blocks.index_set())
                elif tech == 'pv':
                    objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].pv_generation_cost
                                     for t in self.blocks.index_set())
                elif tech == 'wind':
                    objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].wind_generation_cost
                                     for t in self.blocks.index_set())
                elif tech == 'battery':
                    objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].battery_charge_cost
                                     - (1/self.blocks[t].time_weighting_factor) * self.blocks[t].battery_discharge_cost
                                     for t in self.blocks.index_set())
            # TODO: how should battery life cycle costs be accounted
            #objective -= self.model.lifecycle_cost * self.model.lifecycles
            return objective

        self.model.objective = pyomo.Objective(
            rule=gross_profit_objective_rule,
            sense=pyomo.maximize)

    @property
    def time_weighting_factor(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t+1].time_weighting_factor.value

    @time_weighting_factor.setter
    def time_weighting_factor(self, weighting: float):
        for t in self.blocks.index_set():
            self.blocks[t].time_weighting_factor = weighting ** t  # (self.blocks[t].time_duration.value * t)

    @property
    def time_weighting_factor_list(self) -> list:
        return [self.blocks[t].time_weighting_factor.value for t in self.blocks.index_set()]

    # Outputs
    @property
    def objective_value(self):
        return pyomo.value(self.model.objective)

    @property
    def pv_generation_cost(self) -> list:
        return [self.blocks[t].pv_generation_cost.value for t in self.blocks.index_set()]

    @property
    def pv_generation(self) -> list:
        return [self.blocks[t].pv_generation.value for t in self.blocks.index_set()]

    @property
    def wind_generation_cost(self) -> list:
        return [self.blocks[t].wind_generation_cost.value for t in self.blocks.index_set()]

    @property
    def wind_generation(self) -> list:
        return [self.blocks[t].wind_generation.value for t in self.blocks.index_set()]

    @property
    def battery_charge(self) -> list:
        return [self.blocks[t].battery_charge.value for t in self.blocks.index_set()]

    @property
    def battery_discharge(self) -> list:
        return [self.blocks[t].battery_discharge.value for t in self.blocks.index_set()]

    @property
    def battery_charge_cost(self) -> list:
        return [self.blocks[t].battery_charge_cost.value for t in self.blocks.index_set()]

    @property
    def battery_discharge_cost(self) -> list:
        return [self.blocks[t].battery_discharge_cost.value for t in self.blocks.index_set()]

    @property
    def system_generation(self) -> list:
        return [self.blocks[t].system_generation.value for t in self.blocks.index_set()]

    @property
    def system_load(self) -> list:
        return [self.blocks[t].system_load.value for t in self.blocks.index_set()]

    @property
    def electricity_sales(self) -> list:
        return [self.blocks[t].electricity_sale.value for t in self.blocks.index_set()]

    @property
    def electricity_purchases(self) -> list:
        return [self.blocks[t].electricity_purchases.value for t in self.blocks.index_set()]
