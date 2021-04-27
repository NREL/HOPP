import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hybrid.dispatch.dispatch import Dispatch


class HybridDispatch(Dispatch):
    """

    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 power_sources: dict,
                 block_set_name: str = 'hybrid'):
        self.power_sources = power_sources

        super().__init__(pyomo_model,
                         index_set,
                         None,
                         None,
                         block_set_name=block_set_name)

    def dispatch_block_rule(self, hybrid):
        ##################################
        # Parameters                     #
        ##################################
        self._create_parameters(hybrid)
        ##################################
        # Variables / Ports              #
        ##################################
        for tech in self.power_sources.keys():
            if tech is 'solar':
                self._create_solar_pv_variables(hybrid)
                self._create_solar_port(hybrid)
            elif tech is 'wind':
                self._create_wind_variables(hybrid)
                self._create_wind_port(hybrid)
            elif tech is 'battery':
                self._create_battery_variables(hybrid)
                self._create_battery_port(hybrid)
            elif tech is 'grid':
                self._create_grid_variables(hybrid)
                self._create_grid_port(hybrid)
            else:
                raise ValueError("'{}' is not supported in the hybrid dispatch model.".format(tech))
        ##################################
        # Constraints                    #
        ##################################
        if 'grid' in self.power_sources.keys():
            self._create_grid_constraints(hybrid)
            if 'battery' in self.power_sources.keys():
                # TODO: this should be enabled and disabled
                self._create_grid_battery_limitation(hybrid)

    @staticmethod
    def _create_parameters(hybrid):
        hybrid.time_weighting_factor = pyomo.Param(doc="Exponential time weighting factor [-]",
                                                   initialize=1.0,
                                                   within=pyomo.PercentFraction,
                                                   mutable=True,
                                                   units=u.dimensionless)

    @staticmethod
    def _create_solar_pv_variables(hybrid):
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

    @staticmethod
    def _create_solar_port(hybrid):
        hybrid.pv_port = Port(initialize={'generation_cost': hybrid.pv_generation_cost,
                                          'generation': hybrid.pv_generation})

    @staticmethod
    def _create_wind_variables(hybrid):
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

    @staticmethod
    def _create_wind_port(hybrid):
        hybrid.wind_port = Port(initialize={'generation_cost': hybrid.wind_generation_cost,
                                            'generation': hybrid.wind_generation})

    @staticmethod
    def _create_battery_variables(hybrid):
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
        # TODO: add lifecycle cost port variables...

    @staticmethod
    def _create_battery_port(hybrid):
        hybrid.battery_port = Port(initialize={'charge_power': hybrid.battery_charge,
                                               'discharge_power': hybrid.battery_discharge,
                                               'charge_cost': hybrid.battery_charge_cost,
                                               'discharge_cost': hybrid.battery_discharge_cost})

    @staticmethod
    def _create_grid_variables(hybrid):
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

    @staticmethod
    def _create_grid_port(hybrid):
        hybrid.grid_port = Port(initialize={'system_generation': hybrid.system_generation,
                                            'system_load': hybrid.system_load,
                                            'electricity_sales': hybrid.electricity_sales,
                                            'electricity_purchases': hybrid.electricity_purchases})

    def _create_grid_constraints(self, hybrid):
        hybrid.generation_total = pyomo.Constraint(
            doc="hybrid system generation total",
            rule=self._generation_total(hybrid))
        hybrid.load_total = pyomo.Constraint(
            doc="hybrid system load total",
            rule=self._load_total(hybrid))

    def _generation_total(self, hybrid):
        for tech in self.power_sources.keys():
            term = None
            if tech is 'solar':
                term = hybrid.pv_generation
            elif tech is 'wind':
                term = hybrid.wind_generation
            elif tech is 'battery':
                term = hybrid.battery_discharge

            if term is not None:
                try:
                    rhs += term
                except NameError:
                    rhs = term
        return hybrid.system_generation == rhs

    def _load_total(self, hybrid):
        for tech in self.power_sources.keys():
            term = None
            if tech is 'battery':
                term = hybrid.battery_charge

            if term is not None:
                try:
                    rhs += term
                except NameError:
                    rhs = term
        return hybrid.system_load == rhs

    @staticmethod
    def _create_grid_battery_limitation(hybrid):
        hybrid.no_grid_battery_charge = pyomo.Constraint(
            doc="Battery storage cannot charge via the grid",
            expr=hybrid.system_generation >= hybrid.battery_charge)

    def create_arcs(self):
        ##################################
        # Arcs                           #
        ##################################
        for tech in self.power_sources.keys():
            if tech is 'solar':
                self.model.pv_hybrid_arc = Arc(self.blocks.index_set(), rule=self._pv_solar_arc_rule)
            elif tech is 'wind':
                self.model.wind_hybrid_arc = Arc(self.blocks.index_set(), rule=self._wind_arc_rule)
            elif tech is 'battery':
                self.model.battery_hybrid_arc = Arc(self.blocks.index_set(), rule=self._battery_arc_rule)
            elif tech is 'grid':
                self.model.hybrid_hybrid_arc = Arc(self.blocks.index_set(), rule=self._grid_arc_rule)

        pyomo.TransformationFactory("network.expand_arcs").apply_to(self.model)

    def _pv_solar_arc_rule(self, m, t):
        source_port = self.power_sources['solar'].dispatch.blocks[t].port
        destination_port = self.blocks[t].pv_port
        return {'source': source_port, 'destination': destination_port}

    def _wind_arc_rule(self, m, t):
        source_port = self.power_sources['wind'].dispatch.blocks[t].port
        destination_port = self.blocks[t].wind_port
        return {'source': source_port, 'destination': destination_port}

    def _battery_arc_rule(self, m, t):
        source_port = self.power_sources['battery'].dispatch.blocks[t].port
        destination_port = self.blocks[t].battery_port
        return {'source': source_port, 'destination': destination_port}

    def _grid_arc_rule(self, m, t):
        source_port = self.power_sources['grid'].dispatch.blocks[t].port
        destination_port = self.blocks[t].grid_port
        return {'source': source_port, 'destination': destination_port}

    def create_gross_profit_objective(self):
        self.model.gross_profit_objective = pyomo.Objective(
            rule=self.gross_profit_objective_rule,
            sense=pyomo.maximize)

    def gross_profit_objective_rule(self, m):
        objective = 0.0
        for tech in self.power_sources.keys():
            if tech is 'grid':
                objective += sum(self.blocks[t].time_weighting_factor * self.blocks[t].electricity_sales
                                 - (1/self.blocks[t].time_weighting_factor) * self.blocks[t].electricity_purchases
                                 for t in self.blocks.index_set())
            elif tech is 'solar':
                objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].pv_generation_cost
                                 for t in self.blocks.index_set())
            elif tech is 'wind':
                objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].wind_generation_cost
                                 for t in self.blocks.index_set())
            elif tech is 'battery':
                objective += sum(- (1/self.blocks[t].time_weighting_factor) * self.blocks[t].battery_charge_cost
                                 - (1/self.blocks[t].time_weighting_factor) * self.blocks[t].battery_discharge_cost
                                 for t in self.blocks.index_set())
        # TODO: how should battery life cycle costs be accounted
        #objective -= self.model.lifecycle_cost * self.model.lifecycles
        return objective

    def delete_gross_profit_objective(self):
        self.model.del_component(self.model.gross_profit_objective)

    def initialize_dispatch_model_parameters(self):
        self.time_weighting_factor = 1.0
        for tech in self.power_sources.values():
            tech.dispatch.initialize_dispatch_model_parameters()

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        for tech in self.power_sources.values():
            tech.dispatch.update_time_series_dispatch_model_parameters(start_time)

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
    def gross_profit_objective(self):
        return pyomo.value(self.model.gross_profit_objective)

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
