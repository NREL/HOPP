import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hopp.dispatch.dispatch import Dispatch
from hopp.dispatch.hybrid_dispatch_options import HybridDispatchOptions


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
        hybrid.pv_generation = pyomo.Var(
            doc="Power generation of photovoltaics [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.pv_generation)

    def _create_pv_port(self, hybrid, t):
        hybrid.pv_port = Port(initialize={'generation': hybrid.pv_generation})
        self.ports[t].append(hybrid.pv_port)

    def _create_wind_variables(self, hybrid, t):
        hybrid.wind_generation = pyomo.Var(
            doc="Power generation of wind turbines [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.wind_generation)

    def _create_wind_port(self, hybrid, t):
        hybrid.wind_port = Port(initialize={'generation': hybrid.wind_generation})
        self.ports[t].append(hybrid.wind_port)

    def _create_tower_variables(self, hybrid, t):
        hybrid.tower_generation = pyomo.Var(
            doc="Power generation of CSP tower [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        hybrid.tower_load = pyomo.Var(
            doc="Load of CSP tower [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.tower_generation)
        self.load_vars[t].append(hybrid.tower_load)

    def _create_tower_port(self, hybrid, t):
        hybrid.tower_port = Port(initialize={'cycle_generation': hybrid.tower_generation,
                                             'system_load': hybrid.tower_load})
        self.ports[t].append(hybrid.tower_port)

    def _create_trough_variables(self, hybrid, t):
        hybrid.trough_generation = pyomo.Var(
            doc="Power generation of CSP trough [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        hybrid.trough_load = pyomo.Var(
            doc="Load of CSP trough [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
            initialize=0.0)
        self.power_source_gen_vars[t].append(hybrid.trough_generation)
        self.load_vars[t].append(hybrid.trough_load)

    def _create_trough_port(self, hybrid, t):
        hybrid.trough_port = Port(initialize={'cycle_generation': hybrid.trough_generation,
                                              'system_load': hybrid.trough_load})
        self.ports[t].append(hybrid.trough_port)

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
        self.power_source_gen_vars[t].append(hybrid.battery_discharge)
        self.load_vars[t].append(hybrid.battery_charge)

    def _create_battery_port(self, hybrid, t):
        hybrid.battery_port = Port(initialize={'charge_power': hybrid.battery_charge,
                                               'discharge_power': hybrid.battery_discharge})
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
        hybrid.electricity_sold = pyomo.Var(
            doc="Electricity sold [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        hybrid.electricity_purchased = pyomo.Var(
            doc="Electricity purchased [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)

    def _create_grid_port(self, hybrid, t):
        hybrid.grid_port = Port(initialize={'system_generation': hybrid.system_generation,
                                            'system_load': hybrid.system_load,
                                            'electricity_sold': hybrid.electricity_sold,
                                            'electricity_purchased': hybrid.electricity_purchased})
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

    def initialize_parameters(self):
        self.time_weighting_factor = 0.995  # Discount factor
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

        def gross_profit_objective_rule(m):
            objective = 0.0
            for tech in self.power_sources.keys():
                if tech == 'grid':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * tb[t].electricity_sell_price * self.blocks[t].electricity_sold
                                     - (1/self.blocks[t].time_weighting_factor) * tb[t].time_duration
                                     * tb[t].electricity_purchase_price * self.blocks[t].electricity_purchased
                                     - tb[t].epsilon * tb[t].is_generating
                                     for t in self.blocks.index_set())
                elif tech == 'pv':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(- (1/self.blocks[t].time_weighting_factor)
                                     * tb[t].time_duration * tb[t].cost_per_generation * self.blocks[t].pv_generation
                                     for t in self.blocks.index_set())
                elif tech == 'wind':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(- (1/self.blocks[t].time_weighting_factor)
                                     * tb[t].time_duration * tb[t].cost_per_generation * self.blocks[t].wind_generation
                                     for t in self.blocks.index_set())
                elif tech == 'tower' or tech == 'trough':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(- (1/self.blocks[t].time_weighting_factor)
                                     * ((tb[t].cost_per_field_generation
                                         * tb[t].receiver_thermal_power
                                         * tb[t].time_duration)
                                        + tb[t].cost_per_field_start * tb[t].incur_field_start
                                        + (tb[t].cost_per_cycle_generation
                                           * tb[t].cycle_generation
                                           * tb[t].time_duration)
                                        + tb[t].cost_per_cycle_start * tb[t].incur_cycle_start
                                        + tb[t].cost_per_change_thermal_input * tb[t].cycle_thermal_ramp)
                                     for t in self.blocks.index_set())
                elif tech == 'battery':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(- (1/self.blocks[t].time_weighting_factor) * tb[t].time_duration
                                     * (tb[t].cost_per_charge * self.blocks[t].battery_charge
                                        + tb[t].cost_per_discharge * self.blocks[t].battery_discharge)
                                     for t in self.blocks.index_set())
                    tb = self.power_sources['battery'].dispatch
                    if tb.include_lifecycle_count:
                        objective -= tb.model.lifecycle_cost * tb.model.lifecycles
            return objective

        self.model.objective = pyomo.Objective(
            rule=gross_profit_objective_rule,
            sense=pyomo.maximize)

    def create_min_operating_cost_objective(self):
        self._delete_objective()

        def operating_cost_objective_rule(m):
            objective = 0.0
            for tech in self.power_sources.keys():
                if tech == 'grid':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * tb[t].electricity_sell_price * (tb[t].generation_transmission_limit
                                                                       - self.blocks[t].electricity_sold)
                                     + self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * tb[t].electricity_purchase_price * self.blocks[t].electricity_purchased
                                     + tb[t].epsilon * tb[t].is_generating
                                     for t in self.blocks.index_set())
                elif tech == 'pv':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * tb[t].cost_per_generation * self.blocks[t].pv_generation
                                     for t in self.blocks.index_set())
                elif tech == 'wind':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * tb[t].cost_per_generation * self.blocks[t].wind_generation
                                     for t in self.blocks.index_set())
                elif tech == 'tower' or tech == 'trough':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor
                                     * (tb[t].cost_per_field_start * tb[t].incur_field_start
                                        - (tb[t].cost_per_field_generation
                                           * tb[t].receiver_thermal_power
                                           * tb[t].time_duration)   # Trying to incentivize TES generation
                                        + (tb[t].cost_per_cycle_generation
                                           * tb[t].cycle_generation
                                           * tb[t].time_duration)
                                        + tb[t].cost_per_cycle_start * tb[t].incur_cycle_start
                                        + tb[t].cost_per_change_thermal_input * tb[t].cycle_thermal_ramp)
                                     for t in self.blocks.index_set())
                elif tech == 'battery':
                    tb = self.power_sources[tech].dispatch.blocks
                    objective += sum(self.blocks[t].time_weighting_factor * tb[t].time_duration
                                     * (tb[t].cost_per_discharge * self.blocks[t].battery_discharge
                                        - tb[t].cost_per_charge * self.blocks[t].battery_charge)
                                     # Try to incentivize battery charging
                                     for t in self.blocks.index_set())
                    tb = self.power_sources['battery'].dispatch
                    if tb.include_lifecycle_count:
                        objective += tb.model.lifecycle_cost * tb.model.lifecycles
            return objective

        self.model.objective = pyomo.Objective(
            rule=operating_cost_objective_rule,
            sense=pyomo.minimize)

    @property
    def time_weighting_factor(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t+1].time_weighting_factor.value

    @time_weighting_factor.setter
    def time_weighting_factor(self, weighting: float):
        for t in self.blocks.index_set():
            self.blocks[t].time_weighting_factor = round(weighting ** t, self.round_digits)

    @property
    def time_weighting_factor_list(self) -> list:
        return [self.blocks[t].time_weighting_factor.value for t in self.blocks.index_set()]

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
        if 'grid' in self.power_sources:
            tb = self.power_sources['grid'].dispatch.blocks
            return [tb[t].time_duration.value * tb[t].electricity_sell_price.value
                    * self.blocks[t].electricity_sold.value for t in self.blocks.index_set()]

    @property
    def electricity_purchases(self) -> list:
        if 'grid' in self.power_sources:
            tb = self.power_sources['grid'].dispatch.blocks
            return [tb[t].time_duration.value * tb[t].electricity_purchase_price.value
                    * self.blocks[t].electricity_purchased.value for t in self.blocks.index_set()]
