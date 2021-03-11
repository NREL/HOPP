import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hybrid.dispatch.power_source_dispatch import PowerSourceDispatch

try:
    u.USD
except AttributeError:
    u.load_definitions_from_strings(['USD = [currency]'])


class GridDispatch(PowerSourceDispatch):
    _model: pyomo.ConcreteModel
    _blocks: pyomo.Block
    """

    """

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 power_sources: dict,
                 block_set_name: str = 'grid'):
        self.power_sources = power_sources

        super().__init__(pyomo_model, index_set, block_set_name=block_set_name)

    def dispatch_block_rule(self, grid):
        ##################################
        # Parameters                     #
        ##################################
        grid.transmission_limit = pyomo.Param(doc="Net grid transmission upper limit [MW]",
                                              within=pyomo.NonNegativeReals,
                                              mutable=True,
                                              units=u.MW)
        ##################################
        # Variables                      #
        ##################################
        for tech in self.power_sources.keys():
            if tech is 'solar':
                grid.pv_generation = pyomo.Var(doc="Power generation of photovoltaics [MW]",
                                               domain=pyomo.NonNegativeReals,
                                               units=u.MW,
                                               initialize=0.0)
            elif tech is 'wind':
                grid.wind_generation = pyomo.Var(doc="Power generation of wind turbines [MW]",
                                                 domain=pyomo.NonNegativeReals,
                                                 units=u.MW,
                                                 initialize=0.0)
            elif tech is 'battery':
                grid.battery_charge = pyomo.Var(doc="Power charging the electric battery [MW]",
                                                domain=pyomo.NonNegativeReals,
                                                units=u.MW,
                                                initialize=0.0)
                grid.battery_discharge = pyomo.Var(doc="Power discharging the electric battery [MW]",
                                                   domain=pyomo.NonNegativeReals,
                                                   units=u.MW,
                                                   initialize=0.0)
            elif tech is not 'grid':
                raise ValueError("'{}' is not supported in the grid dispatch model.".format(tech))

        grid.electricity_sold = pyomo.Var(doc="Electricity sold to the grid [MW]",
                                          domain=pyomo.NonNegativeReals,
                                          bounds=(0, grid.transmission_limit),
                                          units=u.MW)

        grid.gross_profit = pyomo.Var(doc="Sub-system gross profit [USD]",
                                      domain=pyomo.Reals,
                                      units=u.USD)
        ##################################
        # Constraints                    #
        ##################################
        grid.power_balance = pyomo.Constraint(
            doc="System power balance to impose transmission limits and no grid battery charging",
            rule=self.power_balance_rule)

        # Currently, this model does not include profit or costs associated with the grid
        grid.gross_profit_calculation = pyomo.Constraint(
            doc="Calculation of gross profit for objective function",
            expr=grid.gross_profit == 0.0)
        ##################################
        # Ports                          #
        ##################################
        for tech in self.power_sources.keys():
            if tech is 'solar':
                grid.pv_port = Port(initialize={'generation': grid.pv_generation})
            elif tech is 'wind':
                grid.wind_port = Port(initialize={'generation': grid.wind_generation})
            elif tech is 'battery':
                grid.battery_port = Port(initialize={'charge_power': grid.battery_charge,
                                                     'discharge_power': grid.battery_discharge})

    def power_balance_rule(self, grid):
        for tech in self.power_sources.keys():
            term = None
            if tech is 'solar':
                term = grid.pv_generation
            elif tech is 'wind':
                term = grid.wind_generation
            elif tech is 'battery':
                term = grid.battery_discharge - grid.battery_charge
            elif tech is not 'grid':
                raise ValueError("'{}' is not supported in the grid dispatch model.".format(tech))

            if term is not None:
                try:
                    rhs += term
                except NameError:
                    rhs = term
        return grid.electricity_sold == rhs

    def create_arcs(self):
        ##################################
        # Arcs                           #
        ##################################
        for tech in self.power_sources.keys():
            if tech is 'solar':
                self.model.pv_grid_arc = Arc(self.blocks.index_set(), rule=self.pv_solar_arc_rule)
            elif tech is 'wind':
                self.model.wind_grid_arc = Arc(self.blocks.index_set(), rule=self.wind_arc_rule)
            elif tech is 'battery':
                self.model.battery_grid_arc = Arc(self.blocks.index_set(), rule=self.battery_arc_rule)

    def pv_solar_arc_rule(self, m, t):
        source_port = self.power_sources['solar'].dispatch.blocks[t].port
        destination_port = self.power_sources['grid'].dispatch.blocks[t].pv_port
        return {'source': source_port, 'destination': destination_port}

    def wind_arc_rule(self, m, t):
        source_port = self.power_sources['wind'].dispatch.blocks[t].port
        destination_port = self.power_sources['grid'].dispatch.blocks[t].wind_port
        return {'source': source_port, 'destination': destination_port}

    def battery_arc_rule(self, m, t):
        source_port = self.power_sources['battery'].dispatch.blocks[t].port
        destination_port = self.power_sources['grid'].dispatch.blocks[t].battery_port
        return {'source': source_port, 'destination': destination_port}

    @property
    def transmission_limit(self) -> list:
        return [self.blocks[t].transmission_limit.value for t in self.blocks.index_set()]

    @transmission_limit.setter
    def transmission_limit(self, limit_mw: list):
        if len(limit_mw) == len(self.blocks):
            for t, limit in zip(self.blocks, limit_mw):
                self.blocks[t].transmission_limit = round(limit, self.round_digits)
        else:
            raise ValueError("'limit_mw' list must be the same length as time horizon")

    @property
    def pv_generation(self) -> list:
        return [self.blocks[t].pv_generation.value for t in self.blocks.index_set()]

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
    def electricity_sold(self) -> list:
        return [self.blocks[t].electricity_sold.value for t in self.blocks.index_set()]
