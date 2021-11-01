import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from hybrid.dispatch.dispatch import Dispatch


class GridDispatch(Dispatch):
    _model: pyomo.ConcreteModel
    _blocks: pyomo.Block
    """

    """

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model,
                 financial_model,
                 block_set_name: str = 'grid'):

        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name)

    def dispatch_block_rule(self, grid):
        # Parameters
        self._create_grid_parameters(grid)
        # Variables
        self._create_grid_variables(grid)
        # Constraints
        self._create_grid_constraints(grid)
        # Ports
        self._create_grid_ports(grid)

    @staticmethod
    def _create_grid_parameters(grid):
        ##################################
        # Parameters                     #
        ##################################
        grid.time_duration = pyomo.Param(
            doc="Time step [hour]",
            default=1.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.hr)
        grid.electricity_sell_price = pyomo.Param(
            doc="Electricity sell price [$/MWh]",
            default=0.0,
            within=pyomo.Reals,
            mutable=True,
            units=u.USD / u.MWh)
        grid.electricity_purchase_price = pyomo.Param(
            doc="Electricity purchase price [$/MWh]",
            default=0.0,
            within=pyomo.Reals,
            mutable=True,
            units=u.USD / u.MWh)
        grid.transmission_limit = pyomo.Param(
            doc="Net grid transmission upper limit [MW]",
            default=1000.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)

    def _create_grid_variables(self, grid):
        ##################################
        # Variables                      #
        ##################################
        grid.electricity_sold = pyomo.Var(
            doc="Electricity sold to the grid [MW]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, grid.transmission_limit),
            units=u.MW)
        grid.electricity_purchased = pyomo.Var(
            doc="Electricity purchased to the grid [MW]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, grid.transmission_limit),
            units=u.MW)
        grid.is_system_producing = pyomo.Var(
            doc="1 if system is net producing electricity; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        grid.system_generation = pyomo.Var(
            doc="System generation [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        grid.system_load = pyomo.Var(
            doc="System load [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        grid.electricity_sales = pyomo.Var(
            doc="Value of electricity sales [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD)
        grid.electricity_purchases = pyomo.Var(
            doc="Value of electricity purchases [$]",
            domain=pyomo.NonNegativeReals,
            units=u.USD)

        self._create_grid_aux_variables(grid)

    @staticmethod
    def _create_grid_aux_variables(grid):
        # Auxiliary variables for continuous * binary
        grid.aux_system_generation = pyomo.Var(
            doc="Auxiliary bi-linear term equal to the product of system generation and producing binary [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        grid.aux_system_load = pyomo.Var(
            doc="Auxiliary bi-linear term equal to the product of system load and producing binary [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)

    def _create_grid_constraints(self, grid):
        ##################################
        # Constraints                    #
        ##################################
        grid.sell_limit = pyomo.Constraint(
            doc="system electricity sell limit",
            expr=grid.electricity_sold <= grid.aux_system_generation - grid.aux_system_load)
        grid.purchase_limit = pyomo.Constraint(
            doc="system electricity purchase requirement",
            expr=grid.electricity_purchased >= ((grid.system_load - grid.system_generation)
                                                - (grid.aux_system_load - grid.aux_system_generation)))
        grid.electricity_sales_calc = pyomo.Constraint(
            doc="Calculation of electricity sales for objective function",
            expr=grid.electricity_sales == grid.time_duration * grid.electricity_sell_price * grid.electricity_sold)
        grid.electricity_purchases_calc = pyomo.Constraint(
            doc="Calculation of electricity purchases for objective function",
            expr=grid.electricity_purchases == (grid.time_duration
                                                * grid.electricity_purchase_price * grid.electricity_purchased))

        self._create_grid_aux_constraints(grid)

    @staticmethod
    def _create_grid_aux_constraints(grid):
        # Aux system generation variable
        grid.generation_ub = pyomo.Constraint(
            doc="Auxiliary variable upper bound",
            expr=grid.aux_system_generation <= grid.system_generation)
        grid.generation_ub_binary = pyomo.Constraint(
            doc="Auxiliary variable upper bound with binary",
            expr=grid.aux_system_generation <= 1.5 * grid.transmission_limit * grid.is_system_producing)
        grid.generation_lb_binary = pyomo.Constraint(
            doc="Auxiliary variable lower bound with binary",
            expr=grid.aux_system_generation >= (grid.system_generation
                                                - 1.5 * grid.transmission_limit * (1 - grid.is_system_producing)))
        # Aux system load variable
        grid.load_ub = pyomo.Constraint(
            doc="Auxiliary variable upper bound",
            expr=grid.aux_system_load <= grid.system_load)
        grid.load_ub_binary = pyomo.Constraint(
            doc="Auxiliary variable upper bound with binary",
            expr=grid.aux_system_load <= 1.5 * grid.transmission_limit * grid.is_system_producing)
        grid.load_lb_binary = pyomo.Constraint(
            doc="Auxiliary variable lower bound with binary",
            expr=grid.aux_system_load >= (grid.system_load
                                          - 1.5 * grid.transmission_limit * (1 - grid.is_system_producing)))

    @staticmethod
    def _create_grid_ports(grid):
        ##################################
        # Ports                          #
        ##################################
        grid.port = Port()
        grid.port.add(grid.system_generation)
        grid.port.add(grid.system_load)
        grid.port.add(grid.electricity_sales)
        grid.port.add(grid.electricity_purchases)

    def initialize_dispatch_model_parameters(self):
        grid_limit_kw = self._system_model.value('grid_interconnection_limit_kwac')
        self.transmission_limit = [grid_limit_kw / 1e3] * len(self.blocks.index_set())

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        n_horizon = len(self.blocks.index_set())
        dispatch_factors = self._financial_model.value("dispatch_factors_ts")
        ppa_price = self._financial_model.value("ppa_price_input")[0]
        if start_time + n_horizon > len(dispatch_factors):
            prices = list(dispatch_factors[start_time:])
            prices.extend(list(dispatch_factors[0:n_horizon - len(prices)]))
        else:
            prices = dispatch_factors[start_time:start_time + n_horizon]
        # NOTE: Assuming the same prices
        self.electricity_sell_price = [norm_price * ppa_price * 1e3 for norm_price in prices]
        self.electricity_purchase_price = [norm_price * ppa_price * 1e3 for norm_price in prices]

    @property
    def electricity_sell_price(self) -> list:
        return [self.blocks[t].electricity_sell_price.value for t in self.blocks.index_set()]

    @electricity_sell_price.setter
    def electricity_sell_price(self, price_per_mwh: list):
        if len(price_per_mwh) == len(self.blocks):
            for t, price in zip(self.blocks, price_per_mwh):
                self.blocks[t].electricity_sell_price.set_value(round(price, self.round_digits))
        else:
            raise ValueError("'price_per_mwh' list must be the same length as time horizon")

    @property
    def electricity_purchase_price(self) -> list:
        return [self.blocks[t].electricity_purchase_price.value for t in self.blocks.index_set()]

    @electricity_purchase_price.setter
    def electricity_purchase_price(self, price_per_mwh: list):
        if len(price_per_mwh) == len(self.blocks):
            for t, price in zip(self.blocks, price_per_mwh):
                self.blocks[t].electricity_purchase_price.set_value(round(price, self.round_digits))
        else:
            raise ValueError("'price_per_mwh' list must be the same length as time horizon")

    @property
    def transmission_limit(self) -> list:
        return [self.blocks[t].transmission_limit.value for t in self.blocks.index_set()]

    @transmission_limit.setter
    def transmission_limit(self, limit_mw: list):
        if len(limit_mw) == len(self.blocks):
            for t, limit in zip(self.blocks, limit_mw):
                self.blocks[t].transmission_limit.set_value(round(limit, self.round_digits))
        else:
            raise ValueError("'limit_mw' list must be the same length as time horizon")

    @property
    def system_generation(self) -> list:
        return [self.blocks[t].system_generation.value for t in self.blocks.index_set()]

    @system_generation.setter
    def system_generation(self, system_gen_mw: list):
        if len(system_gen_mw) == len(self.blocks):
            for t, gen in zip(self.blocks, system_gen_mw):
                self.blocks[t].system_generation.set_value(round(gen, self.round_digits))
        else:
            raise ValueError("'system_gen_mw' list must be the same length as time horizon")

    @property
    def system_load(self) -> list:
        return [self.blocks[t].system_load.value for t in self.blocks.index_set()]

    @system_load.setter
    def system_load(self, system_load_mw: list):
        if len(system_load_mw) == len(self.blocks):
            for t, load in zip(self.blocks, system_load_mw):
                self.blocks[t].system_load.set_value(round(load, self.round_digits))
        else:
            raise ValueError("'system_load_mw' list must be the same length as time horizon")

    @property
    def electricity_sold(self) -> list:
        return [self.blocks[t].electricity_sold.value for t in self.blocks.index_set()]

    @property
    def electricity_purchased(self) -> list:
        return [self.blocks[t].electricity_purchased.value for t in self.blocks.index_set()]

    @property
    def electricity_sales(self) -> list:
        return [self.blocks[t].electricity_sales.value for t in self.blocks.index_set()]

    @property
    def electricity_purchases(self) -> list:
        return [self.blocks[t].electricity_purchases.value for t in self.blocks.index_set()]

