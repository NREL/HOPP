import pyomo.environ as pyomo
from pyomo.network import Port
from pyomo.environ import units as u

try:
    u.USD
except AttributeError:
    u.load_definitions_from_strings(['USD = [currency]'])


class PowerSourceDispatch:
    """

    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 block_set_name: str = 'gen'):
        self._model = pyomo_model
        self._blocks = pyomo.Block(index_set, rule=self.dispatch_block_rule)
        setattr(self.model, block_set_name, self.blocks)

        self.block_set_name = block_set_name
        self.round_digits = int(6)

    @staticmethod
    def dispatch_block_rule(gen):
        ##################################
        # Parameters                     #
        ##################################
        gen.time_weighting_factor = pyomo.Param(doc="Exponential time weighting factor [-]",
                                                initialize=1.0,
                                                within=pyomo.PercentFraction,
                                                mutable=True,
                                                units=u.dimensionless)
        gen.time_duration = pyomo.Param(doc="Time step [hour]",
                                        default=1.0,
                                        within=pyomo.NonNegativeReals,
                                        mutable=True,
                                        units=u.hr)
        gen.electricity_sell_price = pyomo.Param(doc="Electricity sell price [$/MWh]",
                                                 default=0.0,
                                                 within=pyomo.Reals,
                                                 mutable=True,
                                                 units=u.USD / u.MWh)
        gen.generation_cost = pyomo.Param(doc="Generation cost for generator [$/MWh]",
                                          default=0.0,
                                          within=pyomo.NonNegativeReals,
                                          mutable=True,
                                          units=u.USD / u.MWh)
        gen.available_generation = pyomo.Param(doc="Available generation for the generator [MW]",
                                               default=0.0,
                                               within=pyomo.NonNegativeReals,
                                               mutable=True,
                                               units=u.MW)
        ##################################
        # Variables                      #
        ##################################
        gen.generation = pyomo.Var(doc="Power generation of generator [MW]",
                                   domain=pyomo.NonNegativeReals,
                                   bounds=(0, gen.available_generation),
                                   units=u.MW)
        gen.gross_profit = pyomo.Var(doc="Sub-system gross profit [USD]",
                                     domain=pyomo.Reals,
                                     units=u.USD)
        ##################################
        # Constraints                    #
        ##################################
        gen.gross_profit_calculation = pyomo.Constraint(
            doc="Calculation of gross profit for objective function with time weighting factor",
            expr=(gen.gross_profit == (gen.time_weighting_factor * gen.electricity_sell_price
                                       - (1/gen.time_weighting_factor) * gen.generation_cost
                                       ) * gen.time_duration * gen.generation))
        ##################################
        # Ports                          #
        ##################################
        gen.port = Port()
        gen.port.add(gen.generation)

    def gross_profit_objective_rule(self, m):
        return sum(self.blocks[t].gross_profit for t in self.blocks.index_set())

    def create_gross_profit_objective(self):
        self.model.gross_profit_objective = pyomo.Objective(rule=self.gross_profit_objective_rule, sense=pyomo.maximize)

    def delete_gross_profit_objective(self):
        self.model.del_component(self.model.gross_profit_objective)

    @property
    def time_weighting_factor(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t+1].time_weighting_factor.value

    @time_weighting_factor.setter
    def time_weighting_factor(self, weighting: float):
        for t in self.blocks.index_set():
            self.blocks[t].time_weighting_factor = weighting ** (self.blocks[t].time_duration.value * t)

    @property
    def time_weighting_factor_list(self) -> list:
        return [self.blocks[t].time_weighting_factor.value for t in self.blocks.index_set()]

    @property
    def time_duration(self) -> list:
        return [self.blocks[t].time_duration.value for t in self.blocks.index_set()]

    @time_duration.setter
    def time_duration(self, time_steps_hr: list):
        if len(time_steps_hr) == len(self.blocks):
            for t, delta in zip(self.blocks, time_steps_hr):
                self.blocks[t].time_duration = round(delta, self.round_digits)
        else:
            raise ValueError("'time_steps_hr' list must be the same length as time horizon")

    @property
    def electricity_sell_price(self) -> list:
        return [self.blocks[t].electricity_sell_price.value for t in self.blocks.index_set()]

    @electricity_sell_price.setter
    def electricity_sell_price(self, prices_dollar_per_mwh: list):
        if len(prices_dollar_per_mwh) == len(self.blocks):
            for t, price in zip(self.blocks, prices_dollar_per_mwh):
                self.blocks[t].electricity_sell_price = round(price, self.round_digits)
        else:
            raise ValueError("'prices_dollar_per_mwh' list must be the same length as time horizon")

    @property
    def generation_cost(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].generation_cost.value

    @generation_cost.setter
    def generation_cost(self, om_dollar_per_mwh: float):
        for t in self.blocks.index_set():
            self.blocks[t].generation_cost = round(om_dollar_per_mwh, self.round_digits)

    @property
    def available_generation(self) -> list:
        return [self.blocks[t].available_generation.value for t in self.blocks.index_set()]

    @available_generation.setter
    def available_generation(self, resource: list):
        if len(resource) == len(self.blocks):
            for t, gen in zip(self.blocks, resource):
                self.blocks[t].available_generation = round(gen, self.round_digits)
        else:
            raise ValueError("'resource' list must be the same length as time horizon")

    @property
    def generation(self) -> list:
        return [round(self.blocks[t].generation.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def gross_profit(self) -> list:
        return [round(self.blocks[t].gross_profit.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def gross_profit_objective(self):
        return pyomo.value(self.model.gross_profit_objective)

    @property
    def blocks(self) -> pyomo.Block:
        return self._blocks

    @property
    def model(self) -> pyomo.ConcreteModel:
        return self._model

