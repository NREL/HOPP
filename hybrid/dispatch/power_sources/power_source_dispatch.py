import pyomo.environ as pyomo
from pyomo.network import Port
from pyomo.environ import units as u

from hybrid.dispatch.dispatch import Dispatch


class PowerSourceDispatch(Dispatch):
    """

    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model,
                 financial_model,
                 block_set_name: str = 'generator'):
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name)

    @staticmethod
    def dispatch_block_rule(gen):
        ##################################
        # Parameters                     #
        ##################################
        gen.time_duration = pyomo.Param(
            doc="Time step [hour]",
            default=1.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.hr)
        gen.cost_per_generation = pyomo.Param(
            doc="Generation cost for generator [$/MWh]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.USD / u.MWh)
        gen.available_generation = pyomo.Param(
            doc="Available generation for the generator [MW]",
            default=0.0,
            within=pyomo.Reals,
            mutable=True,
            units=u.MW)
        ##################################
        # Variables                      #
        ##################################
        gen.generation = pyomo.Var(
            doc="Power generation of generator [MW]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, gen.available_generation),
            units=u.MW)
        ##################################
        # Constraints                    #
        ##################################
        ##################################
        # Ports                          #
        ##################################
        gen.port = Port()
        gen.port.add(gen.generation)

    def initialize_parameters(self):
        self.cost_per_generation = self._financial_model.value("om_capacity")[0]*1e3/8760

    def update_time_series_parameters(self, start_time: int):
        n_horizon = len(self.blocks.index_set())
        generation = self._system_model.value("gen")
        if start_time + n_horizon > len(generation):
            horizon_gen = list(generation[start_time:])
            horizon_gen.extend(list(generation[0:n_horizon - len(horizon_gen)]))
        else:
            horizon_gen = generation[start_time:start_time + n_horizon]

        if len(horizon_gen) < len(self.blocks):
            raise RuntimeError(f"Dispatch parameter update error at start_time {start_time}: System model "
                               f"{type(self._system_model)} generation profile should have at least {len(self.blocks)} "
                               f"length but has only {len(generation)}")
        self.available_generation = [gen_kw / 1e3 for gen_kw in horizon_gen]

    @property
    def cost_per_generation(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_generation.value

    @cost_per_generation.setter
    def cost_per_generation(self, om_dollar_per_mwh: float):
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_generation.set_value(round(om_dollar_per_mwh, self.round_digits))

    @property
    def available_generation(self) -> list:
        return [self.blocks[t].available_generation.value for t in self.blocks.index_set()]

    @available_generation.setter
    def available_generation(self, resource: list):
        if len(resource) == len(self.blocks):
            for t, gen in zip(self.blocks, resource):
                self.blocks[t].available_generation.set_value(round(gen, self.round_digits))
        else:
            raise ValueError(f"'resource' list ({len(resource)}) must be the same length as time horizon ({len(self.blocks)})")

    @property
    def generation(self) -> list:
        return [round(self.blocks[t].generation.value, self.round_digits) for t in self.blocks.index_set()]



