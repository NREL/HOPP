import pyomo.environ as pyomo
from pyomo.environ import units as u


class Dispatch:
    """

    """
    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model,
                 financial_model,
                 block_set_name: str = 'dispatch'):

        try:
            u.USD
        except AttributeError:
            u.load_definitions_from_strings(['USD = [currency]'])

        self.block_set_name = block_set_name
        self.round_digits = int(4)

        self._model = pyomo_model
        self._blocks = pyomo.Block(index_set, rule=self.dispatch_block_rule)
        setattr(self.model, self.block_set_name, self.blocks)

        self._system_model = system_model
        self._financial_model = financial_model

    @staticmethod
    def dispatch_block_rule(block, t):
        raise NotImplemented("This function must be overridden for specific dispatch model")

    def initialize_dispatch_model_parameters(self):
        raise NotImplemented("This function must be overridden for specific dispatch model")

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        raise NotImplemented("This function must be overridden for specific dispatch model")

    @property
    def blocks(self) -> pyomo.Block:
        return self._blocks

    @property
    def model(self) -> pyomo.ConcreteModel:
        return self._model
