from pyomo.environ import ConcreteModel, Set

from hybrid.dispatch.power_source_dispatch import PowerSourceDispatch


class SolarDispatch(PowerSourceDispatch):
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 block_set_name: str = 'pv'):
        super().__init__(pyomo_model, indexed_set, block_set_name=block_set_name)

