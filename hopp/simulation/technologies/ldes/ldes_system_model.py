from attrs import define, field, validators

from hopp.simulation.base import BaseClass
from tests.hopp.utils import DEFAULT_FIN_CONFIG

@define
class LDES(BaseClass):
    valid_chemistry_options = ["LDES", "AES"]
    chemistry: str = field(default=None, validator=validators.in_(valid_chemistry_options))

    @classmethod
    def default(cls, chemistry):

        return cls(chemistry)
    

class LDESTools:

    dummy: 0