from attrs import define, field, validators, fields_dict, setters

from hopp.simulation.base import BaseClass
from hopp.utilities.validators import range_val

@define
class Params:
    cp: float 
    h: float
    nominal_energy: float # nominal installed energy [kWh]
    nominal_voltage: float # nominal DC voltage [V]
    mass: float 
    surface_area: float

@define
class LDES(BaseClass):
    valid_chemistry_options = ["LDES", "AES"]
    chemistry: str = field(default=None, validator=validators.in_(valid_chemistry_options))

    valid_control_modes = [0.0]
    control_mode: float = field(default=0.0, validator=validators.in_(valid_control_modes))

    dt_hr: float = field(default=1.0, validator=validators.gt(0.0))
    minimum_SOC: float = field(default=10, validator=range_val(0, 100))
    maximum_SOC: float = field(default=90, validator=range_val(0, 100))
    initial_SOC: float = field(default=10, validator=range_val(0, 100))

    system_capacity_kw: float = field(default=None)
    system_capacity_kwh: float = field(default=None)

    @classmethod
    def default(cls, chemistry):
        return cls(chemistry=chemistry)
    
    def __attrs_post_init__(self):
        """Auto-populate _parameters with class attributes on initialization. Method generated using ChatGPT"""
        self._parameters = {name: getattr(self, name) for name in fields_dict(self.__class__) if not name.startswith("_")}

    def value(self, key: str, val=None):
        """Getter and setter for attributes using _parameters. Method generated using ChatGPT"""
        if val is None:
            return getattr(self, key)  # Get the attribute directly
        setattr(self, key, val)  # Set attribute, which auto-updates _parameters

    def export(self):
        """Export class data as a dictionary. Method generated using ChatGPT"""
        # Create a dictionary of the instance's attributes
        data = {name: getattr(self, name) for name in fields_dict(self.__class__) if not name.startswith("_")}
        return data
    
    def model_type(self):
        if self.chemistry == "LDES":
            return 0
        elif self.chemistry == "AES":
            return 1
        
    def sizing(self, rating_kw, rating_kwh):
        self.system_capacity_kw = rating_kw
        self.system_capacity_kwh = rating_kwh

    @property
    def nominal_voltage() -> float:
        return None
    
    # @ property
    # def footprint_area() -> float:
    #     return None
    
    # @property
    # def system_mas() -> float:
    #     return None

class LDESTools:

    dummy: 0