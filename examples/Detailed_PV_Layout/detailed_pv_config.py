import warnings
from dataclasses import dataclass, astuple, asdict

@dataclass
class PVLayoutConfig:
    """
    This class contains configuration parameters for constructing the example detailed PV layout class DetailedPVLayout.
    These parameters are needed to compute the layout, are fixed for a given design and known at initialization.

    This dataclass can be populated from a dictionary that's loaded from file:
        e.g. `PVLayoutConfig(**layout_config_dict)`

    The module and inverter data stored here are for use with the simple technology/yield model (Pvwattsv8). If the detailed
    yield model is used (Pvsamv1), then the module and inverter data from that model are used instead of these.

    This PVLayoutConfig class and the associated DetailedPVLayout class are basic examples for computing a detailed PV
    layout. More features can be added like wire routing, road construction and land border setbacks, but those are not
    implemented here and would likely require more parameters to be added to this class.
    """

    # Subarray config
    module_power: float         # kW
    module_width: float         # m
    module_height: float        # m
    subarray1_nmodx: int        # count
    subarray1_nmody: int        # count
    subarray1_track_mode: int   # 0 for fixed, 1 for 1-axis tracking
    subarray1_modules_per_string: int

    # Inverter config
    inverter_power: int         # kW
    nb_inputs_inverter: int     # count

    # Rack config
    interrack_spac: float       # m

    # Wiring config
    nb_inputs_combiner: int     # count

    # Road config
    perimetral_road: bool   
    setback_distance: float     # m

    def __iter__(self):
        yield from astuple(self)

    def items(self):
        return asdict(self).items()

    def __getitem__(self, item):
        return getattr(self, item)

    def __post_init__(self):
        self.setup()
        self.verify()

    def verify(self):
        """
        This function verifies that all required inputs are provided and are within expected ranges and types.
        """
        print("verify")
        if self.subarray1_track_mode > 1:
            warnings.warn(f"Warning: Track modes besides fixed and 1-axis have not been tested.")

    def setup(self):
        """
        This function computes any intermediate variables from the input variables
        This could also fill missing variables with defaults
        """
        print("setup")
