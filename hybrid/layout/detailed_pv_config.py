import warnings
from dataclasses import dataclass, astuple, asdict

@dataclass
class PVLayoutConfig:
    """
    This class contains all the detailed PV layout configuration variables that are fixed for a given design

    These variables include all the pvsamv1 variables that are fixed and known at the beginning; 
    and all other layout variables that are needed to compute the layout and any pvsamv1 variables.

    This dataclass can be populated from a dictionary that's loaded from file:
        e.g. `PVLayoutConfig(**layout_config_dict)`

    This dataclass will be used to construct the DetailedPVPlant, and passed along to DetailedPVLayout

    Module and inverter data is not stored here, but this could be changed

    Any pvsamv1 variables besides module and inverter variables that are not listed here will remain unassigned in the yield model. 
    Some pvsamv1 variables have default-if-not-assigned values, while others are required and will cause an exception if not provided.
    The former kind of variable should be checked to see if they are worth adding to this class, while the
    second kind should be added to this class.
    """

    # Subarray config
    module_power: float         # kW
    module_width: float         # m
    module_height: float        # m
    subarray1_nmodx: int        # count
    subarray1_nmody: int        # count
    subarray1_track_mode: int   # 0 for fixed, 1 for 1-axis tracking
    # subarray1_modules_per_string

    # Inverter config
    nb_inputs_inverter: int     # count

    # Rack config
    interrack_spac: float       # m

    # Wiring config

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
