from hybrid.layout.detailed_pv_constants import *
from hybrid.layout.detailed_pv_config import *
from hybrid.layout.pv_design_utils import *
from hybrid.layout.pv_layout import *
from PySAM import Pvwattsv8

class DetailedPVParameters(NamedTuple):
    """
    All variables in the design vector for the PV Layout

    x_position: ratio of solar's x coords to site width (0, 1)
    y_position: ratio of solar's y coords to site height (0, 1)
    aspect_power: aspect ratio of solar to site width = 2^solar_aspect_power
    s_buffer: south side buffer ratio (0, 1)
    x_buffer: east and west side buffer ratio (0, 1)
    gcr: (0.1, 1)
    azimuth: (0, 180)
    tilt_tracker_angle: for 0-axis tracking, the module tilt; 1-axis tracking, the tracker's rotation limit
    string_voltage_ratio: relative position of string voltage within MPPT voltage window (0, 1)
    dc_ac_ratio: target dc_ac_ratio

    TODO: Add any other layout variables that should be optimized in the optimization loop
    """
    x_position: float
    y_position: float
    aspect_power: float
    s_buffer: float
    x_buffer: float
    gcr: float
    azimuth: float
    tilt_tracker_angle: float
    string_voltage_ratio: float
    dc_ac_ratio: int


class DetailedPVLayout(PVLayout):
    """
    This class creates a PV Layout design consisting of a Solar Region polygon, a set of roads, a substation, a flicker loss
    and module-inverter counts for a single subarray. The design can be expanded to include rack configs and locations, 
    a set of wires, any losses, other power electronics, etc.

    The PV layout design is calculated using the inputs from PVLayoutConfig and from the design vector DetailedPVParameters
    so that the layout best matches the target design vector

    Layout is not computed during construction, but happens during these function calls:
        1. `compute_pv_layout`: this uses the existing DetailedPVParameters
        2. `set_layout_params`: this uses new DetailedPVParameters

    """
    def __init__(self, 
                 site_info: SiteInfo, 
                 solar_source: pv_detailed.Pvsamv1, 
                 parameters: DetailedPVParameters, 
                 config: PVLayoutConfig):
        self.site: SiteInfo = site_info
        self._system_model: pv_detailed.Pvsamv1 = solar_source
        self.config = config
        self.parameters = parameters
        self.module_power = None
        self.num_modules = None
        
        # Design Outputs
        self.solar_region = None
        self.roads = None
        self.substation_coord = None
        self.nstrings = None
        self.modules_per_string = None
        self.ninverters = None
        self.flicker_loss = 0
    
    def _compute_string_config(self):
        """
        Compute the modules_per_string, ninverters and nstrings to fit the target solar capacity, dc_ac_ratio and relative_string_voltage
        """
        self.modules_per_string = find_target_string_voltage(self._system_model, self.parameters.string_voltage_ratio)
        self.nstrings, self.ninverters = find_target_dc_ac_ratio(self._system_model, self.parameters.dc_ac_ratio)

    def _set_system_layout(self):
        """
        Sets all Pvsamv1 variables using computed layout's variables, so that any future yield simulation has up-to-date values
        """
        self._system_model.SystemDesign.subarray1_gcr = self.parameters.gcr
        self._system_model.SystemDesign.subarray1_azimuth = self.parameters.azimuth
        # other vars ..
        self._system_model.SystemDesign.subarray1_nstrings = self.nstrings
        # other vars ...
        self._system_model.AdjustmentFactors.constant = self.flicker_loss * 100  # percent

    def compute_pv_layout(self,
                        target_solar_kw: float):
        """
        Internal function computes the layout using the config and design variables to fit the target capacity
        """
        if isinstance(self._system_model, Pvwattsv8.Pvwattsv8):
            self.module_power = self.config['module_power']
            self.num_modules = round(target_solar_kw / self.module_power)
        else:
            self._compute_string_config(target_solar_kw)

        # find where the solar_region should be centered
        
        # create a solar region that fits the config and design and capacity
        self.solar_region = None

        # create the roads, substation

        # create any additional objects

        # update design output properties

        # pass design output values to yield model
        if isinstance(self._system_model, Pvwattsv8.Pvwattsv8):
            super()._set_system_layout()
        else:
            self._set_system_layout()

        return self.solar_region

    def set_layout_params(self,
                          solar_kw: float,
                          params: DetailedPVParameters):
        """
        Function computes the layout given the target capacity and the layout parameters
        """
        self.parameters = params
        self.compute_pv_layout(solar_kw)

    def set_flicker_loss(self,
                         flicker_loss_multipler: float):
        self.flicker_loss = flicker_loss_multipler
        self._set_system_layout()

    def plot(self,
             figure=None,
             axes=None,
             solar_color='darkorange',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0):
        pass