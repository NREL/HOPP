from hybrid.layout.detailed_pv_config import *
from hybrid.layout.pv_design_utils import *
from hybrid.layout.pv_layout import *

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
                 parameters: DetailedPVParameters,
                 config: PVLayoutConfig,
                 solar_source: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1]=None):
        self.site: SiteInfo = site_info
        self._system_model = solar_source
        self.config = config
        self.parameters = parameters
        
        # Design Outputs
        self.solar_region = None
        self.roads = None
        self.substation_coord = None
        self.modules_per_string = None
        self.n_strings = None
        self.n_inverters = None
        self.calculated_system_capacity = None
        self.flicker_loss = 0
    
    def set_layout_params(self,
                          solar_kw: float,
                          params: DetailedPVParameters):
        """
        Function computes the layout given the target capacity and the layout parameters
        """
        if self._system_model is None:
            raise Exception('Detailed PV layout not initialized with system model reference.')

        self.parameters = params
        self.compute_pv_layout(solar_kw)


    def compute_pv_layout(self,
                          target_solar_kw: float):
        """
        Internal function computes the layout using the config and design variables to fit the target capacity
        """
        if self._system_model is None:
            raise Exception('Detailed PV layout not initialized with system model reference.')

        self._compute_string_config(target_solar_kw)

        # find where the solar_region should be centered
        
        # create a solar region that fits the config and design and capacity
        self.solar_region = None

        # create the roads, substation

        # create any additional objects

        # update design output properties

        # pass design output values to yield model
        self._set_system_layout()

        return self.solar_region


    def set_system_capacity(self, size_kw):
        """
        Overrides base function to call detailed layout instead of simple layout
        """
        return self.compute_pv_layout(size_kw)


    def _compute_string_config(self,
                               target_solar_kw: float):
        """
        Compute the modules_per_string, ninverters and nstrings to fit the target solar capacity, dc_ac_ratio and relative_string_voltage
        """
        if isinstance(self._system_model, pv_detailed.Pvsamv1):
            module_attribs = get_module_attribs(self._system_model)
            inverter_attribs = get_inverter_attribs(self._system_model)
            self.modules_per_string = find_modules_per_string(
                v_mppt_min=inverter_attribs['V_mppt_min'],
                v_mppt_max=inverter_attribs['V_mppt_max'],
                v_mp_module=module_attribs['V_mp_ref'],
                v_oc_module=module_attribs['V_oc_ref'],
                inv_vdcmax=inverter_attribs['V_dc_max'],
                target_relative_string_voltage=self.parameters.string_voltage_ratio,
            )
            self.n_strings, self.n_inverters, self.calculated_system_capacity = size_electrical_parameters(
                target_system_capacity=target_solar_kw,
                target_dc_ac_ratio=self.parameters.dc_ac_ratio,
                modules_per_string=self.modules_per_string,
                module_power=get_module_power(self._system_model),
                inverter_power=get_inverter_power(self._system_model),
                n_inputs_inverter=self.config['nb_inputs_inverter']
            )
        else:   # PVWattsv8
            self.modules_per_string = self.config.subarray1_modules_per_string
            self.n_strings, self.n_inverters, self.calculated_system_capacity = size_electrical_parameters(
                target_system_capacity=target_solar_kw,
                target_dc_ac_ratio=self.parameters.dc_ac_ratio,
                modules_per_string=self.modules_per_string,
                module_power=self.config.module_power,
                inverter_power=self.config.inverter_power,
                n_inputs_inverter=self.config.nb_inputs_inverter
            )


    def _set_system_layout(self):
        """
        Sets all Pvsamv1 variables using computed layout's variables, so that any future yield simulation has up-to-date values
        """
        if isinstance(self._system_model, pv_detailed.Pvsamv1):
            self._system_model.value('subarray1_modules_per_string', self.modules_per_string)
            self._system_model.value('subarray1_nstrings', self.n_strings)
            self._system_model.value('inverter_count', self.n_inverters)
            self._system_model.value('system_capacity', self.calculated_system_capacity)
            self._system_model.value('subarray1_gcr', self.parameters.gcr)
            self._system_model.value('subarray1_azimuth', self.parameters.azimuth)
            if self._system_model.value('subarray1_track_mode') == 0:
                self._system_model.value('subarray1_tilt', self.parameters.tilt_tracker_angle)
            elif self._system_model.value('subarray1_track_mode') == 1:
                self._system_model.value('subarray1_rotlim', self.parameters.tilt_tracker_angle)
            self._system_model.value('constant', self.flicker_loss * 100)  # percent
            self._system_model.value('subarray2_enable', 0)
            self._system_model.value('subarray3_enable', 0)
            self._system_model.value('subarray4_enable', 0)
        else:   # PVWatts
            self._system_model.value('system_capacity', self.calculated_system_capacity)

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
