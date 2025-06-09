from typing import TYPE_CHECKING, Dict, Optional
import PySAM.MhkCosts as MhkCost

from attrs import define, field
from hopp.simulation.base import BaseClass

from hopp.utilities.validators import gt_zero, range_val

# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.wave.mhk_wave_plant import MHKConfig

@define
class MHKCostModelInputs(BaseClass):
    """
    Configuration class for MHK Cost Model.
        
    Args:
        reference_model_num: Reference model number from the 
            Department of Energy Reference Model Project
            (1, 3, 5, or 6).
        water_depth: Water depth in meters
        distance_to_shore: Distance to shore in meters
        number_rows: Number of rows in the device layout
        row_spacing: Spacing between rows in meters
            (default 'device_spacing')
        cable_system_overbuild: Cable system overbuild percentage
            (default 10%)
    Note:
        More information about the reference models and their 
        associated costs can be found in the 
        [Reference Model Project](https://energy.sandia.gov/programs/renewable-energy/water-power/projects/reference-model-project-rmp/)
        
        The supported reference models in this cost model are:
            - Reference Model 1: Tidal Current Turbine
            - Reference Model 3: Wave Point Absorber
            - Reference Model 5: Oscillating Surge Flap
            - Reference Model 6: Oscillating Water Column
        
        Additional MHK cost model information can be found 
        through the [System Advisor Model](https://sam.nrel.gov/)
    """
    reference_model_num: int
    water_depth: float = field(validator=gt_zero)
    distance_to_shore: float = field(validator=gt_zero)
    number_rows: int = field(validator=gt_zero)
    device_spacing: float = field(validator=gt_zero)
    row_spacing: Optional[float] = field(default=None)
    cable_system_overbuild: float = field(default=10., validator=range_val(0, 100))
    

@define
class MHKCosts(BaseClass):
    """
    A class for calculating the costs associated with Marine Hydrokinetic (MHK) energy systems.

    This class initializes and configures cost calculations for MHK systems based on provided input parameters.
    It uses the PySAM library for cost modeling which is based on the [Department of Energy Reference Model Project](https://energy.sandia.gov/programs/renewable-energy/water-power/projects/reference-model-project-rmp/).
    
        Args:
            mhk_config: MHK system configuration parameters.
            cost_model_inputs: Input parameters for cost modeling.

        Raises:
            ValueError: If any of the required keys in `mhk_config` or
                `cost_model_inputs` are missing.
    """
    mhk_config: "MHKConfig"
    cost_model_inputs: MHKCostModelInputs

    _device_rated_power: float = field(init=False)
    _number_devices: int = field(init=False)
    _water_depth: float = field(init=False)
    _distance_to_shore: float = field(init=False)
    _number_rows: int = field(init=False)
    _ref_model_num: str = field(init=False)
    _device_spacing: float = field(init=False)
    _row_spacing: float = field(init=False)
    _cable_sys_overbuild: float = field(init=False)

    def __attrs_post_init__(self):
        self._cost_model = MhkCost.new()

        self._device_rated_power = self.mhk_config.device_rating_kw
        self._number_devices = self.mhk_config.num_devices
        self._water_depth = self.cost_model_inputs.water_depth
        self._distance_to_shore = self.cost_model_inputs.distance_to_shore
        self._number_rows = self.cost_model_inputs.number_rows
        self._device_spacing = self.cost_model_inputs.device_spacing
        self._cable_sys_overbuild = self.cost_model_inputs.cable_system_overbuild
        
        ref_model_numbers = {1,3,5,6}
        if self.cost_model_inputs.reference_model_num in ref_model_numbers:
            self._ref_model_num = f"RM{self.cost_model_inputs.reference_model_num}"
        else:
            raise ValueError("reference_model_num can be 1, 3, 5 or 6")

        if self.cost_model_inputs.row_spacing is None:
            self._row_spacing = self.cost_model_inputs.device_spacing
        else:
            self._row_spacing = self.cost_model_inputs.row_spacing

        # Define a mapping of cost keys to their respective method and input keys
        cost_keys_mapping: Dict[str, tuple] = {
            'structural_assembly_cost_kw': ('structural_assembly_cost_method', 'structural_assembly_cost_input'),
            'power_takeoff_system_cost_kw': ('power_takeoff_system_cost_method', 'power_takeoff_system_cost_input'),
            'mooring_found_substruc_cost_kw': ('mooring_found_substruc_cost_method', 'mooring_found_substruc_cost_input'),
            'development_cost_kw': ('development_cost_method', 'development_cost_input'),
            'eng_and_mgmt_cost_kw': ('eng_and_mgmt_cost_method', 'eng_and_mgmt_cost_input'),
            'assembly_and_install_cost_kw': ('assembly_and_install_cost_method', 'assembly_and_install_cost_input'),
            'other_infrastructure_cost_kw': ('other_infrastructure_cost_method', 'other_infrastructure_cost_input'),
            'array_cable_system_cost_kw': ('array_cable_system_cost_method', 'array_cable_system_cost_input'),
            'export_cable_system_cost_kw': ('export_cable_system_cost_method', 'export_cable_system_cost_input'),
            'onshore_substation_cost_kw': ('onshore_substation_cost_method', 'onshore_substation_cost_input'),
            'offshore_substation_cost_kw': ('offshore_substation_cost_method', 'offshore_substation_cost_input'),
            'other_elec_infra_cost_kw': ('other_elec_infra_cost_method', 'other_elec_infra_cost_input')
        }

        # Loop through the cost keys and set the values accordingly
        for cost_key, (method_key, input_key) in cost_keys_mapping.items():
            attr = getattr(self.cost_model_inputs, cost_key, None)
            if attr is None:
                self._cost_model.value(method_key, 2)
                self._cost_model.value(input_key, 0)
            else:
                self._cost_model.value(method_key, 0)
                self._cost_model.value(input_key, attr)
    
        self.initialize()

    def initialize(self):
        self._cost_model.value("device_rated_power", self._device_rated_power)
        self._cost_model.value("system_capacity", self._cost_model.value("device_rated_power") * self._number_devices) 
        
        if self._row_spacing is None:
            raise AttributeError("row_spacing must be provided")

        if self._number_devices < self._number_rows:
            raise Exception("number_of_rows exceeds number_devices")
        else:
            if (self._number_devices/self._number_rows).is_integer():
                self._cost_model.value("devices_per_row", \
                    self._number_devices / self._number_rows)
            else:
                raise Exception("Layout must be square or rectangular. Modify 'number_rows' or 'num_devices'.")
        self._cost_model.value("lib_wave_device", self._ref_model_num)
        if self._ref_model_num == "RM3" or self._ref_model_num == "RM5" or self._ref_model_num == "RM6":
            self._cost_model.value("marine_energy_tech", 0) # Wave
        elif self._ref_model_num == "RM1":
            self._cost_model.value('marine_energy_tech',1) # Tidal
            self._cost_model.value('lib_tidal_device', self._ref_model_num)
        else:
            self._cost_model.value("marine_energy_tech", 0) # Generic
        self._cost_model.value("library_or_input_wec", 0)
        # Inter-array cable length, m
        # The total length of cable used within the array of devices
        self._array_cable_length = (self._cost_model.value("devices_per_row") -1) * \
            (self._device_spacing * self._number_rows) \
            + self._row_spacing * (self._number_rows-1)
        self._cost_model.value("inter_array_cable_length", self._array_cable_length)

        # Export cable length, m
        # The length of cable between the array and onshore grid connection point
        self._export_cable_length = (self._water_depth + self._distance_to_shore) * \
            (1 + self._cable_sys_overbuild/100)
        self._cost_model.value("export_cable_length", self._export_cable_length)

        #Riser cable length, m
        # The length of cable from the seabed to the water surface that 
        # connects the floating device to the seabed cabling.
        # Applies only to floating array
        self.riser_cable_length = 1.5 * self._water_depth * self._number_devices * \
            (1 + self._cable_sys_overbuild/100)
        self._cost_model.value("riser_cable_length", self.riser_cable_length)

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices.
        """
        new_num_devices = round(wave_size_kw / self._device_rated_power)
        if self._number_devices != new_num_devices:
            self._number_devices = new_num_devices
            self.initialize()

    def simulate_costs(self):
        # TODO: what is the correct int_verbosity?
        self._cost_model.execute(1)

    @property
    def device_rated_power(self):
        return self._device_rated_power

    @device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._device_rated_power = device_rate_power
        self.initialize()

    @property
    def number_devices(self):
        return self._number_devices

    @number_devices.setter
    def number_devices(self, number_devices: int):
        self._number_devices = number_devices
        self.initialize()

    @property
    def system_capacity_kw(self):
        self._cost_model.value("system_capacity", self._cost_model.value("device_rated_power") * self._number_devices)
        return self._cost_model.value("system_capacity")

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity by updates the number of wave devices using
        device rating.
        """
        self.system_capacity_by_num_devices(size_kw)

    @property
    def ref_model_num(self):
        return self._ref_model_num

    @ref_model_num.setter
    def ref_model_num(self, ref_model_number: int):
        model_numbers = {1,3,5,6}
        if ref_model_number in model_numbers:
            self._ref_model_num = f"RM{ref_model_number}"
            self.initialize()
        else:
            raise ValueError(f"Reference model number {ref_model_number} is not supported. Choose from {model_numbers}.")
    
    @property
    def library_or_input_wec(self):
        return self._cost_model.value("library_or_input_wec")

    @library_or_input_wec.setter
    def library_or_input_wec(self):
        model_numbers = {1,3,5,6}
        if self.ref_model_num in model_numbers:
            self._cost_model.value("library_or_input_wec", 0)
        else:
            raise NotImplementedError
        
    ### Output dictionary of costs
    @property
    def cost_outputs(self) -> dict:
        return self._cost_model.Outputs.export()