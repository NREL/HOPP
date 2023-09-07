import PySAM.MhkCosts as MhkCost

class MHKCosts():
    """
    A class for calculating the costs associated with Marine Hydrokinetic (MHK) energy systems.

    This class initializes and configures cost calculations for MHK systems based on provided input parameters.
    It uses the PySAM library for cost modeling which is based on the [Sandia Reference Model Project](https://energy.sandia.gov/programs/renewable-energy/water-power/projects/reference-model-project-rmp/).

    Attributes:
        Various attributes are set based on the input parameters, and cost models are configured accordingly.

    Methods:
        - initialize(): Initialize the cost model with the provided parameters.
        - system_capacity_by_num_devices(wave_size_kw): Update the number of devices based on a given wave size in kilowatts.
        - simulate_costs(): Execute the cost model to simulate MHK system costs.

    Properties:
        Various properties provide access to specific attributes and calculations:
        - device_rated_power
        - number_devices
        - system_capacity_kw
        - ref_model_num
        - library_or_input_wec
        - cost_outputs: Returns a dictionary of cost outputs from the cost model.
    """
    def __init__(self,
                mhk_config:dict,
                cost_model_inputs:dict
                ):
        """
        Initialize MHKCosts class.

        Args:
        `mhk_config` (dict): A dictionary containing MHK system configuration parameters.
            Required keys:
                `'device_rating_kw'`: float, Rated power of the MHK device in kilowatts.
                `'num_devices'`: int, Number of MHK devices in the system.
        `cost_model_inputs` (dict): A dictionary containing input parameters for cost modeling.
            Required keys:
                `'reference_model_num'`: int, Reference model number from Sandia Project (3, 5, or 6).
                `'water_depth'`: float, Water depth in meters.
                `'distance_to_shore'`: float, Distance to shore in meters.
                `'number_rows'`: int, Number of rows in the device layout.
            Optional keys:
                `'row_spacing'`: float, Spacing between rows in meters (default 'device_spacing')
                `'cable_system_overbuild'`: float, Cable system overbuild percentage (default 10%)

        Raises:
        ValueError: If any of the required keys in `mhk_config` or `cost_model_inputs` are missing.
        """
        required_keys = ['device_rating_kw', 'num_devices', 'reference_model_num', 'water_depth', 'distance_to_shore', 'number_rows']

        for key in required_keys:
            if key not in mhk_config.keys() and key not in cost_model_inputs.keys():
                raise ValueError(f"'{key}' for MHKCosts")

        self._cost_model = MhkCost.new()

        self._device_rated_power = mhk_config['device_rating_kw']
        self._number_devices = mhk_config['num_devices'] 

        self._water_depth = cost_model_inputs['water_depth']
        self._distance_to_shore = cost_model_inputs['distance_to_shore']  

        self._number_of_rows = cost_model_inputs['number_rows']
        
        self._ref_model_num = "RM"+str(cost_model_inputs['reference_model_num'])

        self._device_spacing = cost_model_inputs['device_spacing']
        if 'row_spacing' not in cost_model_inputs.keys():
            self._row_spacing = cost_model_inputs['device_spacing']
        else:
            self._row_spacing = cost_model_inputs['row_spacing']
        if 'cable_system_overbuild' not in cost_model_inputs.keys():
            self._cable_sys_overbuild = 10
        else:
            self._cable_sys_overbuild = cost_model_inputs['cable_system_overbuild']

        # Define a mapping of cost keys to their respective method and input keys
        cost_keys_mapping = {
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
            if cost_key not in cost_model_inputs:
                self._cost_model.value(method_key, 2)
                self._cost_model.value(input_key, 0)
            else:
                self._cost_model.value(method_key, 0)
                self._cost_model.value(input_key, cost_model_inputs[cost_key])
    
        self.initialize()

    def initialize(self):
        self._cost_model.value("device_rated_power", self._device_rated_power)
        self._cost_model.value("system_capacity", self._cost_model.value("device_rated_power") * self._number_devices) 
        
        if self._number_devices < self._number_of_rows:
            raise Exception("number_of_rows exceeds number_devices")
        else:
            if (self._number_devices/self._number_of_rows).is_integer():
                self._cost_model.value("devices_per_row", \
                    self._number_devices / self._number_of_rows)
            else:
                raise Exception("Layout must be square or rectangular. Modify 'number_rows' or 'num_devices'.")
        self._cost_model.value("lib_wave_device", self._ref_model_num)
        self._cost_model.value("marine_energy_tech", 0)
        self._cost_model.value("library_or_input_wec", 0)
        # Inter-array cable length, m
        # The total length of cable used within the array of devices
        self._array_cable_length = (self._cost_model.value("devices_per_row") -1) * \
            (self._device_spacing * self._number_of_rows) \
            + self._row_spacing * (self._number_of_rows-1)
        self._cost_model.value("inter_array_cable_length", self._array_cable_length)

        # Export cable length, m
        # The length of cable between the array and onshore grid connection point
        self._export_cable_length = (self._water_depth + self._distance_to_shore) * \
            (1 + self._cable_sys_overbuild/100)
        self._cost_model.value("export_cable_length", self._export_cable_length)

        #Riser cable length, m
        # The length of cable from the seabed to the water surfacethat 
        # connects the floating device to the seabed cabling.
        # Applies only to floating array
        self.riser_cable_length = 1.5 * self._water_depth * self._number_devices * \
            (1 + self._cable_sys_overbuild/100)
        self._cost_model.value("riser_cable_length", self.riser_cable_length)

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices

        :param wave_size_kw: desired system capacity in kW
        """
        new_num_devices = round(wave_size_kw / self._device_rated_power)
        if self._number_devices != new_num_devices:
            self._number_devices = new_num_devices
            self.initialize()

    def simulate_costs(self):
        self._cost_model.execute()

    @property
    def device_rated_power(self):
        return self._device_rated_power

    @ device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._device_rated_power = device_rate_power
        self.initialize()

    @property
    def number_devices(self):
        return self._number_devices

    @ number_devices.setter
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
        Sets the system capacity by updates the number of wave devices using device rating
        :param size_kw:
        :return:
        """
        self.system_capacity_by_num_devices(size_kw)

    @property
    def ref_model_num(self):
        return self._ref_model_num

    @ref_model_num.setter
    def ref_model_num(self, ref_model_number: int):
        if ref_model_number == 3 or ref_model_number == 5 or ref_model_number == 6:
            self._ref_model_num = "RM"+ str(ref_model_number)
            self.initialize()
        else:
            raise NotImplementedError
    
    @property
    def library_or_input_wec(self):
        return self._cost_model.value("library_or_input_wec")
    @library_or_input_wec.setter
    def library_or_input_wec(self):
        if self.ref_model_num == 3 or self.ref_model_num == 5 or self.ref_model_num == 6:
            self._cost_model.value("library_or_input_wec", 0)
        else:
            raise NotImplementedError
        
    ### Output dictionary of costs
    @property
    def cost_outputs(self) -> dict:
        return self._cost_model.Outputs.export()