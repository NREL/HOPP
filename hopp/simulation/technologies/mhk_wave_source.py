from typing import Any, Union, Optional
import PySAM.MhkWave as MhkWave
import PySAM.MhkCosts as MhkCost

from hopp.simulation.technologies.power_source import PowerSource, SiteInfo, Sequence, logger
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    _financial_model: Union[Any, CustomFinancialModel]
    # _layout:
    # _dispatch:

    def __init__(self,
                 site: SiteInfo,
                 mhk_config: dict,
                 cost_model_inputs: Optional[dict] = None
                 ):
        """
        Set up a MhkWavePlant

        :param 'mhk_config': ``dict``, with keys ('device_rating_kw', 'num_devices', 'wave_power_matrix',
                    'loss_array_spacing', 'loss_resource_overprediction', 'loss_transmission',
                    'loss_downtime', 'loss_additional', 'fin_model', 'layout_mode')
                where `losses` are optional inputs otherwise default to 0%
                where `fin_model` is a financial model object to use instead of singleowner model #TODO: Update with ProFAST
                where `layout_mode` is from MhkGridParameters #TODO: make MhkGridParameters
        :param cost_model_inputs: ``dict``, optional
            with keys('reference_model_num','water_depth','distance_to_shore',
                'number_rows','device_spacing','row_spacing','cable_system_overbuild')
        """
        self.mhk_config = mhk_config

        required_keys = ['device_rating_kw', 'num_devices', 'wave_power_matrix']

        for key in required_keys:
            if key not in mhk_config:
                raise ValueError(f"'{key}' required for MHKWavePlant")

        self.config_name = "MhkWave"
        system_model = MhkWave.new()

        if 'fin_model' in mhk_config.keys():
            financial_model = self.import_financial_model(mhk_config['fin_model'], system_model, self.config_name)
        else:
            raise NotImplementedError
        
        if cost_model_inputs != None:
            self.mhk_costs = MHKCosts(mhk_config,cost_model_inputs)
        else:
            self.mhk_costs = None
            
        super().__init__("MHKWavePlant", site, system_model, financial_model)

        # Set wave resource model choice
        system_model.MHKWave.wave_resource_model_choice = 1  # Time-series data=1 JPD=0

        # Copy values from self.site.wave_resource.data to system_model.MHKWave
        attributes_to_copy = ['significant_wave_height', 'energy_period', 'year', 'month', 'day', 'hour', 'minute']
        for attribute in attributes_to_copy:
            setattr(system_model.MHKWave, attribute, self.site.wave_resource.data[attribute])

        # System parameter inputs
        self._system_model.value("device_rated_power", mhk_config['device_rating_kw'])
        self._system_model.value("number_devices",  mhk_config['num_devices'])
        self._system_model.value("wave_power_matrix", mhk_config['wave_power_matrix'])

        # Losses
        loss_attributes = ['loss_array_spacing', 'loss_downtime', 'loss_resource_overprediction', 'loss_transmission', 'loss_additional']

        for attribute in loss_attributes:
            if attribute in mhk_config.keys():
                setattr(self._system_model.MHKWave, attribute, mhk_config[attribute])
            else:
                setattr(self._system_model.MHKWave, attribute, 0)
        
    def create_mhk_cost_calculator(self,mhk_config,cost_model_inputs):
        """
        Instantiates MHKCosts, cost calculator for MHKWavePlant

        :param cost_model_inputs: ``dict``
            with keys('reference_model_num','water_depth','distance_to_shore',
                'number_rows','device_spacing','row_spacing','cable_system_overbuild')
        """
        self.mhk_costs = MHKCosts(mhk_config,cost_model_inputs)
    
    def calculate_total_installed_cost(self):
        self.mhk_costs.simulate_costs()
        cost_dict = self.mhk_costs.cost_outputs

        capex = cost_dict['structural_assembly_cost_modeled']+\
            cost_dict['power_takeoff_system_cost_modeled']+\
            cost_dict['mooring_found_substruc_cost_modeled']
        bos = cost_dict['development_cost_modeled']+\
            cost_dict['eng_and_mgmt_cost_modeled']+\
            cost_dict['plant_commissioning_cost_modeled']+\
            cost_dict['site_access_port_staging_cost_modeled']+\
            cost_dict['assembly_and_install_cost_modeled']+\
            cost_dict['other_infrastructure_cost_modeled']
        elec_infrastruc_costs = cost_dict['array_cable_system_cost_modeled']+\
            cost_dict['export_cable_system_cost_modeled']+\
            cost_dict['onshore_substation_cost_modeled']+\
            cost_dict['offshore_substation_cost_modeled']+\
            cost_dict['other_elec_infra_cost_modeled']
        financial = cost_dict['project_contingency']+\
            cost_dict['insurance_during_construction']+\
            cost_dict['reserve_accounts']

        total_installed_cost = capex+bos+elec_infrastruc_costs+financial
        
        return self._financial_model.value("total_installed_cost",total_installed_cost)

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices

        :param wave_size_kw: desired system capacity in kW
        """
        new_num_devices = round(wave_size_kw / self.device_rated_power)
        if self.number_devices != new_num_devices:
            self.number_devices = new_num_devices

    def simulate(self, interconnect_kw: float, project_life: int = 25, lifetime_sim=False):
        """
        Run the system and financial model

        :param project_life: ``int``,
            Number of year in the analysis period (execepted project lifetime) [years]
        :param lifetime_sim: ``bool``,
            For simulation modules which support simulating each year of the project_life, whether or not to do so; otherwise the first year data is repeated
        """

        self.calculate_total_installed_cost()
        super().simulate(interconnect_kw,project_life)

    @property
    def device_rated_power(self):
        return self._system_model.MHKWave.device_rated_power

    @ device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._system_model.MHKWave.device_rated_power = device_rate_power
        if self.mhk_costs != None:
            self.mhk_costs.device_rated_power = device_rate_power

    @property
    def number_devices(self):
        return self._system_model.MHKWave.number_devices

    @ number_devices.setter
    def number_devices(self, number_devices: int):
        self._system_model.MHKWave.number_devices = number_devices
        if self.mhk_costs != None:
            self.mhk_costs.number_devices = number_devices

    @property
    def wave_power_matrix(self):
        return self._system_model.MHKWave.wave_power_matrix

    @wave_power_matrix.setter
    def wave_power_matrix(self, wave_power_matrix: Sequence):
        if len(wave_power_matrix) != 21 and len(wave_power_matrix[0]) != 22:
            raise Exception("Wave power matrix must be dimensions 21 by 22")
        else:    
            self._system_model.MHKWave.wave_power_matrix = wave_power_matrix

    @property
    def system_capacity_kw(self):
        self._system_model.value("system_capacity", self._system_model.MHKWave.device_rated_power * self._system_model.MHKWave.number_devices)
        return self._system_model.value("system_capacity")

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity by updates the number of wave devices using device rating
        :param size_kw:
        :return:
        """
        self.system_capacity_by_num_devices(size_kw)

    #### These are also in Power Source but overwritten here because MhkWave 
    #### Expects 3-hr timeseries data so values are inflated by 3x
    #### TODO: If additional system models are added will need to revise these properties so correct values are assigned
    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("annual_energy") / 3 
        else:
            return 0
    
    @property
    def capacity_factor(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor") / 3
        else:
            return 0
        
    ### Not in Power Source but affected by hourly data
    @property
    def numberHours(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("numberHours") / 3
        else:
            return 0

class MHKCosts():
    """
    MHKCosts class contains tools to determine wave or tidal plant costs.
    """
    def __init__(self,
                mhk_config:dict,
                cost_model_inputs:dict
                ):
        """
        :mhk_config param: dict, with keys('device_rating_kw','num_devices')
        :cost_model_inputs param: dict, with keys('reference_model_num','water_depth','distance_to_shore',
                    'number_rows','device_spacing','row_spacing','cable_system_overbuild')
            where 'row_spacing' is optional: default 'device_spacing'
            where 'cable_system_overbuild' is optional: default 10%
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