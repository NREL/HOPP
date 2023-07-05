from asyncio import run_coroutine_threadsafe
from logging import raiseExceptions
from weakref import ref
import PySAM.MhkWave as MhkWave
import PySAM.MhkCosts as MhkCost
import PySAM.Singleowner as Singleowner


from hopp.power_source import *
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    # _financial_model:
    # _layout:
    # _dispatch:

    def __init__(self,
                 site: SiteInfo,
                 mhk_config: dict,
                 ):
        """
        Set up a MhkWavePlant

        :mhk_config param: dict, with keys ('device_rating_kw', 'num_devices', 'wave_power_matrix',
                    'loss_array_spacing', 'loss_resource_overprediction', 'loss_transmission',
                    'loss_downtime', 'loss_additional', 'layout_mode')
            where losses are optional inputs otherwise default to 0%
            where 'layout_mode' is from MhkGridParameters #TODO: make MhkGridParameters
        """
        if 'device_rating_kw' not in mhk_config.keys():
            raise ValueError("'device_rating_kw' for MHKWavePlant")

        if 'num_devices' not in mhk_config.keys():
            raise ValueError("'num_devices' required for MHKWavePlant")

        if 'wave_power_matrix' not in mhk_config.keys():
            raise ValueError("'wave_power_matrix' required for MHKWavePlant")

        system_model = MhkWave.new()
        financial_model = None #Singleowner.from_existing(system_model)
        self.mhk_costs = None

        super().__init__("MHKWavePlant", site, system_model, financial_model)

        system_model.MHKWave.wave_resource_model_choice = 1 #Time-series data=1 JPD=0
        system_model.MHKWave.significant_wave_height = self.site.wave_resource.data['significant_wave_height'] 
        system_model.MHKWave.energy_period = self.site.wave_resource.data['energy_period'] 
        system_model.MHKWave.year = self.site.wave_resource.data['year'] 
        system_model.MHKWave.month = self.site.wave_resource.data['month'] 
        system_model.MHKWave.day = self.site.wave_resource.data['day'] 
        system_model.MHKWave.hour = self.site.wave_resource.data['hour'] 
        system_model.MHKWave.minute = self.site.wave_resource.data['minute'] 

        # System parameter inputs
        self._system_model.value("device_rated_power", mhk_config['device_rating_kw'])
        self._system_model.value("number_devices",  mhk_config['num_devices'])
        self._system_model.value("wave_power_matrix", mhk_config['wave_power_matrix'])
        # Losses
        if 'loss_array_spacing' not in mhk_config.keys():
            self._system_model.MHKWave.loss_array_spacing = 0
        else:
            self._system_model.MHKWave.loss_array_spacing = mhk_config['loss_array_spacing']
        if 'loss_downtime' not in mhk_config.keys():
            self._system_model.MHKWave.loss_downtime = 0
        else:
            self._system_model.MHKWave.loss_downtime = mhk_config['loss_downtime'] 
        if 'loss_resource_overprediction' not in mhk_config.keys():
            self._system_model.MHKWave.loss_resource_overprediction = 0
        else:
            self._system_model.MHKWave.loss_resource_overprediction = mhk_config['loss_resource_overprediction'] 
        if 'loss_transmission' not in mhk_config.keys():
            self._system_model.MHKWave.loss_transmission = 0
        else:
            self._system_model.MHKWave.loss_transmission = mhk_config['loss_transmission']
        if 'loss_additional' not in mhk_config.keys():
            self._system_model.MHKWave.loss_additional = 0
        else:
            self._system_model.MHKWave.loss_additional = mhk_config['loss_additional']
    
    def create_mhk_cost_calculator(self,mhk_config,cost_model_inputs):
        self.mhk_costs = MHKCosts(mhk_config,cost_model_inputs)

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

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices

        :param wave_size_kw: desired system capacity in kW
        """
        new_num_devices = round(wave_size_kw / self.device_rated_power)
        if self.number_devices != new_num_devices:
            self.number_devices = new_num_devices

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
    def annual_energy_kw(self) -> float:
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
        if 'device_rating_kw' not in mhk_config.keys():
            raise ValueError("'device_rating_kw' for MHKCosts")

        if 'num_devices' not in mhk_config.keys():
            raise ValueError("'num_devices' required for MHKCosts")

        if 'reference_model_num' not in cost_model_inputs.keys():
            raise ValueError("'reference_model_num' for MHKCosts")
        
        if 'water_depth' not in cost_model_inputs.keys():
            raise ValueError("'water_depth' for MHKCosts")
        
        if 'distance_to_shore' not in cost_model_inputs.keys():
            raise ValueError("'distance_to_shore' for MHKCosts")

        if 'number_rows' not in cost_model_inputs.keys():
            raise ValueError("'number_rows' for MHKCosts")

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

        if 'structural_assembly_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('structural_assembly_cost_method',2)
            self._cost_model.value('structural_assembly_cost_input', 0)
        else:
            self._cost_model.value('structural_assembly_cost_method',0)
            self._cost_model.value('structural_assembly_cost_input', cost_model_inputs['structural_assembly_cost_kw'])

        if 'power_takeoff_system_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('power_takeoff_system_cost_method',2)
            self._cost_model.value('power_takeoff_system_cost_input', 0)
        else:
            self._cost_model.value('power_takeoff_system_cost_method',0)
            self._cost_model.value('power_takeoff_system_cost_input', cost_model_inputs['power_takeoff_system_cost_kw'])

        if 'mooring_found_substruc_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('mooring_found_substruc_cost_method',2)
            self._cost_model.value('mooring_found_substruc_cost_input', 0)
        else:
            self._cost_model.value('mooring_found_substruc_cost_method',0)
            self._cost_model.value('mooring_found_substruc_cost_input', cost_model_inputs['mooring_found_substruc_cost_kw'])

        if 'development_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('development_cost_method',2)
            self._cost_model.value('development_cost_input', 0)
        else:
            self._cost_model.value('development_cost_method',0)
            self._cost_model.value('development_cost_input', cost_model_inputs['development_cost_kw'])

        if 'eng_and_mgmt_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('eng_and_mgmt_cost_method',2)
            self._cost_model.value('eng_and_mgmt_cost_input', 0)
        else:
            self._cost_model.value('eng_and_mgmt_cost_method',0)
            self._cost_model.value('eng_and_mgmt_cost_input', cost_model_inputs['eng_and_mgmt_cost_kw'])

        if 'assembly_and_install_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('assembly_and_install_cost_method',2)
            self._cost_model.value('assembly_and_install_cost_input', 0)
        else:
            self._cost_model.value('assembly_and_install_cost_method',0)
            self._cost_model.value('assembly_and_install_cost_input', cost_model_inputs['assembly_and_install_cost_kw'])

        if 'other_infrastructure_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('other_infrastructure_cost_method',2)
            self._cost_model.value('other_infrastructure_cost_input', 0)
        else:
            self._cost_model.value('other_infrastructure_cost_method',0)
            self._cost_model.value('other_infrastructure_cost_input', cost_model_inputs['other_infrastructure_cost_kw'])

        if 'array_cable_system_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('array_cable_system_cost_method',2)
            self._cost_model.value('array_cable_system_cost_input', 0)
        else:
            self._cost_model.value('array_cable_system_cost_method',0)
            self._cost_model.value('array_cable_system_cost_input', cost_model_inputs['array_cable_system_cost_kw'])

        if 'export_cable_system_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('export_cable_system_cost_method',2)
            self._cost_model.value('export_cable_system_cost_input', 0)
        else:
            self._cost_model.value('export_cable_system_cost_method',0)
            self._cost_model.value('export_cable_system_cost_input', cost_model_inputs['export_cable_system_cost_kw'])

        if 'onshore_substation_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('onshore_substation_cost_method',2)
            self._cost_model.value('onshore_substation_cost_input', 0)
        else:
            self._cost_model.value('onshore_substation_cost_method',0)
            self._cost_model.value('onshore_substation_cost_input', cost_model_inputs['onshore_substation_cost_kw'])

        if 'offshore_substation_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('offshore_substation_cost_method',2)
            self._cost_model.value('offshore_substation_cost_input', 0)
        else:
            self._cost_model.value('offshore_substation_cost_method',0)
            self._cost_model.value('offshore_substation_cost_input', cost_model_inputs['offshore_substation_cost_kw'])

        if 'other_elec_infra_cost_kw' not in cost_model_inputs.keys():
            self._cost_model.value('other_elec_infra_cost_method',2)
            self._cost_model.value('other_elec_infra_cost_input', 0)
        else:
            self._cost_model.value('other_elec_infra_cost_method',0)
            self._cost_model.value('other_elec_infra_cost_input', cost_model_inputs['other_elec_infra_cost_kw'])
    
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

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices

        :param wave_size_kw: desired system capacity in kW
        """
        new_num_devices = round(wave_size_kw / self._device_rated_power)
        if self._number_devices != new_num_devices:
            self._number_devices = new_num_devices
            self.initialize()

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

    def simulate_costs(self):
        self._cost_model.execute()
    
    ### Output dictionary of costs
    @property
    def cost_outputs(self) -> dict:
        return self._cost_model.Outputs.export()