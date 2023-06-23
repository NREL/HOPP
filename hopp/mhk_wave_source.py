from logging import raiseExceptions
import PySAM.MhkWave as MhkWave
import PySAM.MhkCosts as MhkCost
import PySAM.Singleowner as Singleowner


from hopp.power_source import *
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    #_financial_model: Singleowner.Singleowner
    _cost_model: MhkCost.MhkCosts
    # _layout:
    # _dispatch:

    def __init__(self,
                 site: SiteInfo,
                 mhk_config: dict,
                 cost_model_inputs: dict,
                 ):
        """
        Set up a MhkWavePlant

        :mhk_config param: dict, with keys ('device_rating_kw', 'num_devices', 'wave_power_matrix',
                    'loss_array_spacing', 'loss_resource_overprediction', 'loss_transmission',
                    'loss_downtime', 'loss_additional', 'layout_mode')
            where losses are optional inputs otherwise default to 0%
            where 'layout_mode' is from MhkGridParameters #TODO: make MhkGridParameters

        :cost_model_inputs param: dict, with keys('reference_model_num','water_depth','distance_to_shore',
                    'number_rows','device_spacing','row_spacing','cable_system_overbuild')
        """
        if 'device_rating_kw' not in mhk_config.keys():
            raise ValueError("'device_rating_kw' for MHKWavePlant")

        if 'num_devices' not in mhk_config.keys():
            raise ValueError("'num_devices' required for MHKWavePlant")

        if 'wave_power_matrix' not in mhk_config.keys():
            raise ValueError("'wave_power_matrix' required for MHKWavePlant")

        system_model = MhkWave.new()
        financial_model = None #Singleowner.from_existing(system_model)
        cost_model = MhkCost.new()

        super().__init__("MHKWavePlant", site, system_model, financial_model, cost_model)

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

        # Cost parameter inputs
        self.ref_model_num = cost_model_inputs['reference_model_num']
        self._cost_model.value("lib_wave_device", "RM"+str(cost_model_inputs['reference_model_num']))
        self._cost_model.value("marine_energy_tech", 0)
        self._cost_model.value("library_or_input_wec", 0)
        self.water_depth = cost_model_inputs['water_depth']
        self.distance_to_shore = cost_model_inputs['distance_to_shore']
        self.number_of_rows = cost_model_inputs['number_rows']
        self.devices_per_row = cost_model_inputs['devices_per_row']
        self.device_spacing = cost_model_inputs['device_spacing']
        self.row_spacing = cost_model_inputs['row_spacing']
        self.cable_sys_overbuild = cost_model_inputs['cable_system_overbuild']

        self._cost_model.value("system_capacity", self.system_capacity_kw)
        self._cost_model.value("device_rated_power", self.device_rated_power)
        self._cost_model.value("devices_per_row", self.devices_per_row)



        # Inter-array cable length, m
        # The total length of cable used within the array of devices
        self.array_cable_length = (self.devices_per_row -1) * (self.device_spacing * self.number_of_rows)\
            + self.row_spacing * (self.number_of_rows-1)
        self._cost_model.value("inter_array_cable_length", self.array_cable_length)

        # Export cable length, m
        # The length of cable between the array and onshore grid connection point
        self.export_cable_length = (self.water_depth + self.distance_to_shore) * (1 + self.cable_sys_overbuild/100)
        self._cost_model.value("export_cable_length", self.export_cable_length)

        #Riser cable length, m
        # The length of cable from the seabed to the water surfacethat connects the floating device to the seabed cabling.
        # Applies only to floating array
        self.riser_cable_length = 1.5 * self.water_depth * self.number_devices * (1 + self.cable_sys_overbuild/100)
        self._cost_model.value("riser_cable_length", self.riser_cable_length)

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

    @property
    def device_rated_power(self):
        return self._system_model.MHKWave.device_rated_power

    @ device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._system_model.MHKWave.device_rated_power = device_rate_power
        self._cost_model.MHKCosts.device_rated_power = device_rate_power

    @property
    def number_devices(self):
        return self._system_model.MHKWave.number_devices

    @ number_devices.setter
    def number_devices(self, number_devices: int):
        self._system_model.MHKWave.number_devices = number_devices

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
        self._cost_model.value("system_capacity", self._system_model.value("system_capacity"))
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

    @property
    def ref_model_num(self):
        return self._cost_model.value("lib_wave_device")

    @ref_model_num.setter
    def ref_model_num(self, ref_model_number: int):
        if 0 <= ref_model_number < 6:
            try:
                if ref_model_number == 3:
                    model_type = "RM3"
                    library = 0
                    return model_type
                elif ref_model_number == 5:
                    model_type = "RM5"
                    library = 0
                    return model_type
                elif ref_model_number == 6:
                    model_type = "RM6"
                    library = 0
                    return model_type
                self._cost_model.value("lib_wave_device", model_type)
                self._cost_model.value("library_or_input_wec", library)
            except:
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

    ### Output dictionary of costs
    @property
    def cost_outputs(self) -> dict:
        return self._cost_model.Outputs.export()



