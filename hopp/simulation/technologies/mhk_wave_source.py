from typing import Any, Union, Optional
import PySAM.MhkWave as MhkWave

from hopp.simulation.technologies.power_source import PowerSource, SiteInfo, Sequence, logger
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.technologies.financial.mhk_cost_model import MHKCosts
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    _financial_model: Union[Any, CustomFinancialModel]
    # _layout:
    # _dispatch:

    def __init__(
        self,
        site: SiteInfo,
        mhk_config: dict,
        cost_model_inputs: Optional[dict] = None
    ):
        """
        Initialize MHKWavePlant.

        Args:
            mhk_config: A dictionary containing MHK system configuration parameters.

                Required keys:
                    - device_rating_kw (float): Rated power of the MHK device in kilowatts
                    - num_devices (int): Number of MHK devices in the system
                    - wave_power_matrix (Sequence): Wave power matrix
                    - fin_model: a financial model object to use instead of singleowner
                        model #TODO: Update with ProFAST
                    - layout_mode: #TODO: Add layout_mode

                Optional keys:
                    - loss_array_spacing (float): Array spacing loss in % (default: 0)
                    - loss_resource_overprediction (float): Resource overprediction loss
                        in % (default: 0)
                    - loss_transmission (float): Transmission loss in % (default: 0)
                    - loss_downtime (float): Array/WEC downtime loss in % (default: 0)
                    - loss_additional (float): Additional losses in % (default: 0)

            cost_model_inputs: An optional dictionary containing input parameters for
                cost modeling.

                - Required keys
                    - reference_model_num (int): Reference model number from Sandia
                        Project (3, 5, or 6).
                    - water_depth (float): Water depth in meters
                    - distance_to_shore (float): Distance to shore in meters
                    - number_rows (int): Number of rows in the device layout
                - Optional keys
                    - row_spacing (float): Spacing between rows in meters
                        (default 'device_spacing')
                    - cable_system_overbuild (float): Cable system overbuild percentage
                        (default 10%)

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
        
    def create_mhk_cost_calculator(self, cost_model_inputs: dict):
        """
        Instantiates MHKCosts, cost calculator for MHKWavePlant.

        Args:
            cost_model_inputs: A dictionary containing input parameters for cost
                modeling.

                Required keys:
                    - reference_model_num (int): Reference model number from Sandia
                        Project (3, 5, or 6).
                    - water_depth (float): Water depth in meters
                    - distance_to_shore (float): Distance to shore in meters
                    - number_rows (int): Number of rows in the device layout
                Optional keys:
                    - row_spacing (float): Spacing between rows in meters
                        (default 'device_spacing')
                    - cable_system_overbuild (float): Cable system overbuild percentage
                        (default 10%)

        """
        self.mhk_costs = MHKCosts(self.mhk_config, cost_model_inputs)
    
    def calculate_total_installed_cost(self):
        if self.mhk_costs is None:
            raise AttributeError("mhk_costs must be set before calling this method.")

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
        """
        new_num_devices = round(wave_size_kw / self.device_rated_power)
        if self.number_devices != new_num_devices:
            self.number_devices = new_num_devices

    def simulate(self, interconnect_kw: float, project_life: int = 25, lifetime_sim=False):
        """
        Run the system and financial model

        Args:
            interconnect_kw (float): grid interconnect
            project_life (int): Number of years in the analysis period (expected
                project lifetime)
            lifetime_sim (bool):
                For simulation modules which support simulating each year of the
                    project_life, whether or not to do so; otherwise the first year
                    data is repeated
        """

        self.calculate_total_installed_cost()
        super().simulate(interconnect_kw, project_life)

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