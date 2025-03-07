from typing import Optional, List, Union
import PySAM.MhkTidal as MhkTidal

from attrs import define, field
from hopp.simulation.base import BaseClass

from hopp.simulation.technologies.power_source import PowerSource, SiteInfo, Sequence, logger
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.technologies.financial.mhk_cost_model import MHKCosts, MHKCostModelInputs
from hopp.utilities.validators import gt_zero, range_val


@define
class MHKTidalConfig(BaseClass):
    """
    Configuration class for MHKTidalPlant.

    Args:
        device_rating_kw (float): Rated power of the MHK device [kW]
        num_devices (int): Number of MHK tidal devices in the system
        tidal_power_curve (List[List[float]]): Power curve of tidal energy device as function of stream speeds [kW]
        tidal_resource (List[List[float]]): Required by the PySAM MhkTidal module for initialization. Although this parameter 
            is not actively used in HOPP's timeseries simulation mode, it must still be provided to fully 
            instantiate the PySAM MhkTidal model.
            Frequency distribution of resource as a function of stream speeds.
        fin_model (obj | dict): Optional financial model. Can be any of the following:
            - a dict representing a `CustomFinancialModel`
            - an object representing a `CustomFinancialModel` instance
        loss_array_spacing (float): Array spacing loss in % (default: 0)
        loss_resource_overprediction (float): Resource overprediction loss
            in % (default: 0)
        loss_transmission (float): Transmission loss in % (default: 0)
        loss_downtime (float): Array/WEC downtime loss in % (default: 0)
        loss_additional (float): Additional losses in % (default: 0)
    """
    device_rating_kw: float = field(validator=gt_zero)
    num_devices: int = field(validator=gt_zero)
    tidal_power_curve: List[List[float]]
    tidal_resource: List[List[float]]
    fin_model: Union[dict, CustomFinancialModel]
    loss_array_spacing: float = field(default=0., validator=range_val(0, 100))
    loss_resource_overprediction: float = field(default=0., validator=range_val(0, 100))
    loss_transmission: float = field(default=0., validator=range_val(0, 100))
    loss_downtime: float = field(default=0., validator=range_val(0, 100))
    loss_additional: float = field(default=0., validator=range_val(0, 100))
    name: str = field(default="MHKTidalPlant")


@define
class MHKTidalPlant(PowerSource):
    """
    Marine Hydrokinetic (MHK) Tidal Plant.

    Args:
        site: Site information
        config: MHK system configuration parameters
        cost_model_inputs (dict, Optional): An optional dictionary containing input parameters for
            cost modeling.
    """
    site: SiteInfo
    config: MHKTidalConfig
    cost_model_inputs: Optional[MHKCostModelInputs] = field(default=None)
    config_name: str = field(default="MhkWave")

    mhk_costs: Optional[MHKCosts] = field(init=False)

    def __attrs_post_init__(self):
        system_model = MhkTidal.new()

        if isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model

        financial_model = self.import_financial_model(financial_model, system_model, self.config_name)
        
        if self.cost_model_inputs is not None:
            self.mhk_costs = MHKCosts(self.config, self.cost_model_inputs)
        else:
            self.mhk_costs = None
            
        super().__init__("MHKTidalPlant", self.site, system_model, financial_model)

        # Set tidal resource model choice
        system_model.MHKTidal.tidal_resource_model_choice = 1 # Time-series data=1 JPD=0 (Joint-probability distribution)

        # Copy values from self.site.tidal_resource.data to system_model.MHKTidal
        attributes_to_copy = ['tidal_velocity']
        for attribute in attributes_to_copy:
            setattr(system_model.MHKTidal, attribute, self.site.tidal_resource.data[attribute])

        # System parameter inputs
        self._system_model.device_rated_power = self.config.device_rating_kw
        self._system_model.value("number_devices",  self.config.num_devices)
        self._system_model.value("tidal_power_curve", self.config.tidal_power_curve)
        self._system_model.value("tidal_resource", self.config.tidal_resource)

        # Losses
        loss_attributes = [
            'loss_array_spacing', 
            'loss_downtime', 
            'loss_resource_overprediction', 
            'loss_transmission', 
            'loss_additional'
            ]

        for attribute in loss_attributes:
            attr = getattr(self.config, attribute, 0)
            setattr(self._system_model.MHKTidal, attribute, attr)
        
    def create_mhk_cost_calculator(self, cost_model_inputs: Union[dict, MHKCostModelInputs]):
        """
        Instantiates MHKCosts, cost calculator for MHKTidalPlant.

        Args:
            cost_model_inputs: Input parameters for cost modeling.
        """
        if isinstance(cost_model_inputs, dict):
            cost_model = MHKCostModelInputs.from_dict(cost_model_inputs)
        else:
            cost_model = cost_model_inputs

        self.mhk_costs = MHKCosts(self.config, cost_model)
    
    def calculate_total_installed_cost(self) -> float:
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
        
        return self._financial_model.value("total_installed_cost", total_installed_cost)

    def system_capacity_by_num_devices(self, tidal_size_kw: float):
        """
        Sets the system capacity by adjusting the number of devices
        """
        new_num_devices = round(tidal_size_kw / self.device_rated_power)
        self.number_devices = new_num_devices

    def simulate(self, interconnect_kw: float, project_life: int = 25, lifetime_sim=False):
        """
        Run the system and financial model

        Args:
            interconnect_kw: grid interconnect
            project_life: Number of years in the analysis period (expected
                project lifetime)
            lifetime_sim:
                For simulation modules which support simulating each year of the
                    project_life, whether or not to do so; otherwise the first year
                    data is repeated
        """

        self.calculate_total_installed_cost()
        super().simulate(interconnect_kw, project_life)

    @property
    def device_rated_power(self) -> float:
        return self._system_model.device_rated_power

    @device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._system_model.device_rated_power = device_rate_power
        if self.mhk_costs is not None:
            self.mhk_costs.device_rated_power = device_rate_power

    @property
    def number_devices(self) -> int:
        return self._system_model.MHKTidal.number_devices

    @number_devices.setter
    def number_devices(self, number_devices: int):
        self._system_model.MHKTidal.number_devices = number_devices
        if self.mhk_costs is not None:
            self.mhk_costs.number_devices = number_devices

    @property
    def tidal_power_curve(self) -> List[List[float]]:
        return self._system_model.MHKTidal.tidal_power_curve

    @tidal_power_curve.setter
    def tidal_power_curve(self, tidal_power_curve: Sequence):
        self._system_model.MHKTidal.tidal_tidal_power_curve =tidal_power_curve

    @property
    def system_capacity_kw(self) -> float:
        self._system_model.value("system_capacity", self._system_model.device_rated_power * self._system_model.MHKTidal.number_devices)
        return self._system_model.value("system_capacity")

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity by updates the number of tidal devices using device rating
        """
        self.system_capacity_by_num_devices(size_kw)

    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("annual_energy") 
        else:
            return 0
    
    @property
    def capacity_factor(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor")
        else:
            return 0
        