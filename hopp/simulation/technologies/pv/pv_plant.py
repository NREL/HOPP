from typing import List, Sequence, Optional, Union

from attrs import define, field
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.layout.pv_module import get_module_attribs
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.base import BaseClass
from hopp.utilities.validators import gt_zero


@define
class PVConfig(BaseClass):
    """
    Configuration class for PVPlant. 
    
    Args:
        system_capacity_kw: Design system capacity
        use_pvwatts: Whether to use PVWatts (defaults to True). If False, this
            config should be used in a `DetailedPVPlant`
        layout_params: Optional layout parameters
        layout_model: Optional layout model instance
        fin_model: Financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

        dc_degradation: Annual DC degradation for lifetime simulations [%/year]
        approx_nominal_efficiency: approx nominal efficiency depends on module type (standard crystalline silicon 19%, premium 21%, thin film 18%) [decimal]
        module_unit_mass: Mass of the individual module unit (default to 11.092). [kg/m2]
    """
    system_capacity_kw: float = field(validator=gt_zero)

    use_pvwatts: bool = field(default=True)
    layout_params: Optional[Union[dict, PVGridParameters]] = field(default=None)
    layout_model: Optional[Union[dict, PVLayout]] = field(default=None)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)
    dc_degradation: Optional[List[float]] = field(default=None)
    approx_nominal_efficiency: Optional[float] = field(default=0.19)
    module_unit_mass: Optional[float] = field(default=11.092)

@define
class PVPlant(PowerSource):
    """
    Represents a PV Plant.

    Args:
        site: The site information.
        config: Configuration dictionary representing a PVConfig.

    """
    site: SiteInfo
    config: PVConfig

    config_name: str = field(init=False, default="PVWattsSingleOwner")

    def __attrs_post_init__(self):
        system_model = Pvwatts.default(self.config_name)

        # Parse input for a financial model
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)
        else:
            financial_model = self.config.fin_model

        if financial_model is None:
            # default
            financial_model = Singleowner.from_existing(system_model, self.config_name)
        else:
            financial_model = self.import_financial_model(financial_model, system_model, self.config_name)

        # Parse input for layout params
        if isinstance(self.config.layout_params, dict):
            layout_params = PVGridParameters(**self.config.layout_params)
        else:
            layout_params = self.config.layout_params

        # Parse input for layout model
        if isinstance(self.config.layout_model, dict):
            layout_model = PVLayout(**self.config.layout_model)
        else:
            layout_model = self.config.layout_model

        super().__init__("PVPlant", self.site, system_model, financial_model)

        if self.site.solar_resource is not None:
            self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        if self.config.dc_degradation is not None:
            self.dc_degradation = self.config.dc_degradation
        else:
            self.dc_degradation = [0]
        
        if self.config.approx_nominal_efficiency is not None:
            self.approx_nominal_efficiency = self.config.approx_nominal_efficiency
        else:
            self.approx_nominal_efficiency = 0.19

        if self.config.module_unit_mass is not None:
            self.module_unit_mass = self.config.module_unit_mass
        else:
            self.module_unit_mass = 11.092

        if layout_model is not None:
            self.layout = layout_model
            self.layout._system_model = self._system_model
        else:
            self.layout = PVLayout(self.site, self._system_model, layout_params)

        # TODO: it seems like an anti-pattern to be doing this in each power source,
        # then assigning the relevant class using metaprogramming in 
        # HybridDispatchBuilderSolver._create_dispatch_optimization_model
        self._dispatch = None
        self.system_capacity_kw = self.config.system_capacity_kw

    @property
    def system_capacity_kw(self) -> float:
        """Gets the system capacity."""
        # TODO: This is currently DC power; however, all other systems are rated by AC power
        # return self._system_model.SystemDesign.system_capacity / self._system_model.SystemDesign.dc_ac_ratio
        return self._system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model.
        """
        self._system_model.SystemDesign.system_capacity = size_kw
        self._financial_model.value('system_capacity', size_kw) # needed for custom financial models
        self.layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]."""
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        """Sets annual DC degradation for lifetime simulations [%/year]."""
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year

    @property
    def dc_ac_ratio(self) -> float:
        """DC to AC inverter loading ratio [ratio]."""
        return self._system_model.SystemDesign.dc_ac_ratio

    @dc_ac_ratio.setter
    def dc_ac_ratio(self, inverter_loading_ratio: float):
        """Sets DC to AC inverter loading ratio [ratio]."""
        self._system_model.SystemDesign.dc_ac_ratio = inverter_loading_ratio
    
    @property
    def module_type(self) -> int:
        """ Module type: standard, premium, thin film [0/1/2]"""
        return self._system_model.value("module_type")

    @module_type.setter
    def module_type(self, solar_module_type: int):
        """ Sets module type: standard, premium, thin film [0/1/2]"""
        if 0 <= solar_module_type <= 2:
            self._system_model.value("module_type", solar_module_type)
        else:
            raise ValueError("Invalid module type")
        if solar_module_type is not None:
            efficiency_mapping = {0: 0.19, 1: 0.21, 2: 0.18}
            if solar_module_type in efficiency_mapping:
                self.approx_nominal_efficiency = efficiency_mapping[solar_module_type]
            else:
                raise ValueError("Module type not recognized")
        else:
            raise NotImplementedError("Module type not set")

    @property
    def footprint_area(self):
        """Estimate Total Module Footprint Area [m^2]"""
        module_attribs = get_module_attribs(self._system_model)
        num_modules = self.system_capacity_kw / module_attribs['P_mp_ref']
        area = num_modules * module_attribs['area']
        return  area

    @property
    def system_mass(self):
        """Estimate Total Module Mass [kg]"""
        return self.footprint_area * self.module_unit_mass

    @property
    def capacity_factor(self) -> float:
        """System capacity factor [%]"""
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor")*self._system_model.value("dc_ac_ratio")
        else:
            return 0
        ### Use this version when updated to PySAM 4.2.0
        # if self.system_capacity_kw > 0:
        #     return self._system_model.value("capacity_factor_ac")
        # else:
        #     return 0
