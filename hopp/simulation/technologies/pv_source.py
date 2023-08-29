from typing import Union, Sequence, Optional

from attrs import define, field
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel

FIN_MODEL_TYPES = Union[Singleowner.Singleowner, CustomFinancialModel]

@define
class PVConfig:
    """
    Configuration class for PVPlant.

    Args:
        system_capacity_kw (float): Design system capacity.
        layout_params (Optional[PVGridParameters]): Optional layout parameters.
        layout_model (Optional[PVLayout]): Optional layout model instance.
        fin_model (Optional[FIN_MODEL_TYPES]): Optional financial model instance.
    """
    system_capacity_kw: float
    layout_params: Optional[PVGridParameters] = field(default=None)
    layout_model: Optional[PVLayout] = field(default=None)
    fin_model: Optional[FIN_MODEL_TYPES] = field(default=None)


@define
class PVPlant(PowerSource):
    """
    Represents a PV Plant.

    Args:
        site (SiteInfo): The site information.
        config (dict): Configuration dictionary representing a PVConfig.
    """

    site: SiteInfo
    config: dict    

    pv_config: PVConfig = field(init=False)
    system_model: Pvwatts.Pvwattsv8 = field(init=False)
    financial_model: FIN_MODEL_TYPES = field(init=False)
    config_name: str = field(init=False, default="PVWattsSingleOwner")

    def __attrs_post_init__(self):
        self.pv_config = PVConfig(**self.config)

        self.system_model = Pvwatts.default(self.config_name)

        if self.pv_config.fin_model is not None:
            self.financial_model = self.import_financial_model(self.pv_config.fin_model, self.system_model, self.config_name)
        else:
            self.financial_model = Singleowner.from_existing(self.system_model, self.config_name)

        super().__init__("SolarPlant", self.site, self.system_model, self.financial_model)

        if self.site.solar_resource is not None:
            self.system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.dc_degradation = [0]

        if self.pv_config.layout_model is not None:
            self.layout = self.pv_config.layout_model
            self.layout._system_model = self.system_model
        else:
            self.layout = PVLayout(self.site, self.system_model, self.pv_config.layout_params)

        self.system_capacity_kw = self.pv_config.system_capacity_kw

    @property
    def system_capacity_kw(self) -> float:
        # TODO: This is currently DC power; however, all other systems are rated by AC power
        # return self._system_model.SystemDesign.system_capacity / self._system_model.SystemDesign.dc_ac_ratio
        return self.system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        self.system_model.SystemDesign.system_capacity = size_kw
        self.financial_model.value('system_capacity', size_kw) # needed for custom financial models
        self.layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]"""
        return self.system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self.system_model.Lifetime.dc_degradation = dc_deg_per_year