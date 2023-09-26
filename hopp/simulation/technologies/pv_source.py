from typing import Sequence, Optional

from attrs import define, field
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.base import BaseClass
from hopp.utilities.validators import gt_zero


@define
class PVConfig(BaseClass):
    """
    Configuration class for PVPlant.

    Args:
        system_capacity_kw: Design system capacity
        use_pvwatts: Whether to use PVWatts (defaults to True). If False, this
            config should be used in a `DetailedPVPlant`.
        layout_params: Optional layout parameters
        layout_model: Optional layout model instance
        fin_model: Optional financial model instance

    """
    system_capacity_kw: float = field(validator=gt_zero)

    use_pvwatts: bool = field(default=True)
    layout_params: Optional[PVGridParameters] = field(default=None)
    layout_model: Optional[PVLayout] = field(default=None)
    fin_model: Optional[FinancialModelType] = field(default=None)


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

    system_model: Pvwatts.Pvwattsv8 = field(init=False)
    financial_model: FinancialModelType = field(init=False)
    config_name: str = field(init=False, default="PVWattsSingleOwner")

    def __attrs_post_init__(self):
        self.system_model = Pvwatts.default(self.config_name)

        if self.config.fin_model is not None:
            self.financial_model = self.import_financial_model(self.config.fin_model, self.system_model, self.config_name)
        else:
            self.financial_model = Singleowner.from_existing(self.system_model, self.config_name)

        super().__init__("PVPlant", self.site, self.system_model, self.financial_model)

        if self.site.solar_resource is not None:
            self.system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.dc_degradation = [0]

        if self.config.layout_model is not None:
            self.layout = self.config.layout_model
            self.layout._system_model = self.system_model
        else:
            self.layout = PVLayout(self.site, self.system_model, self.config.layout_params)

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
        return self.system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model.
        """
        self.system_model.SystemDesign.system_capacity = size_kw
        self.financial_model.value('system_capacity', size_kw) # needed for custom financial models
        self.layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]."""
        return self.system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        """Sets annual DC degradation for lifetime simulations [%/year]."""
        self.system_model.Lifetime.dc_degradation = dc_deg_per_year