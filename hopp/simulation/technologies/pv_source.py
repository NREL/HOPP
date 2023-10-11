from typing import Sequence, Optional, Union

from attrs import define, field
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.financial import FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.base import BaseClass
from hopp.utilities.validators import gt_zero


@define
class PVConfig(BaseClass):
    """
    Configuration class for PVPlant. Converts nested dicts into relevant instances for
    layout and financial configurations.

    Args:
        system_capacity_kw: Design system capacity
        use_pvwatts: Whether to use PVWatts (defaults to True). If False, this
            config should be used in a `DetailedPVPlant`.
        layout_params: Optional layout parameters
        layout_model: Optional layout model instance
        fin_model: Optional financial model config. Can either be a string representing
            a `Singleowner` default config, or a dict representing a 
            `CustomFinancialModel`

    """
    system_capacity_kw: float = field(validator=gt_zero)

    use_pvwatts: bool = field(default=True)
    layout_params: Optional[Union[dict, PVGridParameters]] = field(default=None)
    layout_model: Optional[Union[dict, PVLayout]] = field(default=None)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)

    # converted instances
    fin_model_inst: Optional[FinancialModelType] = field(init=False)
    layout_params_inst: Optional[PVGridParameters] = field(init=False)
    layout_model_inst: Optional[PVLayout] = field(init=False)

    def __attrs_post_init__(self):
        if isinstance(self.fin_model, str):
            self.fin_model_inst = Singleowner.default(self.fin_model)
        elif isinstance(self.fin_model, dict):
            self.fin_model_inst = CustomFinancialModel(self.fin_model)
        else:
            self.fin_model_inst = self.fin_model

        if isinstance(self.layout_params, dict):
            self.layout_params_inst = PVGridParameters(**self.layout_params)
        else:
            self.layout_params_inst = self.layout_params

        if isinstance(self.layout_model, dict):
            self.layout_model_inst = PVLayout(**self.layout_model)
        else:
            self.layout_model_inst = self.layout_model


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

        if self.config.fin_model_inst is not None:
            self.financial_model = self.import_financial_model(self.config.fin_model_inst, self.system_model, self.config_name)
        else:
            self.financial_model = Singleowner.from_existing(self.system_model, self.config_name)

        super().__init__("PVPlant", self.site, self.system_model, self.financial_model)

        if self.site.solar_resource is not None:
            self.system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.dc_degradation = [0]

        if self.config.layout_model_inst is not None:
            self.layout = self.config.layout_model_inst
            self.layout._system_model = self.system_model
        else:
            self.layout = PVLayout(self.site, self.system_model, self.config.layout_params_inst)

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