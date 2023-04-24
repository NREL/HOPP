from typing import Union, Optional, Sequence

import PySAM.Pvsamv1 as Pvsam
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.power_source import *
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.technologies.dispatch.power_sources.pv_dispatch import PvDispatch


class PVPlant(PowerSource):
    _system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv8]
    _financial_model: Singleowner.Singleowner
    _layout: PVLayout
    _dispatch: PvDispatch

    def __init__(self,
                 site: SiteInfo,
                 pv_config: dict,
                 detailed_not_simple: bool = False):
        """

        :param pv_config: dict, with keys ('system_capacity_kw', 'layout_params')
            where 'layout_params' is of the SolarGridParameters type
        :param detailed_not_simple:
            Detailed model uses Pvsamv1, simple uses PVWatts
        """
        if 'system_capacity_kw' not in pv_config.keys():
            raise ValueError

        self._detailed_not_simple: bool = detailed_not_simple

        if not detailed_not_simple:
            system_model = Pvwatts.default("PVWattsSingleOwner")
            financial_model = Singleowner.from_existing(system_model, "PVWattsSingleOwner")
        else:
            system_model = Pvsam.default("FlatPlatePVSingleOwner")
            financial_model = Singleowner.from_existing(system_model, "FlatPlatePVSingleOwner")

        super().__init__("SolarPlant", site, system_model, financial_model)

        self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.dc_degradation = [0]

        params: Optional[PVGridParameters] = None
        if 'layout_params' in pv_config.keys():
            params: PVGridParameters = pv_config['layout_params']
        self._layout = PVLayout(site, system_model, params)

        self._dispatch: PvDispatch = None

        self.system_capacity_kw: float = pv_config['system_capacity_kw']
        print(self.system_capacity_kw)

    @property
    def system_capacity_kw(self) -> float:
        return self._system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        if self._detailed_not_simple:
            raise NotImplementedError("SolarPlant error: system_capacity setter for detailed pv")
        self._system_model.SystemDesign.system_capacity = size_kw
        self._layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year
