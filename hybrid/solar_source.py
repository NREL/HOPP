from typing import Union, Optional

import PySAM.Pvsamv1 as Pvsam
import PySAM.Pvwattsv7 as Pvwatts

from hybrid.power_source import *
from hybrid.layout.solar_layout import SolarLayout, SolarGridParameters
from hybrid.dispatch.solar_dispatch import SolarDispatch


class SolarPlant(PowerSource):
    _system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv7]
    _financial_model: Singleowner.Singleowner
    _layout: SolarLayout
    _dispatch: SolarDispatch

    def __init__(self,
                 site: SiteInfo,
                 solar_config: dict,
                 detailed_not_simple: bool = False):
        """

        :param solar_config: dict, with keys ('system_capacity_kw', 'layout_params')
            where 'layout_params' is of the SolarGridParameters type
        :param detailed_not_simple:
            Detailed model uses Pvsamv1, simple uses PVWatts
        """
        if 'system_capacity_kw' not in solar_config.keys():
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

        params: Optional[SolarGridParameters] = None
        if 'layout_params' in solar_config.keys():
            params: SolarGridParameters = solar_config['layout_params']
        self._layout = SolarLayout(site, system_model, params)

        self._dispatch: SolarDispatch = None

        self.system_capacity_kw: float = solar_config['system_capacity_kw']

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



