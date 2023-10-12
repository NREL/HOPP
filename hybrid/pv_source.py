from typing import Union, Optional, Sequence

import PySAM.Pvsamv1 as Pvsam
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *
from hybrid.layout.pv_layout import PVLayout, PVGridParameters
from hybrid.dispatch.power_sources.pv_dispatch import PvDispatch


class PVPlant(PowerSource):
    _system_model: Union[Pvsam.Pvsamv1, Pvwatts.Pvwattsv8]
    _financial_model: Singleowner.Singleowner
    _layout: PVLayout
    _dispatch: PvDispatch

    def __init__(self,
                 site: SiteInfo,
                 pv_config: dict):
        """

        :param pv_config: dict, with following keys:
            'system_capacity_kw': float, design system capacity
            'layout_params': dict, optional layout parameters of the SolarGridParameters type for PVLayout
            'layout_model': optional layout model object to use instead of the PVLayout model
        """
        if 'system_capacity_kw' not in pv_config.keys():
            raise ValueError

        self.config_name = "PVWattsSingleOwner"
        system_model = Pvwatts.default(self.config_name)

        if 'fin_model' in pv_config.keys():
            financial_model = self.import_financial_model(pv_config['fin_model'], system_model, self.config_name)
        else:
            financial_model = Singleowner.from_existing(system_model, self.config_name)

        super().__init__("SolarPlant", site, system_model, financial_model)

        self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self.dc_degradation = [0]

        if 'layout_model' in pv_config.keys():
            self._layout = pv_config['layout_model']
            self._layout._system_model = self._system_model
        else:
            if 'layout_params' in pv_config.keys():
                params: PVGridParameters = pv_config['layout_params']
            else:
                params = None
            self._layout = PVLayout(site, system_model, params)

        self._dispatch: PvDispatch = None

        self.system_capacity_kw: float = pv_config['system_capacity_kw']

    @property
    def system_capacity_kw(self) -> float:
        # TODO: This is currently DC power; however, all other systems are rated by AC power
        # return self._system_model.SystemDesign.system_capacity / self._system_model.SystemDesign.dc_ac_ratio
        return self._system_model.SystemDesign.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        self._system_model.SystemDesign.system_capacity = size_kw
        self._financial_model.value('system_capacity', size_kw) # needed for custom financial models
        self._layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]"""
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year
