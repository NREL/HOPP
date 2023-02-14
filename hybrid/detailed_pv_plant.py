from typing import Union, Optional, Sequence, Any

import PySAM.Pvsamv1 as Pvsam
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *
from hybrid.layout.pv_design_utils import *
from hybrid.layout.pv_layout import PVLayout, PVGridParameters
from hybrid.dispatch.power_sources.pv_dispatch import PvDispatch


class DetailedPVPlant(PowerSource):
    _system_model: Pvsam.Pvsamv1
    _financial_model: Union[Any, Singleowner.Singleowner]
    _layout: Union[Any, PVLayout]
    _dispatch: PvDispatch

    def __init__(self,
                 site: SiteInfo,
                 pv_config: dict):
        """

        :param pv_config: dict, with following keys:
            'tech_config': dict, contains parameters for pvsamv1 technology model
            'fin_config': dict, contains `model_type` and any inputs for chosen financial model type
            'layout_params': optional DetailedPVParameters, the design vector w/ values. Required for layout modeling
            'layout_config': optional dict, contains all keys for PVLayoutConfig dataclass. Required for layout modeling
        """
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

        if 'tech_config' in pv_config.keys():
            self.processed_assign(pv_config['tech_config'])

    def processed_assign(self, params):
        """
        Assign attributes from dictionary with additional processing
        to enforce coherence between attributes
        """
        self.assign(params)
        calculated_system_capacity = verify_capacity_from_electrical_parameters(
            system_capacity_target=self.system_capacity,
            n_strings=self.n_strings,
            modules_per_string=self.modules_per_string,
            module_power=self.module_power
        )
        self._system_model.SystemDesign.system_capacity = calculated_system_capacity

    @property
    def system_capacity(self) -> float:
        """pass through to established name property"""
        return self.system_capacity_kw

    @system_capacity.setter
    def system_capacity(self, size_kw: float):
        """pass through to established name setter"""
        self.system_capacity_kw = size_kw

    @property
    def system_capacity_kw(self) -> float:
        return self._system_model.value('system_capacity')      # [kW] DC

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity
        :param size_kw: DC system size in kW
        :return:
        """
        self._system_model.value('system_capacity', size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]"""
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year

    @property
    def dc_ac_ratio(self) -> float:
        return self.system_capacity / (self.n_inverters * self.inverter_power)

    @property
    def module_power(self) -> float:
        """Module power in kW"""
        return get_module_power(self._system_model) * 1e-3

    @property
    def inverter_power(self) -> float:
        """Inverter power in kW"""
        return get_inverter_power(self._system_model) * 1e-3

    @property
    def modules_per_string(self) -> float:
        """Modules per string"""
        return self._system_model.SystemDesign.subarray1_modules_per_string

    @modules_per_string.setter
    def modules_per_string(self, _modules_per_string: float):
        """Sets the modules per string and updates the system capacity"""
        self._system_model.SystemDesign.subarray1_modules_per_string = _modules_per_string
        self._system_model.SystemDesign.subarray1_modules_per_string = 0 
        self._system_model.SystemDesign.subarray1_modules_per_string = 0
        self._system_model.SystemDesign.subarray1_modules_per_string = 0
        self.system_capacity = self.module_power * _modules_per_string * self.n_strings

    @property
    def n_strings(self) -> float:
        """Total number of strings"""
        return self._system_model.SystemDesign.subarray1_nstrings \
               + self._system_model.SystemDesign.subarray2_nstrings \
               + self._system_model.SystemDesign.subarray3_nstrings \
               + self._system_model.SystemDesign.subarray4_nstrings

    @n_strings.setter
    def n_strings(self, _n_strings: float):
        """Sets the total number of strings and updates the system capacity"""
        self._system_model.SystemDesign.subarray1_nstrings = _n_strings
        self._system_model.SystemDesign.subarray2_nstrings = 0 
        self._system_model.SystemDesign.subarray3_nstrings = 0
        self._system_model.SystemDesign.subarray4_nstrings = 0
        self.system_capacity = self.module_power * self.modules_per_string * _n_strings

    @property
    def n_inverters(self) -> float:
        """Total number of inverters"""
        return self._system_model.SystemDesign.inverter_count

    @n_inverters.setter
    def n_inverters(self, _n_inverters: float):
        """Sets the total number of inverters"""
        self._system_model.SystemDesign.inverter_count = _n_inverters
