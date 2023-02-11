from typing import Union, Sequence, Any

import PySAM.Pvsamv1 as Pvsam

from hybrid.power_source import *
from hybrid.layout.detailed_pv_layout import DetailedPVLayout, DetailedPVParameters
from hybrid.dispatch.power_sources.pv_dispatch import PvDispatch
from hybrid.financial.custom_financial_model import CustomFinancialModel


class DetailedPVPlant(PowerSource):
    """
    
    """
    _system_model: Pvsam.Pvsamv1
    _financial_model: Union[CustomFinancialModel, Any]
    _layout: Union[DetailedPVLayout, Any]
    _dispatch: PvDispatch

    def __init__(self,
                 site: SiteInfo,
                 pv_config: dict):
        """

        :param pv_config: dict, with following keys:
            'system_capacity_kw': float, desired solar capacity
            'fin_config': dict, contains `model_type` and any inputs for chosen financial model type
            'layout_params': optional DetailedPVParameters, the design vector w/ values. Required for layout modeling
            'layout_config': optional dict, contains all keys for PVLayoutConfig dataclass. Required for layout modeling
            'pan_file': optional str, PVSyst pan file will be read for the module model. See Pvsamv1.MermoudLejeuneSingleDiodeModel
            'ond_file': optional str, PVSyst ond file will be read for the inverter model. See Pvsamv1.InverterMermoudLejeuneModel
        """
        if 'system_capacity_kw' not in pv_config.keys() or 'fin_config' not in pv_config.keys():
            raise ValueError

        system_model = Pvsam.default("FlatPlatePVSingleOwner")
        financial_model = CustomFinancialModel(system_model, pv_config['fin_config'])

        super().__init__("SolarPlant", site, system_model, financial_model)

        self._dispatch: PvDispatch = None

        self._layout: DetailedPVLayout = None
        if 'layout_params' in pv_config.keys() and 'layout_config' in pv_config.keys():
            self._layout = DetailedPVLayout(site, self._system_model, pv_config['layout_params'], pv_config['layout_config'])
        elif not 'layout_params' in pv_config.keys() and not 'layout_config' in pv_config.keys():
            raise ValueError("Layout modeling requires both 'layout_params' and 'layout_config' inputs to 'pv_config'.")

        self.boq_config = None
        if 'pan_file' in pv_config.keys():
            self.load_pan_parameters(pv_config['pan_file'])
        if 'ond_file' in pv_config.keys():
            self.load_pan_parameters(pv_config['pan_file'])

        self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data
        self.dc_degradation = [0]
        self.system_capacity_kw: float = pv_config['system_capacity_kw']

    def initialize_financial_values(self):
        # fill if needed, otherwise delete
        pass

    def calc_nominal_capacity(self, interconnect_kw: float):
        # overwrite PowerSource's base function here
        pass

    def calc_capacity_credit_percent(self, interconnect_kw: float):
        # overwrite PowerSource's base function here
        pass

    def setup_performance_model(self):
        # overwrite PowerSource's base function here if needed
        pass

    def simulate_power(self, project_life, lifetime_sim=False):
        # overwrite PowerSource's base function here if needed
        pass

    def simulate_financials(self, interconnect_kw: float, project_life: int):
        # overwrite PowerSource's base function here if needed
        # this is where system-performance-dependent financial inputs should be transferred over to financial model
        pass

    def load_pan_parameters(self, pan_file):
        """
        Checks file exists, reads data, sets Pvsamv1.Module.module_model and variables in Pvsamv1.MermoudLejeuneSingleDiodeModel
        """
        pass

    def load_ond_parameters(self, ond_file):
        """
        Checks file exists, reads data, sets Pvsamv1.Inverter.inverter_model and variables in Pvsamv1.InverterMermoudLejeuneModel
        """
        pass

    def export_BOQ(self, bos_variable_access_map: dict):
        """
        This function exports all variables that are keys in the `bos_variable_access_map` dict.
        For each variable named as a key in the `bos_variable_access_map` dict, its value is the accessor method.
        If the accessor is a str, use `self.value(value)`.
        If the accessor is a function, call that function with the system model, `value(self._system_model)`
        """
        pass

    def export_financials(self):
        """
        """
        self._financial_model.export()

    #
    # Inputs
    #

    @property
    def system_capacity_kw(self) -> float:
        # TODO: Compute system capacity from strings and modules setup, figure out if this should be DC or AC
        pass

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        self._layout.set_system_capacity(size_kw)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]"""
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year

    # add any other properties