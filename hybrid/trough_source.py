from typing import Optional, Union, Sequence
import PySAM.TroughPhysical as Trough
import PySAM.Singleowner as Singleowner
from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch

from hybrid.power_source import *


class TroughPlant(PowerSource):
    _system_model: Trough
    _financial_model: Singleowner.Singleowner
    # _layout: TroughLayout
    _dispatch: CspDispatch

    def __init__(self,
                 site: SiteInfo,
                 trough_config: dict):
        """

        :param trough_config: dict, with keys ('system_capacity_kw', 'solar_multiple', 'tes_hours')
        """
        # TODO: update required keys in trough_config
        required_keys = ['system_capacity_kw', 'solar_multiple', 'tes_hours']
        if all(key not in trough_config.keys() for key in required_keys):
            raise ValueError

        system_model = Trough.default('PhysicalTroughSingleOwner')
        financial_model = Singleowner.from_existing(system_model, 'PhysicalTroughSingleOwner')

        super().__init__("TroughPlant", site, system_model, financial_model)

        self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self._dispatch: CspDispatch = None

        self.cycle_capacity_kw: float = trough_config['cycle_capacity_kw']
        self.solar_multiple: float = trough_config['solar_multiple']
        self.tes_hours: float = trough_config['tes_hours']

    @property
    def system_capacity_kw(self) -> float:
        return self.cycle_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model
        :param size_kw:
        :return:
        """
        self.cycle_capacity_kw = size_kw

    @property
    def cycle_capacity_kw(self) -> float:
        return self._system_model.SystemDesign.P_ref

    @cycle_capacity_kw.setter
    def cycle_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model TODO:, cost and financial model
        :param size_kw:
        :return:
        """
        self._system_model.SystemDesign.P_ref = size_kw

    @property
    def solar_multiple(self) -> float:
        return self._system_model.SystemDesign.solarm

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self._system_model.SystemDesign.solarm = solar_multiple

    @property
    def tes_hours(self) -> float:
        return self._system_model.SystemDesign.tshours

    @tes_hours.setter
    def tes_hours(self, tes_hours: float):
        """
        Equivalent full-load thermal storage hours [hr]
        :param tes_hours:
        :return:
        """
        self._system_model.SystemDesign.tshours = tes_hours