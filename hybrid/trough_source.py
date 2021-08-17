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
