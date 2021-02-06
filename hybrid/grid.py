from typing import Sequence

import PySAM.Grid as GridModel

from hybrid.power_source import *


class Grid(PowerSource):
    _system_model: GridModel.Grid
    _financial_model: Singleowner.Singleowner

    def __init__(self, site: SiteInfo, interconnect_kw):
        system_model = GridModel.default("GenericSystemSingleOwner")

        financial_model: Singleowner.Singleowner = Singleowner.from_existing(system_model,
                                                                             "GenericSystemSingleOwner")
        super().__init__("Grid", site, system_model, financial_model)

        self._system_model.GridLimits.enable_interconnection_limit = 1
        self._system_model.GridLimits.grid_interconnection_limit_kwac = interconnect_kw

    @property
    def system_capacity_kw(self) -> float:
        return self._financial_model.SystemOutput.system_capacity

    @property
    def interconnect_kw(self):
        return self._system_model.GridLimits.grid_interconnection_limit_kwac

    @interconnect_kw.setter
    def interconnect_kw(self, interconnect_limit_kw: float):
        self._system_model.GridLimits.grid_interconnection_limit_kwac = interconnect_limit_kw

    @property
    def curtailment_ts_kw(self):
        """
        :return: a time series of max energy (kW) exportable to grid
        """
        return [i for i in self._system_model.GridLimits.grid_curtailment]

    @curtailment_ts_kw.setter
    def curtailment_ts_kw(self, curtailment_limit_timeseries_kw: Sequence):
        if len(curtailment_limit_timeseries_kw) != self.site.n_timesteps:
            raise ValueError("Grid error: length of curtailment_ts_kw must be ", self.site.n_timesteps)
        self._system_model.GridLimits.grid_curtailment = curtailment_limit_timeseries_kw

    @property
    def generation_profile_from_system(self):
        return self._system_model.SystemOutput.gen

    @generation_profile_from_system.setter
    def generation_profile_from_system(self, system_generation_kw: Sequence):
        if len(system_generation_kw) != self.site.n_timesteps:
            raise ValueError("Grid error: length of system_generation_kw must be ", self.site.n_timesteps)
        self._system_model.SystemOutput.gen = system_generation_kw

    def generation_profile_pre_curtailment(self) -> Sequence:
        return self._system_model.Outputs.system_pre_interconnect_kwac

    def generation_profile(self) -> Sequence:
        return self._system_model.Outputs.gen

    def generation_curtailed(self) -> Sequence:
        curtailed = self.generation_profile()
        pre_curtailed = self.generation_profile_pre_curtailment()
        return [pre_curtailed[i] - curtailed[i] for i in range(len(curtailed))]

    def curtailment_percent(self) -> float:
        return self._system_model.Outputs.annual_ac_curtailment_loss_percent \
               + self._system_model.Outputs.annual_ac_interconnect_loss_percent
