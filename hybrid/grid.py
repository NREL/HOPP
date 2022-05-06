from typing import Sequence

import PySAM.Grid as GridModel
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *
from hybrid.dispatch.grid_dispatch import GridDispatch


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

        # financial calculations set up
        self._financial_model.value("add_om_num_types", 1)

        self._dispatch: GridDispatch = None

    @property
    def system_capacity_kw(self) -> float:
        return self._financial_model.SystemOutput.system_capacity

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._financial_model.SystemOutput.system_capacity = size_kw

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
        self._system_model.GridLimits.grid_curtailment = curtailment_limit_timeseries_kw

    @property
    def generation_profile(self):
        return self._system_model.SystemOutput.gen

    @generation_profile.setter
    def generation_profile(self, system_generation_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_kw

    @property
    def generation_profile_wo_battery(self):
        return self._financial_model.SystemOutput.gen_without_battery

    @generation_profile_wo_battery.setter
    def generation_profile_wo_battery(self, system_generation_wo_battery_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_wo_battery_kw

    @property
    def generation_profile_pre_curtailment(self) -> Sequence:
        return self._system_model.Outputs.system_pre_interconnect_kwac

    @property
    def generation_curtailed(self) -> Sequence:
        curtailed = self.generation_profile
        pre_curtailed = self.generation_profile_pre_curtailment
        return [pre_curtailed[i] - curtailed[i] for i in range(len(curtailed))]

    @property
    def curtailment_percent(self) -> float:
        return self._system_model.Outputs.annual_ac_curtailment_loss_percent \
               + self._system_model.Outputs.annual_ac_interconnect_loss_percent

    @property
    def capacity_factor_after_curtailment(self) -> float:
        return self._system_model.Outputs.capacity_factor_curtailment_ac_curtailment

    @property
    def capacity_factor_at_interconnect(self) -> float:
        return self._system_model.Outputs.capacity_factor_interconnect_ac


