from typing import Sequence, Union, Any

import PySAM.Grid as GridModel
import PySAM.Singleowner as Singleowner

from hybrid.power_source import *
from hybrid.dispatch.grid_dispatch import GridDispatch


class Grid(PowerSource):
    _system_model: GridModel.Grid
    _financial_model: Union[Any, Singleowner.Singleowner]

    def __init__(self,
                 site: SiteInfo,
                 grid_config: dict):
        """
        Class that houses the hybrid system performance and financials. Enforces interconnection and curtailment
        limits based on PySAM's Grid module

        :param site: Power source site information (SiteInfo object)
        :param grid_config: dict, with keys ('interconnect_kw', 'fin_model')
            where:
            'interconnect_kw' is the interconnection limit [kW]
            'fin_model' is a financial model (optional)
        """
        system_model = GridModel.default("GenericSystemSingleOwner")

        if 'fin_model' in grid_config.keys():
            if isinstance(grid_config['fin_model'], Singleowner.Singleowner):
                financial_model = Singleowner.from_existing(system_model, "GenericSystemSingleOwner")
                financial_model.assign(grid_config['fin_model'].export())    
            else:
                financial_model = grid_config['fin_model']
        else:
            financial_model = Singleowner.from_existing(system_model, "GenericSystemSingleOwner")
            financial_model.value("add_om_num_types", 1)

        super().__init__("Grid", site, system_model, financial_model)

        self._system_model.GridLimits.enable_interconnection_limit = 1
        self._system_model.GridLimits.grid_interconnection_limit_kwac = grid_config['interconnect_kw']
        self._dispatch: GridDispatch = None

        # TODO: figure out if this is the best place for these
        self.missed_load = [0.]
        self.missed_load_percentage = 0.0
        self.schedule_curtailed = [0.]
        self.schedule_curtailed_percentage = 0.0

    def simulate_grid_connection(self, hybrid_size_kw: float, total_gen: list, project_life: int, lifetime_sim: bool, total_gen_max_feasible_year1: list):
        """
        Sets up and simulates hybrid system grid connection. Additionally, calculates missed load and curtailment (due to schedule) when a desired load is provided.

        :param hybrid_size_kw: ``float``,
            Hybrid system capacity [kW]
        :param total_gen: ``list``,
            Hybrid system generation profile [kWh]
        :param project_life: ``int``,
            Number of year in the analysis period (execepted project lifetime) [years]
        :param lifetime_sim: ``bool``,
            For simulation modules which support simulating each year of the project_life, whether or not to do so; otherwise the first year data is repeated
        :param total_gen_max_feasible_year1: ``list``,
            Maximum generation profile of the hybrid system (for capacity payments) [kWh]
        """
        if self.site.follow_desired_schedule:
            # Desired schedule sets the upper bound of the system output, any over generation is curtailed
            lifetime_schedule = np.tile([x * 1e3 for x in self.site.desired_schedule],
                                        int(project_life / (len(self.site.desired_schedule) // self.site.n_timesteps)))
            self.generation_profile = np.minimum(total_gen, lifetime_schedule)

            self.missed_load = [schedule - gen if gen > 0 else schedule for (schedule, gen) in
                                     zip(lifetime_schedule, self.generation_profile)]
            self.missed_load_percentage = sum(self.missed_load)/sum(lifetime_schedule)

            self.schedule_curtailed = [gen - schedule if gen > schedule else 0. for (gen, schedule) in
                                            zip(total_gen, lifetime_schedule)]
            self.schedule_curtailed_percentage = sum(self.schedule_curtailed)/sum(lifetime_schedule)
        else:
            self.generation_profile = total_gen
        self.system_capacity_kw = hybrid_size_kw  # TODO: Should this be interconnection limit?
        self.gen_max_feasible = np.minimum(total_gen_max_feasible_year1, self.interconnect_kw * self.site.interval / 60)
        self.simulate_power(project_life, lifetime_sim)

        # FIXME: updating capacity credit for reporting only.
        self.capacity_credit_percent = self.capacity_credit_percent * (self.system_capacity_kw / self.interconnect_kw)

    def calc_gen_max_feasible_kwh(self, interconnect_kw: float) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred (year 1)

        :param interconnect_kw: Interconnection limit [kW]

        :return: maximum feasible generation [kWh]
        """
        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        t_step = self.site.interval / 60                                                # hr
        E_net_max_feasible = [min(x,W_ac_nom) * t_step for x in self.total_gen_max_feasible_year1[0:self.site.n_timesteps]]      # [kWh]
        return E_net_max_feasible

    @property
    def system_capacity_kw(self) -> float:
        return self._financial_model.value('system_capacity')

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._financial_model.value('system_capacity', size_kw)

    @property
    def interconnect_kw(self) -> float:
        """Interconnection limit [kW]"""
        return self._system_model.GridLimits.grid_interconnection_limit_kwac

    @interconnect_kw.setter
    def interconnect_kw(self, interconnect_limit_kw: float):
        self._system_model.GridLimits.grid_interconnection_limit_kwac = interconnect_limit_kw

    @property
    def curtailment_ts_kw(self) -> list:
        """Grid curtailment as energy delivery limit (first year) [MW]"""
        return [i for i in self._system_model.GridLimits.grid_curtailment]

    @curtailment_ts_kw.setter
    def curtailment_ts_kw(self, curtailment_limit_timeseries_kw: Sequence):
        self._system_model.GridLimits.grid_curtailment = curtailment_limit_timeseries_kw

    @property
    def generation_profile(self) -> Sequence:
        """System power generated [kW]"""
        return self._system_model.SystemOutput.gen

    @generation_profile.setter
    def generation_profile(self, system_generation_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_kw

    @property
    def generation_profile_wo_battery(self) -> Sequence:
        """System power generated without battery [kW]"""
        return self._financial_model.value('gen_without_battery')

    @generation_profile_wo_battery.setter
    def generation_profile_wo_battery(self, system_generation_wo_battery_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_wo_battery_kw

    @property
    def generation_profile_pre_curtailment(self) -> Sequence:
        """System power before grid interconnect [kW]"""
        return self._system_model.Outputs.system_pre_interconnect_kwac

    @property
    def generation_curtailed(self) -> Sequence:
        """Generation curtailed due to interconnect limit [kW]"""
        curtailed = self.generation_profile
        pre_curtailed = self.generation_profile_pre_curtailment
        return [pre_curtailed[i] - curtailed[i] for i in range(len(curtailed))]

    @property
    def curtailment_percent(self) -> float:
        """Annual energy loss from curtailment and interconnection limit [%]"""
        return self._system_model.Outputs.annual_ac_curtailment_loss_percent \
               + self._system_model.Outputs.annual_ac_interconnect_loss_percent

    @property
    def capacity_factor_after_curtailment(self) -> float:
        """Capacity factor of the curtailment (year 1) [%]"""
        return self._system_model.Outputs.capacity_factor_curtailment_ac

    @property
    def capacity_factor_at_interconnect(self) -> float:
        """Capacity factor of the curtailment (year 1) [%]"""
        return self._system_model.Outputs.capacity_factor_interconnect_ac


