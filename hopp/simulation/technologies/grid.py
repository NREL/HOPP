from typing import Iterable, List, Sequence, Optional, Union, TYPE_CHECKING

import numpy as np
from attrs import define, field
import PySAM.Grid as GridModel
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import FinancialModelType, CustomFinancialModel
from hopp.type_dec import NDArrayFloat
from hopp.utilities.validators import gt_zero
from hopp.utilities.log import hybrid_logger as logger

if TYPE_CHECKING:
    from hopp.simulation.technologies.dispatch.hybrid_dispatch_options import HybridDispatchOptions

@define
class GridConfig(BaseClass):
    """
    Configuration data class for Grid. 

    Args:
        interconnect_kw: grid interconnection limit (kW)
        fin_model: Financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

        ppa_price: PPA price [$/kWh] used in the financial model
    """
    interconnect_kw: float = field(validator=gt_zero)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = None
    ppa_price: Optional[Union[Iterable, float]] = None


@define
class Grid(PowerSource):
    site: SiteInfo
    config: GridConfig

    # TODO: figure out if this is the best place for these
    missed_load: NDArrayFloat = field(init=False)
    missed_load_percentage: float = field(init=False, default=0.)
    schedule_curtailed: NDArrayFloat = field(init=False)
    schedule_curtailed_percentage: float = field(init=False, default=0.)
    total_gen_max_feasible_year1: NDArrayFloat = field(init=False)

    def __attrs_post_init__(self):
        """
        Class that houses the hybrid system performance and financials. Enforces interconnection and curtailment
        limits based on PySAM's Grid module.

        Args:
            site: Power source site information
            config: dict, used to instantiate a `GridConfig` instance
        """
        system_model = GridModel.default("GenericSystemSingleOwner")

        # parse user input for financial model
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)
        else:
            financial_model = self.config.fin_model

        # default
        if financial_model is None:
            financial_model = Singleowner.from_existing(system_model, "GenericSystemSingleOwner")
            financial_model.value("add_om_num_types", 1)

        super().__init__("Grid", self.site, system_model, financial_model)

        if self.config.ppa_price is not None:
            self.ppa_price = self.config.ppa_price

        self._system_model.GridLimits.enable_interconnection_limit = 1
        self._system_model.GridLimits.grid_interconnection_limit_kwac = self.config.interconnect_kw
        self._dispatch = None

        self.missed_load = np.array([0.])
        self.schedule_curtailed = np.array([0.])
        self.total_gen_max_feasible_year1 = np.array([0.])

    def simulate_grid_connection(
        # TODO: update args to use numpy types, once PowerSource is refactored
        self,
        hybrid_size_kw: float, 
        total_gen: Union[List[float], NDArrayFloat], 
        project_life: int, 
        lifetime_sim: bool, 
        total_gen_max_feasible_year1: Union[List[float], NDArrayFloat],
        dispatch_options: Optional["HybridDispatchOptions"] = None
    ):
        """
        Sets up and simulates hybrid system grid connection. Additionally,
        calculates missed load and curtailment (due to schedule) when a
        desired load is provided.

        Args:
            hybrid_size_kw: Hybrid system capacity [kW]
            total_gen: Hybrid system generation profile [kWh]
            project_life: Number of year in the analysis period (expected project
                lifetime) [years]
            lifetime_sim: For simulation modules which support simulating each year of
                the project_life, whether or not to do so; otherwise the first year
                data is repeated
            total_gen_max_feasible_year1: Maximum generation profile of the hybrid
                system (for capacity payments) [kWh]
            dispatch_options: Hybrid dispatch options class, deliminates if the higher
                power analysis for frequency regulation is run

        """
        if self.site.follow_desired_schedule:
            # Desired schedule sets the upper bound of the system output, any over generation is curtailed
            lifetime_schedule: NDArrayFloat = np.tile([
                x * 1e3 for x in self.site.desired_schedule],
                int(project_life / (len(self.site.desired_schedule) // self.site.n_timesteps))
            )
            self.generation_profile = list(np.minimum(total_gen, lifetime_schedule)) # TODO: remove list() cast once parent class uses numpy 

            self.missed_load = np.array([schedule - gen if gen > 0 else schedule for (schedule, gen) in
                                     zip(lifetime_schedule, self.generation_profile)])
            self.missed_load_percentage = sum(self.missed_load)/sum(lifetime_schedule)

            self.schedule_curtailed = np.array([gen - schedule if gen > schedule else 0. for (gen, schedule) in
                                            zip(total_gen, lifetime_schedule)])
            self.schedule_curtailed_percentage = sum(self.schedule_curtailed)/sum(lifetime_schedule)

            # NOTE: This is currently only happening for load following, would be good to make it more general
            #           i.e. so that this analysis can be used when load following isn't being used (without storage)
            #           for comparison 
            N_hybrid = len(self.generation_profile)

            final_power_production = total_gen
            schedule = [x for x in lifetime_schedule]
            hybrid_power = [(final_power_production[x] - (schedule[x]*0.95)) for x in range(len(final_power_production))]

            load_met = len([i for i in hybrid_power if i  >= 0])
            self.time_load_met = 100 * load_met/N_hybrid

            final_power_array = np.array(final_power_production)
            power_met = np.where(final_power_array > schedule, schedule, final_power_array)
            self.capacity_factor_load = np.sum(power_met) / np.sum(schedule) * 100

            logger.info('Percent of time firm power requirement is met: ', np.round(self.time_load_met,2))
            logger.info('Percent total firm power requirement is satisfied: ', np.round(self.capacity_factor_load,2))

            ERS_keys = ['min_regulation_hours', 'min_regulation_power']
            if dispatch_options is not None and dispatch_options.use_higher_hours:
                """
                Frequency regulation analysis for providing essential reliability services (ERS) availability operating case:
                        Finds how many hours (in the group specified group size above the specified minimum
                        power requirement) that the system has available to extra power that could be used to 
                        provide ERS
                Args:
                    :param dispatch_options: need additional ERS arguments
                                        'min_regulation_hours': minimum size of hours in a group to be considered for ERS (>= 1)
                                        'min_regulation_power': minimum power available over the whole group of hours to be 
                                                considered for ERS (> 0, in kW)

                :returns: total_number_hours

                """

                # Performing frequency regulation analysis:
                #    finding how many groups of hours satisfiy the ERS minimum power requirement
                min_regulation_hours = dispatch_options.higher_hours['min_regulation_hours']
                min_regulation_power = dispatch_options.higher_hours['min_regulation_power']

                frequency_power_array = np.array(hybrid_power)
                frequency_test = np.where(frequency_power_array > min_regulation_power, frequency_power_array, 0)
                mask = (frequency_test!=0).astype(int)
                padded_mask = np.pad(mask,(1,), "constant")
                edge_mask = padded_mask[1:] - padded_mask[:-1]  # finding the difference between each array value

                group_starts = np.where(edge_mask == 1)[0]
                group_stops = np.where(edge_mask == -1)[0]

                # Find groups and drop groups that are too small
                groups = [group for group in zip(group_starts,group_stops) if ((group[1]-group[0]) >= min_regulation_hours)]
                group_lengths = [len(final_power_production[group[0]:group[1]]) for group in groups]
                self.total_number_hours = sum(group_lengths)

                logger.info('Total number of hours available for ERS: ', np.round(self.total_number_hours,2))
        else:
            self.generation_profile = total_gen

        self.total_gen_max_feasible_year1 = np.array(total_gen_max_feasible_year1)
        self.system_capacity_kw = hybrid_size_kw  # TODO: Should this be interconnection limit?
        self.gen_max_feasible = list(np.minimum(  # TODO: remove list() cast once parent class uses numpy 
            total_gen_max_feasible_year1, 
            self.interconnect_kw * self.site.interval / 60
        ))
        self.simulate_power(project_life, lifetime_sim)

        # FIXME: updating capacity credit for reporting only.
        self.capacity_credit_percent = [i * (self.system_capacity_kw / self.interconnect_kw) for i in self.capacity_credit_percent]

    def calc_gen_max_feasible_kwh(self, interconnect_kw: float) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred (year 1)

        Args:
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