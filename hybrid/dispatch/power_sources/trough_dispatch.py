from pyomo.environ import ConcreteModel, Set

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


class TroughDispatch(CspDispatch):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: None,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'trough'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

    def initialize_dispatch_model_parameters(self):
        cycle_rated_thermal = self._system_model.value('P_ref') / self._system_model.value('eta_ref')
        field_rated_thermal = self._system_model.value('specified_solar_multiple') * cycle_rated_thermal
        # TODO: This doesn't work with specified field area option

        # TODO: set these values here
        # Cost Parameters
        self.cost_per_field_generation = 3.0
        self.cost_per_field_start = self._system_model.value('disp_rsu_cost')
        self.cost_per_cycle_generation = 2.0
        self.cost_per_cycle_start = self._system_model.value('disp_csu_cost')
        self.cost_per_change_thermal_input = 0.3
        # Solar field and thermal energy storage performance parameters
        # TODO: look how these are set in SSC
        # TODO: Check units
        self.field_startup_losses = 0.0
        self.receiver_required_startup_energy = self._system_model.value('rec_qf_delay') * field_rated_thermal
        self.storage_capacity = self._system_model.value('tshours') * cycle_rated_thermal
        self.minimum_receiver_power = 0.25 * field_rated_thermal
        self.allowable_receiver_startup_power = self._system_model.value('rec_su_delay') * field_rated_thermal / 1.0
        self.receiver_pumping_losses = 0.0
        self.field_track_losses = 0.0
        self.heat_trace_losses = 0.0
        # Power cycle performance
        self.cycle_required_startup_energy = self._system_model.value('startup_frac') * cycle_rated_thermal
        self.cycle_nominal_efficiency = self._system_model.value('eta_ref')
        self.cycle_pumping_losses = self._system_model.value('pb_pump_coef')  # TODO: this is kW/kg ->
        self.allowable_cycle_startup_power = self._system_model.value('startup_time') * cycle_rated_thermal / 1.0
        self.minimum_cycle_thermal_power = self._system_model.value('cycle_cutoff_frac') * cycle_rated_thermal
        self.maximum_cycle_thermal_power = self._system_model.value('cycle_max_frac') * cycle_rated_thermal
        #self.minimum_cycle_power = ???
        self.maximum_cycle_power = self._system_model.value('P_ref')
        self.cycle_performance_slope = ((self.maximum_cycle_power - 0.0)  # TODO: need low point evaluated...
                                        / (self.maximum_cycle_thermal_power - self.minimum_cycle_thermal_power))

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        """
        This is where we need to simulate the future and capture performance for dispatch parameters
        """
        n_horizon = len(self.blocks.index_set())
        self.time_duration = [1.0] * len(self.blocks.index_set())   # assume hourly for now
        # Setting simulation times
        time_start = start_time * 3600  # seconds since new year
        # Handling end of simulation horizon
        if start_time + n_horizon > 8760:
            time_end = (start_time + (8760 - start_time)) * 3600
        else:
            time_end = (start_time + n_horizon) * 3600
        self._system_model.value('time_start', time_start)
        self._system_model.value('time_end', time_end)
        self._system_model.execute()

        # TODO: set up simulation for the next n_horizon...
        #  Simulate and get outputs for the below values

        self.available_thermal_generation = [0.0]*n_horizon
        self.cycle_ambient_efficiency_correction = [1.0]*n_horizon
        self.condenser_losses = [0.0]*n_horizon
