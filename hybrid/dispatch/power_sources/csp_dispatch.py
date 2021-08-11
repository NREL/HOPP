import pyomo.environ as pyomo
from pyomo.network import Port
from pyomo.environ import units as u
from typing import Union

from hybrid.dispatch.dispatch import Dispatch


class CspDispatch(Dispatch):
    """
    Dispatch model for Concentrating Solar Power (CSP) with thermal energy storage.
    """

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 system_model,
                 financial_model,
                 block_set_name: str = 'csp'):
        super().__init__(pyomo_model,
                         index_set,
                         system_model,
                         financial_model,
                         block_set_name=block_set_name)

        super().__init__(pyomo_model, index_set, system_model, financial_model, block_set_name=block_set_name)
        self._create_linking_constraints()

    def dispatch_block_rule(self, csp):
        """
        Called during Dispatch's __init__
        """
        # Parameters
        self._create_storage_parameters(csp)
        self._create_receiver_parameters(csp)
        self._create_cycle_parameters(csp)
        # Variables
        self._create_storage_variables(csp)
        self._create_receiver_variables(csp)
        self._create_cycle_variables(csp)
        # Constraints
        self._create_storage_constraints(csp)
        self._create_receiver_constraints(csp)
        self._create_cycle_constraints(csp)
        # Ports
        self._create_csp_port(csp)

    ##################################
    # Parameters                     #
    ##################################
    # TODO: Commenting out all of the parameters currently not be used.

    @staticmethod
    def _create_storage_parameters(csp):
        csp.time_duration = pyomo.Param(
            doc="Time step [hour]",
            default=1.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.hr)
        csp.storage_capacity = pyomo.Param(
            doc="Thermal energy storage capacity [MWht]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)

    @staticmethod
    def _create_receiver_parameters(csp):
        # Cost Parameters
        # csp.cost_per_field_generation = pyomo.Param(
        #     doc="Generation cost for the csp field [$/MWht]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.USD / u.MWh)
        # csp.cost_per_field_start = pyomo.Param(
        #     doc="Penalty for field start-up [$/start]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.USD)  # $/start
        # Performance Parameters
        csp.available_thermal_generation = pyomo.Param(
            doc="Available solar thermal generation from the csp field [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.field_startup_losses = pyomo.Param(
            doc="Solar field startup or shutdown parasitic loss [MWhe]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)
        csp.receiver_required_startup_energy = pyomo.Param(
            doc="Required energy expended to start receiver [MWht]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)
        csp.receiver_pumping_losses = pyomo.Param(
            doc="Solar field and/or receiver pumping power per unit power produced [MWe/MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.dimensionless)
        csp.minimum_receiver_power = pyomo.Param(
            doc="Minimum operational thermal power delivered by receiver [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.allowable_receiver_startup_power = pyomo.Param(
            doc="Allowable power per period for receiver start-up [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.field_track_losses = pyomo.Param(
            doc="Solar field tracking parasitic loss [MWe]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.heat_trace_losses = pyomo.Param(
            doc="Piping heat trace parasitic loss [MWe]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)

    @staticmethod
    def _create_cycle_parameters(csp):
        # Cost parameters
        # csp.cost_per_cycle_generation = pyomo.Param(
        #     doc="Generation cost for power cycle [$/MWh]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.USD / u.MWh)  # Electric
        # csp.cost_per_cycle_start = pyomo.Param(
        #     doc="Penalty for power cycle start [$/start]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.USD)  # $/start
        # csp.cost_per_change_thermal_input = pyomo.Param(
        #     doc="Penalty for change in power cycle thermal input [$/MWt]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.USD / u.MW)  # $/(Delta)MW (thermal)
        # Performance parameters
        csp.cycle_ambient_efficiency_correction = pyomo.Param(
            doc="Cycle efficiency ambient temperature adjustment factor [-]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.dimensionless)
        csp.condenser_losses = pyomo.Param(
            doc="Normalized condenser parasitic losses [-]",
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.dimensionless)
        csp.cycle_required_startup_energy = pyomo.Param(
            doc="Required energy expended to start cycle [MWht]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)
        csp.cycle_nominal_efficiency = pyomo.Param(
            doc="Power cycle nominal efficiency [-]",
            default=0.0,
            within=pyomo.PercentFraction,
            mutable=True,
            units=u.dimensionless)
        csp.cycle_performance_slope = pyomo.Param(
            doc="Slope of linear approximation of power cycle performance curve [MWe/MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.dimensionless)
        csp.cycle_pumping_losses = pyomo.Param(
            doc="Cycle heat transfer fluid pumping power per unit energy expended [MWe/MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.dimensionless)
        csp.allowable_cycle_startup_power = pyomo.Param(
            doc="Allowable power per period for cycle start-up [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.minimum_cycle_thermal_power = pyomo.Param(
            doc="Minimum operational thermal power delivered to the power cycle [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        csp.maximum_cycle_thermal_power = pyomo.Param(
            doc="Maximum operational thermal power delivered to the power cycle [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)
        # csp.minimum_cycle_power = pyomo.Param(
        #     doc="Minimum cycle electric power output [MWe]",
        #     default=0.0,
        #     within=pyomo.NonNegativeReals,
        #     mutable=True,
        #     units=u.MW)
        csp.maximum_cycle_power = pyomo.Param(
            doc="Maximum cycle electric power output [MWe]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW)

    ##################################
    # Variables                      #
    ##################################

    @staticmethod
    def _create_storage_variables(csp):
        csp.thermal_energy_storage = pyomo.Var(
            doc="Thermal energy storage reserve quantity [MWht]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, csp.storage_capacity),
            units=u.MWh)
        # initial variables
        csp.previous_thermal_energy_storage = pyomo.Var(
            doc="Thermal energy storage reserve quantity at the beginning of the period [MWht]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, csp.storage_capacity),
            units=u.MWh)

    @staticmethod
    def _create_receiver_variables(csp):
        csp.receiver_startup_inventory = pyomo.Var(
            doc="Receiver start-up energy inventory [MWht]",
            domain=pyomo.NonNegativeReals,
            units=u.MWh)
        csp.receiver_thermal_power = pyomo.Var(
            doc="Thermal power delivered by the receiver [MWt]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        csp.receiver_startup_consumption = pyomo.Var(
            doc="Receiver start-up power consumption [MWt]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        csp.is_field_generating = pyomo.Var(
            doc="1 if solar field is generating 'usable' thermal power; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.is_field_starting = pyomo.Var(
            doc="1 if solar field is starting up; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.incur_field_start = pyomo.Var(
            doc="1 if solar field start-up penalty is incurred; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        # initial variables
        csp.previous_receiver_startup_inventory = pyomo.Var(
            doc="Previous receiver start-up energy inventory [MWht]",
            domain=pyomo.NonNegativeReals,
            units=u.MWh)
        csp.was_field_generating = pyomo.Var(
            doc="1 if solar field was generating 'usable' thermal power in the previous time period; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.was_field_starting = pyomo.Var(
            doc="1 if solar field was starting up in the previous time period; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)

    @staticmethod
    def _create_cycle_variables(csp):
        csp.net_generation = pyomo.Var(
            doc="Net generation of csp system [MWe]",
            domain=pyomo.Reals,
            units=u.MW)
        csp.cycle_startup_inventory = pyomo.Var(
            doc="Cycle start-up energy inventory [MWht]",
            domain=pyomo.NonNegativeReals,
            units=u.MWh)
        csp.cycle_generation = pyomo.Var(
            doc="Power cycle electricity generation [MWe]",
            domain=pyomo.NonNegativeReals,
            units=u.MW)
        csp.cycle_thermal_ramp = pyomo.Var(
            doc="Power cycle positive change in thermal energy input [MWt]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, csp.maximum_cycle_thermal_power),
            units=u.MW)
        csp.cycle_thermal_power = pyomo.Var(
            doc="Cycle thermal power utilization [MWt]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, csp.maximum_cycle_thermal_power),
            units=u.MW)
        csp.is_cycle_generating = pyomo.Var(
            doc="1 if cycle is generating electric power; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.is_cycle_starting = pyomo.Var(
            doc="1 if cycle is starting up; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.incur_cycle_start = pyomo.Var(
            doc="1 if cycle start-up penalty is incurred; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        # Initial variables
        csp.previous_cycle_startup_inventory = pyomo.Var(
            doc="Previous cycle start-up energy inventory [MWht]",
            domain=pyomo.NonNegativeReals,
            units=u.MWh)
        csp.previous_cycle_thermal_power = pyomo.Var(
            doc="Cycle thermal power in the previous period [MWt]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, csp.maximum_cycle_thermal_power),
            units=u.MW)
        csp.was_cycle_generating = pyomo.Var(
            doc="1 if cycle was generating electric power in previous time period; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)
        csp.was_cycle_starting = pyomo.Var(
            doc="1 if cycle was starting up in previous time period; 0 Otherwise [-]",
            domain=pyomo.Binary,
            units=u.dimensionless)

    ##################################
    # Constraints                    #
    ##################################

    @staticmethod
    def _create_storage_constraints(csp):
        csp.storage_inventory = pyomo.Constraint(
            doc="Thermal energy storage energy balance",
            expr=(csp.thermal_energy_storage - csp.previous_thermal_energy_storage ==
                  csp.time_duration * (csp.receiver_thermal_power
                                       - (csp.allowable_cycle_startup_power * csp.is_cycle_starting
                                          + csp.cycle_thermal_power)
                                       )
                  ))

    @staticmethod
    def _create_receiver_constraints(csp):
        # Start-up
        csp.receiver_startup_inventory_balance = pyomo.Constraint(
            doc="Receiver startup energy inventory balance",
            expr=csp.receiver_startup_inventory <= (csp.previous_receiver_startup_inventory
                                                    + csp.time_duration * csp.receiver_startup_consumption))
        csp.receiver_startup_inventory_reset = pyomo.Constraint(
            doc="Resets receiver and/or field startup inventory when startup is completed",
            expr=csp.receiver_startup_inventory <= csp.receiver_required_startup_energy * csp.is_field_starting)
        csp.receiver_operation_startup = pyomo.Constraint(
            doc="Thermal production is allowed only upon completion of start-up or operating in previous time period",
            expr=csp.is_field_generating <= (csp.receiver_startup_inventory
                                             / csp.receiver_required_startup_energy) + csp.was_field_generating)
        csp.receiver_startup_delay = pyomo.Constraint(
            doc="If field previously was producing, it cannot startup this period",
            expr=csp.is_field_starting + csp.was_field_generating <= 1)
        csp.receiver_startup_limit = pyomo.Constraint(
            doc="Receiver and/or field startup energy consumption limit",
            expr=csp.receiver_startup_consumption <= (csp.allowable_receiver_startup_power
                                                      * csp.is_field_starting))
        csp.receiver_startup_cut = pyomo.Constraint(
            doc="Receiver and/or field trivial resource startup cut",
            expr=csp.is_field_starting <= csp.available_thermal_generation / csp.minimum_receiver_power)
        # Supply and demand
        csp.receiver_energy_balance = pyomo.Constraint(
            doc="Receiver generation and startup usage must be below available",
            expr=csp.available_thermal_generation >= csp.receiver_thermal_power + csp.receiver_startup_consumption)
        csp.maximum_field_generation = pyomo.Constraint(
            doc="Receiver maximum generation limit",
            expr=csp.receiver_thermal_power <= csp.available_thermal_generation * csp.is_field_generating)
        csp.minimum_field_generation = pyomo.Constraint(
            doc="Receiver minimum generation limit",
            expr=csp.receiver_thermal_power >= csp.minimum_receiver_power * csp.is_field_generating)
        csp.receiver_generation_cut = pyomo.Constraint(
            doc="Receiver and/or field trivial resource generation cut",
            expr=csp.is_field_generating <= csp.available_thermal_generation / csp.minimum_receiver_power)
        # Logic associated with receiver modes
        csp.field_startup = pyomo.Constraint(
            doc="Ensures that field start is accounted",
            expr=csp.incur_field_start >= csp.is_field_starting - csp.was_field_starting)

    @staticmethod
    def _create_cycle_constraints(csp):
        # Start-up
        csp.cycle_startup_inventory_balance = pyomo.Constraint(
            doc="Cycle startup energy inventory balance",
            expr=csp.cycle_startup_inventory <= (csp.previous_cycle_startup_inventory
                                                 + (csp.time_duration
                                                    * csp.allowable_cycle_startup_power
                                                    * csp.is_cycle_starting)))
        csp.cycle_startup_inventory_reset = pyomo.Constraint(
            doc="Resets power cycle startup inventory when startup is completed",
            expr=csp.cycle_startup_inventory <= csp.cycle_required_startup_energy * csp.is_cycle_starting)
        csp.cycle_operation_startup = pyomo.Constraint(
            doc="Electric production is allowed only upon completion of start-up or operating in previous time period",
            expr=csp.is_cycle_generating <= (csp.cycle_startup_inventory
                                             / csp.cycle_required_startup_energy) + csp.was_cycle_generating)
        csp.cycle_startup_delay = pyomo.Constraint(
            doc="If cycle previously was generating, it cannot startup this period",
            expr=csp.is_cycle_starting + csp.was_cycle_generating <= 1)
        # Supply and demand
        csp.maximum_cycle_thermal_consumption = pyomo.Constraint(
            doc="Power cycle maximum thermal energy consumption maximum limit",
            expr=csp.cycle_thermal_power <= csp.maximum_cycle_thermal_power * csp.is_cycle_generating)
        csp.minimum_cycle_thermal_consumption = pyomo.Constraint(
            doc="Power cycle minimum thermal energy consumption minimum limit",
            expr=csp.cycle_thermal_power >= csp.minimum_cycle_thermal_power * csp.is_cycle_generating)
        csp.cycle_performance_curve = pyomo.Constraint(
            doc="Power cycle relationship between electrical power and thermal input with corrections "
                "for ambient temperature",
            expr=(csp.cycle_generation ==
                  (csp.cycle_ambient_efficiency_correction / csp.cycle_nominal_efficiency)
                  * (csp.cycle_performance_slope * csp.cycle_thermal_power
                     + (csp.maximum_cycle_power - csp.cycle_performance_slope
                        * csp.maximum_cycle_thermal_power) * csp.is_cycle_generating)))
        csp.cycle_thermal_ramp_constraint = pyomo.Constraint(
            doc="Positive ramping of power cycle thermal power",
            expr=csp.cycle_thermal_ramp >= csp.cycle_thermal_power - csp.previous_cycle_thermal_power)
        # Logic governing cycle modes
        csp.cycle_startup = pyomo.Constraint(
            doc="Ensures that cycle start is accounted",
            expr=csp.incur_cycle_start >= csp.is_cycle_starting - csp.was_cycle_starting)
        # Net generation # TODO: I don't really know if this level of detail is required...
        csp.net_generation_balance = pyomo.Constraint(
            doc="Calculates csp system net generation for grid model",
            expr=csp.net_generation == (csp.cycle_generation * (1 - csp.condenser_losses)
                                        - csp.receiver_pumping_losses * (csp.receiver_thermal_power
                                                                         + csp.receiver_startup_consumption)
                                        - csp.cycle_pumping_losses * csp.cycle_thermal_power
                                        - csp.field_track_losses * csp.is_field_generating
                                        - csp.heat_trace_losses * csp.is_field_starting
                                        - (csp.field_startup_losses/csp.time_duration) * csp.is_field_starting))

    ##################################
    # Ports                          #
    ##################################

    @staticmethod
    def _create_csp_port(csp):
        csp.port = Port()
        csp.port.add(csp.net_generation)
        # TODO: this is going to require all objective variables

    ##################################
    # Linking Constraints            #
    ##################################

    def _create_linking_constraints(self):
        self._create_storage_linking_constraints()
        self._create_receiver_linking_constraints()
        self._create_cycle_linking_constraints()

    ##################################
    # Initial Parameters             #
    ##################################

    def _create_storage_linking_constraints(self):
        self.model.initial_thermal_energy_storage = pyomo.Param(
            doc="Initial thermal energy storage reserve quantity at beginning of the horizon [MWht]",
            within=pyomo.NonNegativeReals,
            # validate= # TODO: Might be worth looking into
            mutable=True,
            units=u.MWh)

        def tes_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].previous_thermal_energy_storage == self.model.initial_thermal_energy_storage
            return self.blocks[t].previous_thermal_energy_storage == self.blocks[t - 1].thermal_energy_storage
        self.model.tes_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Thermal energy storage block linking constraint",
            rule=tes_linking_rule)

    def _create_receiver_linking_constraints(self):
        self.model.initial_receiver_startup_inventory = pyomo.Param(
            doc="Initial receiver start-up energy inventory at beginning of the horizon [MWht]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)
        self.model.is_field_generating_initial = pyomo.Param(
            doc="1 if solar field is generating 'usable' thermal power at beginning of the horizon; 0 Otherwise [-]",
            default=0.0,
            within=pyomo.Binary,
            mutable=True,
            units=u.dimensionless)
        self.model.is_field_starting_initial = pyomo.Param(
            doc="1 if solar field is starting up at beginning of the horizon; 0 Otherwise [-]",
            default=0.0,
            within=pyomo.Binary,
            mutable=True,
            units=u.dimensionless)

        def receiver_startup_inventory_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].previous_receiver_startup_inventory == self.model.initial_receiver_startup_inventory
            return self.blocks[t].previous_receiver_startup_inventory == self.blocks[t - 1].receiver_startup_inventory
        self.model.receiver_startup_inventory_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Receiver startup inventory block linking constraint",
            rule=receiver_startup_inventory_linking_rule)

        def field_generating_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].was_field_generating == self.model.is_field_generating_initial
            return self.blocks[t].was_field_generating == self.blocks[t - 1].is_field_generating
        self.model.field_generating_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Is field generating binary block linking constraint",
            rule=field_generating_linking_rule)

        def field_starting_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].was_field_starting == self.model.is_field_starting_initial
            return self.blocks[t].was_field_starting == self.blocks[t - 1].is_field_starting
        self.model.field_starting_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Is field starting up binary block linking constraint",
            rule=field_starting_linking_rule)

    def _create_cycle_linking_constraints(self):
        self.model.initial_cycle_startup_inventory = pyomo.Param(
            doc="Initial cycle start-up energy inventory at beginning of the horizon [MWht]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MWh)
        self.model.initial_cycle_thermal_power = pyomo.Param(
            doc="Initial cycle thermal power at beginning of the horizon [MWt]",
            default=0.0,
            within=pyomo.NonNegativeReals,
            # validate= # TODO: bounds->(0, csp.maximum_cycle_thermal_power), Sec. 4.7.1
            mutable=True,
            units=u.MW)
        self.model.is_cycle_generating_initial = pyomo.Param(
            doc="1 if cycle is generating electric power at beginning of the horizon; 0 Otherwise [-]",
            default=0.0,
            within=pyomo.Binary,
            mutable=True,
            units=u.dimensionless)
        self.model.is_cycle_starting_initial = pyomo.Param(
            doc="1 if cycle is starting up at beginning of the horizon; 0 Otherwise [-]",
            default=0.0,
            within=pyomo.Binary,
            mutable=True,
            units=u.dimensionless)

        def cycle_startup_inventory_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].previous_cycle_startup_inventory == self.model.initial_cycle_startup_inventory
            return self.blocks[t].previous_cycle_startup_inventory == self.blocks[t - 1].cycle_startup_inventory
        self.model.cycle_startup_inventory_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Cycle startup inventory block linking constraint",
            rule=cycle_startup_inventory_linking_rule)

        def cycle_thermal_power_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].previous_cycle_thermal_power == self.model.initial_cycle_thermal_power
            return self.blocks[t].previous_cycle_thermal_power == self.blocks[t - 1].cycle_thermal_power
        self.model.cycle_thermal_power_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Cycle thermal power block linking constraint",
            rule=cycle_thermal_power_linking_rule)

        def cycle_generating_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].was_cycle_generating == self.model.is_cycle_generating_initial
            return self.blocks[t].was_cycle_generating == self.blocks[t - 1].is_cycle_generating
        self.model.cycle_generating_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Is cycle generating binary block linking constraint",
            rule=cycle_generating_linking_rule)

        def cycle_starting_linking_rule(m, t):
            if t == self.blocks.index_set().first():
                return self.blocks[t].was_cycle_starting == self.model.is_cycle_starting_initial
            return self.blocks[t].was_cycle_starting == self.blocks[t - 1].is_cycle_starting
        self.model.cycle_starting_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="Is cycle starting up binary block linking constraint",
            rule=cycle_starting_linking_rule)

    def initialize_dispatch_model_parameters(self):
        cycle_rated_thermal = self._system_model.value('P_ref') / self._system_model.value('eta_ref')
        field_rated_thermal = self._system_model.value('solar_mult') * cycle_rated_thermal

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
        n_horizon = len(self.blocks.index_set())
        #generation = self._system_model.value("gen")
        # Handling end of simulation horizon
        # if start_time + n_horizon > len(generation):
        #     horizon_gen = list(generation[start_time:])
        #     horizon_gen.extend(list(generation[0:n_horizon - len(horizon_gen)]))
        # else:
        #     horizon_gen = generation[start_time:start_time + n_horizon]

        # FIXME: There is a bit of work to do here
        # TODO: set these values here
        self.time_duration = [1.0] * len(self.blocks.index_set())
        self.available_thermal_generation = [0.0]*n_horizon
        self.cycle_ambient_efficiency_correction = [1.0]*n_horizon
        self.condenser_losses = [0.0]*n_horizon

    def update_initial_conditions(self):
        # FIXME: There is a bit of work to do here
        # TODO: set these values here
        self.initial_thermal_energy_storage = 0.0  # Might need to calculate this

        # TODO: This appears to be coming from AMPL data files... This will take getters to be set up in pySAM...
        self.initial_receiver_startup_inventory = (self.receiver_required_startup_energy
                                                   - self._system_model.value('rec_startup_energy_remain_final') )
        self.is_field_generating_initial = self._system_model.value('is_field_tracking_final')
        self.is_field_starting_initial = self._system_model.value('rec_op_mode_final') # TODO: this is not right

        self.initial_cycle_startup_inventory = (self.cycle_required_startup_energy
                                                - self._system_model.value('pc_startup_energy_remain_final') )
        self.initial_cycle_thermal_power = self._system_model.value('q_pb')
        self.is_cycle_generating_initial = self._system_model.value('pc_op_mode_final')  # TODO: figure out what this is...
        self.is_cycle_starting_initial = False

    # INPUTS
    @property
    def time_duration(self) -> list:
        """Dispatch horizon time steps [hour]"""
        # TODO: Should we make this constant within dispatch horizon?
        return [self.blocks[t].time_duration.value for t in self.blocks.index_set()]

    @time_duration.setter
    def time_duration(self, time_duration: list):
        """Dispatch horizon time steps [hour]"""
        if len(time_duration) == len(self.blocks):
            for t, delta in zip(self.blocks, time_duration):
                self.blocks[t].time_duration = round(delta, self.round_digits)
        else:
            raise ValueError(self.time_duration.__name__ + " list must be the same length as time horizon")

    @property
    def available_thermal_generation(self) -> list:
        """Available solar thermal generation from the csp field [MWt]"""
        return [self.blocks[t].available_thermal_generation.value for t in self.blocks.index_set()]

    @available_thermal_generation.setter
    def available_thermal_generation(self, available_thermal_generation: list):
        """Available solar thermal generation from the csp field [MWt]"""
        if len(available_thermal_generation) == len(self.blocks):
            for t, value in zip(self.blocks, available_thermal_generation):
                self.blocks[t].available_thermal_generation = round(value, self.round_digits)
        else:
            raise ValueError(self.available_thermal_generation.__name__ + " list must be the same length as time horizon")

    @property
    def cycle_ambient_efficiency_correction(self) -> list:
        """Cycle efficiency ambient temperature adjustment factor [-]"""
        return [self.blocks[t].cycle_ambient_efficiency_correction.value for t in self.blocks.index_set()]

    @cycle_ambient_efficiency_correction.setter
    def cycle_ambient_efficiency_correction(self, cycle_ambient_efficiency_correction: list):
        """Cycle efficiency ambient temperature adjustment factor [-]"""
        if len(cycle_ambient_efficiency_correction) == len(self.blocks):
            for t, value in zip(self.blocks, cycle_ambient_efficiency_correction):
                self.blocks[t].cycle_ambient_efficiency_correction = round(value, self.round_digits)
        else:
            raise ValueError(self.cycle_ambient_efficiency_correction.__name__ + " list must be the same length as time horizon")

    @property
    def condenser_losses(self) -> list:
        """Normalized condenser parasitic losses [-]"""
        return [self.blocks[t].condenser_losses.value for t in self.blocks.index_set()]

    @condenser_losses.setter
    def condenser_losses(self, condenser_losses: list):
        """Normalized condenser parasitic losses [-]"""
        if len(condenser_losses) == len(self.blocks):
            for t, value in zip(self.blocks, condenser_losses):
                self.blocks[t].condenser_losses = round(value, self.round_digits)
        else:
            raise ValueError(self.condenser_losses.__name__ + " list must be the same length as time horizon")

    @property
    def cost_per_field_generation(self) -> float:
        """Generation cost for the csp field [$/MWht]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_field_generation.value

    @cost_per_field_generation.setter
    def cost_per_field_generation(self, om_dollar_per_mwh_thermal: float):
        """Generation cost for the csp field [$/MWht]"""
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_field_generation.set_value(round(om_dollar_per_mwh_thermal, self.round_digits))

    @property
    def cost_per_field_start(self) -> float:
        """Penalty for field start-up [$/start]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_field_start.value

    @cost_per_field_start.setter
    def cost_per_field_start(self, dollars_per_start: float):
        """Penalty for field start-up [$/start]"""
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_field_start.set_value(round(dollars_per_start, self.round_digits))

    @property
    def cost_per_cycle_generation(self) -> float:
        """Generation cost for power cycle [$/MWhe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_cycle_generation.value

    @cost_per_cycle_generation.setter
    def cost_per_cycle_generation(self, om_dollar_per_mwh_electric: float):
        """Generation cost for power cycle [$/MWhe]"""
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_cycle_generation.set_value(round(om_dollar_per_mwh_electric, self.round_digits))

    @property
    def cost_per_cycle_start(self) -> float:
        """Penalty for power cycle start [$/start]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_cycle_start.value

    @cost_per_cycle_start.setter
    def cost_per_cycle_start(self, dollars_per_start: float):
        """Penalty for power cycle start [$/start]"""
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_cycle_start.set_value(round(dollars_per_start, self.round_digits))

    @property
    def cost_per_change_thermal_input(self) -> float:
        """Penalty for change in power cycle thermal input [$/MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cost_per_change_thermal_input.value

    @cost_per_change_thermal_input.setter
    def cost_per_change_thermal_input(self, dollars_per_thermal_power: float):
        """Penalty for change in power cycle thermal input [$/MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].cost_per_change_thermal_input.set_value(round(dollars_per_thermal_power, self.round_digits))

    @property
    def field_startup_losses(self) -> float:
        """Solar field startup or shutdown parasitic loss [MWhe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].field_startup_losses.value

    @field_startup_losses.setter
    def field_startup_losses(self, field_startup_losses: float):
        """Solar field startup or shutdown parasitic loss [MWhe]"""
        for t in self.blocks.index_set():
            self.blocks[t].field_startup_losses.set_value(round(field_startup_losses, self.round_digits))

    @property
    def receiver_required_startup_energy(self) -> float:
        """Required energy expended to start receiver [MWht]"""
        for t in self.blocks.index_set():
            return self.blocks[t].receiver_required_startup_energy.value

    @receiver_required_startup_energy.setter
    def receiver_required_startup_energy(self, energy: float):
        """Required energy expended to start receiver [MWht]"""
        for t in self.blocks.index_set():
            self.blocks[t].receiver_required_startup_energy.set_value(round(energy, self.round_digits))

    @property
    def storage_capacity(self) -> float:
        """Thermal energy storage capacity [MWht]"""
        for t in self.blocks.index_set():
            return self.blocks[t].storage_capacity.value

    @storage_capacity.setter
    def storage_capacity(self, energy: float):
        """Thermal energy storage capacity [MWht]"""
        for t in self.blocks.index_set():
            self.blocks[t].storage_capacity.set_value(round(energy, self.round_digits))

    @property
    def receiver_pumping_losses(self) -> float:
        """Solar field and/or receiver pumping power per unit power produced [MWe/MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].receiver_pumping_losses.value

    @receiver_pumping_losses.setter
    def receiver_pumping_losses(self, electric_per_thermal: float):
        """Solar field and/or receiver pumping power per unit power produced [MWe/MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].receiver_pumping_losses.set_value(round(electric_per_thermal, self.round_digits))

    @property
    def minimum_receiver_power(self) -> float:
        """Minimum operational thermal power delivered by receiver [MWht]"""
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_receiver_power.value

    @minimum_receiver_power.setter
    def minimum_receiver_power(self, thermal_power: float):
        """Minimum operational thermal power delivered by receiver [MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].minimum_receiver_power.set_value(round(thermal_power, self.round_digits))

    @property
    def allowable_receiver_startup_power(self) -> float:
        """Allowable power per period for receiver start-up [MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].allowable_receiver_startup_power.value

    @allowable_receiver_startup_power.setter
    def allowable_receiver_startup_power(self, thermal_power: float):
        """Allowable power per period for receiver start-up [MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].allowable_receiver_startup_power.set_value(round(thermal_power, self.round_digits))

    @property
    def field_track_losses(self) -> float:
        """Solar field tracking parasitic loss [MWe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].field_track_losses.value

    @field_track_losses.setter
    def field_track_losses(self, electric_power: float):
        """Solar field tracking parasitic loss [MWe]"""
        for t in self.blocks.index_set():
            self.blocks[t].field_track_losses.set_value(round(electric_power, self.round_digits))

    @property
    def heat_trace_losses(self) -> float:
        """Piping heat trace parasitic loss [MWe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].heat_trace_losses.value

    @heat_trace_losses.setter
    def heat_trace_losses(self, electric_power: float):
        """Piping heat trace parasitic loss [MWe]"""
        for t in self.blocks.index_set():
            self.blocks[t].heat_trace_losses.set_value(round(electric_power, self.round_digits))

    @property
    def cycle_required_startup_energy(self) -> float:
        """Required energy expended to start cycle [MWht]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cycle_required_startup_energy.value

    @cycle_required_startup_energy.setter
    def cycle_required_startup_energy(self, thermal_energy: float):
        """Required energy expended to start cycle [MWht]"""
        for t in self.blocks.index_set():
            self.blocks[t].cycle_required_startup_energy.set_value(round(thermal_energy, self.round_digits))

    @property
    def cycle_nominal_efficiency(self) -> float:
        """Power cycle nominal efficiency [-]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cycle_nominal_efficiency.value * 100.

    @cycle_nominal_efficiency.setter
    def cycle_nominal_efficiency(self, efficiency: float):
        """Power cycle nominal efficiency [-]"""
        efficiency = self._check_efficiency_value(efficiency)
        for t in self.blocks.index_set():
            self.blocks[t].cycle_nominal_efficiency.set_value(round(efficiency, self.round_digits))

    @property
    def cycle_performance_slope(self) -> float:
        """Slope of linear approximation of power cycle performance curve [MWe/MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cycle_performance_slope.value

    @cycle_performance_slope.setter
    def cycle_performance_slope(self, slope: float):
        """Slope of linear approximation of power cycle performance curve [MWe/MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].cycle_performance_slope.set_value(round(slope, self.round_digits))

    @property
    def cycle_pumping_losses(self) -> float:
        """Cycle heat transfer fluid pumping power per unit energy expended [MWe/MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].cycle_pumping_losses.value

    @cycle_pumping_losses.setter
    def cycle_pumping_losses(self, electric_per_thermal: float):
        """Cycle heat transfer fluid pumping power per unit energy expended [MWe/MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].cycle_pumping_losses.set_value(round(electric_per_thermal, self.round_digits))

    @property
    def allowable_cycle_startup_power(self) -> float:
        """Allowable power per period for cycle start-up [MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].allowable_cycle_startup_power.value

    @allowable_cycle_startup_power.setter
    def allowable_cycle_startup_power(self, thermal_power: float):
        """Allowable power per period for cycle start-up [MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].allowable_cycle_startup_power.set_value(round(thermal_power, self.round_digits))

    @property
    def minimum_cycle_thermal_power(self) -> float:
        """Minimum operational thermal power delivered to the power cycle [MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_cycle_thermal_power.value

    @minimum_cycle_thermal_power.setter
    def minimum_cycle_thermal_power(self, thermal_power: float):
        """Minimum operational thermal power delivered to the power cycle [MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].minimum_cycle_thermal_power.set_value(round(thermal_power, self.round_digits))

    @property
    def maximum_cycle_thermal_power(self) -> float:
        """Maximum operational thermal power delivered to the power cycle [MWt]"""
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_cycle_thermal_power.value

    @maximum_cycle_thermal_power.setter
    def maximum_cycle_thermal_power(self, thermal_power: float):
        """Maximum operational thermal power delivered to the power cycle [MWt]"""
        for t in self.blocks.index_set():
            self.blocks[t].maximum_cycle_thermal_power.set_value(round(thermal_power, self.round_digits))

    @property
    def minimum_cycle_power(self) -> float:
        """Minimum cycle electric power output [MWe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_cycle_power.value

    @minimum_cycle_power.setter
    def minimum_cycle_power(self, electric_power: float):
        """Minimum cycle electric power output [MWe]"""
        for t in self.blocks.index_set():
            self.blocks[t].minimum_cycle_power.set_value(round(electric_power, self.round_digits))

    @property
    def maximum_cycle_power(self) -> float:
        """Maximum cycle electric power output [MWe]"""
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_cycle_power.value

    @maximum_cycle_power.setter
    def maximum_cycle_power(self, electric_power: float):
        """Maximum cycle electric power output [MWe]"""
        for t in self.blocks.index_set():
            self.blocks[t].maximum_cycle_power.set_value(round(electric_power, self.round_digits))

    # INITIAL CONDITIONS
    @property
    def initial_thermal_energy_storage(self) -> float:
        """Initial thermal energy storage reserve quantity at beginning of the horizon [MWht]"""
        return self.model.initial_thermal_energy_storage.value

    @initial_thermal_energy_storage.setter
    def initial_thermal_energy_storage(self, initial_energy: float):
        """Initial thermal energy storage reserve quantity at beginning of the horizon [MWht]"""
        self.model.initial_thermal_energy_storage = round(initial_energy, self.round_digits)

    @property
    def initial_receiver_startup_inventory(self) -> float:
        """Initial receiver start-up energy inventory at beginning of the horizon [MWht]"""
        return self.model.initial_receiver_startup_inventory.value

    @initial_receiver_startup_inventory.setter
    def initial_receiver_startup_inventory(self, initial_energy: float):
        """Initial receiver start-up energy inventory at beginning of the horizon [MWht]"""
        self.model.initial_receiver_startup_inventory = round(initial_energy, self.round_digits)

    @property
    def is_field_generating_initial(self) -> bool:
        """True (1) if solar field is generating 'usable' thermal power at beginning of the horizon;
         False (0) Otherwise [-]"""
        return bool(self.model.is_field_generating_initial.value)

    @is_field_generating_initial.setter
    def is_field_generating_initial(self, is_field_generating: Union[bool, int]):
        """True (1) if solar field is generating 'usable' thermal power at beginning of the horizon;
         False (0) Otherwise [-]"""
        self.model.is_field_generating_initial = int(is_field_generating)

    @property
    def is_field_starting_initial(self) -> bool:
        """True (1) if solar field  is starting up at beginning of the horizon; False (0) Otherwise [-]"""
        return bool(self.model.is_field_starting_initial.value)

    @is_field_starting_initial.setter
    def is_field_starting_initial(self, is_field_starting: Union[bool, int]):
        """True (1) if solar field  is starting up at beginning of the horizon; False (0) Otherwise [-]"""
        self.model.is_field_starting_initial = int(is_field_starting)

    @property
    def initial_cycle_startup_inventory(self) -> float:
        """Initial cycle start-up energy inventory at beginning of the horizon [MWht]"""
        return self.model.initial_cycle_startup_inventory.value

    @initial_cycle_startup_inventory.setter
    def initial_cycle_startup_inventory(self, initial_energy: float):
        """Initial cycle start-up energy inventory at beginning of the horizon [MWht]"""
        self.model.initial_cycle_startup_inventory = round(initial_energy, self.round_digits)

    @property
    def initial_cycle_thermal_power(self) -> float:
        """Initial cycle thermal power at beginning of the horizon [MWt]"""
        return self.model.initial_cycle_thermal_power.value

    @initial_cycle_thermal_power.setter
    def initial_cycle_thermal_power(self, initial_power: float):
        """Initial cycle thermal power at beginning of the horizon [MWt]"""
        self.model.initial_cycle_thermal_power = round(initial_power, self.round_digits)

    @property
    def is_cycle_generating_initial(self) -> bool:
        """True (1) if cycle is generating electric power at beginning of the horizon; False (0) Otherwise [-]"""
        return bool(self.model.is_cycle_generating_initial.value)

    @is_cycle_generating_initial.setter
    def is_cycle_generating_initial(self, is_cycle_generating: Union[bool, int]):
        """True (1) if cycle is generating electric power at beginning of the horizon; False (0) Otherwise [-]"""
        self.model.is_cycle_generating_initial = int(is_cycle_generating)

    @property
    def is_cycle_starting_initial(self) -> bool:
        """True (1) if cycle is starting up at beginning of the horizon; False (0) Otherwise [-]"""
        return bool(self.model.is_cycle_starting_initial.value)

    @is_cycle_starting_initial.setter
    def is_cycle_starting_initial(self, is_cycle_starting: Union[bool, int]):
        """True (1) if cycle is starting up at beginning of the horizon; False (0) Otherwise [-]"""
        self.model.is_cycle_starting_initial = int(is_cycle_starting)

    # OUTPUTS
    @property
    def thermal_energy_storage(self) -> list:
        """Thermal energy storage reserve quantity [MWht]"""
        return [round(self.blocks[t].thermal_energy_storage.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def receiver_startup_inventory(self) -> list:
        """Receiver start-up energy inventory [MWht]"""
        return [round(self.blocks[t].receiver_startup_inventory.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def receiver_thermal_power(self) -> list:
        """Thermal power delivered by the receiver [MWt]"""
        return [round(self.blocks[t].receiver_thermal_power.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def receiver_startup_consumption(self) -> list:
        """Receiver start-up power consumption [MWt]"""
        return [round(self.blocks[t].receiver_startup_consumption.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def is_field_generating(self) -> list:
        """1 if solar field is generating 'usable' thermal power; 0 Otherwise [-]"""
        return [round(self.blocks[t].is_field_generating.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def is_field_starting(self) -> list:
        """1 if solar field is starting up; 0 Otherwise [-]"""
        return [round(self.blocks[t].is_field_starting.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def incur_field_start(self) -> list:
        """1 if solar field start-up penalty is incurred; 0 Otherwise [-]"""
        return [round(self.blocks[t].incur_field_start.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def cycle_startup_inventory(self) -> list:
        """Cycle start-up energy inventory [MWht]"""
        return [round(self.blocks[t].cycle_startup_inventory.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def net_generation(self) -> list:
        """Net generation of csp system [MWe]"""
        return [round(self.blocks[t].net_generation.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def cycle_generation(self) -> list:
        """Power cycle electricity generation [MWe]"""
        return [round(self.blocks[t].cycle_generation.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def cycle_thermal_ramp(self) -> list:
        """Power cycle positive change in thermal energy input [MWt]"""
        return [round(self.blocks[t].cycle_thermal_ramp.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def cycle_thermal_power(self) -> list:
        """Cycle thermal power utilization [MWt]"""
        return [round(self.blocks[t].cycle_thermal_power.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def is_cycle_generating(self) -> list:
        """1 if cycle is generating electric power; 0 Otherwise [-]"""
        return [round(self.blocks[t].is_cycle_generating.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def is_cycle_starting(self) -> list:
        """1 if cycle is starting up; 0 Otherwise [-]"""
        return [round(self.blocks[t].is_cycle_starting.value, self.round_digits) for t in self.blocks.index_set()]

    @property
    def incur_cycle_start(self) -> list:
        """1 if cycle start-up penalty is incurred; 0 Otherwise [-]"""
        return [round(self.blocks[t].incur_cycle_start.value, self.round_digits) for t in self.blocks.index_set()]

