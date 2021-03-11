import pyomo.environ as pyomo
from pyomo.network import Port
from pyomo.environ import units as u

from hybrid.dispatch.power_source_dispatch import PowerSourceDispatch

try:
    u.USD
except AttributeError:
    u.load_definitions_from_strings(['USD = [currency]'])
u.load_definitions_from_strings(['lifecycle = [energy] / [energy]'])


class BatteryDispatch(PowerSourceDispatch):
    _model: pyomo.ConcreteModel
    _blocks: pyomo.Block
    """

    """

    def __init__(self,
                 pyomo_model: pyomo.ConcreteModel,
                 index_set: pyomo.Set,
                 block_set_name: str = 'battery',
                 include_lifecycle_cost: bool = False,
                 use_simple_battery_dispatch: bool = True):
        self.use_simple_battery_dispatch = use_simple_battery_dispatch
        self.control_variable = ""
        if not self.use_simple_battery_dispatch:
            raise ValueError('Detailed battery model has not been integrated')

        super().__init__(pyomo_model, index_set, block_set_name=block_set_name)
        self.create_soc_linking_constraint()
        self.include_lifecycle_cost = include_lifecycle_cost
        if self.include_lifecycle_cost:
            self.create_lifecycle_model()

    @staticmethod
    def dispatch_block_rule(battery):
        ##################################
        # Parameters                     #
        ##################################
        battery.time_weighting_factor = pyomo.Param(doc="Exponential time weighting factor [-]",
                                                    initialize=1.0,
                                                    within=pyomo.PercentFraction,
                                                    mutable=True,
                                                    units=u.dimensionless)
        battery.time_duration = pyomo.Param(doc="Time step [hour]",
                                            default=1.0,
                                            within=pyomo.NonNegativeReals,
                                            mutable=True,
                                            units=u.hr)
        battery.electricity_sell_price = pyomo.Param(doc="Electricity sell price [$/MWh]",
                                                     default=0.0,
                                                     within=pyomo.Reals,
                                                     mutable=True,
                                                     units=u.USD / u.MWh)
        battery.generation_cost = pyomo.Param(doc="Operating cost of battery charging and discharging [$/MWh]",
                                              default=0.,
                                              within=pyomo.NonNegativeReals,
                                              mutable=True,
                                              units=u.USD / u.MWh)
        battery.charge_efficiency = pyomo.Param(doc="Battery Charging efficiency [-]",
                                                default=0.945,
                                                within=pyomo.PercentFraction,
                                                mutable=True,
                                                units=u.dimensionless)
        battery.discharge_efficiency = pyomo.Param(doc="Battery discharging efficiency [-]",
                                                   default=0.945,
                                                   within=pyomo.PercentFraction,
                                                   mutable=True,
                                                   units=u.dimensionless)
        battery.capacity = pyomo.Param(doc="Battery manufacturer-specified capacity [MWh]",
                                       within=pyomo.NonNegativeReals,
                                       mutable=True,
                                       units=u.MWh)
        battery.minimum_power = pyomo.Param(doc="[Battery minimum power rating [MW]",
                                            default=0.0,
                                            within=pyomo.NonNegativeReals,
                                            mutable=True,
                                            units=u.MW)
        battery.maximum_power = pyomo.Param(doc="[Battery maximum power rating [MW]",
                                            within=pyomo.NonNegativeReals,
                                            mutable=True,
                                            units=u.MW)
        battery.minimum_soc = pyomo.Param(doc="Battery minimum state-of-charge [-]",
                                          default=0.1,
                                          within=pyomo.PercentFraction,
                                          mutable=True,
                                          units=u.dimensionless)
        battery.maximum_soc = pyomo.Param(doc="Battery maximum state-of-charge [-]",
                                          default=0.9,
                                          within=pyomo.PercentFraction,
                                          mutable=True,
                                          units=u.dimensionless)
        ##################################
        # Variables                      #
        ##################################
        battery.is_charging = pyomo.Var(doc="1 if battery is charging; 0 Otherwise [-]",
                                        domain=pyomo.Binary,
                                        units=u.dimensionless)
        battery.is_discharging = pyomo.Var(doc="1 if battery is discharging; 0 Otherwise [-]",
                                           domain=pyomo.Binary,
                                           units=u.dimensionless)
        battery.soc0 = pyomo.Var(doc="Battery initial state-of-charge at beginning of period[-]",
                                 domain=pyomo.PercentFraction,
                                 bounds=(battery.minimum_soc, battery.maximum_soc),
                                 units=u.dimensionless)
        battery.soc = pyomo.Var(doc="Battery state of charge at end of period [-]",
                                domain=pyomo.PercentFraction,
                                bounds=(battery.minimum_soc, battery.maximum_soc),
                                units=u.dimensionless)
        battery.charge_power = pyomo.Var(doc="Power into the battery [MW]",
                                         domain=pyomo.NonNegativeReals,
                                         units=u.MW)
        battery.discharge_power = pyomo.Var(doc="Power out of the battery [MW]",
                                            domain=pyomo.NonNegativeReals,
                                            units=u.MW)
        battery.gross_profit = pyomo.Var(doc="Sub-system gross profit [USD]",
                                         domain=pyomo.Reals,
                                         units=u.USD)
        ##################################
        # Constraints                    #
        ##################################
        battery.charge_power_lb = pyomo.Constraint(
            doc="Battery Charging power lower bound",
            expr=battery.charge_power >= battery.minimum_power * battery.is_charging)
        battery.charge_power_ub = pyomo.Constraint(
            doc="Battery Charging power upper bound",
            expr=battery.charge_power <= battery.maximum_power * battery.is_charging)
        battery.discharge_power_lb = pyomo.Constraint(
            doc="Battery Discharging power lower bound",
            expr=battery.discharge_power >= battery.minimum_power * battery.is_discharging)
        battery.discharge_power_ub = pyomo.Constraint(
            doc="Battery Discharging power upper bound",
            expr=battery.discharge_power <= battery.maximum_power * battery.is_discharging)
        battery.charge_discharge_packing = pyomo.Constraint(
            doc="Battery packing constraint for charging and discharging binaries",
            expr=battery.is_charging + battery.is_discharging <= 1)

        def soc_inventory_rule(m):
            return m.soc == (m.soc0
                             + m.time_duration * (m.charge_efficiency * m.charge_power
                                                  - (1 / m.discharge_efficiency) * m.discharge_power
                                                  ) / m.capacity
                             )

        battery.soc_inventory = pyomo.Constraint(
            doc="Battery state-of-charge inventory balance",
            rule=soc_inventory_rule)

        battery.gross_profit_calculation = pyomo.Constraint(
            doc="Calculation of gross profit for objective function",
            expr=(battery.gross_profit == (battery.time_weighting_factor * battery.electricity_sell_price
                                           * (battery.discharge_power - battery.charge_power)
                                           - (1/battery.time_weighting_factor) * battery.generation_cost
                                           * (battery.discharge_power + battery.charge_power)
                                           ) * battery.time_duration))
        ##################################
        # Ports                          #
        ##################################
        battery.port = Port()
        battery.port.add(battery.charge_power)
        battery.port.add(battery.discharge_power)

    # Linking time periods together
    def battery_soc_linking_rule(self, m, t):
        if t == self.blocks.index_set().first():
            return self.blocks[t].soc0 == self.model.battery_initial_soc
        return self.blocks[t].soc0 == self.blocks[t - 1].soc

    def create_soc_linking_constraint(self):
        ##################################
        # Parameters                     #
        ##################################
        self.model.battery_initial_soc = pyomo.Param(
            doc="Battery initial state-of-charge at beginning of the horizon[-]",
            within=pyomo.PercentFraction,
            mutable=True,
            units=u.dimensionless)
        ##################################
        # Constraints                    #
        ##################################
        self.model.soc_linking = pyomo.Constraint(
            self.blocks.index_set(),
            doc="State-of-Charge block linking constraint",
            rule=self.battery_soc_linking_rule)

    def create_lifecycle_model(self):
        self.include_lifecycle_cost = True
        ##################################
        # Parameters                     #
        ##################################
        self.model.lifecycle_cost = pyomo.Param(doc="Lifecycle cost of battery [$/lifecycle]",
                                                default=0.0,
                                                within=pyomo.NonNegativeReals,
                                                mutable=True,
                                                units=u.USD/u.lifecycle)
        ##################################
        # Variables                      #
        ##################################
        self.model.lifecycles = pyomo.Var(doc="Battery lifecycle count",
                                          domain=pyomo.NonNegativeReals,
                                          units=u.lifecycle)
        ##################################
        # Constraints                    #
        ##################################
        self.model.lifecycle_count = pyomo.Constraint(
            doc="Battery lifecycle counting",
            expr=self.model.lifecycles == sum(self.blocks[t].time_duration
                                              * self.blocks[t].discharge_power
                                              / self.blocks[t].capacity for t in self.blocks.index_set()))

    def gross_profit_objective_rule(self, m):
        objective = sum(self.blocks[t].gross_profit for t in self.blocks.index_set())
        if self.include_lifecycle_cost:
            objective -= m.lifecycle_cost * m.lifecycles
        return objective

    @property
    def available_generation(self) -> list:
        print("WARNING: " + type(self).__name__ + " does not support 'available_generation'")
        return None

    @available_generation.setter
    def available_generation(self, resource: list):
        print("WARNING: " + type(self).__name__ + " does not support 'available_generation'")

    @property
    def initial_soc(self) -> float:
        return self.model.battery_initial_soc.value * 100.

    @initial_soc.setter
    def initial_soc(self, initial_soc: float):
        if initial_soc > 1:
            initial_soc /= 100.
        self.model.battery_initial_soc = round(initial_soc, self.round_digits)

    @property
    def lifecycle_cost(self) -> float:
        return self.model.lifecycle_cost.value

    @lifecycle_cost.setter
    def lifecycle_cost(self, cost_per_lifecycle: float):
        self.model.lifecycle_cost = round(cost_per_lifecycle, self.round_digits)

    @property
    def charge_efficiency(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].charge_efficiency.value * 100.

    @charge_efficiency.setter
    def charge_efficiency(self, efficiency: float):
        if efficiency > 1:
            efficiency /= 100
        for t in self.blocks.index_set():
            self.blocks[t].charge_efficiency = round(efficiency, self.round_digits)

    @property
    def discharge_efficiency(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].discharge_efficiency.value * 100.

    @discharge_efficiency.setter
    def discharge_efficiency(self, efficiency: float):
        if efficiency > 1:
            efficiency /= 100
        for t in self.blocks.index_set():
            self.blocks[t].discharge_efficiency = round(efficiency, self.round_digits)

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency / 100.

    @round_trip_efficiency.setter
    def round_trip_efficiency(self, round_trip_efficiency: float):
        if round_trip_efficiency > 1:
            round_trip_efficiency /= 100
        efficiency = round_trip_efficiency ** (1 / 2)  # Assumes equal charge and discharge efficiencies
        self.charge_efficiency = efficiency
        self.discharge_efficiency = efficiency

    @property
    def capacity(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].capacity.value

    @capacity.setter
    def capacity(self, capacity_mwh: float):
        for t in self.blocks.index_set():
            self.blocks[t].capacity = round(capacity_mwh, self.round_digits)

    @property
    def minimum_power(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_power.value

    @minimum_power.setter
    def minimum_power(self, minimum_power_mw: float):
        for t in self.blocks.index_set():
            self.blocks[t].minimum_power = round(minimum_power_mw, self.round_digits)

    @property
    def maximum_power(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_power.value

    @maximum_power.setter
    def maximum_power(self, maximum_power_mw: float):
        for t in self.blocks.index_set():
            self.blocks[t].maximum_power = round(maximum_power_mw, self.round_digits)

    @property
    def minimum_soc(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].minimum_soc.value * 100.

    @minimum_soc.setter
    def minimum_soc(self, minimum_soc: float):
        if minimum_soc > 1:
            minimum_soc /= 100.
        for t in self.blocks.index_set():
            self.blocks[t].minimum_soc = round(minimum_soc, self.round_digits)

    @property
    def maximum_soc(self) -> float:
        for t in self.blocks.index_set():
            return self.blocks[t].maximum_soc.value * 100.

    @maximum_soc.setter
    def maximum_soc(self, maximum_soc: float):
        if maximum_soc > 1:
            maximum_soc /= 100.
        for t in self.blocks.index_set():
            self.blocks[t].maximum_soc = round(maximum_soc, self.round_digits)

    @property
    def generation(self) -> list:
        return [self.blocks[t].discharge_power.value - self.blocks[t].charge_power.value
                for t in self.blocks.index_set()]

    @property
    def is_charging(self) -> list:
        return [self.blocks[t].is_charging.value for t in self.blocks.index_set()]

    @property
    def is_discharging(self) -> list:
        return [self.blocks[t].is_discharging.value for t in self.blocks.index_set()]

    @property
    def soc(self) -> list:
        return [self.blocks[t].soc.value * 100.0 for t in self.blocks.index_set()]

    @property
    def charge_power(self) -> list:
        return [self.blocks[t].charge_power.value for t in self.blocks.index_set()]

    @property
    def discharge_power(self) -> list:
        return [self.blocks[t].discharge_power.value for t in self.blocks.index_set()]

    @property
    def power(self) -> list:
        return [self.blocks[t].discharge_power.value - self.blocks[t].charge_power.value
                for t in self.blocks.index_set()]

    @property
    def lifecycles(self) -> float:
        return self.model.lifecycles.value
