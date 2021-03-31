import PySAM.BatteryStateful as BatteryModel
import PySAM.BatteryTools as BatteryTools

from hybrid.power_source import *

class Battery_Outputs:
    def __init__(self, n_timesteps):
        """ Class of stateful battery outputs

        """
        self.stateful_attributes = ['I', 'P', 'Q', 'SOC', 'T_batt', 'gen']
        for attr in self.stateful_attributes:
            setattr(self, attr, [0]*n_timesteps)

        # dispatch output storage
        dispatch_attributes = ['I', 'P', 'SOC']
        for attr in dispatch_attributes:
            setattr(self, 'dispatch_'+attr, [0]*n_timesteps)


class Battery(PowerSource):
    _system_model: BatteryModel.BatteryStateful
    _financial_model: Singleowner.Singleowner

    def __init__(self,
                 site: SiteInfo,
                 system_capacity_kwh: float,
                 chemistry: str = 'lfpgraphite',
                 system_voltage_volts: float = 500,
                 system_capacity_kw: float = None):
        """

        :param system_capacity_kwh:
        :param system_voltage_volts:
        """
        system_model = BatteryModel.default(chemistry)
        self.Outputs = Battery_Outputs(n_timesteps=site.n_timesteps)
        BatteryTools.battery_model_sizing(system_model,
                                          0.,
                                          system_capacity_kwh,
                                          system_voltage_volts)
                                          #module_specs={'capacity': 400, 'surface_area': 30})  # 400 [kWh] -> 30 [m^2]

        financial_model = Singleowner.from_existing(system_model, "GenericBatterySingleOwner")
        super().__init__("Battery", site, system_model, financial_model)

        if system_capacity_kw is not None:
            self.system_capacity_kw = system_capacity_kw

        # Minimum set of parameters to set to get statefulBattery to work
        self._system_model.value("control_mode", 0.0)
        self._system_model.value("input_current", 0.0)
        self._system_model.value("dt_hr", 1.0)
        self._system_model.value("minimum_SOC", 10)
        self._system_model.value("maximum_SOC", 90)
        self._system_model.value("initial_SOC", 10.0)
        self._system_model.setup()

    @property
    def system_capacity_voltage(self) -> tuple:
        return self._system_model.ParamsPack.nominal_energy, self._system_model.ParamsPack.nominal_voltage

    @system_capacity_voltage.setter
    def system_capacity_voltage(self, capacity_voltage: tuple):
        """
        Sets the system capacity and voltage, and updates the system, cost and financial model
        :param capacity_voltage:
        :return:
        """
        size_kwh = capacity_voltage[0]
        voltage_volts = capacity_voltage[1]

        BatteryTools.battery_model_sizing(self._system_model,
                                          0.,
                                          size_kwh,
                                          voltage_volts,
                                          module_specs={'capacity': 400, 'surface_area': 30})
        self.system_capacity_kwh: float = self.system_model.ParamsPack.nominal_energy
        self.system_voltage_volts: float = self.system_model.ParamsPack.nominal_voltage
        logger.info("Battery set system_capacity to {} kWh".format(size_kwh))
        logger.info("Battery set system_voltage to {} volts".format(voltage_volts))

    @property
    def system_capacity_kwh(self) -> float:
        return self._system_model.ParamsPack.nominal_energy

    @system_capacity_kwh.setter
    def system_capacity_kwh(self, size_kwh: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kwh:
        :return:
        """
        self.system_capacity_voltage = (size_kwh, self.system_voltage_volts)

    @property
    def system_capacity_kw(self) -> float:
        return self._system_model.ParamsPack.nominal_energy * self._system_model.ParamsCell.C_rate

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kw:
        :return:
        """
        self._system_model.value("C_rate", size_kw/self._system_model.ParamsPack.nominal_energy)

    @property
    def system_voltage_volts(self) -> float:
        return self._system_model.ParamsPack.nominal_voltage

    @system_voltage_volts.setter
    def system_voltage_volts(self, voltage_volts: float):
        """
        Sets the system voltage and updates the system, cost and financial model
        :param voltage_volts:
        :return:
        """
        self.system_capacity_voltage = (self.system_capacity_kwh, voltage_volts)

    @property
    def chemistry(self) -> str:
        model_type = self._system_model.ParamsCell.chem
        if model_type == 0:
            return "0 [LeadAcid]"
        elif model_type == 1:
            return "1 [nmcgraphite or lfpgraphite]"  # TODO: Currently, there is no way to tell the difference...
        else:
            return ValueError("chemistry model type unrecognized")

    @chemistry.setter
    def chemistry(self, battery_chemistry: str):
        """
        Sets the system chemistry and updates the system, cost and financial model
        :param battery_chemistry:
        :return:
        """
        BatteryTools.battery_model_change_chemistry(self._system_model, battery_chemistry)
        logger.info("Battery chemistry set to {}".format(battery_chemistry))

    def simulate(self, project_life: int = 25, time_step=None):
        """
        Runs battery simulate stores values if time step is provided
        """
        if not self._system_model:
            return
        self._system_model.execute(0)

        if time_step is not None:
            self.update_battery_stored_values(time_step)

        # TODO: Need to update financial model after battery simulation is complete...
        #  I think this should be handled in the dispatch class

        # TODO: update financial_model... This might need to be taken care of in dispatch class?
        '''
        if not self.financial_model:
            return

        self.financial_model.value("construction_financing_cost", self.get_construction_financing_cost())

        self.financial_model.Revenue.ppa_soln_mode = 1

        self.financial_model.Lifetime.system_use_lifetime_output = 1
        self.financial_model.FinancialParameters.analysis_period = project_life
        single_year_gen = self.financial_model.SystemOutput.gen
        self.financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self.financial_model.SystemOutput.system_pre_curtailment_kwac = self.Outputs.gen * project_life
            self.financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self.Outputs.annual_energy

        self.financial_model.execute(0)
        logger.info("{} simulation executed".format(self.name))
        '''


    def update_battery_stored_values(self, time_step):
        for attr in self.Outputs.stateful_attributes:
            if hasattr(self._system_model.StatePack, attr):
                getattr(self.Outputs, attr)[time_step] = getattr(self._system_model.StatePack, attr)
            else:
                if attr == 'gen':
                    getattr(self.Outputs, attr)[time_step] = self._system_model.StatePack.P

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kwh:
            return self.Outputs.gen
        else:
            return [0] * self.site.n_timesteps