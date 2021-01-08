import PySAM.BatteryStateful as BatteryModel
import PySAM.BatteryTools as BatteryTools

from hybrid.power_source import *

class Battery_Outputs:
    def __init__(self, n_timesteps):
        """ Class of stateful battery outputs

        """
        # TODO: is there anything else to add?
        self.control = [0]*n_timesteps
        self.response = [0]*n_timesteps
        self.I = [0]*n_timesteps
        self.P = [0]*n_timesteps
        self.Q = [0]*n_timesteps
        self.SOC = [0]*n_timesteps
        self.T_batt = [0]*n_timesteps
        self.gen = [0]*n_timesteps

class Battery(PowerSource):
    system_model: BatteryModel.BatteryStateful
    financial_model: Singleowner.Singleowner

    def __init__(self,
                 site: SiteInfo,
                 system_capacity_kwh: float,
                 chemistry: str = 'lfpgraphite',
                 system_voltage_volts: float = 500):
        """

        :param system_capacity_kwh:
        :param system_voltage_volts:
        """
        system_model = BatteryModel.default(chemistry)
        self.Outputs = Battery_Outputs(n_timesteps=site.n_timesteps)
        financial_model = Singleowner.from_existing(system_model, "GenericBatterySingleOwner")
        # TODO: I don't think financial and system models are sufficiently linked.

        super().__init__("Battery", site, system_model, financial_model)
        BatteryTools.battery_model_sizing(system_model,
                                          0.,
                                          system_capacity_kwh,
                                          system_voltage_volts)

    @property
    def system_capacity_voltage(self) -> tuple:
        return self.system_model.ParamsPack.nominal_energy, self.system_model.ParamsPack.nominal_voltage

    @system_capacity_voltage.setter
    def system_capacity_voltage(self, capacity_voltage: tuple):
        """
        Sets the system capacity and voltage, and updates the system, cost and financial model
        :param capacity_voltage:
        :return:
        """
        size_kwh = capacity_voltage[0]
        voltage_volts = capacity_voltage[1]

        BatteryTools.battery_model_sizing(self.system_model,
                                          0.,
                                          size_kwh,
                                          voltage_volts)
        self.system_capacity_kw: float = self.system_model.ParamsPack.nominal_energy
        self.system_voltage_volts: float = self.system_model.ParamsPack.nominal_voltage
        logger.info("Battery set system_capacity to {} kWh".format(size_kwh))
        logger.info("Battery set system_voltage to {} volts".format(voltage_volts))

    @property
    def system_capacity_kw(self) -> float:  # TODO: for batteries capacity is defined in kwh...
        return self.system_model.ParamsPack.nominal_energy

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kwh: float):
        """
        Sets the system capacity and updates the system, cost and financial model
        :param size_kwh:
        :return:
        """
        self.system_capacity_voltage = (size_kwh, self.system_voltage_volts)

    @property
    def system_voltage_volts(self) -> float:
        return self.system_model.ParamsPack.nominal_voltage

    @system_voltage_volts.setter
    def system_voltage_volts(self, voltage_volts: float):
        """
        Sets the system voltage and updates the system, cost and financial model
        :param voltage_volts:
        :return:
        """
        self.system_capacity_voltage = (self.system_capacity_kw, voltage_volts)

    @property
    def chemistry(self) -> str:
        model_type = self.system_model.ParamsCell.chem
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
        BatteryTools.battery_model_change_chemistry(self.system_model, battery_chemistry)
        logger.info("Battery chemistry set to {}".format(battery_chemistry))

    def simulate(self, time_step=None):
        """
        Runs battery simulate stores values if time step is provided
        """
        if not self.system_model:
            return
        self.system_model.execute(0)

        if time_step is not None:
            self.update_battery_stored_values(time_step)


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
            self.financial_model.SystemOutput.system_pre_curtailment_kwac = self.system_model.Outputs.gen * project_life
            self.financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self.system_model.Outputs.annual_energy

        self.financial_model.execute(0)
        logger.info("{} simulation executed".format(self.name))
        '''

    def update_battery_stored_values(self, time_step):
        if self.system_model.Controls.control_mode:
            self.Outputs.control[time_step] = self.system_model.Controls.input_power
            self.Outputs.response[time_step] = self.system_model.StatePack.P
        else:
            self.Outputs.control[time_step] = self.system_model.Controls.input_current
            self.Outputs.response[time_step] = self.system_model.StatePack.I
        self.Outputs.I[time_step] = self.system_model.StatePack.I
        self.Outputs.P[time_step] = self.system_model.StatePack.P
        self.Outputs.Q[time_step] = self.system_model.StatePack.Q
        self.Outputs.SOC[time_step] = self.system_model.StatePack.SOC
        self.Outputs.T_batt[time_step] = self.system_model.StatePack.T_batt
        self.Outputs.gen[time_step] = self.system_model.StatePack.P

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kw:
            return self.Outputs.gen
        else:
            return [0] * self.site.n_timesteps