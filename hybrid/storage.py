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

        # TODO: Scaling of surface area for utility-scale systems needs to be updated
        self._system_model.ParamsPack.surface_area = 30.0 * (system_capacity_kwh / 400.0)    # 500 [kWh] -> 30 [m^2]

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
                                          voltage_volts)
        self.system_capacity_kw: float = self._system_model.ParamsPack.nominal_energy
        self.system_voltage_volts: float = self._system_model.ParamsPack.nominal_voltage
        logger.info("Battery set system_capacity to {} kWh".format(size_kwh))
        logger.info("Battery set system_voltage to {} volts".format(voltage_volts))

    @property
    def system_capacity_kw(self) -> float:  # TODO: for batteries capacity is defined in kwh...
        return self._system_model.ParamsPack.nominal_energy

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
        return self._system_model.ParamsPack.nominal_voltage

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

    def simulate(self, time_step=None):
        """
        Runs battery simulate stores values if time step is provided
        """
        if not self._system_model:
            return
        self._system_model.execute(0)

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
        for attr in self.Outputs.stateful_attributes:
            if hasattr(self._system_model.StatePack, attr):
                getattr(self.Outputs, attr)[time_step] = getattr(self._system_model.StatePack, attr)
            else:
                if attr == 'gen':
                    getattr(self.Outputs, attr)[time_step] = self._system_model.StatePack.P

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kw:
            return self.Outputs.gen
        else:
            return [0] * self.site.n_timesteps