import os
import sys

from hybrid import clustering
import PySAM.BatteryStateful as BatteryModel
import pyomo.environ as pyomo
from pyomo.opt import TerminationCondition

# TODO: This file contains an implementation of clustering...
#  Once clustering has been implemented in the main framework this file can be deleted from the project
# from hybrid.hybrid_simulation import HybridSimulation

def list2dict(lst):
    dct = {}
    for i, v in enumerate(lst):
        dct[i] = v
    return dct


class Param:
    def __init__(self, val, time_index=False):
        # self.val = val
        self.value = val
        self.time_index = time_index

    @property
    def param_value(self):
        return self.value

    @param_value.setter
    def param_value(self, val):
        self.value = val


class HybridDispatchOld:

    def __init__(self,
                 hybrid,  #: HybridSimulation,
                 #time_intervals: list = [(60, 48)],
                 n_roll_periods: int = 24,
                 n_look_ahead_periods: int = 48,
                 is_simple_battery_dispatch: bool = True,
                 is_clustering: bool = False,
                 log_name: str = 'hybrid_dispatch_optimization.log'):
        """

        :param hybrid:
            BatteryStateful Class object
        :param time_intervals:
            [(minutes, -)]  List of tuples containing the duration (in minutes) and the number of time intervals
        :param is_simple_battery_dispatch:
            Dispatch model assumes constant battery charge and discharge -> decisions on power flow
            ELSE: Dispatch model uses linear voltage curve approximation -> decisions on current flow
        """
        self.hybrid = hybrid
        if type(hybrid.battery._system_model) == BatteryModel.BatteryStateful:
            self.battery = hybrid.battery._system_model  # Saving battery class
        else:
            raise TypeError

        self.is_simple_battery_dispatch = is_simple_battery_dispatch
        if self.is_simple_battery_dispatch:
            self.battery.value("control_mode", 1.0)  # Power control
            self.control_variable = "input_power"
        else:
            self.battery.value("control_mode", 0.0)  # Current control
            self.control_variable = "input_current"

        self.log_name = log_name
        if os.path.isfile(self.log_name):
            os.remove(self.log_name)

        #self.time_intervals = time_intervals
        self.n_look_ahead_periods = n_look_ahead_periods
        self.n_roll_periods = n_roll_periods

        self.is_clustering = is_clustering
        if self.is_clustering:
            self.clustering_inputs = dict
            self.initialize_clusters()

        # TODO: Does time_intervals need to be sorted? also, clean up this implementation -> move this to a method,
        #  maybe a property to update if time_intervals change
        '''
        if not len(self.time_intervals) < 1:
            tau_dict = {}  # dictionary containing each set of time steps
            delta_dict = {}
            intervals = []
            delta = []  # time intervals for time step
            tau = []  # time steps
            for interval, n_interval in self.time_intervals:
                if type(n_interval) == int:
                    tau_dict[interval] = range(len(tau), len(tau) + n_interval)
                    delta_dict[interval] = [interval / 60] * n_interval
                    delta.extend(delta_dict[interval])
                    tau.extend(tau_dict[interval])
                    intervals.append(interval)
                else:
                    TypeError('second entry of time_intervals must be an int type')
        else:
            raise ValueError('time_intervals must contain at least one time interval')
        '''

        # ___________________ DISPATCH MODEL (parameters and variables) ____________________________
        self.OptModel = pyomo.ConcreteModel()
        # TODO: Move into initialize model method...
        # ===== Parameters =====================
        self.gamma = Param(0.999)  # [-]       Exponential time weighting factor

        # Time-index Parameters - Initialize to zero
        self.Delta = Param({}, time_index=True)  # [hr]     Time step in hours
        self.P = Param({}, time_index=True)  # [$/kWh]      Normalized grid electricity price in time t
        self.Wnet = Param({}, time_index=True)  # [kW]      Net grid transmission upper limit in time t
        self.Wpv = Param({}, time_index=True)  # [kW]       Available PV array power in time t
        self.Wwf = Param({}, time_index=True)  # [kW]       Available wind farm power in time t

        # Cost Parameters TODO: Update these values in a function
        self.Cb = Param(0.001)  # [$/kWh_DC]        Operating cost of battery charging and discharging
        # [$/lifecycle]     Operating cost of battery lifecycle
        self.Clc = Param(0.01 * self.battery.ParamsPack.nominal_energy)
        # TODO: Simple battery is sensitive to this value
        #  I don't think the detail model is 'Cheating' lifecycle count due to McCormick Envelope
        # self.CdeltaW = Param(0.001)  # [$/DeltaKW_AC]    Penalty for change in net power production
        self.Cpv = Param(0.0015)  # [$/kWh_DC]        Operating cost of photovoltaic array
        self.Cwf = Param(0.005)  # [$/kWh_AC]        Operating cost of wind farm

        # Battery Parameters
        self.etaP = Param(None)  # [-]          Battery Charging efficiency
        self.etaN = Param(None)  # [-]          Battery Discharging efficiency
        self.CB = Param(None)  # [kWh] or [kAh] Battery manufacturer-specified capacity
        self.AV = Param(None)  # [V]            Battery linear voltage model slope coefficient
        self.BV = Param(None)  # [V]            Battery linear voltage model intercept coefficient
        self.alphaP = Param(None)  # [kW_DC]    Bi-directional intercept for charge
        self.betaP = Param(None)  # [-]         Bi-directional slope for charge
        self.alphaN = Param(None)  # [kW_DC]    Bi-directional intercept for discharge
        self.betaN = Param(None)  # [-]         Bi-directional slope for discharge
        self.Iavg = Param(None)  # [A]          Typical battery current for both charge and discharge
        self.Rint = Param(None)  # [Ohms]       Battery internal resistance
        self.ImaxP = Param(None)  # [kA]        Battery charge maximum current
        self.IminP = Param(None)  # [kA]        Battery charge minimum current
        self.ImaxN = Param(None)  # [kA]        Battery discharge maximum current
        self.IminN = Param(None)  # [kA]        Battery discharge minimum current

        self.PminB = Param(None)  # [kW_DC]     Battery minimum power rating
        self.PmaxB = Param(None)  # [kW_DC]     Battery maximum power rating
        self.SmaxB = Param(None)  # [-]         Battery maximum state-of-charge
        self.SminB = Param(None)  # [-]         Battery minimum state-of-charge
        self.bsoc0 = Param(None)  # [-]         Battery initial state-of-charge

        self.init_parameters_stateful_battery()
        self.initialize_OptModel()

    def init_parameters_stateful_battery(self, use_exp_voltage_point: bool = False):
        """
        Initializes dispatch model using BatteryStateful class
        """

        # TODO: update this calculation when more is known about nominal voltage and capacity.
        # Using the Ceiling for both these -> Ceil(a/b) = -(-a//b)
        cells_in_series = - (- self.battery.ParamsPack.nominal_voltage // self.battery.ParamsCell.Vnom_default)
        strings_in_parallel = - (- self.battery.ParamsPack.nominal_energy * 1000 // (
                self.battery.ParamsCell.Qfull * cells_in_series * self.battery.ParamsCell.Vnom_default))

        # Battery parameters
        if self.is_simple_battery_dispatch:
            # TODO: update these using SAM model or an update functions
            self.etaP.param_value = 0.948  # sqrt(0.90) assuming a round-trip efficiency of 90%
            self.etaN.param_value = 0.948
            self.CB.param_value = (self.battery.ParamsCell.Qfull * strings_in_parallel
                                   * self.battery.ParamsCell.Vnom_default * cells_in_series) / 1000.  # [kWh]
        else:
            self.CB.param_value = self.battery.ParamsCell.Qfull * strings_in_parallel / 1000  # [kAh]

            # Calculating linear approximation for Voltage as a function of state-of-charge
            soc_nom = (self.battery.ParamsCell.Qfull - self.battery.ParamsCell.Qnom) / self.battery.ParamsCell.Qfull
            if use_exp_voltage_point:
                # Using cell exp and nom voltage points
                #       Using this method makes the problem more difficult for the solver.
                #       TODO: This behavior is not fully understood and
                #        there could be a better way to create the linear approximation
                soc_exp = (self.battery.ParamsCell.Qfull - self.battery.ParamsCell.Qexp) / self.battery.ParamsCell.Qfull

                a = (self.battery.ParamsCell.Vexp - self.battery.ParamsCell.Vnom) / (soc_exp - soc_nom)
                b = self.battery.ParamsCell.Vexp - a * soc_exp
            else:
                # Using Cell full and nom voltage points
                a = (self.battery.ParamsCell.Vfull - self.battery.ParamsCell.Vnom) / (1.0 - soc_nom)
                b = self.battery.ParamsCell.Vfull - a

            self.AV.param_value = cells_in_series * a
            self.BV.param_value = cells_in_series * b

            # TODO: these parameters need to be updated base on inverter performance
            self.alphaP.param_value = 0
            self.betaP.param_value = 0

            self.alphaN.param_value = 0
            self.betaN.param_value = 0

            # TODO: Is Iavg right? Average between zero and average of charge and discharge max
            self.Iavg.param_value = (self.battery.ParamsCell.Qfull * strings_in_parallel
                                     * self.battery.ParamsCell.C_rate / 2.)

            self.Rint.param_value = self.battery.ParamsCell.resistance * cells_in_series / strings_in_parallel

            # TODO: These parameters need to be updated (max charge and discharge)
            # Charge current limits
            self.ImaxP.param_value = (self.battery.ParamsCell.Qfull
                                      * strings_in_parallel
                                      * self.battery.ParamsCell.C_rate) / 1000.0
            self.IminP.param_value = 0.0

            # Discharge current limits
            self.ImaxN.param_value = (self.battery.ParamsCell.Qfull
                                      * strings_in_parallel
                                      * self.battery.ParamsCell.C_rate) / 1000.0
            self.IminN.param_value = 0.0

        # TODO: We might want to add a charge and discharge power limit
        self.PmaxB.param_value = self.battery.ParamsPack.nominal_energy * self.battery.ParamsCell.C_rate
        self.PminB.param_value = 0.0

        self.SmaxB.param_value = self.battery.ParamsCell.maximum_SOC / 100.0
        self.SminB.param_value = self.battery.ParamsCell.minimum_SOC / 100.0
        self.bsoc0.param_value = self.battery.ParamsCell.initial_SOC / 100.0

    def initialize_OptModel(self):
        model = pyomo.ConcreteModel()
        self.OptModel = model  # overwrites previous optimization model

        # ============= Sets ===============
        # TODO: need to figure out why initialize is failing with tau as input
        model.T = pyomo.Set(initialize=range(self.n_look_ahead_periods))  # set of time periods

        n_days_in_horizon = self.n_look_ahead_periods//self.hybrid.site.n_periods_per_day
        model.D = pyomo.Set(initialize=range(n_days_in_horizon))  # set of days

        def dt_init(mod):
            return ((d, t) for d in mod.D for t in list(range(d * self.hybrid.site.n_periods_per_day,
                                                              (d+1) * self.hybrid.site.n_periods_per_day)))
            # TODO: update for multi-time steps and to handle horizons that start mid-day

        model.DT = pyomo.Set(dimen=2, initialize=dt_init)

        # ========== Parameters (defined in HybridDispatch) =======
        for key in self.__dict__.keys():
            attr = getattr(self, key)
            if type(attr) == Param and attr.param_value is not None:
                if attr.time_index:
                    # setattr(model, key, pyomo.Param(model.T, initialize=attr.param_value, mutable=True, within=pyomo.Reals))
                    setattr(model, key, pyomo.Param(model.T, mutable=True, within=pyomo.Reals))
                else:
                    setattr(model, key, pyomo.Param(initialize=attr.param_value, mutable=True, within=pyomo.Reals))

        # ======== Variables =======
        # model.bsocm = pyomo.Var(model.D, domain=pyomo.NonNegativeReals)  # [-]       Minimum SOC per day d
        model.blc = pyomo.Var(domain=pyomo.NonNegativeReals)               # [-]       Battery lifecycle count

        # ================ Constraints ================
        # Create a block for a single time period
        def hybrid_time_block_rule(b, t):
            # ======= variables ==========
            # General
            # b.wdotdelta = pyomo.Var()                          # [kW_AC]   Change in grid electricity production
            # b.wdotP = pyomo.Var(domain=pyomo.NonNegativeReals) # [kW_AC]   Electrical power purchased from the grid
            b.wdotS = pyomo.Var(domain=pyomo.NonNegativeReals)   # [kW_AC]   Electrical power sold to the grid
            b.wdotPV = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kW_DC]   Power from the photovoltaic array
            b.wdotWF = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kW_AC]   Power from the wind farm

            # Battery system variables
            b.bsoc0 = pyomo.Var(domain=pyomo.NonNegativeReals)  # [-]   Battery state of charge at beginning of period
            b.bsoc = pyomo.Var(domain=pyomo.NonNegativeReals)  # [-]   Battery state of charge at end of period
            b.wdotBC = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kW_DC]   Power into the battery
            b.wdotBD = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kW_DC]   Power out of the battery
            b.yP = pyomo.Var(domain=pyomo.Binary)  # [-]       1 if battery is charging; 0 Otherwise
            b.yN = pyomo.Var(domain=pyomo.Binary)  # [-]       1 if battery is discharging; 0 Otherwise

            # Variables for current battery decisions
            if not self.is_simple_battery_dispatch:
                b.iP = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      Battery current for charge
                b.iN = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      Battery current for discharge
                # Auxiliary Variables
                b.zP = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      = iP[t] * bsoc[t-1]
                b.xP = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      = iP[t] * yP[t]
                b.zN = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      = iN[t] * bsoc[t-1]
                b.xN = pyomo.Var(domain=pyomo.NonNegativeReals)  # [kA]      = iN[t] * yN[t]

            # ============= Define the constraints for the time block ====================
            # System power balance with respect to AC bus
            if self.is_simple_battery_dispatch:
                b.power_balance = pyomo.Constraint(expr=(b.wdotS == b.wdotWF
                                                         + b.wdotPV
                                                         + model.etaN * b.wdotBD
                                                         - (1 / model.etaP) * b.wdotBC))
            else:
                b.power_balance = pyomo.Constraint(expr=(b.wdotS == b.wdotWF
                                                         + b.wdotPV
                                                         + (b.wdotBD - model.alphaN * b.yN) / (1 + model.betaN)
                                                         - ((1 + model.betaP) * b.wdotBC + model.alphaP * b.yP)))
            # PV resource bound
            b.PV_resource = pyomo.Constraint(expr=b.wdotPV <= model.Wpv[t])
            # Wind farm resource bound
            b.WF_resource = pyomo.Constraint(expr=b.wdotWF <= model.Wwf[t])
            # Net grid limit imposed
            b.Wnet_limit = pyomo.Constraint(expr=b.wdotS <= model.Wnet[t])

            # Battery State-of-Charge balance
            if self.is_simple_battery_dispatch:
                # power accounting
                b.battery_soc = pyomo.Constraint(expr=(b.bsoc == b.bsoc0
                                                       + model.Delta[t] * (b.wdotBC - b.wdotBD) / model.CB))
            else:
                # current accounting
                b.battery_soc = pyomo.Constraint(expr=(b.bsoc == b.bsoc0
                                                       + model.Delta[t] * (b.iP - b.iN) / model.CB))
                                                    # + model.Delta[t] * (b.iP * 0.95 - b.iN) * 0.95 / model.CB))
                # TODO: adjustment factors -> We could learn these to minimize SOC error

            # Battery State-of-Charge bounds
            b.soc_lower_bound = pyomo.Constraint(expr=b.bsoc >= model.SminB)
            b.soc_upper_bound = pyomo.Constraint(expr=b.bsoc <= model.SmaxB)
            # Battery Charging power bounds
            b.chargeP_lower_bound = pyomo.Constraint(expr=b.wdotBC >= model.PminB * b.yP)
            b.chargeP_upper_bound = pyomo.Constraint(expr=b.wdotBC <= model.PmaxB * b.yP)
            # Battery Discharging power Bounds
            b.dischargeP_lower_bound = pyomo.Constraint(expr=b.wdotBD >= model.PminB * b.yN)
            b.dischargeP_upper_bound = pyomo.Constraint(expr=b.wdotBD <= model.PmaxB * b.yN)
            # Battery can only charge or discharge in a time period
            b.charge_discharge_packing = pyomo.Constraint(expr=b.yP + b.yN <= 1)

            # Detailed battery constraints
            if not self.is_simple_battery_dispatch:
                # Discharge current bounds
                b.discharge_current_upper_soc = pyomo.Constraint(expr=b.iN <= model.ImaxN * b.bsoc0)
                b.discharge_current_upper = pyomo.Constraint(expr=b.iN <= model.ImaxN * b.yN)
                b.discharge_current_lower = pyomo.Constraint(expr=b.iN >= model.IminN * b.yN)
                # Charge current bounds
                b.charge_current_upper_soc = pyomo.Constraint(expr=b.iP <= model.CB * (1 - b.bsoc0) / model.Delta[t])
                b.charge_current_upper = pyomo.Constraint(expr=b.iP <= model.ImaxP * b.yP)
                b.charge_current_lower = pyomo.Constraint(expr=b.iP >= model.IminP * b.yP)
                # Charge power (is equal to current*voltage)
                b.charge_power = pyomo.Constraint(expr=b.wdotBC == model.AV * b.zP + (model.BV
                                                                                      + model.Iavg * model.Rint) * b.xP)
                b.discharge_power = pyomo.Constraint(expr=(b.wdotBD == (model.AV * b.zN
                                                                        + (model.BV - model.Iavg * model.Rint) * b.xN)))
                                                           #* (1 - 0.1)))  # TODO: adjustment factor
                # Aux. Variable bounds (xN[t] and xP[t]) (binary*continuous exact linearization)
                b.auxN_lower_lim = pyomo.Constraint(expr=b.xN >= model.IminN * b.yN)
                b.auxN_upper_lim = pyomo.Constraint(expr=b.xN <= model.ImaxN * b.yN)
                b.auxN_diff_lower_lim = pyomo.Constraint(expr=b.iN - b.xN >= - model.ImaxN * (1 - b.yN))
                b.auxN_diff_upper_lim = pyomo.Constraint(expr=b.iN - b.xN <= model.ImaxN * (1 - b.yN))
                b.auxP_lower_lim = pyomo.Constraint(expr=b.xP >= model.IminP * b.yP)
                b.auxP_upper_lim = pyomo.Constraint(expr=b.xP <= model.ImaxP * b.yP)
                b.auxP_diff_lower_lim = pyomo.Constraint(expr=b.iP - b.xP >= - model.ImaxP * (1 - b.yP))
                b.auxP_diff_upper_lim = pyomo.Constraint(expr=b.iP - b.xP <= model.ImaxP * (1 - b.yP))
                # ======== Aux. Variable bounds (zP[t] and zN[t]) (continuous*continuous approx. linearization)
                # zP[t] Aux variable
                b.zp_lower_1 = pyomo.Constraint(
                    expr=b.zP >= model.ImaxP * b.bsoc0 + model.SmaxB * b.iP - model.SmaxB * model.ImaxP)
                b.zp_lower_2 = pyomo.Constraint(
                    expr=b.zP >= model.IminP * b.bsoc0 + model.SminB * b.iP - model.SminB * model.IminP)
                b.zp_upper_1 = pyomo.Constraint(
                    expr=b.zP <= model.ImaxP * b.bsoc0 + model.SminB * b.iP - model.SminB * model.ImaxP)
                b.zp_upper_2 = pyomo.Constraint(
                    expr=b.zP <= model.IminP * b.bsoc0 + model.SmaxB * b.iP - model.SmaxB * model.IminP)
                # zN[t] Aux variable
                b.zn_lower_1 = pyomo.Constraint(
                    expr=b.zN >= model.ImaxN * b.bsoc0 + model.SmaxB * b.iN - model.SmaxB * model.ImaxN)
                b.zn_lower_2 = pyomo.Constraint(
                    expr=b.zN >= model.IminN * b.bsoc0 + model.SminB * b.iN - model.SminB * model.IminN)
                b.zn_upper_1 = pyomo.Constraint(
                    expr=b.zN <= model.ImaxN * b.bsoc0 + model.SminB * b.iN - model.SminB * model.ImaxN)
                b.zn_upper_2 = pyomo.Constraint(
                    expr=b.zN <= model.IminN * b.bsoc0 + model.SmaxB * b.iN - model.SmaxB * model.IminP)

        model.htb = pyomo.Block(model.T, rule=hybrid_time_block_rule)

        # Linking time periods together
        def battery_soc_linking_rule(m, t):
            if t == m.T.first():
                return m.htb[t].bsoc0 == model.bsoc0
            return m.htb[t].bsoc0 == m.htb[t - 1].bsoc
        model.battery_soc_linking = pyomo.Constraint(model.T, rule=battery_soc_linking_rule)

        # Depth of discharge per day
        # def minDoD_perDay(m, d, t):
        #     return m.bsocm[d] <= m.htb[t].bsoc
        # model.minDoD_perDay = pyomo.Constraint(model.DT, rule=minDoD_perDay)

        # Battery lifecycle counting
        if self.is_simple_battery_dispatch:
            # power accounting
            def bat_lifecycle(m):
                return m.blc == (1 / m.CB) * sum(m.Delta[t] * m.htb[t].wdotBD for t in m.T)
        else:
            # current accounting
            def bat_lifecycle(m):
                return m.blc == (1 / m.CB) * sum(m.Delta[t] * (0.8 * m.htb[t].iN - 0.8 * m.htb[t].zN) for t in m.T)
                # return m.blc == (m.Delta / m.CB) * sum((m.gamma**t)*(m.iP[t]) for t in m.T)
        model.bat_lifecycle = pyomo.Constraint(rule=bat_lifecycle)

        # =============== Objective ==============

        def obj(m):
            return (sum(m.Delta[t] * (
                                    (m.gamma ** t) * m.P[t] * m.htb[t].wdotS
                                    - ((1 / m.gamma) ** t) * (m.Cpv * m.htb[t].wdotPV         #m.Delta[t] *
                                                              + m.Cwf * m.htb[t].wdotWF
                                                              + m.Cb * (m.htb[t].wdotBC + m.htb[t].wdotBD)
                                                              )
                                    ) for t in m.T) - m.Clc * m.blc)  # - 6000. * sum((1 - m.bsocm[d]) for d in m.D))
        # TODO: get a handle on minimum SOC penalty and move to parameters

        model.objective = pyomo.Objective(rule=obj, sense=pyomo.maximize)
        # TODO: Currently, curtailment is free.
        #  Is this activity free in reality? or is there a small operating cost associate with it?

    def initialize_clusters(self):
        # Clustering parameters
        self.clustering_inputs = {'n_days': int(2),
                                  'n_prev': int(1),
                                  'n_next': int(1),
                                  'n_clusters': int(20),
                                  'initial_charge': 0.1,  # TODO: what is a good value here?
                                  }

        price_data = [f * self.hybrid.ppa_price[0] for f in list(self.hybrid.grid.dispatch_factors)]
        cluster_inputs = clustering.setup_clusters(self.hybrid.site.solar_resource.filename, price_data,
                                                   self.clustering_inputs['n_clusters'],
                                                   self.clustering_inputs['n_days'], self.clustering_inputs['n_prev'],
                                                   self.clustering_inputs['n_next'])
        self.clustering_inputs.update(cluster_inputs)

        # Combine consecutive exemplars into a single simulation
        sf_adjust_tot = [1.] * self.hybrid.site.n_timesteps  # dummy data # TODO: remove sf_adjust_tot
        avg_sfadjust = clustering.compute_cluster_avg_from_timeseries(sf_adjust_tot,
                                                                      self.clustering_inputs['partition_matrix'],
                                                                      Ndays=self.clustering_inputs['n_days'],
                                                                      Nprev=self.clustering_inputs['n_prev'],
                                                                      Nnext=self.clustering_inputs['n_next'],
                                                                      adjust_wt=True,
                                                                      k1=self.clustering_inputs['first_pt_cluster'],
                                                                      k2=self.clustering_inputs['last_pt_cluster'])
        # Combine simulations of consecutive exemplars
        combined = clustering.combine_consecutive_exemplars(self.clustering_inputs['day_start'],
                                                            self.clustering_inputs['weights'],
                                                            self.clustering_inputs['avg_ppamult'],
                                                            avg_sfadjust,
                                                            self.clustering_inputs['n_days'],
                                                            Nprev=self.clustering_inputs['n_prev'],
                                                            Nnext=self.clustering_inputs['n_next'])

        self.clustering_inputs['day_start'] = combined['start_days']
        self.clustering_inputs['group_n_days'] = combined['Nsim_days']
        self.clustering_inputs['avg_ppamult'] = combined['avg_ppa']
        self.clustering_inputs['group_weight'] = combined['weights']
        self.clustering_inputs['avg_sfadjust'] = combined['avg_sfadj']

    def hybrid_optimization_call(self, printlogs=False):
        solver = pyomo.SolverFactory('glpk')  # Ref. on solver options: https://en.wikibooks.org/wiki/GLPK/Using_GLPSOL

        solver_opt = {}
        if printlogs:
            solver_opt['log'] = 'dispatch_instance.log'
        solver_opt['cuts'] = None
        solver_opt['mipgap'] = 0.001
        solver_opt['tmlim'] = 30

        results = solver.solve(self.OptModel, options=solver_opt)

        # Appends single problem instance log to annual log file
        if printlogs:
            fin = open('dispatch_instance.log', 'r')
            data = fin.read()
            fin.close()

            ann_log = open(self.log_name, 'a+')
            ann_log.write("=" * 50 + "\n")
            ann_log.write(data)
            ann_log.close()

        if results.solver.termination_condition == TerminationCondition.infeasible:
            original_stdout = sys.stdout
            with open('infeasible_instance.txt', 'w') as f:
                sys.stdout = f
                self.print_all_parameters()
                sys.stdout = original_stdout

            raise ValueError("Dispatch optimization model is infeasible.\n"
                             "See 'infeasible_instance.txt' for parameter values.")

    def simulate(self, is_test: bool = False):

        for t in self.OptModel.T:
            self.OptModel.Wnet[t] = self.hybrid.grid.interconnect_kw
            self.OptModel.Delta[t] = 1.0

        # Dispatch Optimization Simulation with Rolling Horizon #
        if not self.is_clustering:
            # Solving the year in series
            ti = list(range(0, self.hybrid.site.n_timesteps, self.n_roll_periods))
            for i, t in enumerate(ti):
                print('Evaluating day ', i, ' out of ', len(ti))
                self.simulate_with_dispatch(t, 1, self.battery.StatePack.SOC / 100.)

                # TODO: Remove for release
                if is_test and i > 10:
                    break
        else:
            # Clustering similar days of data and aggregating results
            n_groups = len(self.clustering_inputs['day_start'])  # Number of simulation groups
            for g in range(n_groups):
                print('Evaluating group ', g, ' out of ', n_groups)
                # First day to be included in simulation group g
                first_day = self.clustering_inputs['day_start'][g] - self.clustering_inputs['n_prev']
                # Number of previous days actually allowed in the simulation
                n_prev_sim = self.clustering_inputs['day_start'][g] - max(0, first_day)
                n_days_tot = self.clustering_inputs['group_n_days'][g] + n_prev_sim
                start_time = self.hybrid.site.n_periods_per_day*(self.clustering_inputs['day_start'][g] - n_prev_sim)

                # TODO: Bad practice -> need a setter (how do we make the battery model is consistent with clusters?)
                self.hybrid.battery._system_model.value("initial_SOC", self.clustering_inputs['initial_charge']*100.)
                self.simulate_with_dispatch(start_time,
                                            n_days_tot,
                                            self.clustering_inputs['initial_charge'],
                                            n_initial_sims=n_prev_sim)

            clusters = {'exemplars': self.clustering_inputs['exemplars'],
                        'partition_matrix': self.clustering_inputs['partition_matrix']}

            for attr in self.hybrid.battery.Outputs.__dict__.keys():
                original_data = getattr(self.hybrid.battery.Outputs, attr)
                if len(original_data) == self.hybrid.site.n_timesteps:
                    updated_data = clustering.compute_annual_array_from_clusters(original_data, clusters,
                                                                                 self.clustering_inputs['n_days'],
                                                                                 adjust_wt=True,
                                                                                 k1=self.clustering_inputs['first_pt_cluster'],
                                                                                 k2=self.clustering_inputs['last_pt_cluster'])
                    setattr(self.hybrid.battery.Outputs, attr, updated_data)

    def simulate_with_dispatch(self, start_time, n_days, init_soc=None, n_initial_sims=0, print_logs=True):
        # this is needed for clustering effort
        update_dispatch_times = list(range(start_time,
                                           start_time + n_days * self.hybrid.site.n_periods_per_day,
                                           self.n_roll_periods))

        for i, udt in enumerate(update_dispatch_times):
            # Update battery initial state of charge
            if init_soc is not None:
                self.OptModel.bsoc0 = init_soc
                init_soc = None
            else:
                self.OptModel.bsoc0 = self.battery.StatePack.SOC / 100.

            price_data = self.hybrid.grid.dispatch_factors

            # Update Solar, Wind, and price forecasts
            for t in self.OptModel.T:
                self.OptModel.Wwf[t] = float(self.hybrid.wind.generation_profile[udt + t])
                self.OptModel.Wpv[t] = float(self.hybrid.pv.generation_profile[udt + t])
                # TODO: update OptModel.T to a shorter time horizon for last day of year?
                if udt + t >= self.hybrid.site.n_timesteps:
                    self.OptModel.P[t] = (price_data[udt + t - self.hybrid.site.n_timesteps]
                                          * self.hybrid.ppa_price[0])
                else:
                    self.OptModel.P[t] = price_data[udt + t] * self.hybrid.ppa_price[0]
            self.hybrid_optimization_call(printlogs=print_logs)

            # step through dispatch solution for battery and simulate battery
            for t in range(self.n_roll_periods):
                # Set stateful control value [Discharging (+) + Charging (-)]
                if self.is_simple_battery_dispatch:
                    control_value = (self.OptModel.htb[t].wdotBD.value  # self.OptModel.etaN.value *
                                     + (1/self.OptModel.etaP.value) * (- self.OptModel.htb[t].wdotBC.value))
                else:
                    control_value = (self.OptModel.htb[t].iN.value + (- self.OptModel.htb[t].iP.value)) * 1000.0
                    # [kA] -> [A]

                self.battery.value('dt_hr', self.OptModel.Delta[t].value)
                self.battery.value(self.control_variable, control_value)

                # Only store information if passed the previous day simulations (used in clustering)
                if i >= n_initial_sims:
                    self.hybrid.battery.simulate(time_step=udt + t)
                    self.hybrid.battery.Outputs.dispatch_SOC[udt + t] = self.OptModel.htb[t].bsoc.value * 100.0
                    self.hybrid.battery.Outputs.dispatch_P[udt + t] = (self.OptModel.htb[t].wdotBD.value  # self.OptModel.etaN.value *
                                                                       + (1/self.OptModel.etaP.value) * (- self.OptModel.htb[t].wdotBC.value))
                    # (self.OptModel.htb[t].wdotBD.value + (- self.OptModel.htb[t].wdotBC.value))
                    if not self.is_simple_battery_dispatch:
                        self.hybrid.battery.Outputs.dispatch_I[udt + t] = (self.OptModel.htb[t].iN.value +
                                                                           (- self.OptModel.htb[t].iP.value)) * 1000.0
                else:
                    self.hybrid.battery.simulate()

    def print_all_parameters(self):
        for param_object in self.OptModel.component_objects(pyomo.Param, active=True):
            name_to_print = str(str(param_object.name))
            print("Parameter ", name_to_print)
            for index in param_object:
                val_to_print = pyomo.value(param_object[index])
                print("   ", index, val_to_print)

    def print_all_variables(self):
        for variable_object in self.OptModel.component_objects(pyomo.Var, active=True):
            name_to_print = str(variable_object.name)
            print("Variable ", name_to_print)
            for index in variable_object:
                val_to_print = pyomo.value(variable_object[index])
                print("   ", index, val_to_print)
