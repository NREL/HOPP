import pyomo.environ as pyomo
import numpy as np

def list2dict(lst):
    dct = {}
    for i,v in enumerate(lst):
        dct[i] = v
    return dct

class param:
    def __init__(self, val, time_index = False):
        self.val = val
        self.time_index = time_index

class dispatch_problem:
    def __init__(self, prob_horizon, simplebatt = True):
        self.simplebatt = simplebatt
        self.nt = prob_horizon                          # [-]       Number of time steps in problem horizon

        ## =============== Parameters =======================
        self.Delta = param(1.0)                         # [hr]      Time step duration
        self.gamma = param(0.999)                       # [-]       Exponential time weighting factor

        # Time-index Parameters
        self.P = param({}, time_index=True)           # [$/kWh]   Normalized grid electricity price in time t
        self.Wnet = param({}, time_index=True)        # [kW]      Net grid transmission upper limit in time t
        self.Wpv = param({}, time_index=True)         # [kW]      Available PV array power in time t
        self.Wwf = param({}, time_index=True)         # [kW]      Available wind farm power in time t

        # TODO: I would like to add conditions on where the battery is place within the system (i.e., off the AC or DC bus)
        # TODO: update default values in initalization
        # PV array parameters
            # DC-to-AC inverter parameters
        # self.alphaI = param(0.0)                        # [kW_AC]   inverter slope linear approximations of DC-to-AC efficiency
        # self.betaI = param(0.0)                         # [-]       inverter intercept linear approximations of DC-to-AC efficiency
        # self.WminI = param(0.0)                         # [kW_DC]   inverter minimum DC power limit
        # self.WmaxI = param(0.0)                         # [kW_DC]   inverter maximum DC power limit

            # DC-to-DC converter parameters
        # self.alphaC = param(0.0)                        # [kW]      converter slope linear approximations of DC-to-DC efficiency
        # self.betaC = param(0.0)                         # [-]       converter intercept linear approximations of DC-to-DC efficiency
        # self.WminC = param(0.0)                         # [kW_DC]   converter minimum DC power limit
        # self.WmaxC = param(0.0)                         # [kW_DC]   converter maximum DC power limit

        # Wind parameters
        # self.WminWF = param(0.0)                        # [kW_AC]   Minimum power output for wind farm

        # TODO: setup parameter-user interaction
        # Battery parameters
        if self.simplebatt:
            self.etaP = param(0.85)                     # [-]       Battery Charging efficiency
            self.etaN = param(0.85)                     # [-]       Battery Discharging efficiency
        else:
            self.AV = param(0.0)                        # [V]       Battery linear voltage model slope coefficient
            self.BV = param(0.0)                        # [V]       Battery linear voltage model intercept coefficient

            self.alphaP = param(0.0)                    # [kW_DC]   Bi-directional intercept for charge
            self.betaP = param(0.0)                     # [kW_DC]   Bi-directional slope for charge
            
            self.alphaN = param(0.0)                    # [kW_DC]   Bi-directional intercept for discharge
            self.betaN = param(0.0)                     # [kW_DC]   Bi-directional slope for discharge

            self.Iavg = param(0.0)                      # [A]       Typical current expected from the battery for both charge and discharge
            self.Rint = param(0.0)                      # [Ohms]    Battery internal resistance

            self.ImaxP = param(0.0)                     # [kA]      Battery charge maximum current
            self.IminP = param(0.0)                     # [kA]      Battery charge minimum current
            
            self.ImaxN = param(0.0)                     # [kA]      Battery discharge maximum current
            self.IminN = param(0.0)                     # [kA]      Battery discharge minimum current

        self.CB = param(0.0)                            # [kWh] or [kAh] (simple vs. not) Battery manufacturer-specified capacity
        
        self.PmaxB = param(0.0)                         # [kW_DC]   Battery maximum power rating
        self.PminB = param(0.0)                         # [kW_DC]   Battery minimum power rating

        self.SmaxB = param(1.0)                         # [-]       Battery maximum state of charge
        self.SminB = param(0.0)                         # [-]       Battery minimum state of charge

        # Cost Parameters
        # self.CbP = param(0.0)                           # [$/kWh_DC]        Operating cost of battery charging
        # self.CbN = param(0.0)                           # [$/kWh_DC]        Operating cost of battery discharging 
        # self.Clc = param(0.0)                           # [$/lifecycle]     Lifecycle cost for battery
        # self.CdeltaW = param(0.0)                       # [$/DeltaKW_AC]    Penalty for change in net power production
        # self.Cpv = param(0.0)                           # [$/kWh_DC]        Operating cost of photovolaic array
        # self.Cwf = param(0.0)                           # [$/kWh_AC]        Operating cost of wind farm

        ## =============== Variables =======================
        # TODO: create script that includes variables based on configuration

        # Continuous Variables
        self.ContVars = [
            'blc'                                       # [-]       Battery lifecycle count
        ]

        # Continuous Time-indexed Variables 
        self.ContTimeVars = [
            'bsoc',                                     # [-]       Battery state of charge in time period t
            'wdotBC',                                    # [kW_DC]   Power into the battery at time t
            'wdotBD',                                    # [kW_DC]   Power out of the battery at time t 
            #'wdotdelta',                                # [kW_AC]   Change in grid elctricity production at time t
            'wdotPV',                                   # [kW_DC]   Power from the photovoltaic array at time t
            'wdotS',                                    # [kW_AC]   Electrical power sold to the grid at time t
            'wdotP',                                    # [kW_AC]   Electrical power purchased to the grid at time t
            'wdotWF'                                    # [kW_AC]   Power from the wind farm at time t
        ]

        # Binary Time-index Variables
        self.BinTimeVars = [
            'yP', 'yN',                                 # 1 if battery is charging or discharging in time t; 0 Otherwise
            #'yPV',                                      # 1 if the photovoltaic array is generating in time t; 0 Otherwise
            #'yWF'                                       # 1 if the wind farm is generating power at time period t; 0 otherwise
        ]

        if not self.simplebatt:
            det_vars = [
                'iP',                                       # [kA]      Battery current for charge in time period t
                'iN',                                       # [kA]      Battery current for discharge in time period t
                # Aux variables
                'zP',                                       # [kA]      = iP[t] * bsoc[t-1]
                'xP',                                       # [kA]      = iP[t] * yP[t]
                'zN',                                       # [kA]      = iN[t] * bsoc[t-1]
                'xN',                                       # [kA]      = iN[t] * yN[t]
                #'vsoc',                                    # [V]       Battery voltage in time period t (This doesn't actual exist in the linear formulation)
            ]
            self.ContTimeVars.extend(det_vars)


    def updateParameters(self, local_params, time_index=False):
        for p in local_params:
            if p != 'self':
                if type(local_params[p]) == list:
                    setattr(self, p, param(list2dict(local_params[p]), time_index = time_index) )
                else:
                    setattr(self, p, param(local_params[p]) )


    def updateSolarWindResGrid(self, P, Wnet, Wpv, Wwf):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        P_t         [$/kWh]             Normalized grid electricity price in time t \n
        Wnet_t      [kW]                Net grid transmission upper limit in time t \n
        Wpv_t       [kW]                Available PV array power in time t \n
        Wwf_t       [kW]                Available wind farm power in time t \n
        * NOTE: Data type is Lists of problem horizon length    
        """
        self.updateParameters(locals(), time_index=True)

    def updateInverterParams(self, alphaI, betaI, WminI, WmaxI):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        alphaI      [kW_AC]             Inverter slope linear approximations of DC-to-AC efficiency \n
        betaI       [-]                 Inverter intercept linear approximations of DC-to-AC efficiency \n
        WminI       [kW_DC]             Inverter minimum DC power limit \n
        WmaxI       [kW_DC]             Inverter maximum DC power limit
        """
        self.updateParameters(locals())


    def updateConverterParams(self, alphaC, betaC, WminC, WmaxC):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        alphaC      [kW]                Converter slope linear approximations of DC-to-DC efficiency \n
        betaC       [-]                 Converter intercept linear approximations of DC-to-DC efficiency \n
        WminC       [kW_DC]             Converter minimum DC power limit \n
        WmaxC       [kW_DC]             Converter maximum DC power limit
        """
        self.updateParameters(locals())

    def updateCostParams(self, CbP, CbN, Clc, CdeltaW, Cpv, Cwf):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        CbP         [$/kWh_DC]          Operating cost of battery charging \n
        CbN         [$/kWh_DC]          Operating cost of battery discharging \n
        Clc         [$/lifecycle]       Lifecycle cost for battery \n
        CdeltaW     [$/DeltaKW_AC]      Penalty for change in net power production \n
        Cpv         [$/kWh_DC]          Operating cost of photovolaic array \n
        Cwf         [$/kWh_AC]          Operating cost of wind farm
        """
        self.updateParameters(locals())

    def updateSimpleBatteryParams(self, etaP, etaN, CB, PmaxB, PminB, SmaxB, SminB):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        etaP        [-]                 Battery Charging efficiency \n
        etaN        [-]                 Battery Discharging efficiency \n
        CB          [kWh]               Battery manufacturer-specified capacity \n
        PmaxB       [kW_DC]             Battery maximum power rating \n
        PminB       [kW_DC]             Battery minimum power rating \n
        SmaxB       [-]                 Battery maximum state of charge \n
        SminB       [-]                 Battery minimum state of charge 
        """
        self.updateParameters(locals())

    def updateInitialConditions(self, bsoc0):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        bsoc0       [-]                 Initial battery state of charge
        """
        self.updateParameters(locals())

    def buildBattery(self):
        pass

    def initializeOptModel(self):
        mod = pyomo.ConcreteModel()

        # ====== Sets =======
        mod.T = pyomo.Set(initialize = range(self.nt))    #set of time periods

        # ====== Creating Parameters =======
        for key in self.__dict__.keys():
            attr = getattr(self, key)
            if type(attr) == param:
                if attr.time_index:
                    setattr(mod, key, pyomo.Param(mod.T, initialize = attr.val, mutable = True, within = pyomo.Reals))
                else:
                    setattr(mod, key, pyomo.Param(initialize = attr.val, mutable = True))

        # ====== Creating Variables ======
        for var in self.ContVars:
            setattr(mod, var, pyomo.Var(domain = pyomo.NonNegativeReals))
        for var in self.ContTimeVars:
            setattr(mod, var, pyomo.Var( mod.T, domain = pyomo.NonNegativeReals))
        for var in self.BinTimeVars:
            setattr(mod, var, pyomo.Var( mod.T, domain = pyomo.Binary))

        # ====== Objective =======

        # TODO: updated objective to include operating costs
        def obj(mod):
            return sum(mod.Delta*mod.P[t]*(mod.wdotS[t] - mod.wdotP[t]) for t in mod.T)
        mod.objective = pyomo.Objective(rule = obj, sense = pyomo.maximize)

        # ====== Constraints =======

        # System power balance with respect to AC bus
        def power_balance(mod, t):
            return (mod.wdotS[t] #- mod.wdotP[t] 
                                == mod.wdotWF[t] 
                                                    + mod.wdotPV[t] 
                                                    + mod.etaN*mod.wdotBD[t]
                                                    - (1/mod.etaP)*mod.wdotBC[t])
        mod.power_balance = pyomo.Constraint(mod.T, rule = power_balance)

        # PV resource bound
        def PV_resource(mod, t):
            return mod.wdotPV[t] <= mod.Wpv[t]
        mod.PV_resource = pyomo.Constraint(mod.T, rule = PV_resource)

        # Wind farm resource bound
        def WF_resource(mod, t):
            return mod.wdotWF[t] <= mod.Wwf[t]
        mod.WF_resource = pyomo.Constraint(mod.T, rule = WF_resource)

        # Net grid limit imposed
        def Wnet_limit(mod, t):
            return mod.wdotS[t] <= mod.Wnet[t]
        mod.Wnet_limit = pyomo.Constraint(mod.T, rule = Wnet_limit)

        # Battery State-of-Charge balance
        if self.simplebatt:
            # power accounting
            def battery_soc(mod, t):
                return (mod.bsoc[t] == (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) 
                                        + mod.Delta*(mod.wdotBC[t] - mod.wdotBD[t])/mod.CB)
        else:
            # current accounting
            def battery_soc(mod, t):
                return (mod.bsoc[t] == (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) 
                                        + mod.Delta*(mod.iP[t] - mod.iN[t])/mod.CB)
        mod.battery_soc = pyomo.Constraint(mod.T, rule = battery_soc)

        # Battery State-of-Charge bounds
        def soc_lower_bound(mod, t):
            return mod.bsoc[t] >= mod.SminB
        mod.soc_lower_bound = pyomo.Constraint(mod.T, rule = soc_lower_bound)

        def soc_upper_bound(mod, t):
            return mod.bsoc[t] <= mod.SmaxB
        mod.soc_upper_bound = pyomo.Constraint(mod.T, rule = soc_upper_bound)

        # Battery Charging power bounds
        def chargeP_lower_bound(mod, t):
            return mod.wdotBC[t] >= mod.PminB*mod.yP[t]
        mod.chargeP_lower_bound = pyomo.Constraint(mod.T, rule = chargeP_lower_bound)

        def chargeP_upper_bound(mod, t):
            return mod.wdotBC[t] <= mod.PmaxB*mod.yP[t]
        mod.chargeP_upper_bound = pyomo.Constraint(mod.T, rule = chargeP_upper_bound)

        # Battery Discharging power Bounds
        def dischargeP_lower_bound(mod, t):
            return mod.wdotBD[t] >= mod.PminB*mod.yN[t]
        mod.dischargeP_lower_bound = pyomo.Constraint(mod.T, rule = dischargeP_lower_bound)

        def dischargeP_upper_bound(mod, t):
            return mod.wdotBD[t] <= mod.PmaxB*mod.yN[t]
        mod.dischargeP_upper_bound = pyomo.Constraint(mod.T, rule = dischargeP_upper_bound)

        # Battery can only charge or discharge in a time period
        def charge_discharge_packing(mod, t):
            return mod.yP[t] + mod.yN[t] <= 1.
        mod.charge_discharge_packing = pyomo.Constraint(mod.T, rule = charge_discharge_packing)

        # Battery lifecycle counting TODO: should this be based on batter discharge?
        if self.simplebatt:
            # power accounting
            def bat_lifecycle(mod):
                return mod.blc == (mod.Delta/mod.CB)*sum((mod.gamma**t)*mod.wdotBC[t] for t in mod.T)
        else:
            # current accounting
            def bat_lifecycle(mod):
                return mod.blc == (mod.Delta/mod.CB)*sum((mod.gamma**t)*(mod.iP[t] - mod.zP[t]) for t in mod.T)
        mod.bat_lifecyle = pyomo.Constraint(rule = bat_lifecycle)

        ## Detailed battery constraints
        if not self.simplebatt:
            # Discharge current bounds
            def discharge_current_upper_soc(mod, t):
                return mod.iN[t] <= mod.ImaxN * (mod.bsoc0 if t == 0 else mod.bsoc[t-1])
            mod.discharge_current_upper_soc = pyomo.Constraint(mod.T, rule = discharge_current_upper_soc)

            def discharge_current_upper(mod, t):
                return mod.iN[t] <= mod.ImaxN * mod.yN[t]
            mod.discharge_current_upper = pyomo.Constraint(mod.T, rule = discharge_current_upper)

            def discharge_current_lower(mod, t):
                return mod.iN[t] >= mod.IminN * mod.yN[t]
            mod.discharge_current_lower = pyomo.Constraint(mod.T, rule = discharge_current_lower)

            # Charge current bounds
            def charge_current_upper_soc(mod, t):
                return mod.iP[t] <= mod.CB * (1 - (mod.bsoc0 if t == 0 else mod.bsoc[t-1]))/ mod.Delta
            mod.charge_current_upper_soc = pyomo.Constraint(mod.T, rule = charge_current_upper_soc)

            def charge_current_upper(mod, t):
                return mod.iP[t] <= mod.ImaxP * mod.yP[t]
            mod.charge_current_upper = pyomo.Constraint(mod.T, rule = charge_current_upper)

            def charge_current_lower(mod, t):
                return mod.iP[t] >= mod.IminP * mod.yP[t]
            mod.charge_current_lower = pyomo.Constraint(mod.T, rule = charge_current_lower)

            # Charge power (is equal to current*voltage)
            def charge_power(mod, t):
                return mod.wdotBC[t] == mod.AV*mod.zP[t] + (mod.BV + mod.Iavg*mod.Rint)*mod.xP[t]
            mod.charge_power = pyomo.Constraint(mod.T, rule = charge_power)

            def discharge_power(mod, t):
                return mod.wdotBD[t] == mod.AV*mod.zN[t] + (mod.BV + mod.Iavg*mod.Rint)*mod.xN[t]
            mod.discharge_power = pyomo.Constraint(mod.T, rule = discharge_power)

            # Aux. Variable bounds (xN[t] and xP[t]) (binary*continuous exact linearization)
            def auxN_lower_lim(mod, t):
                return mod.xN[t] >= mod.IminN * mod.yN[t]
            mod.auxN_lower_lim = pyomo.Constraint(mod.T, rule = auxN_lower_lim)

            def auxN_upper_lim(mod, t):
                return mod.xN[t] <= mod.ImaxN * mod.yN[t]
            mod.auxN_upper_lim = pyomo.Constraint(mod.T, rule = auxN_upper_lim)

            def auxN_diff_lower_lim(mod, t):
                return mod.iN[t] - mod.xN[t] >= - mod.ImaxN * (1 - mod.yN[t])
            mod.auxN_diff_lower_lim = pyomo.Constraint(mod.T, rule = auxN_diff_lower_lim)

            def auxN_diff_upper_lim(mod, t):
                return mod.iN[t] - mod.xN[t] <= mod.ImaxN * (1 - mod.yN[t])
            mod.auxN_diff_upper_lim = pyomo.Constraint(mod.T, rule = auxN_diff_upper_lim)

            def auxP_lower_lim(mod, t):
                return mod.xP[t] >= mod.IminP * mod.yP[t]
            mod.auxP_lower_lim = pyomo.Constraint(mod.T, rule = auxP_lower_lim)

            def auxP_upper_lim(mod, t):
                return mod.xP[t] <= mod.ImaxP * mod.yP[t]
            mod.auxP_upper_lim = pyomo.Constraint(mod.T, rule = auxP_upper_lim)

            def auxP_diff_lower_lim(mod, t):
                return mod.iP[t] - mod.xP[t] >= - mod.ImaxP * (1 - mod.yP[t])
            mod.auxP_diff_lower_lim = pyomo.Constraint(mod.T, rule = auxP_diff_lower_lim)

            def auxP_diff_upper_lim(mod, t):
                return mod.iP[t] - mod.xP[t] <= mod.ImaxP * (1 - mod.yP[t])
            mod.auxP_diff_upper_lim = pyomo.Constraint(mod.T, rule = auxP_diff_upper_lim)

            # Aux. Variable bounds (zP[t] and zN[t]) (continuous*continuous approx. linearization)
            ## zP[t] Aux variable
            def zp_lower_1(mod, t):
                return mod.zP[t] >= mod.ImaxP * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SmaxB * mod.iP[t] - mod.SmaxB * mod.ImaxP
            mod.zp_lower_1 = pyomo.Constraint(mod.T, rule = zp_lower_1)

            def zp_lower_2(mod, t):
                return mod.zP[t] >= mod.IminP * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SminB * mod.iP[t] - mod.SminB * mod.IminP
            mod.zp_lower_2 = pyomo.Constraint(mod.T, rule = zp_lower_2)

            def zp_upper_1(mod, t):
                return mod.zP[t] <= mod.ImaxP * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SminB * mod.iP[t] - mod.SminB * mod.ImaxP
            mod.zp_upper_1 = pyomo.Constraint(mod.T, rule = zp_upper_1)

            def zp_upper_2(mod, t):
                return mod.zP[t] <= mod.IminP * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SmaxB * mod.iP[t] - mod.SmaxB * mod.IminP
            mod.zp_upper_2 = pyomo.Constraint(mod.T, rule = zp_upper_2)

            ## zN[t] Aux variable
            def zn_lower_1(mod, t):
                return mod.zN[t] >= mod.ImaxN * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SmaxB * mod.iN[t] - mod.SmaxB * mod.ImaxN
            mod.zn_lower_1 = pyomo.Constraint(mod.T, rule = zn_lower_1)

            def zn_lower_2(mod, t):
                return mod.zN[t] >= mod.IminN * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SminB * mod.iN[t] - mod.SminB * mod.IminN
            mod.zn_lower_2 = pyomo.Constraint(mod.T, rule = zn_lower_2)

            def zn_upper_1(mod, t):
                return mod.zN[t] <= mod.ImaxN * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SminB * mod.iN[t] - mod.SminB * mod.ImaxN
            mod.zn_upper_1 = pyomo.Constraint(mod.T, rule = zn_upper_1)

            def zn_upper_2(mod, t):
                return mod.zN[t] <= mod.IminN * (mod.bsoc0 if t == 0 else mod.bsoc[t-1]) + mod.SmaxB * mod.iN[t] - mod.SmaxB * mod.IminP
            mod.zn_upper_2 = pyomo.Constraint(mod.T, rule = zn_upper_2)

        self.OptModel = mod  # stores the optimization model


    def buildOptModel(self):
        #mod = self.OptModel

        # Create all constraints and objectives
        # Create a list to include in problem
        # Create model based on list
            # using .activate() and .deactivate
        pass


    def hybrid_optimization_call(self):
        self.initializeOptModel()
        
        solver = pyomo.SolverFactory('glpk')
        solver_opt = {}
        solver_opt['log'] = 'dispatch.log'

        #--solver-options= "mipgap=0.01??? "
        solver.solve(self.OptModel, tee=True, options= solver_opt)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dispatch_plotting import gen_plot 

    ndays = 3
    phorizon = 24*ndays

    HP = dispatch_problem(phorizon, simplebatt = True)
    
    # Making up input data
    P = list(np.random.rand(phorizon))
    Wnet = [10000.0] * phorizon
    
    Wwf = list(np.random.rand(phorizon))
    Wwf = [x*7500 for x in Wwf]

    Wpv = list(np.random.rand(phorizon))
    Wpv = [x*7500 for x in Wpv]
    for day in range(ndays):
       Wpv[24*day:24*day + 6] = [0.] * len(Wpv[24*day:24*day + 6])
       Wpv[24*day+19:24*(day+1)] = [0.] * len(Wpv[24*day+19:24*(day+1)])
    HP.updateSolarWindResGrid(P, Wnet, Wpv, Wwf)

    etaP = 0.85
    etaN = 0.85
    CB = 30000.0
    PmaxB = 5000.0
    PminB = 0.0
    SmaxB = 1.0
    SminB = 0.1
    HP.updateSimpleBatteryParams(etaP, etaN, CB, PmaxB, PminB, SmaxB, SminB)

    bsoc0 = 0.1
    HP.updateInitialConditions(bsoc0)

    HP.hybrid_optimization_call()

    time = [t for t in range(phorizon)]
    #plt.figure()
    # fig, ax1 = plt.subplots()
    # ax1.plot(time, Wpv, 'r', label = 'PV Resource')
    # ax1.plot(time, Wwf, 'b', label = 'Wind Resource')
    # ax1.plot(time, HP.OptModel.wdotS[:](), 'g', label = 'Sold Electricity')
    # plt.legend()
    
    # ax2 = ax1.twinx()
    # ax2.plot(time, P, 'k', label = 'Grid Price')
    # plt.legend()
    # fig.tight_layout()
    # plt.show()
    



    gen_plot(HP.OptModel, is_battery = True) #title = None, grid_check = []):


