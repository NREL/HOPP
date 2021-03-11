import pyomo.environ as pyomo
import numpy as np
import os

def list2dict(lst):
    dct = {}
    for i,v in enumerate(lst):
        dct[i] = v
    return dct


class batteryCell:
    def __init__(self):
        self.a = 0.8587             # [V]   Slope of linear voltage SOC model
        self.b = 3.2768             # [V]   Intercept of linear voltage SOC model
        self.avgV = 3.70615         # [V]   Average cell voltage through SOC range
        self.nomV = 4.1355          # [V]   Nominal cell voltage 
        self.intR = 0.1394          # [Ohm] Internal cell resistance
        self.cap = 3.4              # [Ah]  Cell capacity
        self.C_p = 3.               # [hr]  Charge C-rate?
        self.C_n = 0.0401           # [hr]  Discharge C-rate?
        self.socLB = 0.1            # [-]   State-of-charge lower bound
        self.socUB = 0.9            # [-]   State-of-charge upper bound

    def updateBatteryCellSpecs(self, **kwargs):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        a           [V]                 Slope of linear voltage SOC model  \n
        b           [V]                 Intercept of linear voltage SOC model \n
        avgV        [V]                 Average cell voltage through SOC range \n
        nomV        [V]                 Nominal cell voltage \n
        intR        [Ohm]               Internal cell resistance \n
        cap         [Ah]                Cell capacity \n
        C_p         [hr]                Charge C-rate \n
        C_n         [hr]                Discharge C-rate \n
        socLB       [-]                 State-of-charge lower bound \n
        socUB       [-]                 State-of-charge upper bound
        """
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                print("Error: " + key + " is not a attribute name in the BatteryCell class.")

class SimpleBattery:
    def __init__(self, Cell, capD, voltD, powD):
        """
        Parameter   [Units]      Description
        ----------------------------------------------------
        Cell        [-]          batteryCell class \n
        capD        [kWh]        Desired battery capacity \n
        voltD       [V]          Desired battery voltage \n
        powD        [kW]         Desired battery power
        """
        ## TODO: Should calculations be based on nominal voltage? or average voltage?
        self.cell = Cell    # Save batteryCell

        # cells in series, nominal battery voltage, and avgerage battery voltage
        self.series_Ncells = int((voltD + self.cell.nomV)/self.cell.nomV)     # ceiling
        self.nomV = self.series_Ncells*self.cell.nomV
        self.avgV = self.series_Ncells*self.cell.avgV
        self.desV = voltD

        # cells in parallel, total cells in battery, and nominal battery capacity
        self.par_Nstrings = int((capD*1000.0 + self.nomV*self.cell.cap)/(self.nomV*self.cell.cap))   # ceiling
        self.Ncells = self.series_Ncells * self.par_Nstrings
        self.nomC = self.par_Nstrings*self.nomV*self.cell.cap/1000.0       # [kWh]
        self.desC = capD

        self.desP = powD    ## TODO: should we have a charge and a discharge spec?  Should this be based on the C-rate?

        ## TODO: add DC/AC inverter and AC/DC converter BOS specs (could use SAM inverter library to get these values)
        # TODO: size BoS based on power inputs and outputs
        # Charging: AC/DC converter
        self.alphaP = 50.0       # [kW] No-load losses                       (conv['inv_snl_pso']*Bat['number_converters']/1.e3)
        self.betaP = 0.02        # [-] Converting efficiency (ish)           ((conv['inv_snl_pdco'] - conv['inv_snl_pso'])/conv['inv_snl_paco'] - 1)

        # Discharging: DC/AC inverter
        self.alphaN = 50.0       # [kW] No-load losses                       (conv['inv_snl_pso']*Bat['number_converters']/1.e3)
        self.betaN = 0.02        # [-] Converting efficiency (ish)           ((conv['inv_snl_pdco'] - conv['inv_snl_pso'])/conv['inv_snl_paco'] - 1)

        # Charge and Discharge efficiencies for Simple battery - Assuming a round-trip efficiency of 85%
        self.etaP = 0.922        # [-] charge efficiency
        self.etaN = 0.922        # [-] discharge efficiency

    def setChargeDisChargeEff(self, etaP, etaN):
        """
        Parameter   [Units]      Description
        ----------------------------------------------------
        etaP        [-]          Charge efficiency \n
        etaN        [-]          Discharge efficiency
        """
        self.etaP = etaP
        self.etaN = etaN

class param:
    def __init__(self, val, time_index = False):
        self.val = val
        self.time_index = time_index

class dispatch_problem:
    def __init__(self, prob_horizon, battery, simplebatt = True):
        """

        Parameter       [Units]      Description
        ----------------------------------------------------
        prob_horizon    [-]          Number of time steps in problem horizon \n`
        battery         [-]          Battery class object \n
        simplebatt      bool         Model battery using simple or detailed approach
        """
        self.logname = 'battery_dispatch_optimization.log'
        try:
            os.remove(self.logname)
        except:
            pass

        ## condition off if battery is a StandAloneBattery or a Battery Class
        self.battery = battery          # Saving battery class
        self.simplebatt = simplebatt

        ## ============= Dispatch Model =====================
        self.nt = prob_horizon                          # [-]       Number of time steps in problem horizon

        ## =============== Parameters =======================
        self.gamma = param(0.999)                       # [-]       Exponential time weighting factor

        # Time-index Parameters - Initialize but empty 
        self.P = param({}, time_index=True)           # [$/kWh]   Normalized grid electricity price in time t
        self.Wnet = param({}, time_index=True)        # [kW]      Net grid transmission upper limit in time t
        self.Wpv = param({}, time_index=True)         # [kW]      Available PV array power in time t
        self.Wwf = param({}, time_index=True)         # [kW]      Available wind farm power in time t

        if self.battery.__class__.__name__ == 'Battery':
            self.Delta = param(8760. / len(self.battery.BatteryCell.batt_room_temperature_celsius) )   # There might be a better way to get Delta however this work
            self.NTday = int(round(24/self.Delta.val))      # [-]       Number of time periods in day
            self.initStandAloneBattery()

        elif self.battery.__class__.__name__ == 'SimpleBattery':
            self.Delta = param(1.0)                         # [hr]      Time step duration
                # NOTE: this assumes hourly step size, this should be adjusted in future support
            self.NTday = int(round(24/self.Delta.val))      # [-]       Number of time periods in day
            self.initSimpleBattery()
        else:
            raise TypeError('battery class object must be either a SimpleBattery or a StandAloneBattery. Other classes are not currently supported.')

        # Cost Parameters TODO: Update these values
        self.CbP = param(0.002)                         # [$/kWh_DC]        Operating cost of battery charging
        self.CbN = param(0.002)                         # [$/kWh_DC]        Operating cost of battery discharging 
        if self.battery.__class__.__name__ == 'Battery':
            self.Clc = param(0.06*self.battery.BatterySystem.batt_computed_bank_capacity)
        elif self.battery.__class__.__name__ == 'SimpleBattery':
            self.Clc = param(0.06*self.battery.nomC)        # [$/lifecycle]     Lifecycle cost for battery
        self.CdeltaW = param(0.001)                     # [$/DeltaKW_AC]    Penalty for change in net power production
        self.Cpv = param(0.0015)                        # [$/kWh_DC]        Operating cost of photovolaic array
        self.Cwf = param(0.005)                         # [$/kWh_AC]        Operating cost of wind farm

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

    def initStandAloneBattery(self):
        ## Initializes dispatch model using SimpleBattery class

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

        B = self.battery.BatterySystem
        BC = self.battery.BatteryCell

        # Battery parameters
        if self.simplebatt:
            # TODO: update these using SAM model
            self.etaP = param(0.95)          # [-]       Battery Charging efficiency
            self.etaN = param(0.95)          # [-]       Battery Discharging efficiency

            self.CB = param(B.batt_computed_bank_capacity)                                            # [kWh]     Battery manufacturer-specified capacity
        else:
            self.CB = param(B.batt_computed_bank_capacity/(BC.batt_Vfull*B.batt_computed_series) )       # [kAh] Battery manufacturer-specified capacity

            # Calculating linear approximation for Voltage as a function of state-of-charge
            SOCnom = (BC.batt_Qfull - BC.batt_Qnom)/BC.batt_Qfull
            if False:
                # Using cell exp and nom voltage points
                SOCexp = (BC.batt_Qfull - BC.batt_Qexp)/BC.batt_Qfull

                a = (BC.batt_Vexp - BC.batt_Vnom)/(SOCexp - SOCnom)
                b = BC.batt_Vexp - a*SOCexp
            else:
                # Using Cell full and nom voltage points 
                a = (BC.batt_Vfull - BC.batt_Vnom)/(1.0 - SOCnom)
                b = BC.batt_Vfull - a

            self.AV = param(B.batt_computed_series * a)   # [V]       Battery linear voltage model slope coefficient
            self.BV = param(B.batt_computed_series * b)   # [V]       Battery linear voltage model intercept coefficient

            ## TODO: how should this be handled?
            self.alphaP = param(0)    # [kW_DC]   Bi-directional intercept for charge
            self.betaP = param(0)      # [-]   Bi-directional slope for charge
            
            self.alphaN = param(0)    # [kW_DC]   Bi-directional intercept for discharge
            self.betaN = param(0)      # [-]   Bi-directional slope for discharge

            # TODO: Is Iavg right? Average between zero and averge of charge and discharge max
            self.Iavg = param((B.batt_current_discharge_max + B.batt_current_charge_max)/4.)       # [A]       Typical current expected from the battery for both charge and discharge
            self.Rint = param(BC.batt_resistance*B.batt_computed_series/B.batt_computed_strings)  # [Ohms]    Battery internal resistance

            ## Charge current limits
            self.ImaxP = param(B.batt_current_charge_max/1000.0)        # [kA]      Battery charge maximum current
            self.IminP = param(0.0)                                     # [kA]      Battery charge minimum current
            
            ## Discharge current limits
            self.ImaxN = param(B.batt_current_discharge_max/1000.0)     # [kA]      Battery discharge maximum current
            self.IminN = param(0.0)                                     # [kA]      Battery discharge minimum current

        self.PmaxB = param(B.batt_power_discharge_max_kwdc)             # [kW_DC]   Battery maximum power rating
        self.PminB = param(0.0)                                         # [kW_DC]   Battery minimum power rating

        self.SmaxB = param(BC.batt_maximum_SOC/100.0)     # [-]       Battery maximum state of charge
        self.SminB = param(BC.batt_minimum_SOC/100.0)     # [-]       Battery minimum state of charge

    def initSimpleBattery(self):
        ## Initializes dispatch model using SimpleBattery class

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

        # Battery parameters
        if self.simplebatt:
            self.etaP = param(self.battery.etaP)          # [-]       Battery Charging efficiency
            self.etaN = param(self.battery.etaN)          # [-]       Battery Discharging efficiency

            self.CB = param(self.battery.nomC)            # [kWh]     Battery manufacturer-specified capacity
        else:
            self.CB = param(self.battery.nomC/self.battery.nomV)    # [kAh] Battery manufacturer-specified capacity (was average voltage??, changed it to nominal)

            self.AV = param(self.battery.series_Ncells*self.battery.cell.a)   # [V]       Battery linear voltage model slope coefficient
            self.BV = param(self.battery.series_Ncells*self.battery.cell.b)   # [V]       Battery linear voltage model intercept coefficient

            self.alphaP = param(self.battery.alphaP)    # [kW_DC]   Bi-directional intercept for charge
            self.betaP = param(self.battery.betaP)      # [-]   Bi-directional slope for charge
            
            self.alphaN = param(self.battery.alphaN)    # [kW_DC]   Bi-directional intercept for discharge
            self.betaN = param(self.battery.betaN)      # [-]   Bi-directional slope for discharge

            ## Charge current limits
            self.ImaxP = param(self.CB.val/(self.battery.cell.C_p + self.Delta.val))        # [kA]      Battery charge maximum current
            self.IminP = param(0.0)                                                         # [kA]      Battery charge minimum current
            
            ## Discharge current limits
            self.ImaxN = param(min(2.*self.CB.val,self.CB.val/(self.battery.cell.C_n + self.Delta.val)))  # [kA]      Battery discharge maximum current
            self.IminN = param(0.0)                                                                       # [kA]      Battery discharge minimum current

            # TODO: Is Iavg right? Seems like C_n and C_p should be taken into account
            ## Average between zero and Average charge and discharge current
            self.Iavg = param(((self.ImaxP.val + self.ImaxN)/4.)*1000.0)                                    # [A]       Typical current expected from the battery for both charge and discharge
            #self.Iavg = param((self.battery.nomC/self.battery.avgV)*1000.0)                                # [A]       Typical current expected from the battery for both charge and discharge
            self.Rint = param(self.battery.cell.intR*self.battery.series_Ncells/self.battery.par_Nstrings)  # [Ohms]    Battery internal resistance


        self.PmaxB = param(self.battery.desP)           # [kW_DC]   Battery maximum power rating
        self.PminB = param(0.0)                         # [kW_DC]   Battery minimum power rating

        self.SmaxB = param(self.battery.cell.socUB)     # [-]       Battery maximum state of charge
        self.SminB = param(self.battery.cell.socLB)     # [-]       Battery minimum state of charge

    # TODO: updated this using **kwargs
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
        * NOTE: Data type is Lists of problem horizon length TODO: add a check here
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
        #self.updateParameters(locals())
        pass

    def updateConverterParams(self, alphaC, betaC, WminC, WmaxC):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        alphaC      [kW]                Converter slope linear approximations of DC-to-DC efficiency \n
        betaC       [-]                 Converter intercept linear approximations of DC-to-DC efficiency \n
        WminC       [kW_DC]             Converter minimum DC power limit \n
        WmaxC       [kW_DC]             Converter maximum DC power limit
        """
        #self.updateParameters(locals())
        pass

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
        #self.updateParameters(locals())
        pass

    def updateInitialConditions(self, bsoc0):
        """
        Parameter   [Units]             Description
        ----------------------------------------------------
        bsoc0       [-]                 Initial battery state of charge
        """
        self.updateParameters(locals())

    def initializeOptModel(self):
        mod = pyomo.ConcreteModel()

        # ====== Sets =======
        mod.T = pyomo.Set(initialize = range(self.nt) )    #set of time periods
        mod.D = pyomo.Set(initialize = range(int(self.nt/self.NTday)) )  #set of days

        def dt_init(mod):
            return ((d,t) for d in mod.D for t in list(range((d)*self.NTday, (d)*self.NTday + (self.NTday - 1))))
        mod.DT = pyomo.Set(dimen=2, initialize = dt_init)

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

        setattr(mod,'bsocm', pyomo.Var( mod.D, domain = pyomo.NonNegativeReals))

        # ====== Objective =======

        def obj(mod):
            return ( sum((mod.gamma**t)*mod.Delta*mod.P[t]*(mod.wdotS[t] - mod.wdotP[t]) 
                                - ((1/mod.gamma)**t)*mod.Delta*(mod.Cpv*mod.wdotPV[t] 
                                                                    + mod.Cwf*mod.wdotWF[t] 
                                                                    + mod.CbP*mod.wdotBC[t] 
                                                                    + mod.CbN*mod.wdotBD[t]) 
                                for t in mod.T) - mod.Clc*mod.blc - 3000*sum((1 - mod.bsocm[d]) for d in mod.D))
        mod.objective = pyomo.Objective(rule = obj, sense = pyomo.maximize)
        # NOTE: Currently, curtailment is free.  Is this activity free in reality? or is there a small operating cost assocate with it?

        # ====== Constraints =======

        #### Depth of discharge per day
        def minDoD_perDay(mod, d, t):
                return (mod.bsocm[d] <= mod.bsoc[t])
        mod.minDoD_perDay = pyomo.Constraint(mod.DT, rule = minDoD_perDay)

        # System power balance with respect to AC bus
        if self.simplebatt:
            def power_balance(mod, t):
                return (mod.wdotS[t] #- mod.wdotP[t]            # Currently cannot purchase from the grid
                                        == mod.wdotWF[t] 
                                                + mod.wdotPV[t] 
                                                + mod.etaN*mod.wdotBD[t]
                                                - (1/mod.etaP)*mod.wdotBC[t])
        else:
            def power_balance(mod, t):
                return (mod.wdotS[t] #- mod.wdotP[t]
                                        == mod.wdotWF[t]
                                                + mod.wdotPV[t]
                                                + (mod.wdotBD[t] - mod.alphaN*mod.yN[t])/(1+mod.betaN)
                                                - ((1+mod.betaP)*mod.wdotBC[t] + mod.alphaP*mod.yP[t])  )
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
                                        + mod.Delta*(mod.iP[t]*(0.95) - mod.iN[t])*(0.95)/mod.CB)       # TODO: adjustment factor
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

        # Battery lifecycle counting TODO: should this be based on battery discharge?
        
        if self.simplebatt:
            # power accounting
            def bat_lifecycle(mod):
                return mod.blc == (mod.Delta/mod.CB)*sum(mod.wdotBC[t] for t in mod.T)
        else:
            # current accounting
            def bat_lifecycle(mod):
                return mod.blc == (mod.Delta/mod.CB)*sum((0.8*mod.iN[t] - 0.8*mod.zN[t]) for t in mod.T)
                #return mod.blc == (mod.Delta/mod.CB)*sum((mod.gamma**t)*(mod.iP[t]) for t in mod.T)
        
        #def bat_lifecycle(mod):
        #    return mod.blc == (mod.Delta/mod.CB)*sum(mod.wdotBC[t] for t in mod.T)
        mod.bat_lifecycle = pyomo.Constraint(rule = bat_lifecycle)

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
                return mod.wdotBD[t] == (mod.AV*mod.zN[t] + (mod.BV - mod.Iavg*mod.Rint)*mod.xN[t])*(1-0.1) #TODO:adjustment factor
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


    def hybrid_optimization_call(self, printlogs = False):
        self.initializeOptModel() # Builds model
        
        #solver = pyomo.SolverFactory('scip')    # Haven't been able to install
        solver = pyomo.SolverFactory('glpk')    # Ref. on solver options: https://en.wikibooks.org/wiki/GLPK/Using_GLPSOL

        solver_opt = {}
        if printlogs:
            solver_opt['log'] = 'dispatch_instance.log'
        solver_opt['cuts'] = None
        solver_opt['mipgap'] = 0.1 #[%]?
        #solver_opt['tmlim'] = 300


        solver.solve(self.OptModel, options= solver_opt)
        #solver.solve(self.OptModel, options= solver_opt, tee=True)
        ## NEED to add a flag for infeasiable problems

        # appends dispatch_instance.log to dispatch.log
        if printlogs:
            fin = open('dispatch_instance.log', 'r')
            data = fin.read()
            fin.close()

            fout = open(self.logname, 'a+')
            fout.write("="*50 + "\n")
            fout.write(data)
            fout.close()

def calcMassSurfArea(model, **kwargs):
    """
    model                             StandAloneBattery
    specE_per_mass    [Wh/kg]         Energy content per unit mass
    specE_per_volume  [Wh/m^3]        Energy content per unit Volume
    """
    params = {}
    params['specE_per_mass'] = 197.33   #[Wh/kg]
    params['specE_per_volume'] = 501.25   #[Wh/L]
    params.update(kwargs)

    bcap = model.BatterySystem.batt_computed_bank_capacity     
    # mass calculation
    model.BatterySystem.batt_mass = bcap*1000./params['specE_per_mass']             #[kWh] -> [Wh]

    # suface area calculation (assuming perfect cube)
    model.BatterySystem.batt_surface_area = 6.0*(bcap/params["specE_per_volume"])**(2./3.)      #[kWh] -> [Wh] and [L] -> [m^3] (cancels out)
        
def setStatefulUsingStandAlone(BatteryStateful, StandAloneBattery):
    ## sets up Stateful battery class using StandAloneBattery class
    BCell = StandAloneBattery.BatteryCell
    BSys = StandAloneBattery.BatterySystem

    params = {}
    params['input_current'] = 0
    params['control_mode'] = 0  # control mode has to be updated if using power instead of current

    attrBCell = ['chem', 'initial_SOC', 'maximum_SOC', 'minimum_SOC', 
                    'LeadAcid_tn', 'LeadAcid_qn', 'LeadAcid_q10', 'LeadAcid_q20', 
                    'voltage_choice', 'Vnom_default', 'Vfull', 'Vexp', 'Vnom', 'Qfull', 
                    'Qexp', 'Qnom', 'C_rate', 'Cp', 'calendar_choice', 'calendar_q0', 
                    'calendar_a', 'calendar_b', 'calendar_c'] 
    
    for attr in attrBCell:
        if "LeadAcid" in attr:
            if "_q" in attr:
                params[attr.lower()] = getattr(BCell, attr + '_computed')
            else:
                params[attr.lower()] = getattr(BCell, attr)
        else:
            params[attr] = getattr(BCell,'batt_' + attr)

    attrBSys = ['mass', 'surface_area', 'loss_choice', 'replacement_option', 'replacement_capacity']
    for attr in attrBSys:
        params[attr] = getattr(BSys, 'batt_' + attr)

    params['nominal_energy'] = BSys.batt_computed_bank_capacity
    params['nominal_voltage'] = BSys.batt_computed_series * BCell.batt_Vnom
    params['dt_hr'] = 1.0   # TODO: update
    params['resistance'] = BCell.batt_resistance * BSys.batt_computed_series/BSys.batt_computed_strings  #TODO: is this for the cell or whole battery??
    params['h'] = BCell.batt_h_to_ambient
    params['T_room_init'] = BCell.batt_room_temperature_celsius[0]

    params['cap_vs_temp'] = [list(x) for x in BCell.cap_vs_temp]
    params['cycling_matrix'] = [list(x) for x in BCell.batt_lifetime_matrix]
    params['calendar_matrix'] = [list(x) for x in BCell.batt_calendar_lifetime_matrix]

    params['monthly_charge_loss'] = list(BSys.batt_losses_charging)
    params['monthly_discharge_loss'] = list(BSys.batt_losses_discharging)
    params['monthly_idle_loss'] = list(BSys.batt_losses_idle)
    params['schedule_loss'] = list(BSys.batt_losses)    # TODO: I think this is right
    params['replacement_schedule_percent'] = list(BSys.batt_replacement_schedule_percent)

    # set parameter values in BatteryStateful object
    for k, v in params.items():
        BatteryStateful.value(k, v)

    BatteryStateful.setup()
    return

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from hybrid.dispatch_plotting import gen_plot

    ## Create a battery for problem
    cell = batteryCell()
    battery = SimpleBattery(cell, 15000., 220, 5000)

    ndays = 4
    phorizon = int(24*ndays)

    HP = dispatch_problem(phorizon, battery, simplebatt = True)
    
    # Making up input data
    np.random.seed(1)
    P = list(np.random.rand(phorizon))
    P = [x/10. for x in P]
    Wnet = [10000.0] * phorizon
    
    Wwf = list(np.random.rand(phorizon))
    Wwf = [x*7500 for x in Wwf]

    Wpv = list(np.random.rand(phorizon))
    Wpv = [x*7500 for x in Wpv]
    for day in range(ndays):
       Wpv[24*day:24*day + 6] = [0.] * len(Wpv[24*day:24*day + 6])
       Wpv[24*day+19:24*(day+1)] = [0.] * len(Wpv[24*day+19:24*(day+1)])
    HP.updateSolarWindResGrid(P, Wnet, Wpv, Wwf)

    CbP = 0.002
    CbN = 0.002
    Clc = 0.06*HP.battery.nomC
    CdeltaW = 0.0
    Cpv = 0.001
    Cwf = 0.003
    HP.updateCostParams( CbP, CbN, Clc, CdeltaW, Cpv, Cwf)

    bsoc0 = 1.0
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


