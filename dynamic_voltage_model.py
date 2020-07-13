"""
Dynamic voltage model -> Lives in SAM
Tremblay 2009 "A generic battery model for the dynamic simulation of hybrid electric vehicles 
"""

import PySAM.StandAloneBattery as battery_model
from PySAM.BatteryTools import *
import math
import matplotlib.pyplot as plt


def calc_volt_model_params(battery):
    BCell = battery.BatteryCell

    params = {}
    params['I'] = BCell.batt_Qfull * BCell.batt_C_rate        # [A]
    params['A'] = BCell.batt_Vfull - BCell.batt_Vexp          # [V]
    params['B0'] = 3.0/ BCell.batt_Qexp                       # [1/Ah]
    params['K'] = (( BCell.batt_Vfull - BCell.batt_Vnom + params['A']*(math.exp(-params['B0']*BCell.batt_Qnom) - 1.)) * (BCell.batt_Qfull - BCell.batt_Qnom)) / BCell.batt_Qnom # [V] - polarization voltage
    params['R'] = BCell.batt_resistance
    params['E0'] = BCell.batt_Vfull + params['K'] + params['R']*params['I'] - params['A']
    params['R_battery'] = params['R'] * battery.BatterySystem.batt_computed_series / battery.BatterySystem.batt_computed_strings

    return params

def voltage_tremblay(params, Q_cell, I, q0_cell):
    it = Q_cell - q0_cell
    E = params['E0'] - params['K']*(Q_cell / (Q_cell - it)) + params['A']*math.exp(-params['B0'] * it)
    return E - params['R'] * I
    
def setBatteryCellParams(battery, **kwargs):
    """
    Qfull
    Vfull
    Qexp
    Vexp
    Qnom
    Vnom
    C_rate
    """
    for key, val in kwargs.items():
        setattr(battery.BatteryCell, 'batt_'+key, val)




if __name__ == "__main__":

    desired_power = 50000           # [kW] 
    desired_capacity = 200000.      # [kWh]
    desired_voltage = 500.          # [Volts]
    battery = battery_model.default("GenericBatterySingleOwner")
    battery_size_specs = battery_model_sizing(battery, desired_power, desired_capacity, desired_voltage)
    Q_cell = battery.BatteryCell.batt_Qfull

    if False:
        batt_vals = {}
        batt_vals['Qfull'] = 0.
        batt_vals['Vfull'] = 0.
        batt_vals['Qexp'] = 0.
        batt_vals['Vexp'] = 0.
        batt_vals['Qnom'] = 0.
        batt_vals['Vnom'] = 0.
        batt_vals['C_rate'] = 0.
        setBatteryCellParams(battery, **batt_vals)

    if False:
        params = calc_volt_model_params(battery)
        # testing
        volt = voltage_tremblay(params, Q_cell, 1., Q_cell*0.05)


    C_rate = [x/4. for x in range(1,6+1)]
    N_steps = 500

    plt.figure()

    for cr in C_rate:
        batt_vals = {}
        batt_vals['C_rate'] = cr
        batt_vals['Vexp'] = 4.05 - 0.1*cr           # completely made up
        batt_vals['Qexp'] = 0.04005 + 0.1*cr        # completely made up
        
        batt_vals['Vexp'] = 3.9 - 0.2*cr           # completely made up
        batt_vals['Qexp'] = 0.85 + 0.1*cr        # completely made up
        
        setBatteryCellParams(battery, **batt_vals)
        params = calc_volt_model_params(battery)
        
        ispower = True
        if ispower:
            # Constant power Condition
            P_const = battery.BatteryCell.batt_Vnom_default*Q_cell*cr   # [W]
            V = [battery.BatteryCell.batt_Vfull]
            SOC = [1.0]
            time = [0.0]
            q0_cell = Q_cell
            timestep = (1/cr)/N_steps
            for t in range(1,N_steps):
                I = P_const/V[-1]
                if (q0_cell - I*timestep < 1e-6):
                    break
                V.append(voltage_tremblay(params, Q_cell, I, q0_cell - I*timestep))
                q0_cell -= I*timestep
                SOC.append(q0_cell/Q_cell)
                time.append(t*timestep)
            plt.plot(time, V,'.', label = 'C-rate: {0:5.2f}'.format(cr))
            plt.xlabel("Discharge Time [hr]")

        else:
            SOC = [x/N_steps for x in range(1, N_steps+1)]
            V = []
            for c in SOC:
                # constant current condition
                V.append(voltage_tremblay(params, Q_cell, Q_cell*cr, Q_cell*c))
                

            plt.plot(SOC, V,'.', label = 'C-rate: {0:5.2f}'.format(cr))
            plt.xlabel("State-of-Charge [-]")

        ## Filter voltages greater than a dead voltage (3.4)
        V = [x for x in V if x >= 3.4]
        print("Average Voltage: {0:5.2f}".format(sum(V)/len(V)))

    if not ispower:
        ## linear assumption
        lp_soc = [1. - battery.BatteryCell.batt_Qnom/Q_cell, 1. - battery.BatteryCell.batt_Qexp/Q_cell]
        lp_V = [battery.BatteryCell.batt_Vnom, battery.BatteryCell.batt_Vexp]

        plt.plot(lp_soc, lp_V, label = 'Current assumption')




    plt.legend()
    plt.ylim([3.4, 4.2])
    plt.ylim([2.5, 4.2])
    plt.ylabel("Cell Voltage [V]")
    plt.show()

    pass

