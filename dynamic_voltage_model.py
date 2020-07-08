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
    return E #- params['R'] * I
    
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

    Q_cell = battery.BatteryCell.batt_Qfull
    params = calc_volt_model_params(battery)

    # testing
    volt = voltage_tremblay(params, Q_cell, 1., Q_cell*0.05)

    steps = 500
    SOC = [x/steps for x in range(1,steps+1)]
    V = []
    for c in SOC:
        V.append(voltage_tremblay(params, Q_cell, 1., Q_cell*c))

    plt.figure()
    plt.plot(SOC, V,'.')
    plt.ylim([0,5])
    plt.xlabel("State-of-Charge [-]")
    plt.ylabel("Cell Voltage [V]")
    plt.show()

    pass

