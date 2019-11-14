import math


def size_storage(defaults, battery_kw_desired, battery_kwh_desired, battery_voltage_desired=500):
    batt_C_rate_max_discharge = battery_kw_desired / battery_kwh_desired
    batt_C_rate_max_charge = batt_C_rate_max_discharge
    batt_Qfull = defaults['Battery']['StandAloneBattery']['Battery']['batt_Qfull']
    batt_Vnom_default = defaults['Battery']['StandAloneBattery']['Battery']['batt_Vnom_default']
    bank_desired_voltage = battery_voltage_desired
    num_series = math.ceil(bank_desired_voltage / batt_Vnom_default)
    num_strings = round(battery_kwh_desired * 1000 / (batt_Qfull * batt_Vnom_default * num_series))

    batt_computed_voltage = batt_Vnom_default * num_series
    bank_capacity = batt_Qfull * batt_computed_voltage * num_strings * 0.001
    bank_power = bank_capacity * batt_C_rate_max_discharge
    bank_power_charge = bank_capacity * batt_C_rate_max_charge

    batt_current_charge_max = batt_Qfull * num_strings * batt_C_rate_max_charge
    batt_current_discharge_max = batt_Qfull * num_strings * batt_C_rate_max_discharge

    # assign to relevant places
    defaults['Battery']['StandAloneBattery']['Battery']['batt_computed_strings'] = num_strings
    defaults['Battery']['StandAloneBattery']['Battery']['batt_computed_series'] = num_series
    defaults['Battery']['StandAloneBattery']['Battery']['batt_computed_bank_capacity'] = bank_capacity

    defaults['Battery']['StandAloneBattery']['Battery']['batt_current_charge_max'] = batt_current_charge_max
    defaults['Battery']['StandAloneBattery']['Battery']['batt_current_discharge_max'] = batt_current_discharge_max
    defaults['Battery']['StandAloneBattery']['Battery']['batt_power_charge_max'] = bank_power_charge
    defaults['Battery']['StandAloneBattery']['Battery']['batt_power_discharge_max'] = bank_power

    # these are for some specialized applications considering the charge efficiency of the power electronics and trying to achieve an exact AC output
    defaults['Battery']['StandAloneBattery']['Battery']['batt_power_charge_max_kwdc'] = bank_power_charge
    defaults['Battery']['StandAloneBattery']['Battery']['batt_power_discharge_max_kwdc'] = bank_power

    return defaults
