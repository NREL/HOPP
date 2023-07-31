import math
from typing import List, Optional
import numpy as np
import PySAM.Pvsamv1 as pv_detailed
import hybrid.layout.pv_module as pv_module
from hybrid.layout.pv_inverter import get_inverter_attribs

"""

This file contains all the utilities required to make intermediate calculations of the PV design and layout.

Functions that can be kept separate and self-contained should be here to enable re-use by other scripts and tests.
Making these functions standalone helps clarify the required inputs and function scope.
It also reduces the bulk of the PVPlant classes, making it easier to understand what aggregate logic it performs.

These may include any helper functions for calculating any system variable such as number of inverters, combiner boxes, etc
or for estimating some value given a PV layout

"""
def find_modules_per_string(
    v_mppt_min: float,
    v_mppt_max: float,
    v_mp_module: float,
    v_oc_module: float,
    inv_vdcmax: float,
    target_relative_string_voltage: float=None,
    ) -> float:
    """
    Calculates the number of modules per string to best match target string voltage

    :param v_mppt_min: lower boundary of inverter maximum-power-point operating window, V
    :param v_mppt_max: upper boundary of inverter maximum-power-point operating window, V
    :param v_mp_module: voltage of module at maximum point point at reference conditions, V
    :param v_oc_module: open circuit voltage of module at reference conditions, V
    :param inv_vdcmax: maximum inverter input DC voltage, V
    :param target_relative_string_voltage: relative string voltage within MPPT voltage window, [0, 1]

    :returns: number of modules per string
    """
    if v_mp_module <= 0:
        raise Exception("Module maximum power point voltage must be greater than 0.")
    if target_relative_string_voltage is None:
        target_relative_string_voltage = 0.5

    target_string_voltage = v_mppt_min + target_relative_string_voltage * (v_mppt_max - v_mppt_min)
    modules_per_string = max(1, round(target_string_voltage / v_mp_module))
    if inv_vdcmax > 0:
        while modules_per_string > 0 and modules_per_string * v_oc_module > inv_vdcmax:
            modules_per_string -= 1
    return modules_per_string


def find_inverter_count(
    dc_ac_ratio: float,
    modules_per_string: float,
    n_strings: float,
    module_power: float,
    inverter_power: float,
    ):
    """
    Sizes the number of inverters

    :param dc_ac_ratio: DC-to-AC ratio
    :param modules_per_string: modules per string
    :param n_strings: number of strings in array
    :param module_power: module power at maximum point point at reference conditions, kW
    :param inverter_power: inverter maximum AC power, kW

    :returns: number of inverters in array
    """
    n_inverters_frac = modules_per_string * n_strings * module_power / (dc_ac_ratio * inverter_power)
    n_inverters = max(1, round(n_inverters_frac))
    return n_inverters


def size_electrical_parameters(
    target_system_capacity: float,
    target_dc_ac_ratio: float,
    modules_per_string: float,
    module_power: float,
    inverter_power: float,
    n_inputs_inverter: Optional[float]=None,
    n_inputs_combiner: Optional[float]=None,
    ):
    """
    Calculates the number of strings, combiner boxes and inverters to best match target capacity and DC/AC ratio

    :param target_system_capacity: target system capacity, kW
    :param target_dc_ac_ratio: target DC-to-AC ratio
    :param modules_per_string: modules per string
    :param module_power: module power at maximum point point at reference conditions, kW
    :param inverter_power: inverter maximum AC power, kW
    :param n_inputs_inverter: number of DC inputs per inverter
    :param n_inputs_combiner: number of DC inputs per combiner box

    :returns: number of strings, number of combiner boxes, number of inverters, calculated system capacity, kW
    """
    n_strings_frac = target_system_capacity / (modules_per_string * module_power)
    n_strings = max(1, round(n_strings_frac))

    if target_dc_ac_ratio < 0:
        target_dc_ac_ratio = 1
    n_inverters = find_inverter_count(
        dc_ac_ratio=target_dc_ac_ratio,
        modules_per_string=modules_per_string,
        n_strings=n_strings,
        module_power=module_power,
        inverter_power=inverter_power,
        )

    if n_inputs_combiner is not None and n_inputs_inverter is not None:
        n_combiners = math.ceil(n_strings / n_inputs_combiner)
        # Ensure there are enough inverters for the number of combiner boxes
        n_inverters = max(n_inverters, math.ceil(n_combiners / n_inputs_inverter))
    else:
        n_combiners = None

    # Verify sizing was close to the target size, otherwise error out
    calculated_system_capacity = verify_capacity_from_electrical_parameters(
        system_capacity_target=target_system_capacity,
        n_strings=[n_strings],
        modules_per_string=[modules_per_string],
        module_power=module_power
    )

    return n_strings, n_combiners, n_inverters, calculated_system_capacity


def verify_capacity_from_electrical_parameters(
    system_capacity_target: float,
    n_strings: List[int],
    modules_per_string: List[int],
    module_power: float,
    percent_max_deviation: float = 5
    ) -> float:
    """
    Computes system capacity from specified number of strings, modules per string and module power.
    If computed capacity is significantly different than the specified capacity an exception will be thrown.
    
    :param system_capacity_target: target system capacity, kW
    :param n_strings: number of strings in each subarray, -
    :param modules_per_string: modules per string in each subarray, -
    :param module_power: module power at maximum point point at reference conditions, kW
    :param percent_max_deviation: if calculated system capacity differs from target by this percent or more, raise an exception; if None, do not check

    :returns: calculated system capacity, kW
    """
    PERCENT_MAX_DEVIATION = 5       # [%]
    assert len(n_strings) == len(modules_per_string)
    calculated_system_capacity = sum(np.array(n_strings) * np.array(modules_per_string)) * module_power
    if percent_max_deviation is not None and abs((calculated_system_capacity / system_capacity_target - 1)) * 100 > percent_max_deviation:
        raise Exception(f"The specified system capacity of {system_capacity_target} kW is more than " \
                        f"{percent_max_deviation}% from the value calculated from the specified number " \
                        f"of strings, modules per string and module power ({int(calculated_system_capacity)} kW).")

    return calculated_system_capacity


def align_from_capacity(
    system_capacity_target: float,
    dc_ac_ratio: float,
    modules_per_string: float,
    module_power: float,
    inverter_power: float,
    ) -> list:
    """
    Ensure coherence between parameters for detailed PV model (pvsamv1),
    keeping the DC-to-AC ratio approximately the same

    :param system_capacity_target: target system capacity, kW
    :param dc_ac_ratio: DC-to-AC ratio
    :param modules_per_string: modules per string, -
    :param module_power: module power at maximum point point at reference conditions, kW
    :param inverter_power: inverter maximum AC power, kW
    :param n_inverters_orig: original number of inverters

    :returns: number strings, calculated system capacity [kW], number of inverters
    """
    n_strings_frac = system_capacity_target / (modules_per_string * module_power)
    n_strings = max(1, round(n_strings_frac))
    system_capacity = module_power * n_strings * modules_per_string

    if dc_ac_ratio > 0:
        n_inverters_frac = modules_per_string * n_strings * module_power \
                           / (dc_ac_ratio * inverter_power)
    else:
        n_inverters_frac = modules_per_string * n_strings * module_power / inverter_power
    n_inverters = max(1, round(n_inverters_frac))

    return n_strings, system_capacity, n_inverters


def get_num_modules(pvsam_model: pv_detailed.Pvsamv1) -> float:
    """
    Return the number of modules in all subarrays
    """
    n_modules = 0
    for i in range(1, 4+1):
        if i == 1 or pvsam_model.value(f'subarray{i}_enable') == 1:
            n_modules += pvsam_model.value(f'subarray{i}_nstrings') \
                       * pvsam_model.value(f'subarray{i}_modules_per_string')
    return n_modules


def get_modules_per_string(system_model) -> float:
    if isinstance(system_model, pv_detailed.Pvsamv1):
        return system_model.value('subarray1_modules_per_string')
    else:
        return pv_module.modules_per_string


def get_inverter_power(pvsam_model: pv_detailed.Pvsamv1) -> float:
    inverter_attribs = get_inverter_attribs(pvsam_model)
    return inverter_attribs['P_ac']
