import PySAM.Pvsamv1 as pv

"""

This file contains all the utilities required to make intermediate calculations of the PV design and layout.

Functions that can be kept separate and self-contained should be here to enable re-use by other scripts and tests.
Making these functions standalone helps clarify the required inputs and function scope.
It also reduces the bulk of the DetailedPVPlant class, making it easier to understand what aggregate logic it performs.

These may include any helper functions for calculating any system variable such as number of inverters, combiner boxes, etc
or for estimating some value given a PV layout

"""

def get_mppt_voltage_window(pvsam_model: pv.Pvsamv1):
    """
    Gets the Max Point-point tracking voltage window for the modeled inverter
    """
    v_mppt_min = 0
    v_mppt_max = 0
    if pvsam_model.Inverter.inverter_model == 0:
        v_mppt_min = pvsam_model.Inverter.mppt_low_inverter
        v_mppt_max = pvsam_model.Inverter.mppt_hi_inverter
    elif pvsam_model.Inverter.inverter_model == 4:
        v_mppt_min = pvsam_model.InverterMermoudLejeuneModel.ond_VMppMin
        v_mppt_max = pvsam_model.InverterMermoudLejeuneModel.ond_VMPPMax
    else:
        # TODO: fill out the rest
        raise NotImplementedError
    return v_mppt_min, v_mppt_max


def find_target_string_voltage(pvsam_model: pv.Pvsamv1, target_relative_string_voltage):
    """
    Find the number of modules per string to best match target string voltage
    """
    v_module = None
    modules_per_string = None
    v_mppt_min, v_mppt_max = get_mppt_voltage_window(pvsam_model)
    assert target_relative_string_voltage == (v_mppt_max - v_mppt_min) * modules_per_string * v_module + v_mppt_min
    return modules_per_string


def find_target_dc_ac_ratio(pvsam_model: pv.Pvsamv1, target_dc_ac_ratio):
    """
    Find the number of strings per inverter to best match target dc_ac_ratio
    """
    computed_dc_ac_ratio, nstrings, ninverters = 0, 0, 0
    # calulate self.nstrings and self.ninverters from self.config.nb_inputs_inverter for desired dc_ac_ratio
    assert target_dc_ac_ratio == computed_dc_ac_ratio
    return nstrings, ninverters


def get_num_modules(pvsam_model: pv.Pvsamv1) -> float:
    """
    TODO: return the number of modules in all subarrays
    """
    return 0