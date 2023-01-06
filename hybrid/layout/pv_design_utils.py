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
    v_mppt_min = 0.4
    v_mppt_max = 0.8
    return v_mppt_min, v_mppt_max
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
    v_module = 0
    modules_per_string = 0
    v_mppt_min, v_mppt_max = get_mppt_voltage_window(pvsam_model)
    assert target_relative_string_voltage == (v_mppt_max - v_mppt_min) * modules_per_string * v_module + v_mppt_min
    return modules_per_string


def find_target_dc_ac_ratio(pvsam_model: pv.Pvsamv1, target_dc_ac_ratio):
    """
    Find the number of strings per inverter to best match target dc_ac_ratio
    """
    computed_dc_ac_ratio, nstrings, ninverters = 1.3, 0, 0
    # calulate self.nstrings and self.ninverters from self.config.nb_inputs_inverter for desired dc_ac_ratio
    assert target_dc_ac_ratio == computed_dc_ac_ratio
    return nstrings, ninverters


def get_num_modules(pvsam_model: pv.Pvsamv1) -> float:
    """
    Return the number of modules in all subarrays
    """
    n_modules = 0
    for n in range(1, 4+1):
        if n == 1 or pvsam_model.value(f'subarray{n}_enable') == 1:
            n_modules += pvsam_model.value(f'subarray{n}_nstrings') \
                       * pvsam_model.value(f'subarray{n}_modules_per_string')
    return n_modules

def get_module_power(pvsam_model: pv.Pvsamv1) -> float:
    module_model = int(pvsam_model.value('module_model'))   # 0=spe, 1=cec, 2=sixpar_user, #3=snl, 4=sd11-iec61853, 5=PVYield
    if module_model == 0:
        return spe_power(pvsam_model.value('spe_eff4'), pvsam_model.value('spe_rad4'),
            pvsam_model.value('spe_area'))    # 4 = reference conditions
    elif module_model == 1:
        return pvsam_model.value('cec_i_mp_ref') * pvsam_model.value('cec_v_mp_ref')
    elif module_model == 2:
        return pvsam_model.value('sixpar_imp') * pvsam_model.value('sixpar_vmp')
    elif module_model == 3:
        return pvsam_model.value('snl_impo') * pvsam_model.value('snl_vmpo')
    elif module_model == 4:
        return pvsam_model.value('sd11par_Imp0') * pvsam_model.value('sd11par_Vmp0')
    elif module_model == 5:
        return pvsam_model.value('mlm_I_mp_ref') * pvsam_model.value('mlm_V_mp_ref')
    else:
        raise Exception("Module model invalid in module_power.")

def spe_power(spe_eff_level, spe_rad_level, spe_area) -> float:
    return spe_eff_level / 100 * spe_rad_level * spe_area
