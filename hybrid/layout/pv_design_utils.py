import math
import PySAM.Pvsamv1 as pv_detailed
import PySAM.Pvwattsv8 as pv_simple
import hybrid.layout.pv_module as pvwatts_defaults

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
    n_inputs_inverter: float=50,
    n_inputs_combiner: float=32,
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

    n_combiners = math.ceil(n_strings / n_inputs_combiner)

    # Ensure there are enough inverters for the number of combiner boxes
    n_inverters = max(n_inverters, math.ceil(n_combiners / n_inputs_inverter))

    # Verify sizing was close to the target size, otherwise error out
    calculated_system_capacity = verify_capacity_from_electrical_parameters(
        system_capacity_target=target_system_capacity,
        n_strings=n_strings,
        modules_per_string=modules_per_string,
        module_power=module_power
    )

    return n_strings, n_combiners, n_inverters, calculated_system_capacity


def verify_capacity_from_electrical_parameters(
    system_capacity_target: float,
    n_strings: float,
    modules_per_string: float,
    module_power: float,
    ) -> float:
    """
    Computes system capacity from specified number of strings, modules per string and module power.
    If computed capacity is significantly different than the specified capacity an exception will be thrown.
    
    :param system_capacity_target: target system capacity, kW
    :param n_strings: number of strings in array, -
    :param modules_per_string: modules per string, -
    :param module_power: module power at maximum point point at reference conditions, kW

    :returns: calculated system capacity, kW
    """
    PERCENT_MAX_DEVIATION = 5       # [%]
    calculated_system_capacity = n_strings * modules_per_string * module_power
    if abs((calculated_system_capacity / system_capacity_target - 1)) * 100 > PERCENT_MAX_DEVIATION:
        raise Exception(f"The specified system capacity of {system_capacity_target} kW is more than " \
                        f"{PERCENT_MAX_DEVIATION}% from the value calculated from the specified number " \
                        f"of strings, modules per string and module power ({int(calculated_system_capacity)} kW).")

    return calculated_system_capacity


def align_from_capacity(
    system_capacity_target: float,
    modules_per_string: float,
    module_power: float,
    inverter_power: float,
    n_inverters_orig: float,
    ) -> list:
    """
    Ensure coherence between parameters for detailed PV model (pvsamv1),
    keeping the DC-to-AC ratio approximately the same

    :param system_capacity_target: target system capacity, kW
    :param modules_per_string: modules per string, -
    :param module_power: module power at maximum point point at reference conditions, kW
    :param inverter_power: inverter maximum AC power, kW
    :param n_inverters_orig: original number of inverters

    :returns: number strings, calculated system capacity [kW], number of inverters
    """
    n_strings_frac = system_capacity_target / (modules_per_string * module_power)
    n_strings = max(1, round(n_strings_frac))
    system_capacity = module_power * n_strings * modules_per_string

    # Calculate inverter count, keeping the dc/ac ratio the same as before
    dc_ac_ratio_orig = system_capacity / (n_inverters_orig * inverter_power)
    if dc_ac_ratio_orig > 0:
        n_inverters_frac = modules_per_string * n_strings * module_power \
                           / (dc_ac_ratio_orig * inverter_power)
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
        return pvwatts_defaults.modules_per_string


def get_inverter_power(pvsam_model: pv_detailed.Pvsamv1) -> float:
    inverter_attribs = get_inverter_attribs(pvsam_model)
    return inverter_attribs['P_ac']


def spe_power(spe_eff_level, spe_rad_level, spe_area) -> float:
    """
    Computes the module power per the SPE model
    """
    return spe_eff_level / 100 * spe_rad_level * spe_area


def get_module_attribs(model) -> dict:
    """
    Returns the module attributes for either the PVsamv1 or PVWattsv8 models, see:
    https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#module-group

    :param model: PVsamv1 or PVWattsv8 model
    :return: dict, with keys:
        area            [m2]
        aspect_ratio    [-]
        length          [m]
        I_mp_ref        [A]
        I_sc_ref        [A]
        P_mp_ref        [kW]
        V_mp_ref        [V]
        V_oc_ref        [V]
        width           [m]
    """
    if isinstance(model, pv_simple.Pvwattsv8):
        P_mp = pvwatts_defaults.module_power
        I_mp = None
        I_sc = None
        V_oc = None
        V_mp = None
        length = pvwatts_defaults.module_height
        width = pvwatts_defaults.module_width
        area = length * width
        aspect_ratio = length / width
    elif isinstance(model, pv_detailed.Pvsamv1):
        # module_model: 0=spe, 1=cec, 2=sixpar_user, #3=snl, 4=sd11-iec61853, 5=PVYield
        module_model = int(model.value('module_model'))
        if module_model == 0:                   # spe
            SPE_FILL_FACTOR_ASSUMED = 0.79
            P_mp = spe_power(
                model.value('spe_eff4'),
                model.value('spe_rad4'),
                model.value('spe_area'))       # 4 = reference conditions
            I_mp = P_mp / model.value('spe_vmp')
            I_sc = model.value('spe_vmp') * model.value('spe_imp') \
                   / (model.value('spe_voc') * SPE_FILL_FACTOR_ASSUMED)
            V_oc = model.value('spe_voc')
            V_mp = model.value('spe_vmp')
            area = model.value('spe_area')
            aspect_ratio = model.value('module_aspect_ratio')
        elif module_model == 1:                 # cec
            I_mp = model.value('cec_i_mp_ref')
            I_sc = model.value('cec_i_sc_ref')
            V_oc = model.value('cec_v_oc_ref')
            V_mp = model.value('cec_v_mp_ref')
            area = model.value('cec_area')
            try:
                aspect_ratio = model.value('cec_module_length') \
                               / model.value('cec_module_width')
            except:
                aspect_ratio = model.value('module_aspect_ratio')
        elif module_model == 2:                 # sixpar_user
            I_mp = model.value('sixpar_imp')
            I_sc = model.value('sixpar_isc')
            V_oc = model.value('sixpar_voc')
            V_mp = model.value('sixpar_vmp')
            area = model.value('sixpar_area')
            aspect_ratio = model.value('module_aspect_ratio')
        elif module_model == 3:                 # snl
            I_mp = model.value('snl_impo')
            I_sc = model.value('snl_isco')
            V_oc = model.value('snl_voco')
            V_mp = model.value('snl_vmpo')
            area = model.value('snl_area')
            aspect_ratio = model.value('module_aspect_ratio')
        elif module_model == 4:                 # sd11-iec61853
            I_mp = model.value('sd11par_Imp0')
            I_sc = model.value('sd11par_Isc0')
            V_oc = model.value('sd11par_Voc0')
            V_mp = model.value('sd11par_Vmp0')
            area = model.value('sd11par_area')
            aspect_ratio = model.value('module_aspect_ratio')
        elif module_model == 5:                 # PVYield
            I_mp = model.value('mlm_I_mp_ref')
            I_sc = model.value('mlm_I_sc_ref')
            V_oc = model.value('mlm_V_oc_ref')
            V_mp = model.value('mlm_V_mp_ref')
            area = model.value('mlm_Length') * model.value('mlm_Width')
            aspect_ratio = model.value('mlm_Length') / model.value('mlm_Width')
        else:
            raise Exception("Module model number not recognized.")

        P_mp = I_mp * V_mp * 1e-3       # [kW]
        width = math.sqrt(area / aspect_ratio)
        length = math.sqrt(area * aspect_ratio)

    return {
        'area':         area,           # [m2]
        'aspect_ratio': aspect_ratio,   # [-]
        'length':       length,         # [m]
        'I_mp_ref':     I_mp,           # [A]
        'I_sc_ref':     I_sc,           # [A]
        'P_mp_ref':     P_mp,           # [kW]
        'V_mp_ref':     V_mp,           # [V]
        'V_oc_ref':     V_oc,           # [V]
        'width':        width           # [m]
    }


def get_inverter_attribs(pvsam_model: pv_detailed.Pvsamv1) -> dict:
    """
    Returns the inverter attributes for the PVsamv1 model, see:
    https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#inverter-group

    :param pvsam_model: PVsamv1 model
    :return: dict, with keys:
        V_mpp_nom           [V]
        V_dc_max            [V]
        P_ac                [kW]
        P_dc                [kW]
        P_ac_night_loss     [kW]
        n_mppt_inputs       [-]
        V_mppt_min          [V]
        V_mppt_max          [V]
    """
    inverter_model = int(pvsam_model.value('inverter_model'))           # 0=cec, 1=datasheet, 2=partload, 3=coefficientgenerator, 4=PVYield
    if inverter_model == 0:                   # cec
        V_mpp_nom = pvsam_model.value('inv_snl_vdco')
        V_dc_max = pvsam_model.value('inv_snl_vdcmax')
        P_ac = pvsam_model.value('inv_snl_paco')
        P_dc = pvsam_model.value('inv_snl_pdco')
        P_ac_night_loss = pvsam_model.value('inv_snl_pnt')
    elif inverter_model == 1:                 # datasheet
        V_mpp_nom = pvsam_model.value('inv_ds_vdco')
        V_dc_max = pvsam_model.value('inv_ds_vdcmax')
        P_ac = pvsam_model.value('inv_ds_paco')
        P_dc = pvsam_model.value('inv_ds_pdco')
        P_ac_night_loss = pvsam_model.value('inv_ds_pnt')
    elif inverter_model == 2:                 # partload
        V_mpp_nom = pvsam_model.value('inv_pd_vdco')
        V_dc_max = pvsam_model.value('inv_pd_vdcmax')
        P_ac = pvsam_model.value('inv_pd_paco')
        P_dc = pvsam_model.value('inv_pd_pdco')
        P_ac_night_loss = pvsam_model.value('inv_pd_pnt')
    elif inverter_model == 3:                 # coefficientgenerator
        V_mpp_nom = pvsam_model.value('inv_cec_cg_vdco')
        V_dc_max = pvsam_model.value('inv_cec_cg_vdcmax')
        P_ac = pvsam_model.value('inv_cec_cg_paco')
        P_dc = pvsam_model.value('inv_cec_cg_pdco')
        P_ac_night_loss = pvsam_model.value('inv_cec_cg_pnt')
    elif inverter_model == 4:                 # PVYield     TODO: these should be verified
        V_mpp_nom = pvsam_model.value('ond_VNomEff')
        V_dc_max = pvsam_model.value('ond_VAbsMax')
        P_ac = pvsam_model.value('ond_PMaxOUT')
        P_dc = pvsam_model.value('ond_PNomDC')
        P_ac_night_loss = pvsam_model.value('ond_Night_Loss')
    else:
        raise Exception("Inverter model number not recognized.")

    n_mppt_inputs = pvsam_model.value('inv_num_mppt')

    if inverter_model == 4:
        V_mppt_min = pvsam_model.InverterMermoudLejeuneModel.ond_VMppMin
        V_mppt_max = pvsam_model.InverterMermoudLejeuneModel.ond_VMPPMax
    else:
        V_mppt_min = pvsam_model.Inverter.mppt_low_inverter
        V_mppt_max = pvsam_model.Inverter.mppt_hi_inverter

    return {
        'V_mpp_nom':        V_mpp_nom,              # [V]
        'V_dc_max':         V_dc_max,               # [V]
        'P_ac':             P_ac * 1e-3,            # [kW]
        'P_dc':             P_dc * 1e-3,            # [kW]
        'P_ac_night_loss':  P_ac_night_loss * 1e-3, # [kW]
        'n_mppt_inputs':    n_mppt_inputs,          # [-]
        'V_mppt_min':       V_mppt_min,             # [V]
        'V_mppt_max':       V_mppt_max,             # [V]
    }
