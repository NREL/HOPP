import math
import PySAM.Pvsamv1 as pv

"""

This file contains all the utilities required to make intermediate calculations of the PV design and layout.

Functions that can be kept separate and self-contained should be here to enable re-use by other scripts and tests.
Making these functions standalone helps clarify the required inputs and function scope.
It also reduces the bulk of the DetailedPVPlant class, making it easier to understand what aggregate logic it performs.

These may include any helper functions for calculating any system variable such as number of inverters, combiner boxes, etc
or for estimating some value given a PV layout

"""
def find_modules_per_string(
    pvsam_model: pv.Pvsamv1,
    target_relative_string_voltage: float
    ):
    """
    Find the number of modules per string to best match target string voltage

    target_relative_string_voltage: relative string voltage within MPPT voltage window, [0, 1]
    """
    inverter_attribs = get_inverter_attribs(pvsam_model)
    v_mppt_min = inverter_attribs['V_mppt_min']
    v_mppt_max = inverter_attribs['V_mppt_max']
    target_string_voltage = v_mppt_min + target_relative_string_voltage * (v_mppt_max - v_mppt_min)

    module_attribs = get_module_attribs(pvsam_model)
    vmp_module = module_attribs['V_mp_ref']
    if vmp_module > 0:
        modules_per_string = max(1, round(target_string_voltage / vmp_module))

        inv_vdcmax = inverter_attribs['V_dc_max']
        if inv_vdcmax > 0:
            voc_module = module_attribs['V_oc_ref']
            while modules_per_string > 0 and modules_per_string * voc_module > inv_vdcmax:
                modules_per_string -= 1
    else:
        raise Exception("Module maximum power point voltage must be greater than 0.")
    return modules_per_string


def find_strings_per_inverter(
    pvsam_model: pv.Pvsamv1,
    target_solar_kw: float,
    target_dc_ac_ratio: float,
    modules_per_string: float,
    n_inputs_inverter: float
    ):
    """
    Find the number of strings per inverter to best match target dc_ac_ratio
    """
    P_module = get_module_power(pvsam_model)
    n_strings_frac = target_solar_kw / (modules_per_string * P_module * 1e-3)
    n_strings = max(1, round(n_strings_frac))

    inverter_attribs = get_inverter_attribs(pvsam_model)
    P_inverter = inverter_attribs['P_ac']
    if target_dc_ac_ratio > 0:
        n_inverters_frac = modules_per_string * n_strings * P_module / (target_dc_ac_ratio * P_inverter)
    else:
        n_inverters_frac = modules_per_string * n_strings * P_module / P_inverter
    n_inverters = max(1, round(n_inverters_frac))

    # Ensure there are enough enough inverters for the number of field connections
    # TODO: implement get_n_combiner_boxes() and/or string inverter calculations to compute n_field_connections
    n_field_connections = 1
    while math.ceil(n_field_connections / n_inverters) > n_inputs_inverter:
        n_inverters += 1

    # Verify sizing was close to the target size, otherwise error out
    total_modules = modules_per_string * n_strings
    nameplate_dc = total_modules * P_module * 1e-3
    if abs(nameplate_dc - target_solar_kw) / target_solar_kw > 0.2:
        n_strings = None
        n_inverters = None

    return n_strings, n_inverters


def get_num_modules(pvsam_model: pv.Pvsamv1) -> float:
    """
    Return the number of modules in all subarrays
    """
    n_modules = 0
    for i in range(1, 4+1):
        if i == 1 or pvsam_model.value(f'subarray{i}_enable') == 1:
            n_modules += pvsam_model.value(f'subarray{i}_nstrings') \
                       * pvsam_model.value(f'subarray{i}_modules_per_string')
    return n_modules


def get_module_power(pvsam_model: pv.Pvsamv1) -> float:
    module_attribs = get_module_attribs(pvsam_model)
    return module_attribs['P_mp_ref']   # [W]
    # module_model = int(pvsam_model.value('module_model'))   # 0=spe, 1=cec, 2=sixpar_user, #3=snl, 4=sd11-iec61853, 5=PVYield
    # if module_model == 0:
    #     return spe_power(pvsam_model.value('spe_eff4'), pvsam_model.value('spe_rad4'),
    #         pvsam_model.value('spe_area'))    # 4 = reference conditions
    # elif module_model == 1:
    #     return pvsam_model.value('cec_i_mp_ref') * pvsam_model.value('cec_v_mp_ref')
    # elif module_model == 2:
    #     return pvsam_model.value('sixpar_imp') * pvsam_model.value('sixpar_vmp')
    # elif module_model == 3:
    #     return pvsam_model.value('snl_impo') * pvsam_model.value('snl_vmpo')
    # elif module_model == 4:
    #     return pvsam_model.value('sd11par_Imp0') * pvsam_model.value('sd11par_Vmp0')
    # elif module_model == 5:
    #     return pvsam_model.value('mlm_I_mp_ref') * pvsam_model.value('mlm_V_mp_ref')
    # else:
    #     raise Exception("Module model invalid in module_power.")


def get_inverter_power(pvsam_model: pv.Pvsamv1) -> float:
    inverter_attribs = get_inverter_attribs(pvsam_model)
    return inverter_attribs['P_ac']     # [W]

def spe_power(spe_eff_level, spe_rad_level, spe_area) -> float:
    return spe_eff_level / 100 * spe_rad_level * spe_area


def get_module_attribs(pvsam_model: pv.Pvsamv1) -> dict:
        module_model = int(pvsam_model.value('module_model'))           # 0=spe, 1=cec, 2=sixpar_user, #3=snl, 4=sd11-iec61853, 5=PVYield
        if module_model == 0:                   # spe
            SPE_FILL_FACTOR_ASSUMED = 0.79
            P_mp = spe_power(pvsam_model.value('spe_eff4'), pvsam_model.value('spe_rad4'), pvsam_model.value('spe_area'))       # 4 = reference conditions
            I_mp = P_mp / pvsam_model.value('spe_vmp')
            I_sc = pvsam_model.value('spe_vmp') * pvsam_model.value('spe_imp') / (pvsam_model.value('spe_voc') * SPE_FILL_FACTOR_ASSUMED)
            V_oc = pvsam_model.value('spe_voc')
            V_mp = pvsam_model.value('spe_vmp')
            area = pvsam_model.value('spe_area')
            aspect_ratio = pvsam_model.value('module_aspect_ratio')
        elif module_model == 1:                 # cec
            I_mp = pvsam_model.value('cec_i_mp_ref')
            I_sc = pvsam_model.value('cec_i_sc_ref')
            V_oc = pvsam_model.value('cec_v_oc_ref')
            V_mp = pvsam_model.value('cec_v_mp_ref')
            area = pvsam_model.value('cec_area')
            try:
                aspect_ratio = pvsam_model.value('cec_module_length') / pvsam_model.value('cec_module_width')
            except:
                aspect_ratio = pvsam_model.value('module_aspect_ratio')
        elif module_model == 2:                 # sixpar_user
            I_mp = pvsam_model.value('sixpar_imp')
            I_sc = pvsam_model.value('sixpar_isc')
            V_oc = pvsam_model.value('sixpar_voc')
            V_mp = pvsam_model.value('sixpar_vmp')
            area = pvsam_model.value('sixpar_area')
            aspect_ratio = pvsam_model.value('module_aspect_ratio')
        elif module_model == 3:                 # snl
            I_mp = pvsam_model.value('snl_impo')
            I_sc = pvsam_model.value('snl_isco')
            V_oc = pvsam_model.value('snl_voco')
            V_mp = pvsam_model.value('snl_vmpo')
            area = pvsam_model.value('snl_area')
            aspect_ratio = pvsam_model.value('module_aspect_ratio')
        elif module_model == 4:                 # sd11-iec61853
            I_mp = pvsam_model.value('sd11par_Imp0')
            I_sc = pvsam_model.value('sd11par_Isc0')
            V_oc = pvsam_model.value('sd11par_Voc0')
            V_mp = pvsam_model.value('sd11par_Vmp0')
            area = pvsam_model.value('sd11par_area')
            aspect_ratio = pvsam_model.value('module_aspect_ratio')
        elif module_model == 5:                 # PVYield
            I_mp = pvsam_model.value('mlm_I_mp_ref')
            I_sc = pvsam_model.value('mlm_I_sc_ref')
            V_oc = pvsam_model.value('mlm_V_oc_ref')
            V_mp = pvsam_model.value('mlm_V_mp_ref')
            area = pvsam_model.value('mlm_Length') * pvsam_model.value('mlm_Width')
            aspect_ratio = pvsam_model.value('mlm_Length') / pvsam_model.value('mlm_Width')
        else:
            raise Exception("Module model number not recognized.")

        P_mp = I_mp * V_mp
        module_width = math.sqrt(area / aspect_ratio)
        module_length = math.sqrt(area * aspect_ratio)

        return {
            'area':         area,
            'aspect_ratio': aspect_ratio,
            'length':       module_length,
            'I_mp_ref':     I_mp,
            'I_sc_ref':     I_sc,
            'P_mp_ref':     P_mp,
            'V_mp_ref':     V_mp,
            'V_oc_ref':     V_oc,
            'width':        module_width
        }


def get_inverter_attribs(pvsam_model: pv.Pvsamv1) -> dict:
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
        'P_ac':             P_ac,                   # [W]
        'P_dc':             P_dc,                   # [W]
        'P_ac_night_loss':  P_ac_night_loss,        # [W]
        'n_mppt_inputs':    n_mppt_inputs,          # [-]
        'V_mppt_min':       V_mppt_min,             # [V]
        'V_mppt_max':       V_mppt_max,             # [V]
    }
