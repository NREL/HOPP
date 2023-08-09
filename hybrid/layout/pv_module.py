import math
import numpy as np
from typing import Union
import PySAM.Pvsamv1 as pv_detailed
import PySAM.Pvwattsv8 as pv_simple

from tools.utils import flatten_dict

# PVWatts default module
# pvmismatch standard module description
cell_len = 0.124
cell_rows = 12
cell_cols = 8
cell_num_map = [[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24],
                [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                [83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72],
                [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]]
cell_num_map_flat = np.array(cell_num_map).flatten()
module_width = cell_len * cell_rows
module_height = cell_len * cell_cols
modules_per_string = 10
module_power = .321     # kW


def spe_power(spe_eff_level, spe_rad_level, spe_area) -> float:
    """
    Computes the module power per the SPE model
    """
    return spe_eff_level / 100 * spe_rad_level * spe_area


def get_module_attribs(model: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1, dict], only_ref_vals=True) -> dict:
    """
    Returns the module attributes for either the PVsamv1 or PVWattsv8 models, see:
    https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#module-group

    :param model: PVsamv1 or PVWattsv8 model or parameter dictionary
    :param only_ref_vals: if True, only return the reference values (e.g., I_sc_ref)
    :return: dict, with keys (if only_ref_values is True, otherwise will include all model-specific parameters):
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
    MODEL_PREFIX = ['spe', 'cec', '6par', 'snl', 'sd11par', 'mlm']

    if not isinstance(model, dict):
        model = flatten_dict(model.export())

    params = {}
    if 'module_model' not in model:    # Pvwattsv8
        params['P_mp_ref'] = module_power
        params['I_mp_ref'] = None
        params['I_sc_ref'] = None
        params['V_oc_ref'] = None
        params['V_mp_ref'] = None
        params['length'] = module_height
        params['width'] = module_width
        params['area'] = params['length'] * params['width']
        params['aspect_ratio'] = params['length'] / params['width']
    else:                               # Pvsamv1
        module_model = int(model['module_model'])   # 0=spe, 1=cec, 2=sixpar_user, #3=snl, 4=sd11-iec61853, 5=PVYield
        if not only_ref_vals:
            params['module_model'] = model['module_model']
            params['module_aspect_ratio'] = model['module_aspect_ratio']
            for key in model.keys():
                if key.startswith(MODEL_PREFIX[module_model] + '_'):
                    params[key] = model[key]
        elif module_model == 0:                 # spe
            param_map = {
                'V_oc_ref':     'spe_voc',
                'V_mp_ref':     'spe_vmp',
                'area':         'spe_area',
                'aspect_ratio': 'module_aspect_ratio',
            }
            SPE_FILL_FACTOR_ASSUMED = 0.79
            params['P_mp_ref'] = spe_power(
                model['spe_eff4'],
                model['spe_rad4'],
                model['spe_area']
            )       # 4 = reference conditions
            params['I_mp_ref'] = params['P_mp_ref'] / model['spe_vmp']
            params['I_sc_ref'] = model['spe_vmp'] * params['I_mp_ref'] \
                                 / (model['spe_voc'] * SPE_FILL_FACTOR_ASSUMED)
        elif module_model == 1:                 # cec
            param_map = {
                'I_mp_ref':     'cec_i_mp_ref',
                'I_sc_ref':     'cec_i_sc_ref',
                'V_oc_ref':     'cec_v_oc_ref',
                'V_mp_ref':     'cec_v_mp_ref',
                'area':         'cec_area',
                'aspect_ratio': 'module_aspect_ratio',
            }
        elif module_model == 2:                 # sixpar_user
            param_map = {
                'I_mp_ref':     'sixpar_imp',
                'I_sc_ref':     'sixpar_isc',
                'V_oc_ref':     'sixpar_voc',
                'V_mp_ref':     'sixpar_vmp',
                'area':         'sixpar_area',
                'aspect_ratio': 'module_aspect_ratio',
            }
        elif module_model == 3:                 # snl
            param_map = {
                'I_mp_ref':     'snl_impo',
                'I_sc_ref':     'snl_isco',
                'V_oc_ref':     'snl_voco',
                'V_mp_ref':     'snl_vmpo',
                'area':         'snl_area',
                'aspect_ratio': 'module_aspect_ratio',
            }
        elif module_model == 4:                 # sd11-iec61853
            param_map = {
                'I_mp_ref':     'sd11par_Imp0',
                'I_sc_ref':     'sd11par_Isc0',
                'V_oc_ref':     'sd11par_Voc0',
                'V_mp_ref':     'sd11par_Vmp0',
                'area':         'sd11par_area',
                'aspect_ratio': 'module_aspect_ratio',
            }
        elif module_model == 5:                 # PVYield
            param_map = {
                'I_mp_ref':     'mlm_I_mp_ref',
                'I_sc_ref':     'mlm_I_sc_ref',
                'V_oc_ref':     'mlm_V_oc_ref',
                'V_mp_ref':     'mlm_V_mp_ref',
            }
            params['area'] = model['mlm_Length'] * model['mlm_Width']
            params['aspect_ratio'] = model['mlm_Length'] / model['mlm_Width']
        else:
            raise Exception("Module model number not recognized.")

        if only_ref_vals:
            for key, value in param_map.items():
                params[key] = model[value]

            params['P_mp_ref'] = params['I_mp_ref'] * params['V_mp_ref'] * 1e-3       # [kW]
            params['width'] = math.sqrt(params['area'] / params['aspect_ratio'])
            params['length'] = math.sqrt(params['area'] * params['aspect_ratio'])

    return params


def set_module_attribs(model: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1], params: dict):
    """
    Sets the module model parameters for either the PVsamv1 or PVWattsv8 models.
    Will raise exception if not all required parameters are provided.
    
    :param model: PVWattsv8 or PVsamv1 model
    :param params: dictionary of parameters
    """

    if isinstance(model, pv_simple.Pvwattsv8):
        module_model = 'PVWatts'
        req_vals = ['module_type']
    elif isinstance(model, pv_detailed.Pvsamv1):
        if 'module_model' not in params.keys():
            params['module_model'] = model.value('module_model')
        req_vals = ['module_model']

        module_model = params['module_model']
        if module_model == 0:                   # spe
            req_vals += [
                'spe_area',
                'spe_rad0', 'spe_rad1', 'spe_rad2', 'spe_rad3', 'spe_rad4',
                'spe_eff0', 'spe_eff1', 'spe_eff2', 'spe_eff3', 'spe_eff4',
                'spe_reference',
                'spe_module_structure',
                'spe_a', 'spe_b',
                'spe_dT',
                'spe_temp_coeff',
                'spe_fd',
                'spe_vmp',
                'spe_voc',
                'spe_is_bifacial',
                'spe_bifacial_transmission_factor',
                'spe_bifaciality',
                'spe_bifacial_ground_clearance_height',
            ]
        elif module_model == 1:                 # cec
            req_vals += [
                'cec_area',
                'cec_a_ref',
                'cec_adjust',
                'cec_alpha_sc',
                'cec_beta_oc',
                'cec_gamma_r',
                'cec_i_l_ref',
                'cec_i_mp_ref',
                'cec_i_o_ref',
                'cec_i_sc_ref',
                'cec_n_s',
                'cec_r_s',
                'cec_r_sh_ref',
                'cec_t_noct',
                'cec_v_mp_ref',
                'cec_v_oc_ref',
                'cec_temp_corr_mode',
                'cec_is_bifacial',
                'cec_bifacial_transmission_factor',
                'cec_bifaciality',
                'cec_bifacial_ground_clearance_height',
                'cec_standoff',
                'cec_height',
                'cec_transient_thermal_model_unit_mass',
            ]
            if 'cec_temp_corr_mode' in params.keys() and params['cec_temp_corr_mode'] == 1:
                req_vals += [
                    'cec_mounting_config',
                    'cec_heat_transfer',
                    'cec_mounting_orientation',
                    'cec_gap_spacing',
                    'cec_module_width',
                    'cec_module_length',
                    'cec_array_rows',
                    'cec_array_cols',
                    'cec_backside_temp',
                ]
            if 'cec_lacunarity_enable' in params.keys() and params['cec_lacunarity_enable'] == 1:
                req_vals += ['cec_lacunarity_enable']
                if 'cec_temp_corr_mode' in params.keys() and params['cec_temp_corr_mode'] == 1:
                    req_vals += [
                        'cec_lacunarity_length',
                        'cec_ground_clearance_height',
                    ]
        elif module_model == 2:                 # sixpar_user
            req_vals += [
                '6par_celltech',
                '6par_vmp',
                '6par_imp',
                '6par_voc',
                '6par_isc',
                '6par_bvoc',
                '6par_aisc',
                '6par_gpmp',
                '6par_nser',
                '6par_area',
                '6par_tnoct',
                '6par_standoff',
                '6par_mounting',
                '6par_is_bifacial',
                '6par_bifacial_transmission_factor',
                '6par_bifaciality',
                '6par_bifacial_ground_clearance_height',
                '6par_transient_thermal_model_unit_mass',
            ]
        elif module_model == 3:                 # snl
            req_vals += [
                'snl_module_structure',
                'snl_a',
                'snl_b',
                'snl_dtc',
                'snl_ref_a',
                'snl_ref_b',
                'snl_ref_dT',
                'snl_fd',
                'snl_a0', 'snl_a1', 'snl_a2', 'snl_a3', 'snl_a4',
                'snl_aimp',
                'snl_aisc',
                'snl_area',
                'snl_b0', 'snl_b1', 'snl_b2', 'snl_b3', 'snl_b4', 'snl_b5',
                'snl_bvmpo',
                'snl_bvoco',
                'snl_c0', 'snl_c1', 'snl_c2', 'snl_c3', 'snl_c4', 'snl_c5', 'snl_c6', 'snl_c7',
                'snl_impo',
                'snl_isco',
                'snl_ixo',
                'snl_ixxo',
                'snl_mbvmp',
                'snl_mbvoc',
                'snl_n',
                'snl_series_cells',
                'snl_vmpo',
                'snl_voco',
                'snl_transient_thermal_model_unit_mass',
            ]
        elif module_model == 4:                 # sd11-iec61853
            req_vals += [
                'sd11par_nser',
                'sd11par_area',
                'sd11par_AMa0', 'sd11par_AMa1', 'sd11par_AMa2', 'sd11par_AMa3', 'sd11par_AMa4',
                'sd11par_glass',
                'sd11par_tnoct',
                'sd11par_standoff',
                'sd11par_mounting',
                'sd11par_Vmp0',
                'sd11par_Imp0',
                'sd11par_Voc0',
                'sd11par_Isc0',
                'sd11par_alphaIsc',
                'sd11par_n',
                'sd11par_Il',
                'sd11par_Io',
                'sd11par_Egref',
                'sd11par_d1', 'sd11par_d2', 'sd11par_d3',
                'sd11par_c1', 'sd11par_c2', 'sd11par_c3',
            ]
        elif module_model == 5:                 # PVYield
            req_vals += [
                'mlm_N_series',
                'mlm_N_parallel',
                'mlm_N_diodes',
                'mlm_Width',
                'mlm_Length',
                'mlm_V_mp_ref',
                'mlm_I_mp_ref',
                'mlm_V_oc_ref',
                'mlm_I_sc_ref',
                'mlm_S_ref',
                'mlm_T_ref',
                'mlm_R_shref',
                'mlm_R_sh0',
                'mlm_R_shexp',
                'mlm_R_s',
                'mlm_alpha_isc',
                'mlm_beta_voc_spec',
                'mlm_E_g',
                'mlm_n_0',
                'mlm_mu_n',
                'mlm_D2MuTau',
                'mlm_T_mode',
                'mlm_T_c_no_tnoct',
                'mlm_T_c_no_mounting',
                'mlm_T_c_no_standoff',
                'mlm_T_c_fa_alpha',
                'mlm_T_c_fa_U0', 'mlm_T_c_fa_U1',
                'mlm_AM_mode',
                'mlm_AM_c_sa0', 'mlm_AM_c_sa1', 'mlm_AM_c_sa2', 'mlm_AM_c_sa3', 'mlm_AM_c_sa4',
                'mlm_AM_c_lp0', 'mlm_AM_c_lp1', 'mlm_AM_c_lp2', 'mlm_AM_c_lp3', 'mlm_AM_c_lp4', 'mlm_AM_c_lp5',
                'mlm_IAM_mode',
                'mlm_IAM_c_as',
                'mlm_IAM_c_sa0', 'mlm_IAM_c_sa1', 'mlm_IAM_c_sa2', 'mlm_IAM_c_sa3', 'mlm_IAM_c_sa4', 'mlm_IAM_c_sa5',
                'mlm_IAM_c_cs_incAngle',
                'mlm_IAM_c_cs_iamValue',
                'mlm_groundRelfectionFraction',
                'mlm_is_bifacial',
                'mlm_bifacial_transmission_factor',
                'mlm_bifaciality',
                'mlm_bifacial_ground_clearance_height',
            ]
        else:
            raise Exception("Module model number not recognized.")

        if 'module_aspect_ratio' in params.keys():
            req_vals.append('module_aspect_ratio')

    if not set(req_vals).issubset(params.keys()):
        raise Exception("Not all parameters specified for module model {}.".format(module_model))

    for value in req_vals:
        model.value(value, params[value])
