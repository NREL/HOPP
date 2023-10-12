from typing import Union
import PySAM.Pvsamv1 as pv_detailed
import PySAM.Pvwattsv8 as pv_simple

from tools.utils import flatten_dict

def get_inverter_attribs(model: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1, dict], only_ref_values=True) -> dict:
    """
    Returns the inverter attributes for the PVwattsv8 or PVsamv1 model, see:
    https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html#systemdesign-group
    https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#inverter-group

    :param model: PVsamv1 or PVWattsv8 model or parameter dictionary
    :param only_ref_vals: if True, only return the reference values (e.g., P_ac)
    :return: dict, with keys (if only_ref_values is True, otherwise will include all model-specific parameters):
        V_mpp_nom           [V]
        V_dc_max            [V]
        P_ac                [kW]
        P_dc                [kW]
        P_ac_night_loss     [kW]
        n_mppt_inputs       [-]
        V_mppt_min          [V]
        V_mppt_max          [V]
    """
    MODEL_PREFIX = ['inv_snl', 'inv_ds', 'inv_pd', 'inv_cec', 'ond']

    if not isinstance(model, dict):
        model = flatten_dict(model.export())

    params = {}
    if 'inverter_model' not in model:    # Pvwattsv8
        params['V_mpp_nom'] = None
        params['V_dc_max'] = None
        params['P_ac'] = model['system_capacity'] / model['dc_ac_ratio']    # [kW]
        params['P_dc'] = params['P_ac'] / model['inv_eff'] * 1e-3           # [kW]
        params['P_ac_night_loss'] = None
        params['n_mppt_inputs'] = None
        params['V_mppt_min'] = None
        params['V_mppt_max'] = None
    else:                               # Pvsamv1
        inverter_model = int(model['inverter_model'])   # 0=cec, 1=datasheet, 2=partload, 3=coefficientgenerator, 4=PVYield
        if not only_ref_values:
            params['inverter_model'] = model['inverter_model']
            params['mppt_low_inverter'] = model['mppt_low_inverter']
            params['mppt_hi_inverter'] = model['mppt_hi_inverter']
            params['inv_num_mppt'] = model['inv_num_mppt']
            if inverter_model < 4:
                temp_derate_curve = ['inv_tdc_cec_db', 'inv_tdc_ds', 'inv_tdc_plc', 'inv_tdc_cec_cg'][inverter_model]
                params[temp_derate_curve] = model[temp_derate_curve]
            for key in model.keys():
                if key.startswith(MODEL_PREFIX[inverter_model] + '_'):
                    params[key] = model[key]
        elif inverter_model == 0:                 # cec (snl)
            param_map = {
                'V_mpp_nom':        'inv_snl_vdco',
                'V_dc_max':         'inv_snl_vdcmax',
                'P_ac':             'inv_snl_paco',
                'P_dc':             'inv_snl_pdco',
                'P_ac_night_loss':  'inv_snl_pnt',
            }
        elif inverter_model == 1:                 # datasheet
            param_map = {
                'V_mpp_nom':        'inv_ds_vdco',
                'V_dc_max':         'inv_ds_vdcmax',
                'P_ac':             'inv_ds_paco',
                'P_dc':             'inv_ds_pdco',
                'P_ac_night_loss':  'inv_ds_pnt',
            }
        elif inverter_model == 2:                 # partload
            param_map = {
                'V_mpp_nom':        'inv_pd_vdco',
                'V_dc_max':         'inv_pd_vdcmax',
                'P_ac':             'inv_pd_paco',
                'P_dc':             'inv_pd_pdco',
                'P_ac_night_loss':  'inv_pd_pnt',
            }
        elif inverter_model == 3:                 # coefficientgenerator (cec)
            param_map = {
                'V_mpp_nom':        'inv_cec_cg_vdco',
                'V_dc_max':         'inv_cec_cg_vdcmax',
                'P_ac':             'inv_cec_cg_paco',
                'P_dc':             'inv_cec_cg_pdco',
                'P_ac_night_loss':  'inv_cec_cg_pnt',
            }
        elif inverter_model == 4:                 # PVYield     TODO: these should be verified
            param_map = {
                'V_mpp_nom':        'ond_VNomEff',
                'V_dc_max':         'ond_VAbsMax',
                'P_ac':             'ond_PMaxOUT',
                'P_dc':             'ond_PNomDC',
                'P_ac_night_loss':  'ond_Night_Loss',
            }
        else:
            raise Exception("Inverter model number not recognized.")
        
        if only_ref_values:
            for key, value in param_map.items():
                params[key] = model[value]

            params['P_ac'] = params['P_ac'] * 1e-3                          # [kW]
            params['P_dc'] = params['P_dc'] * 1e-3                          # [kW]
            params['P_ac_night_loss'] = params['P_ac_night_loss'] * 1e-3    # [kW]

            if inverter_model == 4:
                params['V_mppt_min'] = model['ond_VMppMin']
                params['V_mppt_max'] = model['ond_VMPPMax']
            else:
                params['V_mppt_min'] = model['mppt_low_inverter']
                params['V_mppt_max'] = model['mppt_hi_inverter']

    return params

def set_inverter_attribs(model: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1], params: dict):
    """
    Sets the inverter model parameters for either the PVsamv1 or PVWattsv8 models.
    Will raise exception if not all required parameters are provided.
    
    :param model: PVWattsv8 or PVsamv1 model
    :param params: dictionary of parameters
    """
    if isinstance(model, pv_simple.Pvwattsv8):
        inverter_model = 'PVWatts'
        req_vals = ['inv_eff']
    elif isinstance(model, pv_detailed.Pvsamv1):
        if 'inverter_model' not in params.keys():
            params['inverter_model'] = model.value('inverter_model')
        req_vals = ['inverter_model']

        inverter_model = params['inverter_model']
        if inverter_model == 0:                   # cec (snl)
            req_vals += [
                'inv_snl_c0', 'inv_snl_c1', 'inv_snl_c2', 'inv_snl_c3',
                'inv_snl_paco',
                'inv_snl_pdco',
                'inv_snl_pnt',
                'inv_snl_pso',
                'inv_snl_vdco',
                'inv_snl_vdcmax',
                'inv_tdc_cec_db',
            ]
        elif inverter_model == 1:                 # datasheet
            req_vals += [
                'inv_ds_paco',
                'inv_ds_eff',
                'inv_ds_pnt',
                'inv_ds_pso',
                'inv_ds_vdco',
                'inv_ds_vdcmax',
                'inv_tdc_ds',
            ]
        elif inverter_model == 2:                 # partload
            req_vals += [
                'inv_pd_paco',
                'inv_pd_pdco',
                'inv_pd_partload',
                'inv_pd_efficiency',
                'inv_pd_pnt',
                'inv_pd_vdco',
                'inv_pd_vdcmax',
                'inv_tdc_plc',
            ]
        elif inverter_model == 3:                 # coefficientgenerator (cec)
            req_vals += [
                'inv_cec_cg_c0', 'inv_cec_cg_c1', 'inv_cec_cg_c2', 'inv_cec_cg_c3',
                'inv_cec_cg_paco',
                'inv_cec_cg_pdco',
                'inv_cec_cg_pnt',
                'inv_cec_cg_psco',
                'inv_cec_cg_vdco',
                'inv_cec_cg_vdcmax',
                'inv_tdc_cec_cg',
            ]
        elif inverter_model == 4:                 # PVYield
            req_vals += [
                'ond_PNomConv',
                'ond_PMaxOUT',
                'ond_VOutConv',
                'ond_VMppMin',
                'ond_VMPPMax',
                'ond_VAbsMax',
                'ond_PSeuil',
                'ond_ModeOper',
                'ond_CompPMax',
                'ond_CompVMax',
                'ond_ModeAffEnum',
                'ond_PNomDC',
                'ond_PMaxDC',
                'ond_IMaxDC',
                'ond_INomDC',
                'ond_INomAC',
                'ond_IMaxAC',
                'ond_TPNom',
                'ond_TPMax',
                'ond_TPLim1',
                'ond_TPLimAbs',
                'ond_PLim1',
                'ond_PLimAbs',
                'ond_VNomEff',
                'ond_NbInputs',
                'ond_NbMPPT',
                'ond_Aux_Loss',
                'ond_Night_Loss',
                'ond_lossRDc',
                'ond_lossRAc',
                'ond_effCurve_elements',
                'ond_effCurve_Pdc',
                'ond_effCurve_Pac',
                'ond_effCurve_eta',
                'ond_Aux_Loss',
                'ond_Aux_Loss',
                'ond_doAllowOverpower',
                'ond_doUseTemperatureLimit',
            ]
        else:
            raise Exception("Inverter model number not recognized.")

        if 'inv_num_mppt' in params.keys():
            req_vals.append('inv_num_mppt')
        if inverter_model == 4 and 'ond_VMppMin' in params.keys():
            req_vals.append('ond_VMppMin')
        if inverter_model == 4 and 'ond_VMPPMax' in params.keys():
            req_vals.append('ond_VMPPMax')
        if inverter_model != 4 and 'mppt_low_inverter' in params.keys():
            req_vals.append('mppt_low_inverter')
        if inverter_model != 4 and 'mppt_hi_inverter' in params.keys():
            req_vals.append('mppt_hi_inverter')

    if not set(req_vals).issubset(params.keys()):
        raise Exception("Not all parameters specified for inverter model {}.".format(inverter_model))

    for value in req_vals:
        model.value(value, params[value])
