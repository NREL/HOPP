import PySAM.Pvsamv1 as pv
#this file shows how to size the module and inverter according to a desired system_kw
#TODO: adapt the function to update the default dictionary rather than the PySAM model itself
# defaults["Solar"]["Pvsamv1"]["SystemDesign"]["inverter_count"] goes into model.SystemDesign.inverter_count

model = pv.default("FlatPlatePVSingleOwner")

#
# Size system for system capacity by using the Inverter partload curve model using single subarray
#
# desired_dc_ac_ratio = 1.2


def size_pv_array(model, desired_kw, desired_dc_ac_ratio):

    for i in range(2, 5):
        model.SystemDesign.__setattr__("subarray"+str(i)+"_enable", False)

    module_vmp = model.CECPerformanceModelWithModuleDatabase.cec_v_mp_ref
    module_voc = model.CECPerformanceModelWithModuleDatabase.cec_v_oc_ref
    module_rated_power = model.CECPerformanceModelWithModuleDatabase.cec_v_mp_ref * model.CECPerformanceModelWithModuleDatabase.cec_i_mp_ref

    n_modules = int(desired_kw * 1000. / module_rated_power)

    system_capacity_kw = module_rated_power * n_modules / 1000.

    string_vmp = n_modules * module_vmp
    string_voc = n_modules * module_voc

    model.Module.module_model = 1
    model.SystemDesign.subarray1_modules_per_string = n_modules
    model.SystemDesign.subarray1_nstrings = 1
    model.SystemDesign.system_capacity = system_capacity_kw

    inverter_capacity_kw = int(system_capacity_kw / desired_dc_ac_ratio)

    model.Inverter.inverter_model = 2
    model.Inverter.inverter_count = 1
    inverter_ac = inverter_capacity_kw * 1000
    model.InverterPartLoadCurve.inv_pd_paco = inverter_ac
    model.InverterPartLoadCurve.inv_pd_pdco = inverter_ac / 0.95551 # efficiency calculated from efficiency curve
    model.InverterPartLoadCurve.inv_pd_vdco = string_vmp
    model.InverterPartLoadCurve.inv_pd_vdcmax = string_voc
    model.Inverter.mppt_low_inverter = 0
    model.Inverter.mppt_hi_inverter = string_voc

    model_required_inputs = dict()
    model_required_inputs['n_modules'] = n_modules
    model_required_inputs['inverter_ac'] = inverter_ac
    model_required_inputs['string_vmp'] = string_vmp
    model_required_inputs['string_voc'] = string_voc
    model_required_inputs['desired_system_capacity_kw'] = desired_kw
    model_required_inputs['system_capacity_kw'] = system_capacity_kw

    #model_required_inputs['Module']['module_model'] = model.Module.module_model


    #print(model_required_inputs)
    return model

#Sample run
# model.SolarResource.solar_resource_file = "Solar_Resource_File_Sample.csv"
#
# gen_vs_nameplate = []
# for desired_kw in range(5000, 15000, 1000):
#     model_required_inputs = size_pv_array(model, desired_kw, desired_dc_ac_ratio)
#     model.execute(0)
#     gen_vs_nameplate.append(model.Outputs.annual_ac_gross / desired_kw)
#     print(model_required_inputs)
# print(gen_vs_nameplate)

