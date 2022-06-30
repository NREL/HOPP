"""
Pipeline Model
Converted file from NREL/NRWAL repository:
https://github.com/NREL/NRWAL/blob/main/NRWAL/analysis_library/hydrogen/pipeline.yaml
"""
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals

def pipeline_model(dist_to_h2_load_km, h2_flow_kg_h, capex_pipeline_linear: bool):
    """
    parm: dist_to_h2_load_km [km]
    parm: h2_flow_kg_h [kg/hr]
    parm: capex_pipeline_linear: bool
        if True capex is linear relationship
        if False capex is nonlinear relationship
    """
    dist_to_h2_load_miles = dist_to_h2_load_km * 0.621371

    opex_pipeline = 6 * dist_to_h2_load_miles

    if capex_pipeline_linear == True:
        capex_pipeline = (6.2385 * h2_flow_kg_h + 339713) * dist_to_h2_load_km
    else:
        capex_pipeline = (-0.020744 * (h2_flow_kg_h**2) + 710.144314 * h2_flow_kg_h + 212320.312500) * dist_to_h2_load_km
    
    pipeline_annuals = simple_cash_annuals(30, 30, capex_pipeline,opex_pipeline, 0.03)
    return capex_pipeline, opex_pipeline, pipeline_annuals

if __name__ == '__main__':
    test = pipeline_model(100, 60, True)
    print(test)

