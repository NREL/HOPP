## Pipeline model referenced from NREL/NRWAL repository 

import math
import numpy as np 
from matplotlib import pyplot as plt 
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals

class Pipeline: 
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict 
        self.output_dict = output_dict
    
        self.pipeline_model = input_dict['pipeline_model']
        self.pipe_diam_in = input_dict['pipe_diam_in']
        self.dist_to_h2_load_km = input_dict['dist_to_h2_load_km']
        self.h2_flow_kg_h = input_dict['flow_rate_kg_hr']
        self.offshore_param = input_dict['offshore_bool']
        self.plant_life = input_dict['plant_life']
        self.useful_life = input_dict['useful_life']
       
    def pipeline_cost(self):
     
        # Natural Gas to H2 cost factor 
        ng_to_h2_factor = 1.1 
        # Convert pipeline length from km to miles
        self.dist_to_h2_load_miles = 0.621371 * self.dist_to_h2_load_km
        
        self.output_dict['len_pipeline_miles'] = self.dist_to_h2_load_miles
        # TODO: convert pipeline distance to length of pipeline (bends, drops, connections)
        
        # TODO: Relate pipe diameter, flow-rate and pressure? 
        
        # Install Factor based on on-land or undersea 
        # Offshore
        if self.offshore_param:
            install_opex_fact = 7.0/100 # 7% of CapEx # https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub 
            
            install_capex_fact = 2 # offshore cost factor 1.3-2.3 (Hydrogen insight paper)
        # On Land
        else:
            install_opex_fact = 4.0/100 # 4% of CapEx 
            install_capex_fact = 1 
       
        # CapEx model from NRWL code repository 
        if self.pipeline_model == 'nrwl': 
            '''
            This code section was ported from .yaml file 
            found at: https://github.com/NREL/NRWAL/blob/main/NRWAL/analysis_library/hydrogen/pipeline.yaml 
 
            Inputs
            distance to load (miles), float rate (kg/hr) 
            Returns
            CapEx of pipeline($USD)
            '''
            # TODO: Pipeline CapEx model (linear vs. nonlinear) 
            capex_pipe = (6.2385 * self.h2_flow_kg_h + 339713) * self.dist_to_h2_load_km
            #capex_pipe = (-0.020744 * (self.h2_flow_kg_h**2) + \
            #                    710.144314 * self.h2_flow_kg_h + 212320.312500) * self.dist_to_h2_load_km 
        
            self.output_dict['pipeline_capex'] = capex_pipe
        
        # CapEx model for NG pipeline from Nexant Report (citing Nathan Parker Oil and Gas Journal)
        elif self.pipeline_model == 'nexant':
            '''
            This code section was copied from an H2A Report (NREL and Nexant)
            DE-FG36-05GO15032
            
            Inputs
            distance to load (miles), pipe diameter (inches)
            Returns
            CapEx of pipeline($USD)
            '''
            mat_cost = (330.5 * self.pipe_diam_in**2 + \
                         687 * self.pipe_diam_in + 26960) * self.dist_to_h2_load_miles + \
                          35000
            misc_cost = (8417 * self.pipe_diam_in + 7324) * self.dist_to_h2_load_miles + \
                          95000
            labor_cost = (343 * self.pipe_diam_in**2 + \
                           2047 * self.pipe_diam_in + 170013) * self.dist_to_h2_load_miles + \
                            185000
            row_cost = (577 * self.pipe_diam_in + 29788) * self.dist_to_h2_load_miles + 40000 
            
            capex_pipe = ng_to_h2_factor * (mat_cost + misc_cost + labor_cost + row_cost)
            self.output_dict['pipeline_capex'] = install_capex_fact * capex_pipe 

        '''
        This code section referenced the following report: 
        https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub 
        
        Inputs
        CapEx ($USD), install_factor(%)
        Returns
        OpEx of pipeline($USD)
        '''    
        opex_pipe = install_opex_fact * capex_pipe
        
        self.output_dict['pipeline_opex'] = opex_pipe

        pipeline_annuals = simple_cash_annuals(self.plant_life, self.useful_life,\
            self.output_dict['pipeline_capex'],self.output_dict['pipeline_opex'], 0.03)
        
        self.output_dict['pipeline_annuals'] = pipeline_annuals
        
        return opex_pipe, capex_pipe, pipeline_annuals

# Test sections 
if __name__ == '__main__': 
    print('### pipeline test ###')
    in_dict = dict()
    #in_dict['pipeline_model'] = 'nrwl'
    in_dict['pipeline_model'] = 'nexant'
    in_dict['pipe_diam_in'] = 24.0
    in_dict['offshore_bool'] = True 
    
    in_dict['flow_rate_kg_hr'] = 200
    in_dict['dist_to_h2_load_km'] = np.linspace(20, 2000, 5) 

    out_dict = dict()
    opex_sweep = np.zeros_like(in_dict['flow_rate_kg_hr'])
    capex_sweep = np.zeros_like(in_dict['flow_rate_kg_hr'])
    for i in range(in_dict['dist_to_h2_load_km'].size):
        
        standalone_test = Pipeline(in_dict,out_dict)
        standalone_test.pipeline_cost()
        
    plt.plot(in_dict['dist_to_h2_load_km'],out_dict['pipeline_capex']/10**6,label='CapEx')
    plt.plot(in_dict['dist_to_h2_load_km'],out_dict['pipeline_opex']/10**6,label='OpEx')
    plt.xlabel('dist (km)')
    plt.ylabel('Cost($mUSD)')
    plt.title('CapEx/OpEx for H2 Pipeline')
    plt.legend()

    print('Pipeline Model:',in_dict['pipeline_model'])
    print('Pipeline length (miles):',out_dict['len_pipeline_miles'])
    print('Pipeline CapEx Cost ($USD):',out_dict['pipeline_capex'])
    print('Pipeline OpEx Cost ($USD):', out_dict['pipeline_opex'])

#OpEx NRWL Code that I don't think makes sense 
 #install_opex_fact = 6 # from NRWL code $USD per mile? This number is sus

 #opex_pipe = install_opex_fact * self.dist_to_h2_load_miles # from NRWL 
  
