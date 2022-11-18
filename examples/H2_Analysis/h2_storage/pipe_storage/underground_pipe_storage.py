from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals

class Underground_Pipe_Storage():
    """
    Oversize pipe: pipe OD = 24'' schedule 60
    Max pressure: 100 bar
    https://www.nrel.gov/docs/fy14osti/58564.pdf
    https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
    https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub
    """

    def __init__(self, input_dict, output_dict):           
        self.input_dict = input_dict
        self.output_dict = output_dict
        """
        input_dict requires:
            parm: compressor_output_pressure == 100 [bar]
            parm: 'H2_storage_kg' [kg] or 'storage_duration_hrs' [hrs] and 'flow_rate_kg_hr' [kg/hr]
        """
        #inputs
        if input_dict['compressor_output_pressure'] == 100:
            self.compressor_output_pressure = input_dict['compressor_output_pressure'] #[bar]
        else:
            raise Exception('Error. compressor_output_pressure must = 100bar for pressure vessel storage.')

        
        if 'H2_storage_kg' in input_dict:
            self.H2_storage_kg = input_dict['H2_storage_kg']        #[kg]
        elif 'storage_duration_hrs' and 'flow_rate_kg_hr' in input_dict:
            self.H2_storage_kg = input_dict['storage_duration_hrs'] * input_dict['flow_rate_kg_hr']  
        else:
            raise Exception('Error. input_dict must contain H2_storage_kg or storage_duration_hrs and flow_rate_kg_hr')

        #assumptions
        self.useful_life = 30       #[years]
        self.plant_life = 30        # [years]
    
    def pipe_storage_costs(self):
        # Capex = $560/kg
        pipe_storage_capex = 560 * self.H2_storage_kg  #[USD]
        self.output_dict['pipe_storage_capex'] = pipe_storage_capex
    
        #Opex = 2.85% of capital investment
        pipe_storage_opex = 0.0285 * pipe_storage_capex
        self.output_dict['pipe_storage_opex'] = pipe_storage_opex

        """Assumed useful life = payment period for capital expenditure.
           compressor amortization interest = 3%"""
        
        pipe_storage_annuals = simple_cash_annuals(self.plant_life, self.useful_life,\
            self.output_dict['pipe_storage_capex'],self.output_dict['pipe_storage_opex'], 0.03)

        self.output_dict['pipe_storage_annuals'] = pipe_storage_annuals

        return pipe_storage_capex, pipe_storage_opex, pipe_storage_annuals

if __name__ == '__main__':
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000
    in_dict['storage_duration_hrs'] = 4
    in_dict['flow_rate_kg_hr'] = 126        #[kg-H2/hr]
    in_dict['compressor_output_pressure'] = 100

    test = Underground_Pipe_Storage(in_dict,out_dict)
    test.pipe_storage_costs()
    print('Underground pipe storage capex [USD]: ', out_dict['pipe_storage_capex'])
    print('Underground pipe storage annuals [USD/yr]: ', out_dict['pipe_storage_annuals'])
