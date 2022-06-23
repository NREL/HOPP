
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

        #inputs
        self.H2_storage_kg = input_dict['H2_storage_kg']        #[kg]
        self.storage_duration_hrs = input_dict['storage_duration_hrs']  #[hr]
        self.flow_rate_kg_hr = input_dict['flow_rate_kg_hr']        #[kg-H2/hr]
        self.useful_life = 30       #[years]
        self.plant_life = 30
        self.compressor_output_pressure = input_dict['compressor_output_pressure'] #[bar]
    
    def pipe_storage_costs(self):
        # Capex = $560/kg
        pipe_storage_capex = 560 * self.H2_storage_kg  #[USD]
        self.output_dict['pipe_storage_capex'] = pipe_storage_capex
    
        #Opex = 2.85% of capital investment
        pipe_storage_opex = 0.285 * self.output_dict['pressure_vessel_capex']
        self.output_dict['pipe_storage_opex'] = pipe_storage_opex

        return pipe_storage_capex, pipe_storage_opex

    def pipe_storage_annuals(self):
        """Assumed useful life = payment period for capital expenditure.
           compressor amortization interest = 3%"""
        a = 0.03
        pipe_storage_annuals = [0] * self.plant_life

        if self.useful_life <= self.plant_life:
            pipe_storage_amortization = self.output_dict['pipe_storage_capex'] * \
            ((a*(1+a)**self.useful_life)/((1+a)**self.useful_life - 1))
        else:
            pipe_storage_amortization = self.output_dict['pipe_storage_capex'] * \
            ((a*(1+a)**self.plant_life)/((1+a)**self.plant_life - 1))
        
        for i in range(len(pipe_storage_annuals)):
            if pipe_storage_annuals[i] == 0:
                pipe_storage_annuals[i] = pipe_storage_amortization + self.output_dict['pipe_storage_opex']
        self.output_dict['pipe_storage_annuals'] = pipe_storage_annuals
        return pipe_storage_annuals

if __name__ == '__main__':
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000
    in_dict['storage_duration_hrs'] = 4
    in_dict['flow_rate_kg_hr'] = 126        #[kg-H2/hr]
    in_dict['compressor_output_pressure'] = 250

    test = Underground_Pipe_Storage(in_dict,out_dict)
    test.pipe_storage_costs()
    test.pipe_storage_annuals()
    print(out_dict['pipe_storage_capex'])
    print(out_dict['pipe_storage_annuals'])
