
class Pressure_Vessel_Storage():
    """
    Each pressure vessel can hold ~122 kg of hydrogen.
    Type 4 pressure vessel
    Max pressure: 250 bar
    TODO: Add additional pressure vessels based on market availability.
    https://www.nrel.gov/docs/fy14osti/58564.pdf
    https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
    """

    def __init__(self, input_dict, output_dict):           
        self.input_dict = input_dict
        self.output_dict = output_dict

        #inputs
        self.H2_storage_kg = input_dict['H2_storage_kg']        #[kg]
        self.storage_duration_hrs = input_dict['storage_duration_hrs']  #[hr]
        self.flow_rate_kg_hr = input_dict['flow_rate_kg_hr']        #[kg-H2/hr]
        self.useful_life = 10       #[years] - assumed range 5-20 years
        self.plant_life = 30
        self.compressor_output_pressure = input_dict['compressor_output_pressure'] #[bar]
        self.pressure_vessel_capacity = 122     #[kg-H2/vessel]
    
    def pressure_vessel_costs(self):
        F_install = 1.3     # installation factor
        F_indir = 1.27      # direct and indirect capital cost factor
         
        # Capex = $1,035/kg
        pressure_vessel_capex = 1035 * self.H2_storage_kg * F_install * F_indir #[USD]
        self.output_dict['pressure_vessel_capex'] = pressure_vessel_capex
    
        #Opex = 2.85% of capital investment
        pressure_vessel_opex = 0.285 * self.output_dict['pressure_vessel_capex']
        self.output_dict['pressure_vessel_opex'] = pressure_vessel_opex

        """Assumed useful life = payment period for capital expenditure.
           compressor amortization interest = 3%"""
        a = 0.03
        pressure_vessel_annuals = [0] * self.plant_life

        if self.useful_life <= self.plant_life:
            pressure_vessel_amortization = self.output_dict['pressure_vessel_capex'] * \
            ((a*(1+a)**self.useful_life)/((1+a)**self.useful_life - 1))
        else:
            pressure_vessel_amortization = self.output_dict['pressure_vessel_capex'] * \
            ((a*(1+a)**self.plant_life)/((1+a)**self.plant_life - 1))
        
        for i in range(len(pressure_vessel_annuals)):
            if pressure_vessel_annuals[i] == 0:
                pressure_vessel_annuals[i] = pressure_vessel_amortization + self.output_dict['pressure_vessel_opex']
        self.output_dict['pressure_vessel_annuals'] = pressure_vessel_annuals
        return pressure_vessel_capex, pressure_vessel_opex, pressure_vessel_annuals

if __name__ == '__main__':
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000
    in_dict['storage_duration_hrs'] = 4
    in_dict['flow_rate_kg_hr'] = 126        #[kg-H2/hr]
    in_dict['compressor_output_pressure'] = 250

    test = Pressure_Vessel_Storage(in_dict,out_dict)
    test.pressure_vessel_costs()
    
    print(out_dict['pressure_vessel_capex'])
    print(out_dict['pressure_vessel_annuals'])
