from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals

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
        """
        input_dict requires:
            parm: compressor_output_pressure == 250 [bar]
            parm: 'H2_storage_kg' [kg] or 'storage_duration_hrs' [hrs] and 'flow_rate_kg_hr' [kg/hr]
        """
        #inputs
        if input_dict['compressor_output_pressure'] == 250:
            self.compressor_output_pressure = input_dict['compressor_output_pressure'] #[bar]
        else:
            raise Exception('Error. compressor_output_pressure must = 250bar for pressure vessel storage.')

        
        if 'H2_storage_kg' in input_dict:
            self.H2_storage_kg = input_dict['H2_storage_kg']        #[kg]
        elif 'storage_duration_hrs' and 'flow_rate_kg_hr' in input_dict:
            self.H2_storage_kg = input_dict['storage_duration_hrs'] * input_dict['flow_rate_kg_hr']  
        else:
            raise Exception('Error. input_dict must contain H2_storage_kg or storage_duration_hrs and flow_rate_kg_hr')

        #assumptions
        self.useful_life = 10       #[years] - assumed range 5-20 years
        self.plant_life = 30
        self.pressure_vessel_capacity = 122     #[kg-H2/vessel]
    
    def pressure_vessel_costs(self):
        F_install = 1.3     # installation factor
        F_indir = 1.27      # direct and indirect capital cost factor
         
        # Capex = $1,035/kg
        pressure_vessel_capex = 1035 * self.H2_storage_kg * F_install * F_indir #[USD]
        self.output_dict['pressure_vessel_capex'] = pressure_vessel_capex
    
        #Opex = 2.85% of capital investment
        pressure_vessel_opex = 0.0285 * self.output_dict['pressure_vessel_capex']
        self.output_dict['pressure_vessel_opex'] = pressure_vessel_opex

        # Annuals = opex + payment period for capex
        pressure_vessel_annuals = simple_cash_annuals(self.plant_life, self.useful_life,\
            self.output_dict['pressure_vessel_capex'],self.output_dict['pressure_vessel_opex'], 0.03)

        self.output_dict['pressure_vessel_annuals'] = pressure_vessel_annuals

        return pressure_vessel_capex, pressure_vessel_opex, pressure_vessel_annuals

if __name__ == '__main__':
    in_dict = dict()
    out_dict = dict()
    in_dict['compressor_output_pressure'] = 250 #[bar]
    in_dict['H2_storage_kg'] = 122        #[kg]
    in_dict['storage_duration_hrs'] = 4     #[hrs]
    in_dict['flow_rate_kg_hr'] = 126        #[kg-H2/hr]
    

    test = Pressure_Vessel_Storage(in_dict,out_dict)
    test.pressure_vessel_costs()
    
    print('Pressure Vessel capex [USD]: ', out_dict['pressure_vessel_capex'])
    print(out_dict['pressure_vessel_opex'])
    print('Pressure Vessel Annuals [USD/year]: ',out_dict['pressure_vessel_annuals'])
