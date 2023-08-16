from hopp.simulation.technologies.steel.eaf_model import eaf_model
from hopp.simulation.technologies.steel.hdri_model import hdri_model


def greensteel_run(steel_output_desired_kg_hr=1000):
    '''
    This function is an example of how to run the steel functions from eaf_model and hdri_model

    The main argument for most of the functions is steel_output_desired. 
    The argument needs to be in kg or kg/hr. It is recommended that a rate is 
    used (kg/hr) as the financials are dependent on capacity (metric ton liquid steel (tls) per year).
    
    Sources are in the functions

    Notes: Most steel plants run 95% of the year so the yearly hours would be 24*365*.95
    1000 kg are in 1 metric ton
    (tls) stands for tonne liquid steel
    (tco2) is metric tonne of CO2
    Electric efficiency for system was assumed to be 60%

    EAF Notes:

        The steel carbon rating for this is low-carbon steel with a .07% carbon composition. Similar to that of rebar
        Carbon composition is the mass of the of the carbon compare to the total mass of the steel.
        In most steel plants, secondary finishing process are used to achieve higher compositions.
        These processes are not modeled in this.

        Lime is a composition of moslty Silica and Magnesium oxide that bonds with impurities in the iron.
        The result is slag. Slag can be used in other industiral processes like cement bases.

    HDRI: Notes
        
        Not all the iron ore is reduced in the HDRI shaft. Abut 95-97% of the iron ore is reduced.

        More H2 is needed than stoichiometrically required.  H2 gas can be captured in the exhaust gas and
        reused into the input gas.  The exhaust steam could also be captured and sent back to the electrolyzer.
    
        Energy balance on the shaft should be negative meaning that heat is leaving the system and not being absorbed.
        If values are positive, heat will need to be inputted into the shaft.  Values should not be positive.  Energy needed
        should be supplied by the heater.

        The recuperator is the heat exhanger between the exhaust gas and input h2. Not required but is a efficiency
        system.
    '''
    hours = 365*24*.95  

    steel_out_year_tls = steel_output_desired_kg_hr * hours / 1000 
    
    eaf_model_instance = eaf_model()
    hdri_model_instance = hdri_model()


    eaf_mass_outputs = eaf_model_instance.mass_model(steel_output_desired_kg_hr)
    eaf_energy_outputs = eaf_model_instance.energy_model(steel_output_desired_kg_hr)
    eaf_emission_outputs = eaf_model_instance.emission_model(steel_output_desired_kg_hr)
    eaf_financial_outputs = eaf_model_instance.financial_model(steel_out_year_tls)

    hdri_mass_outputs = hdri_model_instance.mass_model(steel_output_desired_kg_hr)
    hdri_energy_outputs = hdri_model_instance.energy_model(steel_output_desired_kg_hr)
    recuperator_outputs = hdri_model_instance.recuperator_mass_energy_model(steel_output_desired_kg_hr)
    heater_outputs = hdri_model_instance.heater_mass_energy_model(steel_output_desired_kg_hr)
    hdri_financial_outputs = hdri_model_instance.financial_model(steel_out_year_tls)


    '''
    EAF model outputs
    '''
    print(eaf_mass_outputs[0]) #Prints Dict list of outputs of the function with units
    print(eaf_energy_outputs[0]) #Prints Dict list of outputs of the function with units
    print(eaf_emission_outputs[0]) #Prints Dict list of outputs of the function with units
    print(eaf_financial_outputs[0]) #Prints Dict list of outputs of the function with units

    steel_out_actual_kg = eaf_mass_outputs[1] #(kg or kg/hr) Iron ore will also reduce in EAF so more steel is produced/
                                                # Will produce 4% more than desired can be used as buffer for 4% steel loss.

    carbon_needed = eaf_mass_outputs[2] #(kg or kg/hr) This is the required carbon needed to create low-carbon steel

    lime_needed = eaf_mass_outputs[3] #(kg or kg/hr) This is the required lime/slag formers needed


    electricity_needed = eaf_energy_outputs[1] #(kwh or kw) This is the energy needed in the furnace
                                            #The units depend on the steel_desired units
                                            #Input of kg returns kwh//Input of kg/hr returns kw


    indirect_emissions = eaf_emission_outputs[1] #(tco2 or tco2/hr) Indirect emissions from the grid of the entire system
                                                # Units are dependent on steel_out units (kg -> tco2, kg/hr -> tco2/hr)
                                                # Includes heater in hdri and eaf eletric arc
                                                #If plant is run solely on renewables, this would be 0
    
    direct_emissions = eaf_emission_outputs[2] #(tco2 or tco2/hr) These are emissions directly from the EAF. 
                                                #Units are dependent on steel_out units (kg -> tco2, kg/hr -> tco2/hr).
                                                #This includes the excess carbon turning into CO2, the CaO emissions,
                                                # the CO2 given off by the EAF electrode and the iron ore pellet production
                                                #Iron pellet production commonly uses fossil fuel to compact the iron ore into pellets

    total_emissions = eaf_emission_outputs[3] #(tco2 or tco2/hr) Total emissions includes direct and indirect
                                                #Units are dependent on steel_out units (kg -> tco2, kg/hr -> tco2/hr)


    eaf_capital_cost = eaf_financial_outputs[1] #(Mil USD) Capital cost uses a lang factor of 3.
                                                #Estimated using capital cost rate of $140 USD/tls/yr

    eaf_operation_cost = eaf_financial_outputs[2] #(Mil USD/yr) Estimated using operational cost of $32 USD/tls/yr

    eaf_maintenance_cost = eaf_financial_outputs[3] #(Mil USD/yr) Estimated using a percentage of 1.5% of total Capital cost

    eaf_depreciation_cost = eaf_financial_outputs[4] #(Mil USD/yr) Total Capital cost divided by the plant life(40 years)

    eaf_coal_cost = eaf_financial_outputs[5] #(Mil USD/yr) Total cost of the coal needed for desired capacity
                                            #coal cost rate is assumed $120/ton coal

    eaf_labor_cost = eaf_financial_outputs[6] #(Mil USD/yr) Total labor cost needed for desired capacity
                                            #Labor cost rate assumed $20 USD/year

    eaf_lime_cost = eaf_financial_outputs[7] #(Mil USD/yr) Total lime cost needed for desired capacity

    eaf_total_emission_cost = eaf_financial_outputs[8] #(Mil USD/yr) Emissions multiplied by emissions cost
                                                        #Emission cost assumed to be $30 USD/tco2
                                                        #Currently no emission costs in states that hold steel plants

    '''
    hdri model outputs
    '''
    print(hdri_mass_outputs[0]) #Prints Dict of outputs of the function with units
    print(hdri_energy_outputs[0]) #Prints Dict of outputs of the function with units
    print(recuperator_outputs[0]) #Prints Dict of outputs of the function with units
    print(heater_outputs[0]) #Prints Dict of outputs of the function with units
    print(hdri_financial_outputs[0]) #Prints Dict of outputs of the function with units

    steel_out_desired = hdri_mass_outputs[3] #(kg or kg/hr) Should return inputted arg

    iron_ore_mass = hdri_mass_outputs[4] #(kg or kg/hr) Iron ore needed for desired_steel_out

    mass_h2_in = hdri_mass_outputs[5] #(kg or kg/hr) Mass of hydrogen gas needed to reduce the iron ore in

    mass_h2_out = hdri_mass_outputs[6] #(kg or kg/hr) Mass of hydrogen leaving shaft

    mass_h2o_out = hdri_mass_outputs[7] #(kg or kg/hr) Mass of water (steam) leaving shaft

    mass_pure_iron_out = hdri_mass_outputs[8] #(kg or kg/hr) Mass of the pure iron in stream leaving shaft

    mass_gas_stream_out = hdri_mass_outputs[9] #(kg or kg/hr) mass of the gas steam leaving

    mass_iron_ore_out = hdri_mass_outputs[10] #(kg or kg/hr) mass of iron ore leaving shaft


    energy_balance = hdri_energy_outputs[1] #(kwh or kw) Energy balance of the hdri shaft (Negative denotes heat leaving system)


    heater_electricity_needed = heater_outputs[1] #(kwh or kw) Electricity needed by the heater to heat hydrogen to needed temp to reduce iron


    enthalpy_entering_heater = recuperator_outputs[1] #(kwh or kw) Enthalpy of the hydrogen entering heater from recuperator


    hdri_capital_cost = hdri_financial_outputs[1] #(Mil USD) Capital cost uses a lang factor of 3.
                                                #Estimated using capital cost rate of $80 USD/tls/yr

    hdri_operation_cost = hdri_financial_outputs[2] #(Mil USD/yr) Estimated using operational cost of $13 USD/tls/yr

    hdri_maintenance_cost = hdri_financial_outputs[3] #(Mil USD/yr) Estimated using a percentage of 1.5% of total Capital cost

    hdri_depreciation_cost = hdri_financial_outputs[4] #(Mil USD/yr) Total Capital cost divided by the plant life(40 years)

    iron_ore_cost = hdri_financial_outputs[5] #(Mil USD/yr) Total iron ore cost needed for desirec steel output per year

    hdri_labor_cost = hdri_financial_outputs[6] #(Mil USD/yr) Total labor cost needed for desired capacity
                                                #Labor cost rate assumed $20 USD/year



