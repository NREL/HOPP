from hopp.simulation.technologies.steel.eaf_model import eaf_model
from hopp.simulation.technologies.steel.hdri_model import hdri_model


#from csv import writer
import os
#from dotenv import load_dotenv
import pandas as pd

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site
#from hopp.utilities.keys import set_developer_nrel_gov_key
#from examples.H2_Analysis.plot_reopt_results import plot_reopt_results
from examples.H2_Analysis.run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
#from examples.H2_Analysis.simple_dispatch import SimpleDispatch
#import hopp.simulation.technologies.hydrogen.electrolysis.run_h2_PEM as run_h2_PEM
import numpy as np
#import numpy_financial as npf
#from lcoe.lcoe import lcoe as lcoe_calc
#import matplotlib.pyplot as plt
import warnings
#from pathlib import Path
warnings.filterwarnings("ignore")

def greensteel_run(steel_output_desired_kg_hr=120160):
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

    print('Iron ore mass',iron_ore_mass/steel_out_desired)
    print('Lime Mass',lime_needed/steel_out_actual_kg)
    print('Carbon Mass need',carbon_needed/steel_out_desired)
    print('H2 needed',mass_h2_in/steel_out_desired)
    print('Electricity',electricity_needed/1000)
    print('Cap Cost Eaf',eaf_capital_cost)
    print('Cap Cost HDRI',hdri_capital_cost)
    print('Labor Cost',(eaf_labor_cost+hdri_labor_cost))
    print('Maintenance Cost',(eaf_maintenance_cost+hdri_maintenance_cost))
    return

import os
from dotenv import load_dotenv
import pandas as pd
import json
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site
from hopp.utilities.keys import set_developer_nrel_gov_key
#from examples.H2_Analysis.plot_reopt_results import plot_reopt_results
#from examples.H2_Analysis.run_reopt import run_reopt
#from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
#from examples.H2_Analysis.simple_dispatch import SimpleDispatch
#import hopp.simulation.technologies.hydrogen.electrolysis.run_h2_PEM as run_h2_PEM
import numpy as np
#import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
#import matplotlib.pyplot as plt
import warnings
#from pathlib import Path
warnings.filterwarnings("ignore")

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env



def h2_main_steel(lcoe=.5612,steel_output_desired=120160,efficiency=.67,MW_h2=.6,lang_factor=3,elec_spec=55.5,discount_rate=.10,lifetime=40):
    """
    Example of how to call in steel models to calculate lcos (Levelized Cost of Steel)

    Electrolyzer costs should be brought in from hopp.  Electrolyzer financials are calculated using simple assumptions

    Assumes constant electrical input

    Inputs:
        -lcoe: levelized cost of electricity entering plant (cents/kwh)
        -steel_output_desired:  steel output a day (kg/day) 1 million tonnes per year is about 120160 kg/day
        -efficiency: electrical efficiency of the electrolyzer (%)
        -lang_factor: factor of capital costs to estimate construction and auxilliary processes
        -elec_spec: specification of electrolyzer i.e. how many kilowatthours to produce 1 kilogram h2(kwh/kgh2)
        -discount_rate: (%)
        -lifetime: lifetime of plant (years)
    """


    # Step 1: Establish output structure and special inputs
    # save_all_runs = pd.DataFrame()
    #save_outputs_dict = establish_save_output_dict()
    year = 2013
    sample_site['year'] = year
    useful_life = 30
    critical_load_factor_list = [1]
    run_reopt_flag = False
    custom_powercurve = True
    storage_used = True
    battery_can_grid_charge = False
    grid_connected_hopp = False
    interconnection_size_mw = 100
    electrolyzer_sizes = [50]


    
    eaf_model_instance = eaf_model()
    hdri_model_instance = hdri_model()

    #lcoe = .5612
    
    #steel_output_desired = 120160 #kg/hr or tls/hr

    energy_kwh = eaf_model_instance.energy_model(steel_output_desired)
    energy_mwh = eaf_model_instance.energy_model(steel_output_desired)[1]/1000

    lcoe_cents_kwh = lcoe

    lcoe_USDMWH = lcoe * 100


    #total_electricity_cost = energy_mwh * lcoe_USDMWH


    steel_output_desired_kg_hr = steel_output_desired
    hours = 365*24*.95 #hours plant runs of the time

    h2_kg_hr = hdri_model_instance.mass_model(steel_output_desired_kg_hr)[5]

    steel_out_year_tls = steel_output_desired_kg_hr * hours / 1000 #(tls/yr)
    h2_yr = h2_kg_hr*365*24*.95 #(kg h2/year)

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
    Way to incorporate hopp electrolyzer financials
    '''
    #from hopp.simulation.technologies.hydrogen.electrolysis.H2_cost_model import basic_H2_cost_model

    #electrolyzer_capex_kw = 1100 #(USD/kw)
    #time_between_replacement = 40000 #(hours)
    #electrolyzer_size_mw = electrolyzer_capacity
    #electrical_generation_timeseries_kw = electrical_generation_timeseries
    #hydrogen_annual_output = h2_yr
    #PTC_USD_kg = .60
    #ITC_perc = 0
    #plant_life = 40 

    #electrolyzer_costs = basic_H2_cost_model(electrolyzer_capex_kw,time_between_replacement,electrolyzer_size_mw,
    #                                        plant_life, atb_year,electrical_generation_timeseries_kw, hydrogen_annual_output, PTC_USD_kg,
    #                                         ITC_perc, False, 0)
    
    #electrolyzer_capital_cost = electrolyzer_costs[1]
    #electrolyzer_OM_cost = electrolyzer_costs[2]

    
    '''
    Capital Cost for plant
    '''
    #lang_factor = 3    

    hdri_capital_cost = hdri_financial_outputs[1] #(Mil USD) Capital cost uses a lang factor of 3.
                                                #Estimated using capital cost rate of $80 USD/tls/yr
    eaf_capital_cost = eaf_financial_outputs[1]#(Mil USD) Capital cost uses a lang factor of 3.
                                                #Estimated using capital cost rate of $140 USD/tls/yr


    lhv_h2 = 120.1 #Mj/kg h2  low heating value for hydrogen gas
    h2_kg_s = h2_yr/(365*24*60*60*.95) #kg/s
    #efficiency = .67 #%

    electrolyzer_capacity = h2_kg_s*lhv_h2*efficiency #MW

    #MW_h2 = .6 #Mil USD/MW electrolyzer

    electrolyzer_cap_cost = electrolyzer_capacity * MW_h2 * lang_factor #(Mil USD) 

    total_cap_cost_MilUSD = (hdri_capital_cost + eaf_capital_cost + electrolyzer_cap_cost)
    total_cap_cost = (hdri_capital_cost + eaf_capital_cost + electrolyzer_cap_cost)*10**6 #USD
    
    '''
    Operational Costs for plant
    '''

    hdri_operation_cost = hdri_financial_outputs[2] #(Mil USD/yr) Estimated using operational cost of $13 USD/tls/yr
    hdri_maintenance_cost = hdri_financial_outputs[3] #(Mil USD/yr) Estimated using a percentage of 1.5% of total Capital cost
    hdri_depreciation_cost = hdri_financial_outputs[4] #(Mil USD/yr) Total Capital cost divided by the plant life(40 years)
    iron_ore_cost = hdri_financial_outputs[5] #(Mil USD/yr) Total iron ore cost needed for desirec steel output per year
    hdri_labor_cost = hdri_financial_outputs[6] #(Mil USD/yr) Total labor cost needed for desired capacity


    total_hdri_operating_cost = (hdri_operation_cost + hdri_maintenance_cost + hdri_depreciation_cost
                                +iron_ore_cost+hdri_labor_cost)*10**6#(USD)
    
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


    total_eaf_operating_cost = (eaf_operation_cost + eaf_depreciation_cost + eaf_maintenance_cost + eaf_coal_cost + eaf_labor_cost
                                + eaf_lime_cost + eaf_total_emission_cost)*10**6 #(USD/yr)
    

    #h2_yr = h2*365*24*.95 #(kg h2/year)

    water_cost_USD_kg = .59289/1000 #(USD/kg)
    water_yr = h2_yr * 11 #(kg h20/yr)11 kg water for 1 kg h2
    water_cost = water_yr * water_cost_USD_kg #(USD/yr)

    total_electrolyzer_operating_cost = water_cost
    
    '''
    Electricity costs for Plant
    '''
    
    eaf_electricity_needed = eaf_energy_outputs[1] #(kwh or kw)/yr This is the energy needed in the furnace  kwh need per input tls
                                            #The units depend on the steel_desired units
                                            #Input of kg returns kwh//Input of kg/hr returns kw
    heater_electricity_needed = heater_outputs[1] #(kwh or kw) Electricity needed by the heater to heat hydrogen to needed temp to reduce iron

    eaf_electricity_needed_mwh = eaf_electricity_needed/1000  #mwh needed for input steel in tls per day
    eaf_electricity_needed_mwh_yr = eaf_electricity_needed_mwh*24*365  #mwh needed for input steel in tls per year


    heater_electricity_needed_mwh = heater_electricity_needed/1000 #mwh needed for input steel in tls per day
    heater_electricity_needed_mwh_yr = heater_electricity_needed_mwh*24*365 #mwh needed for input steel in tls per year

    '''
    Checks. Returns steel output per year
    '''
    #check_1 = heater_electricity_needed_mwh_yr*1000/355
    #check_2 = eaf_electricity_needed_mwh_yr*1000/501.315


    #elec_spec = 55.5 #kwh/kgh2
    electrolyzer_electricity_needed = elec_spec * h2_yr/1000 #(mwh/yr)

    total_electricity_mwh = heater_electricity_needed_mwh_yr+eaf_electricity_needed_mwh_yr+electrolyzer_electricity_needed #(mwh/yr)
    
    total_electricity_cost = total_electricity_mwh * lcoe_USDMWH #(USD/yr)

    '''
    Total Annual Operating Costs consist of HDRI, EAF, Electrolyzer operating costs and the total electricity used by all three subsystems
    '''
    
    annual_operating_cost = (total_hdri_operating_cost + total_eaf_operating_cost + total_electrolyzer_operating_cost + total_electricity_cost) #(USD/yr)

    annual_operating_cost_MilUSD = (total_hdri_operating_cost + total_eaf_operating_cost + total_electrolyzer_operating_cost + total_electricity_cost)/10**6
    '''
    LCOE Assumptions
    '''
    #discount_rate = .10 #%
    #lifetime = 20 #years

    lcos = lcoe_calc(steel_out_year_tls,total_cap_cost,annual_operating_cost,discount_rate,lifetime)
    #lcos = lcoe_calc(steel_out_year_tls,total_cap_cost,annual_operating_cost,discount_rate,lifetime)

    #print(lcos)

    return(lcos,total_electricity_cost,total_cap_cost,annual_operating_cost)

