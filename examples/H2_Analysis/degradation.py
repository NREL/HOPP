from tracemalloc import start
import numpy as np
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT

class Degradation:



    def __init__(self,
                 power_sources: dict,
                 electrolyzer: bool,
                 electrolyzer_rating: 0,
                 project_life: int,
                 generation_profile: dict,
                 load: list): 
        """
        Base class for adding technology degredation over time.

        :param power_sources: tuple of strings, float pairs
            names of power sources to include and their kw sizes
            choices include: ('pv', 'wind', 'battery')
        :param electrolyzer: bool, 
            if True electrolyzer is included in degredation model
        :param electrolyzer_rating: electrolyzer rating in MW
        :param project_life: integer,
            duration of hybrid project in years
        :param generation_profile: arrays,
            generation_profile from HybridSimulation for each technology ('wind','pv')
        :param load: list
            absolute desired load profile [kWe]
        """
        self.power_sources = power_sources
        self.electrolyzer = electrolyzer
        self.electrolyzer_rating = electrolyzer_rating
        self.project_life = project_life
        self.max_generation = generation_profile
        self.load = load

        temp = list(power_sources.keys())
        for k in temp:
            power_sources[k.lower()] = power_sources.pop(k)
        if 'battery' in power_sources.keys():
            self.battery_storage = power_sources['battery']['system_capacity_kwh'] / 1000       #[MWh]
            self.battery_charge_rate = power_sources['battery']['system_capacity_kw'] / 1000    #[MW]
            self.battery_discharge_rate = power_sources['battery']['system_capacity_kw'] / 1000 #[MW]

    def simulate_generation_degradation(self):
        if 'pv' in self.power_sources.keys():
            self.pv_degraded_generation = []
            self.pv_degradation_rate = 0.75/100     #Linear degradation of 0.75% per year
            pv_max_generation = self.max_generation.pv[0:8760]
            pv_generation = pv_max_generation
            for years in range(0,self.project_life):
                pv_gen_next = pv_generation
                self.pv_degraded_generation = np.append(self.pv_degraded_generation,pv_gen_next)
                pv_generation = np.multiply(pv_gen_next, 1 - self.pv_degradation_rate)    
                    
        if not 'pv' in self.power_sources.keys():
            self.pv_degraded_generation = [0] * useful_life
            #TODO: add option for non-degraded generation


        if 'wind' in self.power_sources.keys():
            self.wind_degraded_generation = []
            self.wind_degradation_rate = 1.5/100     #Linear degradation of 1.5% per year
            wind_max_generation = self.max_generation.wind[0:8760]
            wind_generation = wind_max_generation
            for years in range(0,self.project_life):
                wind_gen_next = wind_generation
                self.wind_degraded_generation = np.append(self.wind_degraded_generation,wind_gen_next)
                wind_generation = np.multiply(wind_gen_next, 1 - self.wind_degradation_rate)    
                    
        if not 'wind' in self.power_sources.keys():
            self.wind_degraded_generation = [0] * useful_life
            #TODO: add option for non-degraded generation
        
        self.hybrid_degraded_generation = np.add(self.pv_degraded_generation, self.wind_degraded_generation)
      
    def simulate_battery_degradation(self):
        self.hybrid_degraded_generation 

        # energy specific metrics required for battery model
        self.energy_shortfall = [x - y for x, y in
                             zip(self.load,self.hybrid_degraded_generation)]
        self.energy_shortfall = [x if x > 0 else 0 for x in self.energy_shortfall]
        self.combined_pv_wind_curtailment = [x - y for x, y in
                             zip(self.hybrid_degraded_generation,self.load)]
        self.combined_pv_wind_curtailment = [x if x > 0 else 0 for x in self.combined_pv_wind_curtailment]

        # run SimpleDispatch()
        # battery degradation: reduced max state of charge (SOC)
        # battery model will re-run annually to with updated max SOC
        # note when max_SOC is set it will multiply with battery_storage [MWh] so battery_SOC will be > 1
        if 'battery' in self.power_sources.keys():
            self.battery_used = []
            self.excess_energy = []
            self.battery_SOC = []
            self.battery_repair = [] 
            start_year = 0
            full_SOC = 1
            current_Max_SOC = full_SOC
            for years in range(0,self.project_life):
                battery_dispatch = SimpleDispatch()
                end_year = start_year + 8760
                battery_dispatch.Nt = len(self.energy_shortfall[start_year:end_year])
                battery_dispatch.curtailment = self.combined_pv_wind_curtailment[start_year:end_year]
                battery_dispatch.shortfall = self.energy_shortfall[start_year:end_year]
                battery_dispatch.battery_storage = self.battery_storage
                battery_dispatch.charge_rate = self.battery_charge_rate
                battery_dispatch.discharge_rate = self.battery_discharge_rate
                self.battery_degradation_rate = 1/100       #Linear degradation 1% annually (1 cycle a day)
                battery_dispatch.max_SOC = current_Max_SOC
                if battery_dispatch.max_SOC > 0.8:
                    battery_used, excess_energy, battery_SOC = battery_dispatch.run()
                    self.battery_used = np.append(self.battery_used, battery_used)
                    self.excess_energy = np.append(self.excess_energy, excess_energy)
                    self.battery_SOC = np.append(self.battery_SOC, battery_SOC)
                    self.battery_repair = np.append(self.battery_repair,[0])
                    start_year += 8760
                    current_Max_SOC = battery_dispatch.max_SOC - self.battery_degradation_rate
                else:
                    battery_MTTR = 7 #days
                    battery_dispatch.curtailment[0:battery_MTTR*24] = [0] * (battery_MTTR*24)
                    battery_dispatch.shortfall[0:battery_MTTR*24] = [0] * (battery_MTTR*24)
                    battery_dispatch.max_SOC = full_SOC
                    battery_used, excess_energy, battery_SOC = battery_dispatch.run()
                    self.battery_used = np.append(self.battery_used, battery_used)
                    self.excess_energy = np.append(self.excess_energy, excess_energy)
                    self.battery_SOC = np.append(self.battery_SOC, battery_SOC)
                    start_year += 8760
                    self.battery_repair = np.append(self.battery_repair,[1])
                    current_Max_SOC = battery_dispatch.max_SOC - self.battery_degradation_rate
                #TODO: Add flag for charging and discharging to SimpleDispatch
                #TODO: Look at converting from SimpleDispatch to battery model in HOPP
        
        self.combined_pv_wind_storage_power_production = self.hybrid_degraded_generation + self.battery_used
    
    def simulate_electrolyzer_degradation(self):
        #TODO: make it so simulate_battery_degradation isn't necessary to simulate_electrolyzer_degradation
        
        self.combined_pv_wind_storage_power_production 
    
        if self.electrolyzer:
            kw_continuous = self.electrolyzer_rating * 1000
            energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in self.combined_pv_wind_storage_power_production]
            electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
            electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

            in_dict = dict()
            in_dict['electrolyzer_system_size_MW'] = self.electrolyzer_rating
            out_dict = dict()
            self.hydrogen_hourly_production = []
            self.electrolyzer_total_efficiency = []
            self.electrolyzer_repair = []
            
            start_year = 0
            self.electrolyzer_degradation_rate = 0.01314 # 1.5mV/1000 hrs extrapolated to one year TODO: VERIFY!!
            ideal_stack_input_voltage_DC = 250
            current_stack_input_voltage_DC = ideal_stack_input_voltage_DC

            for years in range(0,self.project_life):
                end_year = start_year + 8760
                in_dict['P_input_external_kW'] = electrical_generation_timeseries[start_year:end_year]
                if current_stack_input_voltage_DC < 250.09198:
                    # Set threshold for repair to be in line with H2A model stack life (7 years) 
                    # https://www.nrel.gov/hydrogen/assets/docs/current-central-pem-electrolysis-2019-v3-2018.xlsm
                    
                    el = PEM_electrolyzer_LT(in_dict, out_dict)
                    el.stack_input_voltage_DC = current_stack_input_voltage_DC
                    el.h2_production_rate()

                    hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
                    self.hydrogen_hourly_production = np.append(self.hydrogen_hourly_production, hydrogen_hourly_production)

                    electrolyzer_total_efficiency = out_dict['total_efficiency']
                    self.electrolyzer_total_efficiency = np.append(self.electrolyzer_total_efficiency, electrolyzer_total_efficiency)
                    start_year += 8760
                    current_stack_input_voltage_DC = current_stack_input_voltage_DC + self.electrolyzer_degradation_rate
                    self.electrolyzer_repair = np.append(self.electrolyzer_repair,[0])
                else:
                    electrolyzer_MTTR = 21 #days
                    in_dict['P_input_external_kW'][0:electrolyzer_MTTR*24] = [0] * (electrolyzer_MTTR*24)
                    el = PEM_electrolyzer_LT(in_dict, out_dict)
                    current_stack_input_voltage_DC = ideal_stack_input_voltage_DC
                    el.stack_input_voltage_DC = current_stack_input_voltage_DC
                    el.h2_production_rate()

                    hydrogen_hourly_production = out_dict['h2_produced_kg_hr_system']
                    self.hydrogen_hourly_production = np.append(self.hydrogen_hourly_production, hydrogen_hourly_production)

                    electrolyzer_total_efficiency = out_dict['total_efficiency']
                    self.electrolyzer_total_efficiency = np.append(self.electrolyzer_total_efficiency, electrolyzer_total_efficiency)
                    start_year += 8760
                    current_stack_input_voltage_DC = current_stack_input_voltage_DC + self.electrolyzer_degradation_rate
                    self.electrolyzer_repair = np.append(self.electrolyzer_repair,[1])




if __name__ == '__main__': 
    from pathlib import Path
    import matplotlib.pyplot as plt
    from hybrid.sites import SiteInfo, flatirons_site
    from hybrid.hybrid_simulation import HybridSimulation
    from hybrid.log import hybrid_logger as logger
    from hybrid.keys import set_nrel_key_dot_env

    examples_dir = Path(__file__).resolve().parents[1]
    plot_degradation = True

    # Set API key
    set_nrel_key_dot_env()

    # Set wind, solar, and interconnection capacities (in MW)
    solar_size_mw = 50
    wind_size_mw = 50
    interconnection_size_mw = 50        #Required by HybridSimulation() not currently being used for calculations.
    battery_capacity_mw = 20
    battery_capacity_mwh = battery_capacity_mw * 4 
    electrolyzer_capacity_mw = 40
    useful_life = 30
    load = [electrolyzer_capacity_mw*1000] * useful_life * 8760

    technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': 10,
                    'turbine_rating_kw': 2000},
                'battery': {
                    'system_capacity_kwh': battery_capacity_mwh * 1000,
                    'system_capacity_kw': battery_capacity_mw * 1000
                    }
                }

    # Get resource
    lat = flatirons_site['lat']
    lon = flatirons_site['lon']
    site = SiteInfo(flatirons_site)

    # Create model
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)
    

    hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hybrid_plant.ppa_price = 0.1

    hybrid_plant.simulate(useful_life)
    
    # Save the outputs
    generation_profile = hybrid_plant.generation_profile

    hybrid_degradation = Degradation(technologies, True, electrolyzer_capacity_mw, useful_life, generation_profile, load)

    hybrid_degradation.simulate_generation_degradation()
    hybrid_degradation.simulate_battery_degradation()
    hybrid_degradation.simulate_electrolyzer_degradation()
    print("Number of battery repairs: ", hybrid_degradation.battery_repair)
    print("Number of electrolyzer repairs: ", hybrid_degradation.electrolyzer_repair)
    print("Non-degraded lifetime pv power generation: ", np.sum(hybrid_plant.pv.generation_profile)/1000, "[MW]")
    print("Degraded lifetime pv power generation: ", np.sum(hybrid_degradation.pv_degraded_generation)/1000, "[MW]")
    print("Non-degraded lifetime wind power generation: ", np.sum(hybrid_plant.wind.generation_profile)/1000, "[MW]")
    print("Degraded lifetime wind power generation: ", np.sum(hybrid_degradation.wind_degraded_generation)/1000, "[MW]")
    print("Battery used over lifetime: ", np.sum(hybrid_degradation.battery_used)/1000, "[MW]")
    print("Life-time Hydrogen production: ", np.sum(hybrid_degradation.hydrogen_hourly_production), "[kg]")

    if plot_degradation:
        plt.figure(figsize=(10,6))
        plt.subplot(311)
        plt.title("Max power generation vs degraded power generation")
        plt.plot(hybrid_degradation.wind_degraded_generation[175200:175344],label="degraded wind")
        plt.plot(hybrid_plant.wind.generation_profile[175200:175344],label="max generation")
        plt.ylabel("Power Production (kW)")
        plt.legend()
        
        plt.subplot(312)
        plt.plot(hybrid_degradation.pv_degraded_generation[175200:175344],label="degraded pv")
        plt.plot(hybrid_plant.pv.generation_profile[175200:175344],label="max generation")
        plt.ylabel("Power Production (kW)")
        plt.legend()

        plt.subplot(313)
        plt.plot(hybrid_degradation.hybrid_degraded_generation[175200:175344], label="degraded hybrid generation")
        plt.plot(load[175200:175344], label = "load profile")
        plt.ylabel("Power Production (kW)")
        plt.xlabel("Time (hour)")
        plt.legend()
        plt.show()

