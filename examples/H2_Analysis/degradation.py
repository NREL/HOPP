import numpy as np
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM

class Degradation:



    def __init__(self,
                 power_sources: dict,
                 electrolyzer: bool,
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
        :param project_life: integer,
            duration of hybrid project in years
        :param max_generation: arrays,
            generation_profile from HybridSimulation for each technology
        :param load: list
            absolute desired load profile [kWe]
        """
        self.power_sources = power_sources
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

    def simulate_degradation(self):

        if 'pv' in self.power_sources.keys():
            self.pv_degrad_gen = []
            self.pv_repair = 0
            pv_max_gen = self.max_generation.pv[0:8760]
            pv_gen = pv_max_gen
            for years in range(0,self.project_life):
                pv_gen_next = np.multiply(pv_gen, 0.9925)           #Linear degredation of 0.75% per year
                check = max(pv_gen_next)
                threshold = max(np.multiply(pv_max_gen,0.8))
                if check > threshold:
                    self.pv_degrad_gen = np.append(self.pv_degrad_gen,pv_gen_next)
                    pv_gen = pv_gen_next
                else:   
                    # Add a part availability 
                    # Append pv_degrad_gen with lost generation during repair time 
                    # and set pv_gen back to pv_max_gen
                    # have counter for each time a repair is necessary
                    pv_MTTR = 9     #days
                    pv_gen_next[0:pv_MTTR*24] = 0
                    self.pv_degrad_gen = np.append(self.pv_degrad_gen,pv_gen_next)
                    pv_gen = pv_max_gen
                    self.pv_repair += 1 
        if not 'pv' in self.power_sources.keys():
            self.pv_degrad_gen = [0] * useful_life
            #TODO: add option for non-degraded generation

        if 'wind' in self.power_sources.keys():
            self.wind_degrad_gen = []
            self.wind_repair = 0
            wind_max_gen = self.max_generation.wind[0:8760]
            wind_gen = wind_max_gen
            for years in range(0,self.project_life):
                wind_gen_next = np.multiply(wind_gen, 0.985)            #Linear degredation of 1.5% per year
                check = max(wind_gen_next)
                threshold = max(np.multiply(wind_max_gen,0.8))
                if check > threshold:
                    self.wind_degrad_gen = np.append(self.wind_degrad_gen,wind_gen_next)
                    wind_gen = wind_gen_next
                else:   
                    # Add a part availability 
                    # Append pv_degrad_gen with lost generation during repair time 
                    # and set pv_gen back to pv_max_gen
                    # have counter for each time a repair is necessary
                    wind_MTTR = 7     #days
                    wind_gen_next[0:wind_MTTR*24] = 0
                    self.wind_degrad_gen = np.append(self.wind_degrad_gen,wind_gen_next)
                    wind_gen = wind_max_gen
                    self.wind_repair += 1 

        if not 'wind' in self.power_sources.keys():
            self.wind_degrad_gen = [0] * useful_life
            #TODO: add option for non-degraded generation

        self.hybrid_degraded_generation = np.add(self.pv_degrad_gen, self.wind_degrad_gen)

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
            start_year = 0
            full_SOC = 1
            for years in range(0,self.project_life):
                battery_dispatch = SimpleDispatch()
                end_year = start_year + 8760
                battery_dispatch.Nt = len(self.energy_shortfall[start_year:end_year])
                battery_dispatch.curtailment = self.combined_pv_wind_curtailment[start_year:end_year]
                battery_dispatch.shortfall = self.energy_shortfall[start_year:end_year]
                battery_dispatch.battery_storage = self.battery_storage
                battery_dispatch.charge_rate = self.battery_charge_rate
                battery_dispatch.discharge_rate = self.battery_discharge_rate
                battery_dispatch.max_SOC = full_SOC

                battery_used, excess_energy, battery_SOC = battery_dispatch.run()
                self.battery_used = np.append(self.battery_used, battery_used)
                self.excess_energy = np.append(self.excess_energy, excess_energy)
                self.battery_SOC = np.append(self.battery_SOC, battery_SOC)
                start_year += 8760
                full_SOC = battery_dispatch.max_SOC - 0.01
                #TODO: Add replacement trigger and counter. Add flag for charging and discharging to SimpleDispatch



if __name__ == '__main__': 
    from pathlib import Path
    from math import sin, pi
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
    interconnection_size_mw = 50
    battery_capacity_mw = 20
    battery_capacity_mwh = battery_capacity_mw * 4 
    useful_life = 30
    load = [40000] * useful_life * 8760

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
    prices_file = examples_dir.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
    site = SiteInfo(flatirons_site, grid_resource_file=prices_file)

    # Create model
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

    hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hybrid_plant.ppa_price = 0.1
    # hybrid_plant.pv.dc_degradation = [0] * 25
    # hybrid_plant.wind._system_model.value("env_degrad_loss", 20)
    hybrid_plant.simulate(useful_life)

    # Save the outputs
    generation_profile = hybrid_plant.generation_profile

    hybrid_degradation = Degradation(technologies, False, useful_life, generation_profile, load)

    hybrid_degradation.simulate_degradation()
    print("Number of pv repairs: ", hybrid_degradation.pv_repair)
    print("Number of wind repairs: ", hybrid_degradation.wind_repair)
    print("Non-degraded lifetime wind power generation: ", np.sum(hybrid_plant.wind.generation_profile)/1000, "[MW]")
    print("Degraded lifetime wind power generation: ", np.sum(hybrid_degradation.wind_degrad_gen)/1000, "[MW]")
    print("Battery used over lifetime: ", np.sum(hybrid_degradation.battery_used)/1000, "[MW]")

    if plot_degradation:
        plt.figure(figsize=(10,6))
        plt.subplot(311)
        plt.title("Max power generation vs degraded power generation")
        plt.plot(hybrid_degradation.wind_degrad_gen[6816:6960],label="degraded wind")
        plt.plot(hybrid_plant.wind.generation_profile[6816:6960],label="max generation")
        plt.ylabel("Power Production (kW)")
        plt.legend()
        
        plt.subplot(312)
        plt.plot(hybrid_degradation.pv_degrad_gen[6816:6960],label="degraded pv")
        plt.plot(hybrid_plant.pv.generation_profile[6816:6960],label="max generation")
        plt.ylabel("Power Production (kW)")
        plt.legend()

        plt.subplot(313)
        plt.plot(hybrid_degradation.hybrid_degraded_generation[6816:6960], label="degraded hybrid generation")
        plt.plot(load[6816:6960], label = "load profile")
        plt.ylabel("Power Production (kW)")
        plt.xlabel("Time (hour)")
        plt.legend()
        plt.show()

