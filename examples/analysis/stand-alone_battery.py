import pandas as pd

from matplotlib.pyplot import plot
from hybrid.sites import SiteInfo
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_generation

def print_tech_output(hybrid: HybridSimulation, tech: str):
    if tech != 'hybrid':
        if not getattr(hybrid, tech):
            return

    print(tech + ":")
    print("\tEnergy (year 1) [GWh]: {:.3f}".format(getattr(hybrid.annual_energies, tech)/1e6))
    print("\tCapacity Factor [%]: {:.2f}".format(getattr(hybrid.capacity_factors, tech)))
    if tech == 'hybrid':
        print("\tInstalled Cost [$M]: {:.2f}".format(getattr(hybrid,'grid').total_installed_cost/1e6))
    else:
        print("\tInstalled Cost [$M]: {:.2f}".format(getattr(hybrid,tech).total_installed_cost/1e6))
    print("\tNPV [$M]: {:.3f}".format(getattr(hybrid.net_present_values, tech)/1e6))
    print("\tLCOE (nominal) [$/MWh]: {:.2f}".format(getattr(hybrid.lcoe_nom, tech)*10))
    print("\tLCOE (real) [$/MWh]: {:.2f}".format(getattr(hybrid.lcoe_real, tech)*10))
    print("\tIRR [%]: {:.2f}".format(getattr(hybrid.internal_rate_of_returns, tech)))
    print("\tBenefit Cost Ratio [-]: {:.2f}".format(getattr(hybrid.benefit_cost_ratios, tech)))
    print("\tCapacity credit [%]: {:.2f}".format(getattr(hybrid.capacity_credit_percent, tech)))
    print("\tCapacity payment (year 1): {:.2f}".format(getattr(hybrid.capacity_payments, tech)[1]))
    if tech == 'hybrid':
        print("\tCurtailment percentage [%]: {:.2f}".format(hybrid.grid.curtailment_percent))

def print_hybrid_output(hybrid: HybridSimulation):
    # TODO: Create a function to print this as a table
    techs = ['tower', 'trough', 'pv', 'battery', 'hybrid']
    for t in techs:
        print_tech_output(hybrid, t)

    if hybrid.site.follow_desired_schedule:
        print("\tMissed load [MWh]: {:.2f}".format(sum(hybrid.grid.missed_load[0:8760])/1.e3))
        print("\tMissed load percentage [%]: {:.2f}".format(hybrid.grid.missed_load_percentage*100.0))
        print("\tSchedule curtailed [MWh]: {:.2f}".format(sum(hybrid.grid.schedule_curtailed[0:8760])/1.e3))
        print("\tSchedule curtailed percentage [%]: {:.2f}".format(hybrid.grid.schedule_curtailed_percentage*100.0))

if __name__ == "__main__":
    plot_generation = True
    is_test = False

    # Set plant location
    site_data = {
        "lat": 34.8653,
        "lon": -116.7830,
        "elev": 561,
        "tz": 1,
        }

    prices_file = "./resource_files/grid/LCGS_price_per_MWh.csv"
    site = SiteInfo(site_data, grid_resource_file=prices_file)

    battery_mw = 100
    hours_of_storage = 8
    technologies = {'battery': {'system_capacity_kwh': battery_mw * hours_of_storage * 1.2 * 1000,
                                'system_capacity_kw': battery_mw * 1000}}
    
    hybrid_plant = HybridSimulation(technologies,
                                    site,
                                    interconnect_kw= battery_mw * 1000,
                                    dispatch_options={
                                        'is_test_start_year': is_test,
                                        'is_test_end_year': is_test,
                                        'solver': 'xpress',
                                        'grid_charging': True
                                        })

    # hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.0

    hybrid_plant.simulate()

    if plot_generation:
        figure_dir = "./examples/analysis/figures/"
        for d in range(0, 365, 5):
            plot_battery_generation(hybrid_plant, start_day=d, plot_filename=figure_dir + "battery_gen_{}.png".format(d))
            plot_battery_output(hybrid_plant, start_day=d, plot_filename=figure_dir + "battery_output_{}.png".format(d))

    print_hybrid_output(hybrid_plant)
    hybrid_plant.hybrid_simulation_outputs(filename= "./examples/analysis/stand-alone_battery_outputs.csv")

    print("Battery Annual Energy: {} [GWh]".format(hybrid_plant.battery.annual_energy_kwh/1e6))

    gen_year_1 = hybrid_plant.grid.generation_profile[0:hybrid_plant.site.n_timesteps]
    discharge_year_1 = [dc if dc > 0 else 0 for dc in gen_year_1]
    charge_year_1 = [c if c < 0 else 0 for c in gen_year_1]

    print("Discharge Energy: {:.3f} [GWh]".format(sum(discharge_year_1)/1e6))
    print("Charge Energy: {:.3f} [GWh]".format(-sum(charge_year_1)/1e6))
    print("Round-trip Efficiency: {:.4f} [-]".format(-sum(discharge_year_1)/sum(charge_year_1)))

    batt_dict = {"Generation": gen_year_1,
                 "Discharge": discharge_year_1,
                 "Charge": charge_year_1}

    df = pd.DataFrame(batt_dict)
    df.to_csv("./examples/analysis/battery_gen.csv")
    pass