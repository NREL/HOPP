import openmdao.api as om
import json
from pathlib import Path
import numpy as np

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

     
resource_files_dir = Path(__file__).parent.absolute().parent.absolute() / 'resource_files'

# Set API key
set_nrel_key_dot_env()

class HybridSystem(om.ExplicitComponent):
    """
    """
    def initialize(self):
        self.options.declare('location', default=flatirons_site)
        # self.options.declare('resource_api', default='nrel')        
        self.options.declare('battery', default=True)
        self.options.declare('grid', default=True)
        self.options.declare('sim_duration_years', default=25)
        self.options.declare('turbine_hub_ht', default=80)
        self.options.declare('hybrid_rated_power', default=100)

    def setup(self):
        # Options variables
        self.sim_duration_years = self.options['sim_duration_years']
        self.wind_turbine_hub_ht = self.options['turbine_hub_ht']
        self.hybrid_rated_power = self.options['hybrid_rated_power']
        
        # Battery inputs

        if self.options['battery']:
            self.add_input('battery_capacity_mwh', units='MW*h', val=1.)
            self.add_input('battery_power_mw', units='MW', val=1.)
        else:
            self.add_input('battery_capacity_mwh', units='MW*h', val=0.)
            self.add_input('battery_power_mw', units='MW', val=0.)

        # Electrolysis inputs

        # Fuel Cell inputs

        # Geothermal inputs

        # Grid inputs
        if self.options['grid']:
            self.add_input('interconnection_size_mw', units='MW', val=7.0)
        else:
            self.add_input('interconnection_size_mw', units='MW', val=0.)

        # H2 inputs

        # MHKWave inputs

        # MHKTidal inputs

        # NH3 inputs

        # Solar inputs
        # self.add_input('solar_size_mw', units='MW', val=0.480)
        
        # Wind inputs
        # self.add_input('wind_size_mw', units='MW', val=2.3)
        self.add_discrete_input('turbine_rating_kw', val=2300)
        self.add_input('wind_fraction', val=1.0)
      
        # Hybrid system outputs
        self.add_output('pv_npv', units='USD', val=1.)
        self.add_output('wind_npv', units='USD', val=1.)
        self.add_output('battery_npv', units='USD', val=1.)
        self.add_output('hybrid_npv', units='USD', val=1.)
        self.add_output('grid_npv', units='USD', val=1.)
        self.add_output('pv_lcoe_real', units = 'USD/kW*h', val=1.) # actually in c/kW*h TODO: figure out how to represent as c/kW*h
        self.add_output('wind_lcoe_real', units = 'USD/kW*h', val=1.) # actually in c/kW*h TODO: figure out how to represent as c/kW*h
        self.add_output('battery_lcoe_real', units = 'USD/kW*h', val=1.) # actually in c/kW*h TODO: figure out how to represent as c/kW*h
        self.add_output('hybrid_lcoe_real', units = 'USD/kW*h', val=1.) # actually in c/kW*h TODO: figure out how to represent as c/kW*h
        self.add_output('grid_lcoe_real', units = 'USD/kW*h', val=1.) # actually in c/kW*h TODO: figure out how to represent as c/kW*h
        self.add_output('pv_irr', val=1.)
        self.add_output('wind_irr', val=1.)
        self.add_output('battery_irr', val=1.)
        self.add_output('hybrid_irr', val=1.)
        self.add_output('grid_irr', val=1.)
        self.add_output('pv_pct', val=1.)
        self.add_output('wind_pct', val=1.)
        self.add_output('pv_annual_energy', units='kW', val=1.)
        self.add_output('wind_annual_energy', units='kW', val=1.)
        self.add_output('battery_annual_energy', units='kW', val=1.)
        self.add_output('hybrid_annual_energy', units='kW', val=1.)
        self.add_output('grid_annual_energy', units='kW', val=1.)
        self.add_output('pv_cost_installed', units='USD', val=1.)
        self.add_output('wind_cost_installed', units='USD', val=1.)
        self.add_output('battery_cost_installed', units='USD', val=1.)
        self.add_output('hybrid_cost_installed', units='USD', val=1.)
        self.add_output('grid_cost_installed', units='USD', val=1.)
        self.add_output('pv_total_revenues', units='USD', val=1.)
        self.add_output('wind_total_revenues', units='USD', val=1.)
        self.add_output('battery_total_revenues', units='USD', val=1.)
        self.add_output('hybrid_total_revenues', units='USD', val=1.)
        self.add_output('grid_total_revenue', units='USD', val=1.)
        self.add_output('pv_capacity_factor', val=1.)
        self.add_output('wind_capacity_factor', val=1.)
        self.add_output('hybrid_capacity_factor', val=1.)
        self.add_output('grid_capacity_factor_curtailed', val=1.)
        self.add_output('grid_capacity_factor_at_interconnect', val=1.)
        self.add_output('grid_curtailment_%', val=1.)
        self.add_output('pv_generation_profile', shape=self.sim_duration_years*8760)
        self.add_output('wind_generation_profile', shape=self.sim_duration_years*8760)
        self.add_output('hybrid_generation_profile', shape=self.sim_duration_years*8760)
        self.add_output('pv_resource_gh', shape=8760)
        self.add_output('wind_resource_speed', shape=8760)
        self.add_output('wind_resource_temp', shape=8760)
        self.add_output('wind_resource_pres', shape=8760)
        self.add_output('wind_resource_dir', shape=8760)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Set wind, solar, and interconnection capacities (in MW)
        wind_size_mw = float((self.hybrid_rated_power*inputs['wind_fraction']))
        solar_size_mw = (self.hybrid_rated_power - wind_size_mw)
        interconnection_size_mw = inputs['interconnection_size_mw']
        turbine_rating_kw = discrete_inputs['turbine_rating_kw']
        battery_capacity_mwh = inputs['battery_capacity_mwh']
        battery_power_mw = inputs['battery_power_mw']
        
        technologies = {'pv': {
                            'system_capacity_kw': solar_size_mw * 1000,
                            # 'layout_params' : of the SolarGridParameters type
                        },
                        'wind': {
                            'num_turbines': 17,
                            'turbine_rating_kw': turbine_rating_kw,
                            'hub_height': self.wind_turbine_hub_ht,
                            # 'layout_mode': 'boundarygrid' or 'grid' ,
                            # 'layout_params': 'WindBoundaryGridParameters' if 'layout_mode' = 'boundarygrid'
                        }}
        
        if self.options['battery']:
            technologies['battery'] = {
                                    'system_capacity_kwh': battery_capacity_mwh * 1000,
                                    'system_capacity_kw' : battery_power_mw * 1000
                                        }

        if self.options['grid']:
            technologies['grid'] = interconnection_size_mw
        
        # Get resource
        prices_file = resource_files_dir / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
        # resource_api = self.options['resource_api']
        location = self.options['location']
        site = SiteInfo(location, grid_resource_file=prices_file, hub_height=self.wind_turbine_hub_ht)
        
        # Create model
        
        hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)
        
        hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
        hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
        hybrid_plant.ppa_price = 0.1
        hybrid_plant.pv.dc_degradation = [0] * self.sim_duration_years
        hybrid_plant.simulate(self.sim_duration_years)
        
        # Save the outputs
        annual_energies = hybrid_plant.annual_energies
        wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.pv
        npvs = hybrid_plant.net_present_values        
        wind_installed_cost = hybrid_plant.wind.total_installed_cost
        solar_installed_cost = hybrid_plant.pv.total_installed_cost
        hybrid_installed_cost = hybrid_plant.grid.total_installed_cost
        pv_pct = (hybrid_plant.pv.system_capacity_kw / (hybrid_plant.pv.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)) * 100
        wind_pct = (hybrid_plant.wind.system_capacity_kw / (hybrid_plant.pv.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)) * 100
        
        print("Wind Installed Cost: {}".format(wind_installed_cost))
        print("Solar Installed Cost: {}".format(solar_installed_cost))
        print("Hybrid Installed Cost: {}".format(hybrid_installed_cost))
        if self.options['battery']:
            print("Battery Installed Cost: {}".format(hybrid_plant.battery.total_installed_cost))
        print("Wind NPV: {}".format(hybrid_plant.net_present_values.wind))
        print("Solar NPV: {}".format(hybrid_plant.net_present_values.pv))
        print("Hybrid NPV: {}".format(hybrid_plant.net_present_values.hybrid))
        if self.options['battery']:
            print("Battery NPV: {}".format(hybrid_plant.net_present_values.battery))
        print("Wind + Solar Expected NPV: {}".format(wind_plus_solar_npv))
        print('annual energies ', annual_energies)
        print('npvs ', npvs)
                
        outputs["pv_npv"] = hybrid_plant.net_present_values.pv
        outputs["wind_npv"] = hybrid_plant.net_present_values.wind
        if self.options['battery']:
            outputs["battery_npv"] = hybrid_plant.net_present_values.battery
        outputs["hybrid_npv"] = hybrid_plant.net_present_values.hybrid
        outputs["grid_npv"] = hybrid_plant.grid.net_present_value
        outputs["pv_lcoe_real"] = hybrid_plant.lcoe_real.pv
        outputs["wind_lcoe_real"] = hybrid_plant.lcoe_real.wind
        if self.options['battery']:
            outputs["battery_lcoe_real"] = hybrid_plant.lcoe_real.battery
        outputs["hybrid_lcoe_real"] = hybrid_plant.lcoe_real.hybrid
        outputs["grid_lcoe_real"] = hybrid_plant.grid.levelized_cost_of_energy_real
        outputs["pv_irr"] = hybrid_plant.internal_rate_of_returns.pv
        outputs["wind_irr"] = hybrid_plant.internal_rate_of_returns.wind
        if self.options['battery']:
            outputs["battery_irr"] = hybrid_plant.battery.internal_rate_of_return
        outputs["hybrid_irr"] = hybrid_plant.internal_rate_of_returns.hybrid
        outputs["grid_irr"] = hybrid_plant.grid.internal_rate_of_return
        outputs['pv_pct'] = pv_pct
        outputs['wind_pct'] = wind_pct
        outputs['pv_annual_energy'] = hybrid_plant.annual_energies.pv
        outputs['wind_annual_energy'] = hybrid_plant.annual_energies.wind
        if self.options['battery']:
            outputs['battery_annual_energy'] = hybrid_plant.annual_energies.battery
        outputs['hybrid_annual_energy'] = hybrid_plant.annual_energies.hybrid
        outputs['grid_annual_energy'] = hybrid_plant.grid.annual_energy_kwh
        outputs['pv_cost_installed'] = hybrid_plant.cost_installed.pv
        outputs['wind_cost_installed'] = hybrid_plant.cost_installed.wind
        if self.options['battery']:
            outputs['battery_cost_installed'] = hybrid_plant.cost_installed.battery
        outputs['hybrid_cost_installed'] = hybrid_plant.cost_installed.hybrid
        outputs['grid_cost_installed'] = hybrid_plant.grid.total_installed_cost
        outputs['pv_total_revenues'] = sum(hybrid_plant.total_revenues.pv)
        outputs['wind_total_revenues'] = sum(hybrid_plant.total_revenues.wind)
        if self.options['battery']:
            outputs['battery_total_revenues'] = sum(hybrid_plant.total_revenues.battery)
        outputs['hybrid_total_revenues'] = sum(hybrid_plant.total_revenues.hybrid)
        outputs['grid_total_revenue'] = sum(hybrid_plant.grid.total_revenue)
        outputs['pv_capacity_factor'] = hybrid_plant.capacity_factors.pv
        outputs['wind_capacity_factor'] = hybrid_plant.capacity_factors.wind
        outputs['hybrid_capacity_factor'] = hybrid_plant.capacity_factors.hybrid
        try:
            outputs['grid_capacity_factor_curtailed'] = hybrid_plant.grid.capacity_factor_after_curtailment
        except:
            outputs['grid_capacity_factor_at_interconnect'] = hybrid_plant.grid.capacity_factor_at_interconnect
        outputs['grid_curtailment_%'] = hybrid_plant.grid.curtailment_percent
        outputs['pv_generation_profile'] = hybrid_plant.generation_profile.pv
        outputs['wind_generation_profile'] = hybrid_plant.generation_profile.wind
        outputs['hybrid_generation_profile'] = hybrid_plant.generation_profile.hybrid
        outputs['pv_resource_gh'] = site.solar_resource._data['gh']
        outputs['wind_resource_speed'] = np.array(site.wind_resource._data['data'])[:,2]
        outputs['wind_resource_temp'] = np.array(site.wind_resource._data['data'])[:,0]
        outputs['wind_resource_pres'] = np.array(site.wind_resource._data['data'])[:,1]
        outputs['wind_resource_dir'] = np.array(site.wind_resource._data['data'])[:,3]
        

if __name__ == "__main__":
    ### build the model
    prob = om.Problem()
    prob.model.add_subsystem('hybrid_system', HybridSystem(), promotes=['*'])

    ### setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=76))
    prob.driver.options['debug_print'] = ["desvars", "objs"]
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    prob.driver.recording_options['includes'] = ['*']
    
    ### setup design variables
    # Solar DVs
    # prob.model.add_design_var('solar_size_mw', lower=0.001, upper=15.)

    ## Wind DVs
    prob.model.add_design_var('wind_fraction', lower=0.001, upper=0.999)
    # prob.model.add_design_var('wind_size_mw', lower=0., upper=15.)
    # prob.model.add_design_var('turbine_rating_kw', lower=10, upper=14000)

    ## Battery DVs
    # prob.model.add_design_var('battery_capacity_mwh', lower=0., upper=5.)
    # prob.model.add_design_var('battery_power_mw', lower=0., upper=5.) 

    ## Grid DVs
    # prob.model.add_design_var('interconnection_size_mw', lower=0., upper=5.)

    ### setup objective function
    # prob.model.add_objective('hybrid_npv', ref=1.)
    prob.model.add_objective('hybrid_lcoe_real', ref=-1.)
    # prob.model.add_objective('hybrid_irr', ref=1.)
    
    # prob.model.approx_totals()
    
    prob.setup()
    prob.run_driver()
    
    prob.model.list_outputs()
    

    
    