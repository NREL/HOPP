from pathlib import Path
from typing import Union

import numpy as np
from scipy.stats import pearsonr
import PySAM.GenericSystem as GenericSystem

from tools.analysis import create_cost_calculator
from hybrid.sites import SiteInfo
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
from hybrid.storage import Battery
from hybrid.dispatch import HybridDispatch
from hybrid.grid import Grid
from hybrid.reopt import REopt
from hybrid.layout.hybrid_layout import HybridLayout
from hybrid.log import hybrid_logger as logger


class HybridSimulationOutput:
    def __init__(self, power_sources):
        self.power_sources = power_sources
        self.hybrid = 0
        self.grid = 0
        self.solar = 0
        self.wind = 0
        self.battery = 0

    def create(self):
        return HybridSimulationOutput(self.power_sources)

    def __repr__(self):
        conts = ""
        if 'solar' in self.power_sources.keys():
            conts += "solar: " + str(self.solar) + ", "
        if 'wind' in self.power_sources.keys():
            conts += "wind: " + str(self.wind) + ", "
        if 'battery' in self.power_sources.keys():
            conts += "battery: " + str(self.battery) + ", "
        conts += "hybrid: " + str(self.hybrid)
        return conts


class HybridSimulation:
    hybrid_system: GenericSystem.GenericSystem

    def __init__(self, power_sources: dict, site: SiteInfo, interconnect_kw: float):
        """
        Base class for simulating a hybrid power plant.

        Can be derived to add other sizing methods, financial analyses,
            methods for pre- or post-processing, etc

        :param power_sources: tuple of strings, float pairs
            names of power sources to include and their kw sizes
            choices include:
                    ('solar', 'wind', 'geothermal', 'battery')
        :param site: Site
            layout, location and resource data
        :param interconnect_kw: float
            power limit of interconnect for the site
        """
        self._fileout = Path.cwd() / "results"
        self.site = site
        self.interconnect_kw = interconnect_kw

        self.power_sources = dict()
        self.solar: Union[SolarPlant, None] = None
        self.wind: Union[WindPlant, None] = None
        self.battery: Union[Battery, None] = None
        self.dispatch: Union[HybridDispatch, None] = None
        self.grid: Union[Grid, None] = None

        for k in power_sources.keys():
            power_sources[k.lower()] = power_sources.pop(k)

        if 'solar' in power_sources.keys():
            self.solar = SolarPlant(self.site, power_sources['solar'])
            self.power_sources['solar'] = self.solar
            logger.info("Created HybridSystem.solar with system size {} mW".format(power_sources['solar']))
        if 'wind' in power_sources.keys():
            self.wind = WindPlant(self.site, power_sources['wind'])
            self.power_sources['wind'] = self.wind
            logger.info("Created HybridSystem.wind with system size {} mW".format(power_sources['wind']))
        if 'geothermal' in power_sources.keys():
            raise NotImplementedError("Geothermal plant not yet implemented")
        if 'battery' in power_sources.keys():
            self.battery = Battery(self.site, power_sources['battery'] * 1000)
            self.power_sources['battery'] = self.battery
            logger.info("Created HybridSystem.battery with system capacity {} mWh".format(power_sources['battery']))

        self.layout = HybridLayout(self.site, self.power_sources)

        # Default cost calculator, can be overwritten
        self.cost_model = create_cost_calculator(self.interconnect_kw)

        # performs interconnection and curtailment energy limits
        self.grid = Grid(self.site, interconnect_kw)

        self.outputs_factory = HybridSimulationOutput(power_sources)

        if len(self.site.elec_prices.data):
            self.dispatch_factors = self.site.elec_prices.data

    def setup_cost_calculator(self, cost_calculator: object):
        if hasattr(cost_calculator, "calculate_total_costs"):
            self.cost_model = cost_calculator

    @property
    def ppa_price(self):
        return self.grid.ppa_price

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        for tech, _ in self.power_sources.items():
            if hasattr(self, tech):
                getattr(self, tech).ppa_price = ppa_price
        self.grid.ppa_price = ppa_price

    @property
    def dispatch_factors(self):
        return self.grid.dispatch_factors

    @dispatch_factors.setter
    def dispatch_factors(self, dispatch_factors):
        for tech, _ in self.power_sources.items():
            if hasattr(self, tech):
                getattr(self, tech).dispatch_factors = dispatch_factors
        self.grid.dispatch_factors = dispatch_factors

    @property
    def discount_rate(self):
        return self.grid.get_variable("real_discount_rate")

    @discount_rate.setter
    def discount_rate(self, discount_rate):
        for k, _ in self.power_sources.items():
            if hasattr(self, k):
                getattr(self, k).set_variable("real_discount_rate", discount_rate)
        self.grid.set_variable("real_discount_rate", discount_rate)

    def set_om_costs_per_kw(self, solar_om_per_kw=None, wind_om_per_kw=None, hybrid_om_per_kw=None):
        if solar_om_per_kw and wind_om_per_kw and hybrid_om_per_kw:
            if len(solar_om_per_kw) != len(wind_om_per_kw) != len(hybrid_om_per_kw):
                raise ValueError("Length of yearly om cost per kw arrays must be equal.")

        if solar_om_per_kw and self.solar:
            self.solar.om_capacity = solar_om_per_kw

        if wind_om_per_kw and self.wind:
            self.wind.om_capacity = wind_om_per_kw

        if hybrid_om_per_kw:
            self.grid.om_capacity = hybrid_om_per_kw

    def size_from_reopt(self):
        """
        Calls ReOpt API for optimal sizing with system parameters for each power source.
        :return:
        """
        if not self.site.urdb_label:
            raise ValueError("REopt run requires urdb_label")
        reopt = REopt(lat=self.site.lat,
                      lon=self.site.lon,
                      interconnection_limit_kw=self.interconnect_kw,
                      load_profile=[0] * 8760,
                      urdb_label=self.site.urdb_label,
                      solar_model=self.solar,
                      wind_model=self.wind,
                      fin_model=self.grid._financial_model,
                      fileout=str(self._fileout / "REoptResult.json"))
        results = reopt.get_reopt_results(force_download=False)
        wind_size_kw = results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"]
        self.wind.system_capacity_closest_fit(wind_size_kw)

        solar_size_kw = results["outputs"]["Scenario"]["Site"]["PV"]["size_kw"]
        self.solar.system_capacity_kw = solar_size_kw
        logger.info("HybridSystem set system capacities to REopt output")

    def calculate_installed_cost(self):
        if not self.cost_model:
            raise RuntimeError("'calculate_installed_cost' called before 'setup_cost_calculator'.")

        wind_mw = 0
        solar_mw = 0
        if self.solar:
            solar_mw = self.solar.system_capacity_kw / 1000
        if self.wind:
            wind_mw = self.wind.system_capacity_kw / 1000

        solar_cost, wind_cost, total_cost = self.cost_model.calculate_total_costs(wind_mw, solar_mw)
        if self.solar:
            self.solar.total_installed_cost = solar_cost
        if self.wind:
            self.wind.total_installed_cost = wind_cost

        self.grid.total_installed_cost = total_cost
        logger.info("HybridSystem set hybrid total installed cost to to {}".format(total_cost))

    def calculate_financials(self):
        """
        prepare financial parameters from individual power plants for total performance and financial metrics
        """
        # TODO: need to make financial parameters consistent

        # TODO: generalize this for different plants besides wind and solar
        hybrid_size_kw = sum([v.system_capacity_kw for v in self.power_sources.values()])
        solar_percent = 0
        wind_percent = 0
        solar_financing_cost = 0
        wind_financing_cost = 0
        if self.solar:
            solar_percent = self.solar.system_capacity_kw / hybrid_size_kw
            solar_financing_cost = self.solar.get_construction_financing_cost()
        if self.wind:
            wind_percent = self.wind.system_capacity_kw / hybrid_size_kw
            wind_financing_cost = self.wind.get_construction_financing_cost()

        def average_cost(var_name):
            hybrid_avg = 0
            if self.solar:
                hybrid_avg += np.array(self.solar.get_variable(var_name)) * solar_percent
            if self.wind:
                hybrid_avg += np.array(self.wind.get_variable(var_name)) * wind_percent
            self.grid.set_variable(var_name, hybrid_avg)
            return hybrid_avg

        # FinancialParameters
        hybrid_construction_financing_cost = wind_financing_cost + solar_financing_cost

        self.grid.set_construction_financing_cost_per_kw(hybrid_construction_financing_cost / hybrid_size_kw)
        average_cost("debt_percent")

        # O&M Cost Averaging
        average_cost("om_capacity")
        average_cost("om_fixed")
        average_cost("om_fuel_cost")
        average_cost("om_production")
        average_cost("om_replacement_cost1")

        average_cost("cp_system_nameplate")

        # Tax Incentives
        average_cost("ptc_fed_amount")
        average_cost("ptc_fed_escal")
        average_cost("itc_fed_amount")
        average_cost("itc_fed_percent")
        self.grid.set_variable("ppa_soln_mode", 1)

        # Depreciation, copy from solar for now
        if self.solar:
            self.grid._financial_model.Depreciation.assign(self.solar._financial_model.Depreciation.export())

        average_cost("degradation")

    def simulate(self,
                 project_life: int = 25,
                 is_simple_battery_dispatch: bool = True,
                 is_test: bool = False):
        """
        Runs the individual system models then combines the financials
        :return:
        """
        # TODO: Add battery installed_cost and financials
        self.calculate_installed_cost()
        self.calculate_financials()

        hybrid_size_kw = 0
        total_gen = [0] * self.site.n_timesteps
        if self.solar:
            hybrid_size_kw += self.solar.system_capacity_kw
            self.solar.simulate(project_life)

            gen = self.solar.generation_profile
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        if self.wind:
            hybrid_size_kw += self.wind.system_capacity_kw
            self.wind.simulate(project_life)
            gen = self.wind.generation_profile
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        if self.battery:
            """
            Run dispatch optimization
            """
            self.dispatch = HybridDispatch(self, is_simple_battery_dispatch=is_simple_battery_dispatch)
            self.dispatch.simulate(is_test=is_test)
            gen = self.battery.generation_profile()
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        self.grid.generation_profile_from_system = total_gen
        self.grid.system_capacity_kw = hybrid_size_kw

        self.grid.simulate(project_life)

    @property
    def annual_energies(self):
        aep = self.outputs_factory.create()
        if self.solar:
            aep.solar = self.solar.annual_energy_kw
        if self.wind:
            aep.wind = self.wind.annual_energy_kw
        if self.battery:
            aep.battery = sum(self.battery.Outputs.gen)
        aep.grid = sum(self.grid.generation_profile)
        aep.hybrid = aep.solar + aep.wind + aep.battery
        return aep

    @property
    def generation_profile(self):
        gen = self.outputs_factory.create()
        if self.solar:
            gen.solar = self.solar.generation_profile
        if self.wind:
            gen.wind = self.wind.generation_profile
        gen.grid = self.grid.generation_profile
        gen.hybrid = list(gen.solar) + list(gen.wind)
        return gen

    @property
    def capacity_factors(self):
        cf = self.outputs_factory.create()
        if self.solar:
            cf.solar = self.solar.capacity_factor
        if self.wind:
            cf.wind = self.wind.capacity_factor
        try:
            cf.grid = self.grid.capacity_factor_after_curtailment
        except:
            cf.grid = self.grid.capacity_factor_at_interconnect
        cf.hybrid = (self.solar.annual_energy_kw + self.wind.annual_energy_kw) \
                    / (self.solar.system_capacity_kw + self.wind.system_capacity_kw) / 87.6
        return cf

    @property
    def net_present_values(self):
        npv = self.outputs_factory.create()
        if self.solar:
            npv.solar = self.solar.net_present_value
        if self.wind:
            npv.wind = self.wind.net_present_value
        npv.hybrid = self.grid.net_present_value
        return npv

    @property
    def internal_rate_of_returns(self):
        irr = self.outputs_factory.create()
        if self.solar:
            irr.solar = self.solar.internal_rate_of_return
        if self.wind:
            irr.wind = self.wind.internal_rate_of_return
        irr.hybrid = self.grid.internal_rate_of_return
        return irr

    @property
    def lcoe_real(self):
        lcoes_real = self.outputs_factory.create()
        if self.solar:
            lcoes_real.solar = self.solar.levelized_cost_of_energy_real
        if self.wind:
            lcoes_real.wind = self.wind.levelized_cost_of_energy_real
        lcoes_real.hybrid = self.grid.levelized_cost_of_energy_real
        return lcoes_real

    @property
    def lcoe_nom(self):
        lcoes_nom = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            lcoes_nom.solar = self.solar.levelized_cost_of_energy_nominal
        if self.wind and self.wind.system_capacity_kw > 0:
            lcoes_nom.wind = self.wind.levelized_cost_of_energy_nominal
        lcoes_nom.hybrid = self.grid.levelized_cost_of_energy_nominal
        return lcoes_nom

    def hybrid_outputs(self):
        outputs = dict()
        # outputs['Lat'] = self.site.lat
        # outputs['Lon'] = self.site.lon
        # outputs['PPA Price'] = self.hybrid_financial.Revenue.ppa_price_input[0]
        outputs['Solar (MW)'] = self.solar.system_capacity_kw / 1000
        outputs['Wind (MW)'] = self.wind.system_capacity_kw / 1000
        solar_pct = self.solar.system_capacity_kw / (self.solar.system_capacity_kw + self.wind.system_capacity_kw)
        wind_pct = self.wind.system_capacity_kw / (self.solar.system_capacity_kw + self.wind.system_capacity_kw)
        outputs['Solar (%)'] = solar_pct * 100
        outputs['Wind (%)'] = wind_pct * 100

        annual_energies = self.annual_energies
        outputs['Solar AEP (GWh)'] = annual_energies.solar / 1000000
        outputs['Wind AEP (GWh)'] = annual_energies.wind / 1000000
        outputs["AEP (GWh)"] = annual_energies.hybrid / 1000000

        capacity_factors = self.capacity_factors
        outputs['Solar Capacity Factor'] = capacity_factors.solar
        outputs['Wind Capacity Factor'] = capacity_factors.wind
        outputs["Capacity Factor"] = capacity_factors.hybrid
        outputs['Capacity Factor of Interconnect'] = capacity_factors.grid

        outputs['Percentage Curtailment'] = self.grid.curtailment_percent

        outputs["BOS Cost"] = self.grid.total_installed_cost
        outputs['BOS Cost percent reduction'] = 0
        outputs["Cost / MWh Produced"] = outputs["BOS Cost"] / (outputs['AEP (GWh)'] * 1000)

        outputs["NPV ($-million)"] = self.net_present_values.hybrid / 1000000
        outputs['IRR (%)'] = self.internal_rate_of_returns.hybrid
        outputs['PPA Price Used'] = self.grid.ppa_price[0]

        outputs['LCOE - Real'] = self.lcoe_real.hybrid
        outputs['LCOE - Nominal'] = self.lcoe_nom.hybrid

        # time series dispatch
        if self.grid.get_variable('ppa_multiplier_model') == 1:
            outputs['Revenue (TOD)'] = sum(self.grid.total_revenue)
            outputs['Revenue (PPA)'] = outputs['TOD Profile Used'] = 0

        outputs['Cost / MWh Produced percent reduction'] = 0

        if solar_pct * wind_pct > 0:
            outputs['Pearson R Wind V Solar'] = pearsonr(self.solar.generation_profile[0:8760],
                                                         self.wind.generation_profile[0:8760])[0]

        return outputs

    def copy(self):
        """

        :return: a clone
        """
        # TODO implement deep copy

    def plot_layout(self,
                    figure=None,
                    axes=None,
                    wind_color='b',
                    solar_color='darkorange',
                    site_border_color='k',
                    site_alpha=0.95,
                    linewidth=4.0
                    ):
        self.layout.plot(figure, axes, wind_color, solar_color, site_border_color, site_alpha, linewidth)