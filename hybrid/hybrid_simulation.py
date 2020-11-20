from collections.abc import Iterable
from pathlib import Path
from typing import Union

import numpy as np
from scipy.stats import pearsonr
import PySAM.GenericSystem as GenericSystem

from hybrid.sites import SiteInfo
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
from hybrid.grid import Grid
from hybrid.reopt import REopt

from hybrid.log import hybrid_logger as logger


class HybridSimulationOutput:
    def __init__(self, power_sources):
        self.power_sources = power_sources
        self.hybrid = 0
        self.grid = 0
        if 'solar' in power_sources.keys():
            self.solar = 0
        if 'wind' in power_sources.keys():
            self.wind = 0

    def create(self):
        return HybridSimulationOutput(self.power_sources)

    def __repr__(self):
        conts = ""
        if 'solar' in self.power_sources.keys():
            conts += "Solar: " + str(self.solar)
        if 'wind' in self.power_sources.keys():
            conts += ", "
            conts += "Wind: " + str(self.wind) + ", "
        conts += "Hybrid: " + str(self.hybrid)
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
        self.grid: Union[Grid, None] = None

        if 'solar' in power_sources.keys():
            self.solar = SolarPlant(self.site, power_sources['solar'] * 1000)
            self.power_sources['solar'] = self.solar
            logger.info("Created HybridSystem.solar with system size {} mW".format(power_sources['solar']))
        if 'wind' in power_sources.keys():
            self.wind = WindPlant(self.site, power_sources['wind'] * 1000)
            self.power_sources['wind'] = self.wind
            logger.info("Created HybridSystem.wind with system size {} mW".format(power_sources['wind']))
        if 'geothermal' in power_sources.keys():
            raise NotImplementedError("Geothermal plant not yet implemented")
        if 'battery' in power_sources:
            raise NotImplementedError("Battery not yet implemented")

        self.cost_model = None

        # performs interconnection and curtailment energy limits
        self.grid = Grid(self.site, interconnect_kw)

        self.outputs_factory = HybridSimulationOutput(power_sources)

    def setup_cost_calculator(self, cost_calculator: object):
        if hasattr(cost_calculator, "calculate_total_costs"):
            self.cost_model = cost_calculator

    @property
    def ppa_price(self):
        return self.grid.financial_model.Revenue.ppa_price_input

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        if not isinstance(ppa_price, Iterable):
            ppa_price = (ppa_price, )
        for k, _ in self.power_sources.items():
            if hasattr(self, k):
                getattr(self, k).financial_model.Revenue.ppa_price_input = ppa_price
        self.grid.financial_model.Revenue.ppa_price_input = ppa_price

    @property
    def discount_rate(self):
        return self.grid.financial_model.FinancialParameters.real_discount_rate

    @discount_rate.setter
    def discount_rate(self, discount_rate):
        for k, _ in self.power_sources.items():
            if hasattr(self, k):
                getattr(self, k).financial_model.FinancialParameters.real_discount_rate = discount_rate
        self.grid.financial_model.FinancialParameters.real_discount_rate = discount_rate

    def set_om_costs_per_kw(self, solar_om_per_kw=None, wind_om_per_kw=None, hybrid_om_per_kw=None):
        if solar_om_per_kw and wind_om_per_kw and hybrid_om_per_kw:
            if len(solar_om_per_kw) != len(wind_om_per_kw) != len(hybrid_om_per_kw):
                raise ValueError("Length of yearly om cost per kw arrays must be equal.")

        if solar_om_per_kw and self.solar:
            self.solar.financial_model.SystemCosts.om_capacity = solar_om_per_kw

        if wind_om_per_kw and self.wind:
            self.wind.financial_model.SystemCosts.om_capacity = wind_om_per_kw

        if hybrid_om_per_kw:
            self.grid.financial_model.SystemCosts.om_capacity = hybrid_om_per_kw

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
                      fin_model=self.grid.financial_model,
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

        solar_cost, wind_cost, total_cost = self.cost_model.calculate_total_costs(self.wind.system_capacity_kw / 1000,
                                                                                  self.solar.system_capacity_kw / 1000)
        if self.solar:
            self.solar.set_total_installed_cost_dollars(solar_cost)
        if self.wind:
            self.wind.set_total_installed_cost_dollars(wind_cost)

        self.grid.set_total_installed_cost_dollars(total_cost)
        logger.info("HybridSystem set hybrid total installed cost to to {}".format(total_cost))

    def calculate_financials(self):
        """
        prepare financial parameters from individual power plants for total performance and financial metrics
        """
        # TODO: need to make financial parameters consistent

        # TODO: generalize this for different plants besides wind and solar
        hybrid_size_kw = sum([v.system_capacity_kw for v in self.power_sources.values()])
        solar_percent = self.solar.system_capacity_kw / hybrid_size_kw
        wind_percent = self.wind.system_capacity_kw / hybrid_size_kw

        def average_cost(var_name):
            hybrid_avg = 0
            if self.solar:
                hybrid_avg += np.array(self.solar.financial_model.value(var_name)) * solar_percent
            if self.wind:
                hybrid_avg += np.array(self.wind.financial_model.value(var_name)) * wind_percent
            self.grid.financial_model.value(var_name, hybrid_avg)
            return hybrid_avg

        # FinancialParameters
        hybrid_construction_financing_cost = self.wind.get_construction_financing_cost() + \
                                             self.solar.get_construction_financing_cost()
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
        self.grid.financial_model.Revenue.ppa_soln_mode = 1

        # Depreciation, copy from solar for now
        self.grid.financial_model.Depreciation.assign(self.solar.financial_model.Depreciation.export())

        average_cost("degradation")

    def simulate(self, project_life: int = 25):
        """
        Runs the individual system models then combines the financials
        :return:
        """
        self.calculate_installed_cost()
        self.calculate_financials()

        hybrid_size_kw = 0
        total_gen = [0] * self.site.n_timesteps
        if self.solar.system_capacity_kw > 0:
            hybrid_size_kw += self.solar.system_capacity_kw
            self.solar.simulate(project_life)

            gen = self.solar.generation_profile()
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        if self.wind.system_capacity_kw > 0:
            hybrid_size_kw += self.wind.system_capacity_kw
            self.wind.simulate(project_life)
            gen = self.wind.generation_profile()
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        self.grid.generation_profile_from_system = total_gen
        self.grid.financial_model.SystemOutput.system_capacity = hybrid_size_kw

        self.grid.simulate(project_life)

    @property
    def annual_energies(self):
        aep = self.outputs_factory.create()
        if self.solar.system_capacity_kw > 0:
            aep.solar = self.solar.system_model.Outputs.annual_energy
        if self.wind.system_capacity_kw > 0:
            aep.wind = self.wind.system_model.Outputs.annual_energy
        aep.grid = sum(self.grid.system_model.Outputs.gen)
        aep.hybrid = aep.solar + aep.wind
        return aep

    @property
    def capacity_factors(self):
        cf = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            cf.solar = self.solar.system_model.Outputs.capacity_factor
        if self.wind and self.wind.system_capacity_kw > 0:
            cf.wind = self.wind.system_model.Outputs.capacity_factor
        try:
            cf.grid = self.grid.system_model.Outputs.capacity_factor_curtailment_ac
        except:
            cf.grid = self.grid.system_model.Outputs.capacity_factor_interconnect_ac
        cf.hybrid = (self.solar.annual_energy_kw() + self.wind.annual_energy_kw()) \
                    / (self.solar.system_capacity_kw + self.wind.system_capacity_kw) / 87.6
        return cf

    @property
    def net_present_values(self):
        npv = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            npv.solar = self.solar.financial_model.Outputs.project_return_aftertax_npv
        if self.wind and self.wind.system_capacity_kw > 0:
            npv.wind = self.wind.financial_model.Outputs.project_return_aftertax_npv
        npv.hybrid = self.grid.financial_model.Outputs.project_return_aftertax_npv
        return npv

    @property
    def internal_rate_of_returns(self):
        irr = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            irr.solar = self.solar.financial_model.Outputs.project_return_aftertax_irr
        if self.wind and self.wind.system_capacity_kw > 0:
            irr.wind = self.wind.financial_model.Outputs.project_return_aftertax_irr
        irr.hybrid = self.grid.financial_model.Outputs.project_return_aftertax_irr
        return irr

    @property
    def lcoe_real(self):
        lcoes_real = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            lcoes_real.solar = self.solar.financial_model.Outputs.lcoe_real
        if self.wind and self.wind.system_capacity_kw > 0:
            lcoes_real.wind = self.wind.financial_model.Outputs.lcoe_real
        lcoes_real.hybrid = self.grid.financial_model.Outputs.lcoe_real
        return lcoes_real

    @property
    def lcoe_nom(self):
        lcoes_nom = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            lcoes_nom.solar = self.solar.financial_model.Outputs.lcoe_nom
        if self.wind and self.wind.system_capacity_kw > 0:
            lcoes_nom.wind = self.wind.financial_model.Outputs.lcoe_nom
        lcoes_nom.hybrid = self.grid.financial_model.Outputs.lcoe_nom
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

        outputs['Percentage Curtailment'] = (self.grid.system_model.Outputs.annual_ac_curtailment_loss_percent +
                                                 self.grid.system_model.Outputs.annual_ac_interconnect_loss_percent)

        outputs["BOS Cost"] = self.grid.financial_model.SystemCosts.total_installed_cost
        outputs['BOS Cost percent reduction'] = 0
        outputs["Cost / MWh Produced"] = outputs["BOS Cost"] / (outputs['AEP (GWh)'] * 1000)

        outputs["NPV ($-million)"] = self.net_present_values.hybrid / 1000000
        outputs['IRR (%)'] = self.internal_rate_of_returns.hybrid
        outputs['PPA Price Used'] = self.grid.financial_model.Revenue.ppa_price_input[0]

        outputs['LCOE - Real'] = self.lcoe_real.hybrid
        outputs['LCOE - Nominal'] = self.lcoe_nom.hybrid

        # time series dispatch
        if self.grid.financial_model.Revenue.ppa_multiplier_model == 1:
            outputs['Revenue (TOD)'] = sum(self.grid.financial_model.Outputs.cf_total_revenue)
            outputs['Revenue (PPA)'] = outputs['TOD Profile Used'] = 0

        outputs['Cost / MWh Produced percent reduction'] = 0

        if solar_pct * wind_pct > 0:
            outputs['Pearson R Wind V Solar'] = pearsonr(self.solar.system_model.Outputs.gen[0:8760],
                                                        self.wind.system_model.Outputs.gen[0:8760])[0]

        return outputs

    def copy(self):
        """

        :return: a clone
        """
        # TODO implement deep copy

