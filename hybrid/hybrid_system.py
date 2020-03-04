import logging

from pathlib import Path
from typing import Union

import PySAM.GenericSystem as GenericSystem

from hybrid.site_info import SiteInfo
from hybrid.solar_source import SolarPlant, Singleowner
from hybrid.wind_source import WindPlant
from hybrid.grid import Grid
from hybrid.reopt import REopt

import numpy as np

logger = logging.getLogger('hybrid_system')


class HybridSystemOutput:
    def __init__(self, power_sources):
        self.power_sources = power_sources
        self.Hybrid = 0
        if 'Solar' in power_sources.keys():
            self.Solar = 0
        if 'Wind' in power_sources.keys():
            self.Wind = 0
        if 'Grid' in power_sources.keys():
            self.Grid = 0

    def create(self):
        return HybridSystemOutput(self.power_sources)


class HybridSystem:
    hybrid_system: GenericSystem.GenericSystem

    def __init__(self, power_sources: dict, site: SiteInfo, interconnect_kw: float):
        """
        Base class for simulating a hybrid power plant.

        Can be derived to add other sizing methods, financial analyses,
            methods for pre- or post-processing, etc

        :param power_sources: tuple of strings, float pairs
            names of power sources to include and their kw sizes
            choices include:
                    ('Solar', 'Wind', 'Geothermal', 'Battery')
        :param site: Site
            layout, location and resource data
        :param interconnect_kw: float
            power limit of interconnect for the site
        """
        self._fileout = Path.cwd() / "results"
        self.site = site
        self.interconnect_size = interconnect_kw

        self.systems = []
        self.solar: Union[SolarPlant, None] = None
        self.wind: Union[WindPlant, None] = None
        self.grid: Union[Grid, None] = None

        if 'Solar' in power_sources.keys():
            self.solar = SolarPlant(self.site, power_sources['Solar'] * 1000)
            logger.info("Created HybridSystem.solar with system size {} mW".format(power_sources['Solar']))
        if 'Wind' in power_sources.keys():
            self.wind = WindPlant(self.site, power_sources['Wind'] * 1000)
            logger.info("Created HybridSystem.wind with system size {} mW".format(power_sources['Wind']))
        if 'Geothermal' in power_sources.keys():
            raise ValueError("Geothermal plant not yet implemented")
        if 'Battery' in power_sources:
            raise ValueError("Battery not yet implemented")

        self.hybrid_financial: Singleowner.Singleowner = Singleowner.default("GenericSystemSingleOwner")
        logger.info("Created HybridSystem.hybrid_financial")

        # hybrid will postprocess outputs from individual power plants for total performance and financial metrics
        self.setup_hybrid_financials(power_sources)

        # performs interconnection and curtailment energy limits
        if 'Grid' in power_sources.keys():
            self.grid = Grid(self.site, power_sources['Grid'] * 1000)
            self.grid.financial_model.assign(self.hybrid_financial.export())
            logger.info("Created HybridSystem.grid with interconnect size {} mW".format(power_sources['Grid']))

        self.outputs_factory = HybridSystemOutput(power_sources)

    def setup_hybrid_financials(self, power_sources: dict):
        # TODO: generalize this for different plants besides wind and solar
        # TODO: need to make financial parameters consistent
        # TODO: O&M costs
        self.hybrid_financial.TaxCreditIncentives.ptc_fed_amount = [0]
        self.hybrid_financial.TaxCreditIncentives.itc_fed_amount = 0
        self.hybrid_financial.Revenue.ppa_soln_mode = 1

        hybrid_size_kw = 0

        if self.solar:
            hybrid_size_kw += self.solar.system_capacity_kw
        if self.wind:
            hybrid_size_kw += self.wind.system_capacity_kw

        hybrid_ptc = 0
        hybrid_itc = 0
        hybrid_construction_financing = 0
        if self.solar:
            annual_energy_percent = self.solar.system_capacity_kw / hybrid_size_kw
            hybrid_ptc += self.solar.financial_model.TaxCreditIncentives.ptc_fed_amount[0] * annual_energy_percent
            hybrid_ptc += self.solar.financial_model.TaxCreditIncentives.itc_fed_amount * annual_energy_percent
            hybrid_construction_financing += self.solar.construction_financing_cost_per_kw * self.solar.system_capacity_kw

        if self.wind:
            annual_energy_percent = self.wind.system_capacity_kw / hybrid_size_kw
            hybrid_ptc += self.wind.financial_model.TaxCreditIncentives.ptc_fed_amount[0] * annual_energy_percent
            hybrid_ptc += self.wind.financial_model.TaxCreditIncentives.itc_fed_amount * annual_energy_percent
            hybrid_construction_financing += self.wind.construction_financing_cost_per_kw * self.wind.system_capacity_kw

        self.hybrid_financial.TaxCreditIncentives.ptc_fed_amount = [hybrid_ptc]
        self.hybrid_financial.TaxCreditIncentives.itc_fed_percent = hybrid_itc
        self.hybrid_financial.FinancialParameters.construction_financing_cost = hybrid_construction_financing
        logger.info("HybridSystem set hybrid financial federal PTC amount to {}".format(hybrid_ptc))
        logger.info("HybridSystem set hybrid financial federal ITC amount to {}".format(hybrid_itc))
        logger.info("HybridSystem set hybrid financial construction_financing_cost to {}".format(hybrid_construction_financing))

    def size_from_reopt(self):
        """
        Calls ReOpt API for optimal sizing with system parameters for each power source.
        :return:
        """
        if not self.site.urdb_label:
            raise ValueError("REopt run requires urdb_label")
        reopt = REopt(lat=self.site.lat,
                      lon=self.site.lon,
                      interconnection_limit_kw=self.interconnect_size,
                      load_profile=[0] * 8760,
                      urdb_label=self.site.urdb_label,
                      solar_model=self.solar,
                      wind_model=self.wind,
                      fin_model=self.hybrid_financial,
                      fileout=str(self._fileout / "REoptResult.json"))
        results = reopt.get_reopt_results(force_download=False)
        wind_size_kw = results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"]
        self.wind.system_capacity_closest_fit(wind_size_kw)

        solar_size_kw = results["outputs"]["Scenario"]["Site"]["PV"]["size_kw"]
        self.solar.system_capacity_kw = solar_size_kw
        logger.info("HybridSystem set system capacities to REopt output")

    def calculate_installed_cost(self):
        # TODO
        total_installed_cost = 0
        self.hybrid_financial.SystemCosts.total_installed_cost = total_installed_cost
        logger.info("HybridSystem set hybrid total installed cost to to {}".format(total_installed_cost))
        pass

    def simulate(self):
        """
        Runs the individual system models then combines the financials
        :return:
        """
        hybrid_size_kw = 0
        total_gen = [0] * self.site.n_timesteps
        if self.solar:
            hybrid_size_kw += self.solar.system_capacity_kw
            self.solar.simulate()
            gen = self.solar.generation_profile()
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        if self.wind:
            hybrid_size_kw += self.wind.system_capacity_kw
            self.wind.simulate()
            gen = self.wind.generation_profile()
            total_gen = [total_gen[i] + gen[i] for i in range(self.site.n_timesteps)]

        self.hybrid_financial.SystemOutput.system_capacity = hybrid_size_kw
        self.hybrid_financial.SystemOutput.gen = total_gen
        logger.info("HybridSystem set hybrid system size to {}".format(hybrid_size_kw))

        self.calculate_installed_cost()

        self.hybrid_financial.execute(0)

        if self.grid:
            self.grid.generation_profile_from_system = total_gen
            self.grid.financial_model.assign(self.hybrid_financial.export())
            self.grid.simulate()

    @property
    def annual_energies(self):
        aep = self.outputs_factory.create()
        if self.solar.system_capacity_kw > 0:
            aep.Solar = self.solar.system_model.Outputs.annual_energy
        if self.wind.system_capacity_kw > 0:
            aep.Wind = self.wind.system_model.Outputs.annual_energy
        aep.Hybrid = aep.Solar + aep.Wind
        if self.grid:
            aep.Grid = sum(self.grid.system_model.Outputs.gen)
        return aep

    @property
    def time_series_kW(self):
        # TODO: hard-coded 8760 hours - allow for variable timing

        ts = self.outputs_factory.create()
        if self.solar.system_capacity_kw > 0:
            ts.Solar = np.array(list(self.solar.system_model.Outputs.ac))/1000
        else:
            ts.Solar = np.zeros(8760)
        if self.wind.system_capacity_kw > 0:
            ts.Wind = np.array(list(self.wind.system_model.Outputs.gen))
        else:
            ts.Wind = np.zeros(8760)

        ts.Hybrid = np.array(ts.Solar) + np.array(ts.Wind)
        if self.grid:
            ts.Grid = sum(self.grid.system_model.Outputs.gen)
        return ts

    @property
    def capacity_factors(self):
        cf = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            cf.Solar = self.solar.system_model.Outputs.capacity_factor
        if self.wind and self.wind.system_capacity_kw > 0:
            cf.Wind = self.wind.system_model.Outputs.capacity_factor
        cf.Hybrid = (self.solar.annual_energy_kw() + self.wind.annual_energy_kw()) \
                       / (self.solar.system_capacity_kw + self.wind.system_capacity_kw) / 87.6
        if self.grid:
            try:
                cf.Grid = self.grid.system_model.Outputs.capacity_factor_curtailment_ac
            except:
                cf.Grid = self.grid.system_model.Outputs.capacity_factor_interconnect_ac
        return cf

    @property
    def net_present_values(self):
        npv = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            npv.Solar = self.solar.financial_model.Outputs.project_return_aftertax_npv
        if self.wind and self.wind.system_capacity_kw > 0:
            npv.Wind = self.wind.financial_model.Outputs.project_return_aftertax_npv
        npv.Hybrid = self.hybrid_financial.Outputs.project_return_aftertax_npv
        if self.grid:
            npv.Grid = self.grid.financial_model.Outputs.project_return_aftertax_npv
        return npv

    @property
    def internal_rate_of_returns(self):
        irr = self.outputs_factory.create()
        if self.solar and self.solar.system_capacity_kw > 0:
            irr.Solar = self.solar.financial_model.Outputs.project_return_aftertax_irr
        if self.wind and self.wind.system_capacity_kw > 0:
            irr.Wind = self.wind.financial_model.Outputs.project_return_aftertax_irr
        irr.Hybrid = self.hybrid_financial.Outputs.project_return_aftertax_irr
        if self.grid:
            irr.Grid = self.grid.financial_model.Outputs.project_return_aftertax_irr
        return irr

    def copy(self):
        """

        :return: a clone
        """
        # TODO implement deep copy

