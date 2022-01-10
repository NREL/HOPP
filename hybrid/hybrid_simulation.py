import csv
from pathlib import Path
from typing import Union
import json
from collections import OrderedDict

import numpy as np
from scipy.stats import pearsonr
import PySAM.GenericSystem as GenericSystem

from tools.analysis import create_cost_calculator
from hybrid.sites import SiteInfo
from hybrid.pv_source import PVPlant
from hybrid.wind_source import WindPlant
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant
from hybrid.battery import Battery
from hybrid.grid import Grid
from hybrid.reopt import REopt
from hybrid.layout.hybrid_layout import HybridLayout
from hybrid.dispatch.hybrid_dispatch_builder_solver import HybridDispatchBuilderSolver
from hybrid.log import hybrid_logger as logger


class HybridSimulationOutput:
    def __init__(self, power_sources):
        self.power_sources = power_sources
        self.hybrid = 0
        self.pv = 0
        self.wind = 0
        self.tower = 0
        self.trough = 0
        self.battery = 0

    def create(self):
        return HybridSimulationOutput(self.power_sources)

    def __repr__(self):
        repr_dict = {}
        for k in self.power_sources.keys():
            if k == 'grid':
                repr_dict['hybrid'] = self.hybrid
            else:
                repr_dict[k] = getattr(self, k)
        repr_dict['hybrid'] = self.hybrid
        return json.dumps(repr_dict)


class HybridSimulation:
    hybrid_system: GenericSystem.GenericSystem

    def __init__(self,
                 power_sources: dict,
                 site: SiteInfo,
                 interconnect_kw: float,
                 dispatch_options=None,
                 cost_info=None,
                 simulation_options=None):
        """
        Base class for simulating a hybrid power plant.

        Can be derived to add other sizing methods, financial analyses,
            methods for pre- or post-processing, etc

        :param power_sources: tuple of strings, float pairs
            names of power sources to include and their kw sizes
            choices include:
                    ('pv', 'wind', 'geothermal', 'tower', 'trough', 'battery')
        :param site: Site
            layout, location and resource data

        :param interconnect_kw: float
            power limit of interconnect for the site

        :param dispatch_options: dict
            For details see HybridDispatchOptions in hybrid_dispatch_options.py

        :param cost_info: dict
            optional dictionary of cost information

        :param simulation_options: dict
            optional nested dictionary; ie: {'pv': {'skip_financial'}}
        """
        self._fileout = Path.cwd() / "results"
        self.site = site
        self.sim_options = simulation_options if simulation_options else dict()

        self.power_sources = OrderedDict()
        self.pv: Union[PVPlant, None] = None
        self.wind: Union[WindPlant, None] = None
        self.tower: Union[TowerPlant, None] = None
        self.trough: Union[TroughPlant, None] = None
        self.battery: Union[Battery, None] = None
        self.dispatch_builder: Union[HybridDispatchBuilderSolver, None] = None
        self.grid: Union[Grid, None] = None

        temp = list(power_sources.keys())
        for k in temp:
            power_sources[k.lower()] = power_sources.pop(k)

        if 'pv' in power_sources.keys():
            self.pv = PVPlant(self.site, power_sources['pv'])
            self.power_sources['pv'] = self.pv
            logger.info("Created HybridSystem.pv with system size {} mW".format(power_sources['pv']))
        if 'wind' in power_sources.keys():
            self.wind = WindPlant(self.site, power_sources['wind'])
            self.power_sources['wind'] = self.wind
            logger.info("Created HybridSystem.wind with system size {} mW".format(power_sources['wind']))
        if 'tower' in power_sources.keys():
            self.tower = TowerPlant(self.site, power_sources['tower'])
            self.power_sources['tower'] = self.tower
            logger.info("Created HybridSystem.tower with cycle size {} MW, a solar multiple of {}, {} hours of storage".format(
                self.tower.cycle_capacity_kw/1000., self.tower.solar_multiple, self.tower.tes_hours))
        if 'trough' in power_sources.keys():
            self.trough = TroughPlant(self.site, power_sources['trough'])
            self.power_sources['trough'] = self.trough
            logger.info("Created HybridSystem.trough with cycle size {} MW, a solar multiple of {}, {} hours of storage".format(
                self.trough.cycle_capacity_kw/1000., self.trough.solar_multiple, self.trough.tes_hours))
        if 'battery' in power_sources.keys():
            self.battery = Battery(self.site, power_sources['battery'])
            self.power_sources['battery'] = self.battery
            logger.info("Created HybridSystem.battery with system capacity {} MWh and rating of {} MW".format(
                self.battery.system_capacity_kwh/1000., self.battery.system_capacity_kw/1000.))
        if 'geothermal' in power_sources.keys():
            raise NotImplementedError("Geothermal plant not yet implemented")

        # performs interconnection and curtailment energy limits
        self.grid = Grid(self.site, interconnect_kw)
        self.interconnect_kw = interconnect_kw
        self.power_sources['grid'] = self.grid

        self.layout = HybridLayout(self.site, self.power_sources)

        self.dispatch_builder = HybridDispatchBuilderSolver(self.site,
                                                            self.power_sources,
                                                            dispatch_options=dispatch_options)
        
        # Default cost calculator, can be overwritten
        self.cost_model = create_cost_calculator(self.interconnect_kw, **cost_info if cost_info else {})

        self.outputs_factory = HybridSimulationOutput(power_sources)

        if len(self.site.elec_prices.data):
            # if prices are provided, assume that they are in units of $/MWh so convert to $/KWh
            # if not true, the user should adjust the base ppa price
            self.ppa_price = 0.001
            self.dispatch_factors = self.site.elec_prices.data


    def setup_cost_calculator(self, cost_calculator: object):
        if hasattr(cost_calculator, "calculate_total_costs"):
            self.cost_model = cost_calculator

    @property
    def interconnect_kw(self):
        return self.grid.value("grid_interconnection_limit_kwac")

    @interconnect_kw.setter
    def interconnect_kw(self, ic_kw):
        self.grid.value("grid_interconnection_limit_kwac", ic_kw)

    @property
    def ppa_price(self):
        return self.grid.ppa_price

    @ppa_price.setter
    def ppa_price(self, ppa_price):
        for tech, _ in self.power_sources.items():
            getattr(self, tech).ppa_price = ppa_price
        self.grid.ppa_price = ppa_price

    @property
    def capacity_price(self):
        return self.grid.capacity_price

    @capacity_price.setter
    def capacity_price(self, cap_price_per_mw_year):
        for tech, _ in self.power_sources.items():
            getattr(self, tech).capacity_price = cap_price_per_mw_year
        self.grid.capacity_price = cap_price_per_mw_year

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
        return self.grid.value("real_discount_rate")

    @discount_rate.setter
    def discount_rate(self, discount_rate):
        for k, _ in self.power_sources.items():
            if hasattr(self, k):
                getattr(self, k).value("real_discount_rate", discount_rate)
        self.grid.value("real_discount_rate", discount_rate)

    def set_om_costs_per_kw(self, pv_om_per_kw=None, wind_om_per_kw=None,
                            tower_om_per_kw=None, trough_om_per_kw=None,
                            hybrid_om_per_kw=None):
        if pv_om_per_kw and wind_om_per_kw and tower_om_per_kw and trough_om_per_kw and hybrid_om_per_kw:
            if len(pv_om_per_kw) != len(wind_om_per_kw) != len(tower_om_per_kw) != len(trough_om_per_kw) \
                    != len(hybrid_om_per_kw):
                raise ValueError("Length of yearly om cost per kw arrays must be equal.")

        if pv_om_per_kw and self.pv:
            self.pv.om_capacity = pv_om_per_kw

        if wind_om_per_kw and self.wind:
            self.wind.om_capacity = wind_om_per_kw

        if tower_om_per_kw and self.tower:
            self.tower.om_capacity = tower_om_per_kw

        if trough_om_per_kw and self.trough:
            self.trough.om_capacity = trough_om_per_kw

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
                      solar_model=self.pv,
                      wind_model=self.wind,
                      fin_model=self.grid._financial_model,
                      fileout=str(self._fileout / "REoptResult.json"))
        results = reopt.get_reopt_results(force_download=False)
        wind_size_kw = results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"]
        self.wind.system_capacity_closest_fit(wind_size_kw)

        solar_size_kw = results["outputs"]["Scenario"]["Site"]["PV"]["size_kw"]
        self.pv.system_capacity_kw = solar_size_kw
        logger.info("HybridSystem set system capacities to REopt output")

    def calculate_installed_cost(self):
        if not self.cost_model:
            raise RuntimeError("'calculate_installed_cost' called before 'setup_cost_calculator'.")

        wind_mw = 0
        pv_mw = 0
        battery_mw = 0
        battery_mwh = 0
        if self.pv:
            pv_mw = self.pv.system_capacity_kw / 1000
        if self.wind:
            wind_mw = self.wind.system_capacity_kw / 1000
        if self.battery:
            battery_mw = self.battery.system_capacity_kw / 1000
            battery_mwh = self.battery.system_capacity_kwh / 1000

        pv_cost, wind_cost, storage_cost, total_cost = self.cost_model.calculate_total_costs(wind_mw,
                                                                                             pv_mw,
                                                                                             battery_mw,
                                                                                             battery_mwh)
        # TODO: add tower and trough to cost_model functionality
        if self.pv:
            self.pv.total_installed_cost = pv_cost
        if self.wind:
            self.wind.total_installed_cost = wind_cost
        if self.battery:
            self.battery.total_installed_cost = storage_cost
        if self.tower:
            self.tower.total_installed_cost = self.tower.calculate_total_installed_cost()
            total_cost += self.tower.total_installed_cost
        if self.trough:
            self.trough.total_installed_cost = self.trough.calculate_total_installed_cost()
            total_cost += self.trough.total_installed_cost

        self.grid.total_installed_cost = total_cost
        logger.info("HybridSystem set hybrid total installed cost to to {}".format(total_cost))

    def calculate_financials(self):
        """
        prepare financial parameters from individual power plants for total performance and financial metrics
        """
        # TODO: need to make financial parameters consistent

        # TODO: generalize this for different plants besides wind and solar
        generators = [v for k, v in self.power_sources.items() if k != 'grid']

        # Average based on capacities
        hybrid_size_kw = sum([v.system_capacity_kw for v in generators])
        if hybrid_size_kw == 0:
            return

        size_ratios = []
        for v in generators:
            size_ratios.append(v.system_capacity_kw / hybrid_size_kw)

        # Average based on total production
        production_ratios = []
        total_production = sum([v.annual_energy_kwh for v in generators])
        for v in generators:
            production_ratios.append(v.annual_energy_kwh / total_production)

        # Average based on installed cost
        cost_ratios = []
        total_cost = sum([v.total_installed_cost for v in generators])
        for v in generators:
            cost_ratios.append(v.total_installed_cost / total_cost)

        def set_average_for_hybrid(var_name, weight_factor=None):
            """
            Sets the hybrid plant's financial input to the weighted average of each component's value
            """
            if not weight_factor:
                weight_factor = [1 / len(generators) for _ in generators]
            hybrid_avg = sum(np.array(v.value(var_name)) * weight_factor[n]
                             for n, v in enumerate(generators))
            self.grid.value(var_name, hybrid_avg)
            return hybrid_avg

        def set_logical_or_for_hybrid(var_name):
            """
            Sets the hybrid plant's financial input to the logical or value of each component's value
            """
            hybrid_or = sum(np.array(v.value(var_name)) for n, v in enumerate(generators)) > 0
            self.grid.value(var_name, int(hybrid_or))
            return hybrid_or

        # Debt and Financing should be handled via user customization of the grid's financial model

        # capacity payments
        # for v in generators:
        #     v.value("cp_system_nameplate", v.system_capacity_kw)
        # self.grid.value("cp_system_nameplate", hybrid_size_kw)

        # O&M Cost
        set_average_for_hybrid("om_capacity", size_ratios)
        set_average_for_hybrid("om_fixed")
        set_average_for_hybrid("om_production", production_ratios)

        # Tax Incentives
        set_average_for_hybrid("ptc_fed_amount", production_ratios)
        set_average_for_hybrid("ptc_fed_escal", production_ratios)
        set_average_for_hybrid("itc_fed_amount", cost_ratios)
        set_average_for_hybrid("itc_fed_percent", cost_ratios)

        # Federal Depreciation Allocations are averaged
        set_average_for_hybrid("depr_alloc_macrs_5_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_macrs_15_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_sl_5_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_sl_15_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_sl_20_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_sl_39_percent", cost_ratios)
        set_average_for_hybrid("depr_alloc_custom_percent", cost_ratios)

        # Federal Depreciation Qualification are "hybridized" by taking the logical or
        set_logical_or_for_hybrid("depr_bonus_fed_macrs_5")
        set_logical_or_for_hybrid("depr_bonus_sta_macrs_5")
        set_logical_or_for_hybrid("depr_itc_fed_macrs_5")
        set_logical_or_for_hybrid("depr_itc_sta_macrs_5")
        set_logical_or_for_hybrid("depr_bonus_fed_macrs_15")
        set_logical_or_for_hybrid("depr_bonus_sta_macrs_15")
        set_logical_or_for_hybrid("depr_itc_fed_macrs_15")
        set_logical_or_for_hybrid("depr_itc_sta_macrs_15")
        set_logical_or_for_hybrid("depr_bonus_fed_sl_5")
        set_logical_or_for_hybrid("depr_bonus_sta_sl_5")
        set_logical_or_for_hybrid("depr_itc_fed_sl_5")
        set_logical_or_for_hybrid("depr_itc_sta_sl_5")
        set_logical_or_for_hybrid("depr_bonus_fed_sl_15")
        set_logical_or_for_hybrid("depr_bonus_sta_sl_15")
        set_logical_or_for_hybrid("depr_itc_fed_sl_15")
        set_logical_or_for_hybrid("depr_itc_sta_sl_15")
        set_logical_or_for_hybrid("depr_bonus_fed_sl_20")
        set_logical_or_for_hybrid("depr_bonus_sta_sl_20")
        set_logical_or_for_hybrid("depr_itc_fed_sl_20")
        set_logical_or_for_hybrid("depr_itc_sta_sl_20")
        set_logical_or_for_hybrid("depr_bonus_fed_sl_39")
        set_logical_or_for_hybrid("depr_bonus_sta_sl_39")
        set_logical_or_for_hybrid("depr_itc_fed_sl_39")
        set_logical_or_for_hybrid("depr_itc_sta_sl_39")
        set_logical_or_for_hybrid("depr_bonus_fed_custom")
        set_logical_or_for_hybrid("depr_bonus_sta_custom")
        set_logical_or_for_hybrid("depr_itc_fed_custom")
        set_logical_or_for_hybrid("depr_itc_sta_custom")

        # Degradation of energy output year after year
        set_average_for_hybrid("degradation", production_ratios)

        self.grid.value("ppa_soln_mode", 1)

        if self.battery:
            self.grid._financial_model.SystemCosts.om_batt_replacement_cost = self.battery._financial_model.SystemCosts.om_batt_replacement_cost

    def simulate(self,
                 project_life: int = 25):
        """
        Runs the individual system models then combines the financials
        :return:
        """
        # Calling simulation set-up functions that have to be called before calculate_installed_cost()
        if 'tower' in self.power_sources:
            if self.tower.optimize_field_before_sim:
                self.tower.optimize_field_and_tower()
            else:
                self.tower.generate_field()
        if 'battery' in self.power_sources:
            self.battery.setup_system_model()

        # Set csp thermal resource for dispatch model
        for csp_tech in ['tower', 'trough']:
            if csp_tech in self.power_sources:
                self.power_sources[csp_tech].set_ssc_info_for_dispatch()

        self.calculate_installed_cost()

        hybrid_size_kw = 0
        total_gen = np.zeros(self.site.n_timesteps * project_life)
        total_gen_max_feasible_year1 = np.zeros(self.site.n_timesteps)
        systems = ['pv', 'wind']
        for system in systems:
            model = getattr(self, system)
            if model:
                if type(model).__name__ == 'PVPlant':
                    hybrid_size_kw += model.system_capacity_kw / model._system_model.SystemDesign.dc_ac_ratio  # [kW] (AC output)
                else:
                    hybrid_size_kw += model.system_capacity_kw  # [kW]

                skip_sim = False
                if system in self.sim_options.keys():
                    if 'skip_financial' in self.sim_options[system].keys():
                        skip_sim = self.sim_options[system]['skip_financial']
                model.simulate(self.interconnect_kw, project_life, skip_sim)
                project_life_gen = np.tile(model.generation_profile,
                                           int(project_life / (len(model.generation_profile) // self.site.n_timesteps)))
                if len(project_life_gen) != len(total_gen):
                    raise ValueError("Generation profile, `gen`, from system {} should have length that divides"
                                     " n_timesteps {} * project_life {}".format(system, self.site.n_timesteps,
                                                                                project_life))
                total_gen += project_life_gen
                total_gen_max_feasible_year1 += model.gen_max_feasible

        if self.dispatch_builder.needs_dispatch:
            """
            Run dispatch optimization
            """
            self.dispatch_builder.simulate()

            dispatchable_systems = ['battery', 'tower', 'trough']
            for system in dispatchable_systems:
                model = getattr(self, system)
                if model:
                    hybrid_size_kw += model.system_capacity_kw
                    tech_gen = np.tile(model.generation_profile,
                                       int(project_life / (len(model.generation_profile) // self.site.n_timesteps)))
                    total_gen += tech_gen
                    storage_cc = True
                    if 'storage_capacity_credit' in self.sim_options.keys():
                        storage_cc = self.sim_options['storage_capacity_credit']
                    model.simulate_financials(self.interconnect_kw, project_life, storage_cc)
                    total_gen_max_feasible_year1 += model.gen_max_feasible
                    if system == 'battery':
                        # copy over replacement info
                        self.grid._financial_model.BatterySystem.assign(model._financial_model.BatterySystem.export())

        # Simulate grid, after components are combined
        self.grid.generation_profile_from_system = total_gen
        self.grid.system_capacity_kw = hybrid_size_kw  # TODO: Should this be interconnection limit?
        self.grid.gen_max_feasible = np.minimum(total_gen_max_feasible_year1, self.interconnect_kw * self.site.interval / 60)
        
        self.calculate_financials()

        self.grid.simulate(self.interconnect_kw, project_life)
        # FIXME: updating capacity credit for reporting only.
        self.grid.capacity_credit_percent = self.grid.capacity_credit_percent * \
                                            (self.grid.system_capacity_kw / self.grid.interconnect_kw)
        
        logger.info(f"Hybrid Simulation complete. NPVs are {self.net_present_values}. AEPs are {self.annual_energies}.")
        
    @property
    def system_capacity_kw(self):
        cap = self.outputs_factory.create()
        for v in self.power_sources.keys():
            if v == "grid":
                continue
            if hasattr(self, v):
                setattr(cap, v, getattr(getattr(self, v), "system_capacity_kw"))
        cap.hybrid = self.grid.system_capacity_kw
        return cap

    @property
    def annual_energies(self):
        aep = self.outputs_factory.create()
        if self.pv:
            aep.pv = self.pv.annual_energy_kwh
        if self.wind:
            aep.wind = self.wind.annual_energy_kwh
        if self.tower:
            aep.tower = self.tower.annual_energy_kwh
        if self.trough:
            aep.trough = self.trough.annual_energy_kwh
        if self.battery:
            aep.battery = sum(self.battery.Outputs.gen)
        aep.hybrid = sum(self.grid.generation_profile[0:self.site.n_timesteps])
        return aep

    @property
    def generation_profile(self):
        gen = self.outputs_factory.create()
        if self.pv:
            gen.pv = self.pv.generation_profile
        if self.wind:
            gen.wind = self.wind.generation_profile
        if self.tower:
            gen.tower = self.tower.generation_profile
        if self.trough:
            gen.trough = self.trough.generation_profile
        if self.battery:
            gen.battery = self.battery.generation_profile
        gen.hybrid = self.grid.generation_profile
        return gen

    @property
    def capacity_factors(self):
        cf = self.outputs_factory.create()
        hybrid_generation = 0.0
        hybrid_capacity = 0.0
        if self.pv:
            cf.pv = self.pv.capacity_factor
            hybrid_generation += self.pv.annual_energy_kwh
            hybrid_capacity += self.pv.system_capacity_kw
        if self.wind:
            cf.wind = self.wind.capacity_factor
            hybrid_generation += self.wind.annual_energy_kwh
            hybrid_capacity += self.wind.system_capacity_kw
        if self.tower:
            cf.tower = self.tower.capacity_factor
            hybrid_generation += self.tower.annual_energy_kwh
            hybrid_capacity += self.tower.system_capacity_kw
        if self.trough:
            cf.trough = self.trough.capacity_factor
            hybrid_generation += self.trough.annual_energy_kwh
            hybrid_capacity += self.trough.system_capacity_kw
        if self.battery:
            hybrid_generation += sum(self.battery.Outputs.gen)
            hybrid_capacity += self.battery.system_capacity_kw
        try:
            cf.grid = self.grid.capacity_factor_after_curtailment
        except:
            cf.grid = self.grid.capacity_factor_at_interconnect
        # TODO: how should the battery be handled?
        cf.hybrid = (hybrid_generation / hybrid_capacity) / 87.6
        return cf

    @property
    def capacity_credit_percent(self):
        """
        Capacity credit (eligible portion of nameplate), %
        """
        cap_cred = self.outputs_factory.create()
        if self.pv:
            cap_cred.pv = self.pv.capacity_credit_percent
        if self.wind:
            cap_cred.wind = self.wind.capacity_credit_percent
        if self.tower:
            cap_cred.tower = self.tower.capacity_credit_percent
        if self.trough:
            cap_cred.trough = self.trough.capacity_credit_percent
        if self.battery:
            cap_cred.battery = self.battery.capacity_credit_percent
        cap_cred.hybrid = self.grid.capacity_credit_percent
        return cap_cred

    def _aggregate_financial_output(self, name, start_index=None, end_index=None):
        out = self.outputs_factory.create()
        for k, v in self.power_sources.items():
            if k in self.sim_options.keys():
                if 'skip_financial' in self.sim_options[k].keys():
                    continue
            val = getattr(v, name)
            if start_index and end_index:
                val = list(val[start_index:end_index])
            if k == "grid":
                setattr(out, "hybrid", val)
            else:
                setattr(out, k, val)
        return out

    @property
    def cost_installed(self):
        """
        The total_installed_cost plus any financing costs, $
        """
        return self._aggregate_financial_output("cost_installed")

    @property
    def total_revenues(self):
        """
        Revenue in cashflow, $/year
        """
        return self._aggregate_financial_output("total_revenue", 1)

    @property
    def capacity_payments(self):
        """
        Payments received for capacity, $/year
        """
        return self._aggregate_financial_output("capacity_payment", 1)

    @property
    def energy_purchases_values(self):
        """
        Value of energy purchased, $/year
        """
        return self._aggregate_financial_output("energy_purchases_value", 1)

    @property
    def energy_sales_values(self):
        """
        Value of energy sold, $/year
        """
        return self._aggregate_financial_output("energy_sales_value", 1)

    @property
    def energy_values(self):
        """
        Value of energy sold, $/year
        """
        return self._aggregate_financial_output("energy_value", 1)

    @property
    def federal_depreciation_totals(self):
        """
        Value of all federal depreciation allocations, $/year
        """
        return self._aggregate_financial_output("federal_depreciation_total", 1)

    @property
    def federal_taxes(self):
        """
        Federal taxes paid, $/year
        """
        return self._aggregate_financial_output("federal_taxes", 1)

    @property
    def tax_incentives(self):
        """
        Federal and state Production Tax Credits and Investment Tax Credits, $/year
        """
        return self._aggregate_financial_output("tax_incentives", 1)

    @property
    def debt_payment(self):
        """
        Payment to debt interest and principal, $/year
        """
        return self._aggregate_financial_output("debt_payment", 1)

    @property
    def insurance_expenses(self):
        """
        Payments for insurance, $/year
        """
        return self._aggregate_financial_output("insurance_expense", 1)

    @property
    def om_expenses(self):
        """
        Total O&M expenses including fixed, production-based, and capacity-based, $/year
        """
        return self._aggregate_financial_output("om_expense", 1)

    @property
    def net_present_values(self):
        return self._aggregate_financial_output("net_present_value")

    @property
    def internal_rate_of_returns(self):
        return self._aggregate_financial_output("internal_rate_of_return")

    @property
    def lcoe_real(self):
        return self._aggregate_financial_output("levelized_cost_of_energy_real")

    @property
    def lcoe_nom(self):
        return self._aggregate_financial_output("levelized_cost_of_energy_nominal")

    @property
    def benefit_cost_ratios(self):
        return self._aggregate_financial_output("benefit_cost_ratio")

    def hybrid_outputs(self):
        # TODO: Update test_run_hopp_calc.py to work with hybrid_simulation_outputs
        outputs = dict()
        outputs['PV (MW)'] = self.pv.system_capacity_kw / 1000
        outputs['Wind (MW)'] = self.wind.system_capacity_kw / 1000
        pv_pct = self.pv.system_capacity_kw / (self.pv.system_capacity_kw + self.wind.system_capacity_kw)
        wind_pct = self.wind.system_capacity_kw / (self.pv.system_capacity_kw + self.wind.system_capacity_kw)
        outputs['PV (%)'] = pv_pct * 100
        outputs['Wind (%)'] = wind_pct * 100

        annual_energies = self.annual_energies
        outputs['PV AEP (GWh)'] = annual_energies.pv / 1000000
        outputs['Wind AEP (GWh)'] = annual_energies.wind / 1000000
        outputs["AEP (GWh)"] = annual_energies.hybrid / 1000000

        capacity_factors = self.capacity_factors
        outputs['PV Capacity Factor'] = capacity_factors.pv
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
        if self.grid.value('ppa_multiplier_model') == 1:
            outputs['Revenue (TOD)'] = sum(self.grid.total_revenue)
            outputs['Revenue (PPA)'] = outputs['TOD Profile Used'] = 0

        outputs['Cost / MWh Produced percent reduction'] = 0

        if pv_pct * wind_pct > 0:
            outputs['Pearson R Wind V Solar'] = pearsonr(self.pv.generation_profile[0:8760],
                                                         self.wind.generation_profile[0:8760])[0]

        return outputs

    def hybrid_simulation_outputs(self, filename: str = ""):
        outputs = dict()

        if self.pv:
            outputs['PV (MW)'] = self.pv.system_capacity_kw / 1000
        if self.wind:
            outputs['Wind (MW)'] = self.wind.system_capacity_kw / 1000
        if self.tower:
            outputs['Tower (MW)'] = self.tower.system_capacity_kw / 1000
            outputs['Tower Hours of Storage (hr)'] = self.tower.tes_hours
            outputs['Tower Solar Multiple (-)'] = self.tower.solar_multiple
        if self.trough:
            outputs['Trough (MW)'] = self.trough.system_capacity_kw / 1000
            outputs['Trough Hours of Storage (hr)'] = self.trough.tes_hours
            outputs['Trough Solar Multiple (-)'] = self.trough.solar_multiple
        if self.battery:
            outputs['Battery (MW)'] = self.battery.system_capacity_kw / 1000
            outputs['Battery (MWh)'] = self.battery.system_capacity_kwh / 1000
        if self.grid:
            outputs['Grid System Capacity (MW)'] = self.grid.system_capacity_kw / 1000
            outputs['Grid Interconnect (MW)'] = self.grid.interconnect_kw / 1000
            outputs['Grid Curtailment Percent (%)'] = self.grid.curtailment_percent
            # outputs['Grid Capacity Factor After Curtailment (%)'] = self.grid.capacity_factor_after_curtailment
            outputs['Grid Capacity Factor at Interconnect (%)'] = self.grid.capacity_factor_at_interconnect

        attr_map = {'annual_energies': {'name': 'AEP (GWh)', 'scale': 1/1e6},
                    'capacity_factors': {'name': 'Capacity Factor (-)'},
                    'capacity_credit_percent': {'name': 'Capacity Credit (%)'},
                    'cost_installed': {'name': 'Installed Cost ($-million)', 'scale': 1/1e6},
                    'total_revenues': {'name': 'Total Revenue ($/year)'},  # list
                    'capacity_payments': {'name': 'Capacity Payments ($/year)'},  # tuple
                    'energy_purchases_values': {'name': 'Energy Purchases ($/year)'},  # tuple
                    'energy_sales_values': {'name': 'Energy Sales ($/year)'},  # tuple
                    'energy_values': {'name': 'Energy Values ($/year)'},  # tuple
                    'federal_depreciation_totals': {'name': 'Federal Depreciation Totals ($/year)'},  # tuple
                    'federal_taxes': {'name': 'Federal Taxes ($/year)'},  # tuple
                    'debt_payment': {'name': 'Debt Payments ($/year)'},  # tuple
                    'insurance_expenses': {'name': 'Insurance Expenses ($/year)'},  # tuple
                    'om_expenses': {'name': 'O&M Expenses ($/year)'},  # list
                    'net_present_values': {'name': 'Net Present Value ($-million)', 'scale': 1/1e6},
                    'internal_rate_of_returns': {'name': 'Internal Rate of Return (%)'},
                    'lcoe_real': {'name': 'Real Levelized Cost of Energy ($/kWh)'},
                    'lcoe_nom': {'name': 'Nominal Levelized Cost of Energy ($/kWh)'},
                    'benefit_cost_ratios': {'name': 'Benefit cost Ratio (-)'}}

        skip_attr = ['generation_profile', 'outputs_factory']

        for attr in dir(self):
            if attr in skip_attr:
                continue
            if type(getattr(self, attr)) == HybridSimulationOutput:
                if attr in attr_map:
                    technologies = list(self.power_sources.keys())
                    technologies.append('hybrid')
                    for source in technologies:
                        attr_dict = attr_map[attr]
                        o_name = source.capitalize() + ' ' + attr_dict['name']

                        try:
                            source_output = getattr(getattr(self, attr), source)
                        except AttributeError:
                            continue

                        # Scaling output
                        scale = 1
                        if 'scale' in attr_dict:
                            scale = attr_dict['scale']

                        if type(source_output) == list:
                            value = [x * scale for x in source_output]
                        elif type(source_output) == float:
                            value = source_output * scale
                        else:
                            value = source_output

                        outputs[o_name] = value

        outputs['PPA Price Used'] = self.grid.ppa_price[0]

        # time series dispatch
        if self.grid.value('ppa_multiplier_model') == 1:
            outputs['Grid Total Revenue (TOD)'] = self.grid.total_revenue

        # export to file
        if filename != "":
            with open(filename, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerows(outputs.items())

        return outputs

    def assign(self, input_dict: dict):
        """
        Assign values from a nested dictionary of values which can be for all technologies in the hybrid plant
        or for a specific technology:

        :param input_dict: dict or nested dict
            If not nested, the keys are the parameter names and the values are the parameter values. All components with
            the parameters will have their parameter values changed to these new provided values.
            If a nested dict, the key for the outer dictionary is the name of the component (i.e. "pv") and the dict
            value provides all the parameter name-value pairs to assign to the component.
        """
        for k, v in input_dict.items():
            if not isinstance(v, dict):
                for tech in self.power_sources.keys():
                    self.power_sources[tech.lower()].value(k, v)
            else:
                if k not in self.power_sources.keys():
                    logger.warning(f"Cannot assign {v} to {k}: technology was not included in hybrid plant")
                    continue
                for kk, vv in v.items():
                    self.power_sources[k.lower()].value(kk, vv)

    def copy(self):
        """

        :return: a clone
        """
        # TODO implement deep copy

    def plot_layout(self,
                    figure=None,
                    axes=None,
                    wind_color='b',
                    pv_color='darkorange',
                    site_border_color='k',
                    site_alpha=0.95,
                    linewidth=4.0
                    ):
        self.layout.plot(figure, axes, wind_color, pv_color, site_border_color, site_alpha, linewidth)
