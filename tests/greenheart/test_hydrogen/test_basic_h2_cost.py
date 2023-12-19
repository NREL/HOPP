from greenheart.simulation.technologies.hydrogen.electrolysis.H2_cost_model import basic_H2_cost_model
from pytest import approx
import numpy as np

class TestBasicH2Costs():

    electrolyzer_size_mw = 100
    h2_annual_output = 500
    nturbines = 10
    electrical_generation_timeseries = electrolyzer_size_mw*(np.sin(range(0,500)))*0.5 + electrolyzer_size_mw*0.5

    per_turb_electrolyzer_size_mw = electrolyzer_size_mw/nturbines
    per_turb_h2_annual_output = h2_annual_output/nturbines
    per_turb_electrical_generation_timeseries = electrical_generation_timeseries/nturbines

    elec_capex = 600 # $/kW
    time_between_replacement = 80000 #hours
    useful_life = 30 # years
    atb_year = 2022

    def test_on_turbine_capex(self):
        
        cf_h2_annuals, per_turb_electrolyzer_total_capital_cost, per_turb_electrolyzer_OM_cost, per_turb_electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.per_turb_electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.per_turb_electrical_generation_timeseries, self.per_turb_h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=1)

        electrolyzer_total_capital_cost = per_turb_electrolyzer_total_capital_cost*self.nturbines
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost*self.nturbines

        assert electrolyzer_total_capital_cost == approx(127698560.0)

    def test_on_platform_capex(self):
        
        cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.electrical_generation_timeseries, self.h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=1)

        assert electrolyzer_total_capital_cost == approx(125448560.0)

    def test_on_land_capex(self):
        
        cf_h2_annuals, per_turb_electrolyzer_total_capital_cost, per_turb_electrolyzer_OM_cost, per_turb_electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.per_turb_electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.per_turb_electrical_generation_timeseries, self.per_turb_h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=0)

        electrolyzer_total_capital_cost = per_turb_electrolyzer_total_capital_cost*self.nturbines
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost*self.nturbines

        assert electrolyzer_total_capital_cost == approx(116077280.00000003)

    def test_on_turbine_opex(self):
        
        cf_h2_annuals, per_turb_electrolyzer_total_capital_cost, per_turb_electrolyzer_OM_cost, per_turb_electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.per_turb_electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.per_turb_electrical_generation_timeseries, self.per_turb_h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=1)

        electrolyzer_total_capital_cost = per_turb_electrolyzer_total_capital_cost*self.nturbines
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost*self.nturbines

        assert electrolyzer_OM_cost == approx(1377207.4599629682)

    def test_on_platform_opex(self):
        
        cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.electrical_generation_timeseries, self.h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=1)

        assert electrolyzer_OM_cost == approx(1864249.9310054395)

    def test_on_land_opex(self):
        
        cf_h2_annuals, per_turb_electrolyzer_total_capital_cost, per_turb_electrolyzer_OM_cost, per_turb_electrolyzer_capex_kw, \
        time_between_replacement, h2_tax_credit, h2_itc = \
            basic_H2_cost_model(self.elec_capex, self.time_between_replacement, self.per_turb_electrolyzer_size_mw, self.useful_life, self.atb_year,
            self.per_turb_electrical_generation_timeseries, self.per_turb_h2_annual_output, 0.0, 0.0, include_refurb_in_opex=False, offshore=0)

        electrolyzer_total_capital_cost = per_turb_electrolyzer_total_capital_cost*self.nturbines
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost*self.nturbines

        assert electrolyzer_OM_cost == approx(1254447.4599629682)

if __name__ == "__main__":
    test_set = TestBasicH2Costs()
