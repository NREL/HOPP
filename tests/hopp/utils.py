from hopp import ROOT_DIR
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site

# default resource files
DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"

# default configuration for `CustomFinancialModel`
DEFAULT_FIN_CONFIG = {
    'battery_system': {
        'batt_replacement_schedule_percent': [0],
        'batt_bank_replacement': [0],
        'batt_replacement_option': 0,
        'batt_computed_bank_capacity': 0,
        'batt_meter_position': 0,
    },
    'system_costs': {
        'om_fixed': [1],
        'om_production': [2],
        'om_capacity': (0,),
        'om_batt_fixed_cost': 0,
        'om_batt_variable_cost': [0],
        'om_batt_capacity_cost': 0,
        'om_batt_replacement_cost': 0,
        'om_replacement_cost_escal': 0,
    },
    'revenue': {},
    'system_use_lifetime_output': 0,
    'financial_parameters': {
        'inflation_rate': 2.5,
        'real_discount_rate': 6.4,
    },
    'cp_capacity_credit_percent': [0],
    'degradation': [0],
}

def create_default_site_info(**kwargs):
    return SiteInfo(
        flatirons_site,
        solar_resource_file=DEFAULT_SOLAR_RESOURCE_FILE,
        wind_resource_file=DEFAULT_WIND_RESOURCE_FILE,
        **kwargs
    )