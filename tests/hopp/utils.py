from hopp import ROOT_DIR
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site

# default resource files
DEFAULT_SOLAR_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
DEFAULT_WIND_RESOURCE_FILE = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
DEFAULT_GREET_DATA_FILE = ROOT_DIR / "simulation" / "resource_files" / "greet" / "2023" / "greet_2023_processed.yaml"
DEFAULT_CAMBIUM_DATA_FILE = ROOT_DIR / "simulation" / "resource_files" / "cambium" / "Cambium23_MidCase100by2035_hourly_West_Connect_North_2025.csv"

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
        'om_capacity': [0],
        'om_batt_fixed_cost': 0,
        'om_batt_variable_cost': [0.75],
        'om_batt_capacity_cost': 0,
        'om_batt_replacement_cost': 0,
        'om_replacement_cost_escal': 0,
    },
    'revenue': {
        'ppa_price_input': [25], # cents/kWh
        'ppa_escalation': 2.5 # %
    },
    'system_use_lifetime_output': 0,
    'financial_parameters': {
        'inflation_rate': 2.5,
        'real_discount_rate': 6.4,
        'federal_tax_rate': 21.0,
        'state_tax_rate': 4.0,
        'property_tax_rate': 1.0,
        'insurance_rate': 0.5,
        'debt_percent': 68.5,
        'term_int_rate': 6.0,
        'months_working_reserve': 1,
        'analysis_start_year': 2025,
        'installation_months': 12,
        'sales_tax_rate_state': 4.5,
        'admin_expense_percent_of_sales': 1.0, 
        'capital_gains_tax_rate': 15.0, 
        'debt_type': "Revolving debt",
        'depreciation_method': "MACRS",
        'depreciation_period': 5,
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