from collections import OrderedDict

"""
parameter_data.py

An example of how to define variables to optimize, the limits of variable values, and desired outputs
Define inputs parameters and their ranges for a continuous range, use [min, max]; for discrete values, use (options)
"""

solar_input = OrderedDict({
    'Solar': {
        'Pvsamv1': {
            'SystemDesign': {
                #'subarray1_gcr': [.1, .9],
                'subarray1_azimuth': [160, 200]#,
                #'subarray1_tilt': [20, 36]
            }
        }
    },
})

wind_input = OrderedDict({
    'Wind': {
        'Windpower': {
            'Turbine': {
                'wind_turbine_rotor_diameter': [70, 80]
            }
        }
    }
})

geothermal_input = OrderedDict({
    'Geothermal': {
        'Geothermal': {
            'GeoHourly': {
                'model_choice': [0] # currently don't care about any variations, just populate with something
            }
        }
    }
})


# outputs
solar_output = OrderedDict({
    'Solar': {
        'Pvsamv1': [
            'annual_energy',        # 'Solar Annual Energy (MWh)'
            'gen',                  # 'Solar Power Generated (kWac)'
            'capacity_factor'       # 'Solar Capacity Factor (%)'
        ]
    },
})
wind_output = OrderedDict({
    'Wind': {
        'Windpower': [
            'annual_energy',        #: 'Wind Annual Energy (MWh)',
            'gen',                  #: 'Wind Power Generated (kWac)',
            'capacity_factor',      #: 'Wind Capacity Factor (%)'
            'wind_speed',           #: m/s
            'wind_direction',       #: degrees
            'temp',
            'pressure'
        ]
    },
})

geothermal_output = OrderedDict({
    'Geothermal': {
        'Geothermal': [
            'annual_energy',        #: 'Annual Energy (MWh)',
            'capacity_factor',      #: 'Capacity Factor (%)'
            'gen'                   #: 'Geothermal power generated (kWac)
        ]
    },
})

battery_output = OrderedDict({
    'Battery': {
        'StandAloneBattery': [
            'gen',  #: 'Power generated post battery (kWac)
            'average_battery_roundtrip_efficiency', #: 'Average battery round-trip efficiency (%)
            'batt_power', # battery dispatch (>0 discharge, <0 charge)
            'market_sell_rate_series_yr1', # the market sell rate the battery dispatched against
        ]
    },
})


grid_output = OrderedDict({
    'Grid': {
        'Grid': [
            'annual_energy_pre_interconnect_ac',  #: 'Annual Energy before interconnect (MWh)',
            'annual_energy',  #: 'Annual Energy after interconnect (MWh)',
            'annual_ac_interconnect_loss_percent', #: 'Annual Energy percent loss at interconnect (%)',
            'gen'  #: 'Power generated post interconnection (kWac)
        ]
    },
})


generic_output = OrderedDict({
    'Generic': {
        'GenericSystem': [
            'annual_energy',        #: 'Hybrid Annual Energy (MWh)',
            'gen',                  # 'Hybrid Power Generated (kWac)',
            'capacity_factor'       # 'Hybrid Capacity Factor (%)'
        ],
        'Singleowner': [
            'ppa_price',            # : 'Hybrid PPA price ($/MWh)',
            'analysis_period_irr',  #: 'Hybrid IRR (%)'
            'npv_ppa_revenue',      #: NPV of all PPA revenue
            'npv_annual_costs',     #: NPV of all costs
            'project_return_aftertax_npv', #: NPV of project
            'cf_total_revenue'      # total revenue
        ]
    }
})




def get_input_output_data(systems):
    inputs = OrderedDict()
    outputs = OrderedDict()
    if 'Solar' in systems:
        inputs.update(solar_input)
        outputs.update(solar_output)
    if 'Wind' in systems:
        inputs.update(wind_input)
        outputs.update(wind_output)
    if 'Grid' in systems:
        outputs.update(grid_output)
    if 'Geothermal' in systems:
        inputs.update(geothermal_input)
        outputs.update(geothermal_output)
    if 'Battery' in systems:
        outputs.update(battery_output)
    outputs.update(generic_output)
    return inputs, outputs

