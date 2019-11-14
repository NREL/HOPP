import os
this_directory = os.path.dirname(os.path.abspath(__file__))

grid_none = \
{
    "Lifetime": {
        "analysis_period": 25,
        "system_use_lifetime_output": 0
    },
    "Common": {
        "enable_interconnection_limit": 0,
        "grid_interconnection_limit_kwac": 100000
    }
}