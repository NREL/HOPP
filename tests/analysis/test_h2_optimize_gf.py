import pytest
from examples.H2_Analysis.h2_optimize_gf import optimize_gf
from pathlib import Path

@pytest.mark.skip
def test_h2_optimize_gf():
    """
    Test h2_optimize_gf.py file
    """
    opt_lcoh, opt_electrolyzer_size_mw, opt_solar_capacity_mw, opt_battery_storage_mwh, opt_n_turbines = optimize_gf()

    assert opt_lcoh > 0.0
    assert opt_electrolyzer_size_mw >= 1E-6 and opt_electrolyzer_size_mw <= 450.0
    assert opt_solar_capacity_mw >= 0.0 and opt_solar_capacity_mw <= 450.0
    assert opt_battery_storage_mwh >= 0.0 and opt_battery_storage_mwh <= 450.0
    assert opt_n_turbines >= 0 and opt_n_turbines <= 64
