import pytest
from examples.H2_Analysis.h2_optimize_gf import optimize_gf
import os
from pathlib import Path


def test_h2_optimize_gf():
    """
    Test h2_optimize_gf.py file
    """
    current_path = os.getcwd()
    examples_path = Path(__file__).absolute().parent.parent.parent / "examples" / 'H2_Analysis'
    os.chdir(examples_path)
    opt_lcoh, opt_electrolyzer_size_mw, opt_solar_capacity_mw, opt_battery_storage_mwh, opt_n_turbines = optimize_gf()
    os.chdir(current_path)

    assert opt_lcoh > 0.0
    assert opt_electrolyzer_size_mw >= 1E-6 and opt_electrolyzer_size_mw <= 450.0
    assert opt_solar_capacity_mw >= 0.0 and opt_solar_capacity_mw <= 450.0
    assert opt_battery_storage_mwh >= 0.0 and opt_battery_storage_mwh <= 450.0
    assert opt_n_turbines >= 0 and opt_n_turbines <= 64