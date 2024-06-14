import pytest
from pathlib import Path
import numpy as np
import copy

import openmdao.api as om

from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml

from greenheart.tools.optimization.openmdao import GreenHeartComponent, HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent
from greenheart.tools.optimization.gc_PoseOptimization import PoseOptimization
from greenheart.tools.optimization.gc_run_greenheart import run_greenheart
from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig

from hopp import ROOT_DIR

solar_resource_file = ROOT_DIR / "simulation" / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = ROOT_DIR / "simulation" / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
floris_input_filename = Path(__file__).absolute().parent / "inputs" / "floris_input.yaml"
hopp_config_filename = Path(__file__).absolute().parent / "input_files" / "plant" / "hopp_config_wind_wave_solar_battery.yaml"
greenheart_config_filename = Path(__file__).absolute().parent / "input_files" / "plant" / "greenheart_config.yaml"
turbine_config_filename = Path(__file__).absolute().parent / "input_files" / "turbines" / "osw_18MW.yaml"
rtol = 1E-5

def setup_greenheart():
    config = GreenHeartSimulationConfig(
    filename_hopp_config=hopp_config_filename,
    filename_greenheart_config=greenheart_config_filename,
    filename_turbine_config=turbine_config_filename,
    filename_floris_config=floris_input_filename,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir= str(Path(__file__).absolute().parent / "output"),
    use_profast=True,
    post_processing=False,
    incentive_option=1,
    plant_design_scenario=9,
    output_level=7,
    )
    
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_installed_cost_mw"] = 1434000.0 
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_om_per_kw"] = 29.567
    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][0] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    # set skip_financial to false for onshore wind
    config.hopp_config["config"]["simulation_options"]["wind"]["skip_financial"] = False

    config.greenheart_config["opt_options"] = {
            "opt_flag": True,
            "general": {
                "folder_output": "output",
                "fname_output": "test_run_greenheart_optimization_mpi",
            },
            "design_variables": {
                "electrolyzer_rating_kw": {
                    "flag": True,
                    "lower": 10000.0,
                    "upper": 200000.0,
                    "units": "kW",
                },
                "pv_capacity_kw": {
                    "flag": True,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "wave_capacity_kw": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "battery_capacity_kw": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW",
                },
                "battery_capacity_kwh": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW*h",
                },
                "battery_capacity_kwh": {
                    "flag": False,
                    "lower": 1000.0,
                    "upper": 1500000.0,
                    "units": "kW*h",
                },
                "turbine_x": {
                    "flag": False,
                    "lower": 0.0,
                    "upper": 1500000.0,
                    "units": "m",
                },
                "turbine_y": {
                    "flag": False,
                    "lower": 0.0,
                    "upper": 1500000.0,
                    "units": "m",
                },
            },
            "constraints": {
                "turbine_spacing": {
                    "flag": False,
                    "lower": 0.0,
                },
                "boundary_distance": {
                    "flag": False,
                    "lower": 0.0,
                },
            "pv_to_platform_area_ratio": {
                "flag": False, 
                "upper": 1.0, # relative size of solar pv area to platform area
                },
            "user": {}
            },
            "merit_figure": "lcoh",
            "merit_figure_user": {
                "name": "lcoh",
                "max_flag": False,
                "ref": 1.0, # value of objective that scales to 1.0
            },
            "driver": {
                "optimization": {
                    "flag": True,
                    "solver": "SLSQP",
                    "tol": 1E-6,
                    "max_major_iter": 1,
                    "max_minor_iter": 2,
                    "gradient_method": "openmdao",
                    # "time_limit": 10, # (sec) optional
                    # "hist_file_name": "snopt_history.txt", # optional
                    "verify_level": -1, # optional
                    "step_calc": None,
                    "form": "forward", # type of finite differences to use, can be one of ["forward", "backward", "central"]
                    "debug_print": False,
                },
                "design_of_experiments": {
                    "flag": False,
                    "run_parallel": False,
                    "generator": "FullFact", # [Uniform, FullFact, PlackettBurman, BoxBehnken, LatinHypercube]
                    "num_samples": 1, # Number of samples to evaluate model at (Uniform and LatinHypercube only)
                    "seed": 2,
                    "levels":  50, #  Number of evenly spaced levels between each design variable lower and upper bound (FullFactorial only)
                    "criterion": None, # [None, center, c, maximin, m, centermaximin, cm, correelation, corr]
                    "iterations": 1,
                    "debug_print": False
                },
                "step_size_study": {
                    "flag": False  
                },
            },
            "recorder": {
                "flag": True,
                "file_name": str(Path(__file__).absolute().parent / "output" / "recorder.sql"),
                "includes": False,
            },
        }

    return config

# @pytest.mark.mpi
def test_run_greenheart_optimize_mpi(subtests):
        
    try:
        from mpi4py import MPI
    except:
        MPI = False

    config = setup_greenheart()

    prob, config = run_greenheart(config, run_only=False)

    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    if rank == 0:

        cr = om.CaseReader(Path(__file__).absolute().parent / "output" / "recorder.sql")

        # get initial LCOH
        case = cr.get_case(0)
        lcoh_init = case.get_val("lcoh", units='USD/kg')[0]

        # get final LCOH
        case = cr.get_case(-1)
        lcoh_final = case.get_val("lcoh", units='USD/kg')[0]
        
        with subtests.test("lcoh"):
            assert lcoh_final < lcoh_init
