from unittest.mock import patch, MagicMock

import pytest
from pytest import fixture
import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal

from hopp.simulation.technologies.grid import GridConfig, Grid
from tests.hopp.utils import create_default_site_info


interconnect_kw = 10e3


@fixture
def site():
    return create_default_site_info()


def test_grid_config_initialization(subtests):
    interconnect_kw = 10e3

    with subtests.test("with valid config"):
        grid_config = GridConfig(interconnect_kw)
        assert grid_config.interconnect_kw == interconnect_kw
        assert grid_config.fin_model is None

    with subtests.test("with invalid interconnect_kw"):
        with pytest.raises(ValueError):
            grid_config = GridConfig(0.0)


def test_grid_initialization(site, subtests):
    with subtests.test("initialize attributes"):
        config = GridConfig.from_dict({"interconnect_kw": interconnect_kw})
        grid = Grid(site, config=config)

        assert_array_equal(grid.missed_load, [0.])
        assert_array_equal(grid.missed_load_percentage, 0.)
        assert_array_equal(grid.schedule_curtailed, [0.])
        assert_array_equal(grid.schedule_curtailed_percentage, 0.)
        assert_array_equal(grid.total_gen_max_feasible_year1, [0.])

    with subtests.test("default (SAM) financial model"):
        config = GridConfig.from_dict({"interconnect_kw": interconnect_kw})
        grid = Grid(site, config=config)
        assert grid._financial_model is not None

    with subtests.test("provided SAM financial model"):
        config = GridConfig.from_dict({
            "interconnect_kw": interconnect_kw,
            "fin_model": grid._financial_model
        })
        grid2 = Grid(site, config=config)
        assert grid2._financial_model is not None

    with subtests.test("provided custom financial model"):
        # We'd typically use CustomFinancialModel, but we can provide a dummy
        # for this test for isolation purposes
        custom_fin_model = MagicMock()
        config = GridConfig.from_dict({
            "interconnect_kw": interconnect_kw,
            "fin_model": custom_fin_model
        })
        grid = Grid(site, config=config)
        assert grid._financial_model is not None


# NOTE: simulate_power is a side effect that runs the simulation, so we mock it out
# to maintain isolation
@patch.object(Grid, "simulate_power")
def test_simulate_grid_connection(mock_simulate_power, site, subtests):
    project_life = 25
    hybrid_size_kw = 10e3
    # use constant gen profile for simplicity
    total_gen = np.repeat([5000], site.n_timesteps * project_life)
    lifetime_sim = False
    total_gen_max_feasible_year1 = np.repeat([5000], 8760)

    with subtests.test("no desired schedule"):
        config = GridConfig.from_dict({"interconnect_kw": interconnect_kw})
        grid = Grid(site, config=config)
        grid.simulate_grid_connection(
            hybrid_size_kw,
            total_gen,
            project_life,
            lifetime_sim,
            total_gen_max_feasible_year1
        )
        assert_array_equal(grid.generation_profile, total_gen)

    with subtests.test("update attributes"):
        mock_simulate_power.assert_called_with(project_life, lifetime_sim)
        assert_array_equal(grid.total_gen_max_feasible_year1, total_gen_max_feasible_year1)
        assert grid.system_capacity_kw == hybrid_size_kw
        assert_array_equal(grid.gen_max_feasible, total_gen_max_feasible_year1)
        assert grid.capacity_credit_percent[0] == 0.0

    with subtests.test("follow desired schedule: curtailment"):
        desired_schedule = np.repeat([3], site.n_timesteps)
        site2 = create_default_site_info(
            desired_schedule=desired_schedule
        )
        config = GridConfig.from_dict({"interconnect_kw": interconnect_kw})
        grid = Grid(site2, config=config)
        grid.simulate_grid_connection(
            hybrid_size_kw,
            total_gen,
            project_life,
            lifetime_sim,
            total_gen_max_feasible_year1
        )

        timesteps = site.n_timesteps * project_life
        assert_array_equal(
            grid.generation_profile, 
            np.repeat([3000], timesteps),
            "gen profile should be reduced"
        )

        msg = "no load should be missed"
        assert_array_equal(
            grid.missed_load,
            np.repeat([0], timesteps),
            msg
        )
        assert grid.missed_load_percentage == 0., msg

        assert_array_equal(grid.schedule_curtailed, np.repeat([2000], timesteps)) 
        assert_approx_equal(grid.schedule_curtailed_percentage, 2/3)
