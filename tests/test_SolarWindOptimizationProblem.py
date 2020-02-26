import pytest

from examples.hybrid_opt.solar_wind_optimization_problem import SolarWindOptimizationProblem
from examples.wind_opt.site_info import SiteInfo

from defaults.defaults_data import (
    Site,
    )

from hybrid.solar_wind.shadow_cast import get_unshaded_areas_on_site


def test_SolarWindProblem():
    site = SiteInfo(Site)
    problem = SolarWindOptimizationProblem(site, 20000)
    wind_farm = problem.windmodel

    shade_mask = get_unshaded_areas_on_site(
        wind_farm.Farm.wind_farm_xCoordinates,
        wind_farm.Farm.wind_farm_yCoordinates,
        problem.turbine_rating,
        threshold=.007,
        plot_bool=True)