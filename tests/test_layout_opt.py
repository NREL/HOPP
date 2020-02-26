import pytest
from pytest import approx

from hybrid.scenario import Scenario
from parameters.parameter_data import get_input_output_data
from defaults.flatirons_site import wind_windsingleowner, Site
import hybrid.wind.opt_tools as opt_tools


def run_wind_model(systems):
    windmodel = systems['Wind']['Windpower']
    windmodel.Farm.system_capacity = max(windmodel.Turbine.wind_turbine_powercurve_powerout) \
                                     * len(windmodel.Farm.wind_farm_xCoordinates)
    windmodel.execute()


@pytest.fixture
def scenario():
    systems = {'Wind': run_wind_model}
    defaults = {'Wind': {'Windpower': wind_windsingleowner}}
    input_data, output_data = get_input_output_data(systems)
    return Scenario(defaults, input_data, output_data, systems)


def test_layout_opt(scenario):
    x0 = (1426.651162, 731.8163169, 655.9166205, 1283.272483, 312.0997645,
          1111.885174, 766.9593724, 934.1454174, 317.6134104, 59.25907537,
          1080.395979, 580.5661303, 232.6466948, 822.0345446, 786.5217163,
          212.7902666, 825.6651025, 1080.52743, 607.5484287, 408.7925702)

    verts = Site['site_boundaries']['verts_simple']

    x_opt, power_opt = opt_tools.layout_opt(x0, scenario, verts)
    print(x_opt, power_opt)
    assert(round(x_opt[0], 0) == 1514)
    assert(round(power_opt) == 61505)


