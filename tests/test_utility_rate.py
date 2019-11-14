from hybrid.utility_rate import UtilityRate
import os

path = os.path.dirname(os.path.abspath(__file__))

def test_urdb_response():
    path_rates = os.path.join(path, 'data')

    # these rates sometimes mysteriously disappear from URDB fyi
    urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca4d1175457a39b23b3d45e

    urdb = UtilityRate(path_rates=path_rates, urdb_label=urdb_label)

    assert(isinstance(urdb.get_urdb_response(), dict))