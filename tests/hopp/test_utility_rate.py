from dotenv import load_dotenv
import os
import shutil

from hopp.simulation.utility_rate import UtilityRate
from hopp.utils.keys import set_developer_nrel_gov_key

path = os.path.abspath(os.path.dirname(__file__))
load_dotenv(dotenv_path=os.path.join(path, ".env"))
set_developer_nrel_gov_key(os.getenv("NREL_API_KEY"))


def test_urdb_response():
    path_rates = os.path.join(path, 'data')
    os.mkdir(path_rates)

    # these rates sometimes mysteriously disappear from URDB fyi
    urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca4d1175457a39b23b3d45e

    urdb = UtilityRate(path_rates=path_rates, urdb_label=urdb_label)
    resp = urdb.get_urdb_response()
    assert('label' in resp)

    shutil.rmtree(path_rates)
