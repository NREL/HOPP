import os
import shutil

import responses
from pytest import fixture
import json

from hopp import ROOT_DIR
from hopp.simulation.technologies.utility_rate import UtilityRate, URDB_BASE_URL
from hopp.utilities.keys import get_developer_nrel_gov_key

path = os.path.dirname(os.path.abspath(__file__))
path_rates = ROOT_DIR / "simulation" / "resource_files"

# these rates sometimes mysteriously disappear from URDB fyi
urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca4d1175457a39b23b3d45e

@fixture
def urdb():
    ur = UtilityRate(path_rates=path_rates, urdb_label=urdb_label)
    yield ur

    # clean up file created, if necessary
    f = os.path.join(path_rates, urdb_label + ".json")
    if os.path.exists(f):
        os.remove(f)


def test_urdb_url(urdb):
    key = get_developer_nrel_gov_key()
    assert urdb.urdb_url == f"{URDB_BASE_URL}?version=7&format=json&detail=full&getpage={urdb_label}&api_key={key}" 


@responses.activate
def test_urdb_response(urdb):
    # instead of making a real call, we'll stub using a saved JSON response
    file = str(ROOT_DIR / "simulation" / "resource_files" / "utility_rate_response.json")
    with open(file, 'r') as f:
        body = json.load(f)
    responses.add(
        responses.GET,
        urdb.urdb_url,
        body=body
    )

    resp = urdb.get_urdb_response()
    assert resp is not None
    assert('label' in resp)


def test_from_file():
    label = "utility_rate_response"
    ur =  UtilityRate(path_rates=path_rates, urdb_label=label)
    resp = ur.get_urdb_response()

    assert resp is not None
    assert('label' in resp)