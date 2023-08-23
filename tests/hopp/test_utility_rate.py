import os
import shutil

import responses
from pytest import fixture
import json

from hopp import ROOT_DIR
from hopp.simulation.technologies.utility_rate import UtilityRate, URDB_BASE_URL

path = os.path.dirname(os.path.abspath(__file__))
path_rates = ROOT_DIR.parent / "resource_files"
api_key = "a" * 40 # dummy API key

# these rates sometimes mysteriously disappear from URDB fyi
urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca4d1175457a39b23b3d45e

@fixture
def urdb():
    ur = UtilityRate(path_rates=path_rates, urdb_label=urdb_label, api_key=api_key)
    yield ur

    # clean up file created from stubbed API response (this happens after test finishes)
    os.remove(os.path.join(path_rates, urdb_label + ".json"))


def test_urdb_url(urdb):
    assert urdb.urdb_url == f"{URDB_BASE_URL}?version=7&format=json&detail=full&getpage={urdb_label}&api_key={api_key}" 


@responses.activate
def test_urdb_response(urdb):
    # instead of making a real call, we'll stub using a saved JSON response
    file = str(ROOT_DIR.parent / "resource_files" / "utility_rate_response.json")
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
    ur =  UtilityRate(path_rates=path_rates, urdb_label=label, api_key=api_key)
    resp = ur.get_urdb_response()

    assert resp is not None
    assert('label' in resp)