import os

import requests
import pytest
import responses

from hopp.simulation.technologies.resource import Resource

api_url = "https://api.example.com/data"
fname = "testfile.csv"


@responses.activate
def test_call_api():
    responses.add(
        responses.GET,
        api_url,
        status=200,
    )
    assert Resource.call_api(api_url, fname) == True
    
    # cleanup
    os.remove(fname)    

@responses.activate
def test_call_api_timeout():
    resp = responses.add(
        responses.GET,
        api_url,
        body=requests.exceptions.Timeout()
    )
    assert Resource.call_api(api_url, fname) == False
    assert resp.call_count == 5

@responses.activate
def test_call_api_400_error():
    responses.add(
        responses.GET,
        api_url,
        json={'errors': 'Bad request'},
        status=400
    )
    with pytest.raises(requests.exceptions.HTTPError):
        Resource.call_api(api_url, fname)

@responses.activate
def test_call_api_403_error():
    responses.add(
        responses.GET,
        api_url,
        json={'errors': 'Forbidden'},
        status=403
    )
    with pytest.raises(requests.exceptions.HTTPError):
        Resource.call_api(api_url, fname)

@responses.activate
def test_call_api_404_error():
    responses.add(
        responses.GET,
        api_url,
        status=404
    )
    with pytest.raises(requests.exceptions.HTTPError):
        Resource.call_api(api_url, fname)

@responses.activate
def test_call_api_429_error():
    responses.add(
        responses.GET,
        api_url,
        status=429
    )
    with pytest.raises(RuntimeError):
        Resource.call_api(api_url, fname)