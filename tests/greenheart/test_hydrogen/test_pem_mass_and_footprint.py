import pytest
import os

from greenheart.simulation.technologies.hydrogen.electrolysis.pem_mass_and_footprint import mass, footprint


def test_footprint_0mw():

    assert footprint(0.0) == 0.0

def test_footprint_1mw():

    assert footprint(1) == 48

def test_mass():

    assert mass(0.045) == pytest.approx(900.0, rel=1E-4)