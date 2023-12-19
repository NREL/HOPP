import pytest
from pytest import fixture

from greenheart.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern.lined_rock_cavern import LinedRockCavernStorage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD
in_dict = {
    "H2_storage_kg": 1000000,
    "system_flow_rate":100000,
    "model":'papadias'}

@fixture
def lined_rock_cavern_storage():
    lined_rock_cavern_storage = LinedRockCavernStorage(in_dict)

    return lined_rock_cavern_storage


def test_init():
    lined_rock_cavern_storage = LinedRockCavernStorage(in_dict)

    assert lined_rock_cavern_storage.input_dict is not None
    assert lined_rock_cavern_storage.output_dict is not None

def test_capex_per_kg(lined_rock_cavern_storage):
    lined_rock_cavern_storage_capex_per_kg, _installed_capex, _compressor_capex = lined_rock_cavern_storage.lined_rock_cavern_capex()
    assert lined_rock_cavern_storage_capex_per_kg == pytest.approx(51.532548895265045)

def test_capex(lined_rock_cavern_storage):
    _lined_rock_cavern_storage_capex_per_kg, installed_capex, _compressor_capex = lined_rock_cavern_storage.lined_rock_cavern_capex()
    assert installed_capex == pytest.approx(51136144.673)

def test_compressor_capex(lined_rock_cavern_storage):
    _lined_rock_cavern_storage_capex_per_kg, _installed_capex, compressor_capex = lined_rock_cavern_storage.lined_rock_cavern_capex()
    assert compressor_capex == pytest.approx(9435600.2555)

def test_capex_output_dict(lined_rock_cavern_storage):
    _lined_rock_cavern_storage_capex_per_kg, _installed_capex, _compressor_capex = lined_rock_cavern_storage.lined_rock_cavern_capex()
    assert lined_rock_cavern_storage.output_dict["lined_rock_cavern_storage_capex"] == pytest.approx(51136144.673)


def test_opex(lined_rock_cavern_storage):
    _lined_rock_cavern_storage_capex_per_kg, _installed_capex, _compressor_capex = lined_rock_cavern_storage.lined_rock_cavern_capex()
    lined_rock_cavern_storage.lined_rock_cavern_opex()
    assert lined_rock_cavern_storage.output_dict["lined_rock_cavern_storage_opex"] == pytest.approx(2359700)