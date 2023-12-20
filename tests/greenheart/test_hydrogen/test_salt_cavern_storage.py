import pytest
from pytest import fixture

from greenheart.simulation.technologies.hydrogen.h2_storage.salt_cavern.salt_cavern import SaltCavernStorage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD
in_dict = {
    "H2_storage_kg": 1000000,
    "system_flow_rate":100000,
    "model":'papadias'}

@fixture
def salt_cavern_storage():
    salt_cavern_storage = SaltCavernStorage(in_dict)

    return salt_cavern_storage


def test_init():
    salt_cavern_storage = SaltCavernStorage(in_dict)

    assert salt_cavern_storage.input_dict is not None
    assert salt_cavern_storage.output_dict is not None

def test_capex_per_kg(salt_cavern_storage):
    salt_cavern_storage_capex_per_kg, _installed_capex, _compressor_capex = salt_cavern_storage.salt_cavern_capex()
    assert salt_cavern_storage_capex_per_kg == pytest.approx(25.18622259358959)

def test_capex(salt_cavern_storage):
    _salt_cavern_storage_capex_per_kg, installed_capex, _compressor_capex = salt_cavern_storage.salt_cavern_capex()
    assert installed_capex == pytest.approx(24992482.4198)

def test_compressor_capex(salt_cavern_storage):
    _salt_cavern_storage_capex_per_kg, _installed_capex, compressor_capex = salt_cavern_storage.salt_cavern_capex()
    assert compressor_capex == pytest.approx(6516166.67163)

def test_capex_output_dict(salt_cavern_storage):
    _salt_caven_storage_capex_per_kg, _installed_capex, _compressor_capex = salt_cavern_storage.salt_cavern_capex()
    assert salt_cavern_storage.output_dict["salt_cavern_storage_capex"] == pytest.approx(24992482.4198)


def test_opex(salt_cavern_storage):
    _salt_cavern_storage_capex_per_kg, _installed_capex, _compressor_capex = salt_cavern_storage.salt_cavern_capex()
    salt_cavern_storage.salt_cavern_opex()
    assert salt_cavern_storage.output_dict["salt_cavern_storage_opex"] == pytest.approx(1461664)