import pytest
from pytest import fixture

from greenheart.simulation.technologies.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import UndergroundPipeStorage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD

in_dict = {
    "H2_storage_kg": 1000000,
    "system_flow_rate":100000,
    "model":'papadias',
    "compressor_output_pressure": 100}

@fixture
def pipe_storage():
    pipe_storage = UndergroundPipeStorage(in_dict)

    return pipe_storage


def test_init():
    pipe_storage = UndergroundPipeStorage(in_dict)

    assert pipe_storage.input_dict is not None
    assert pipe_storage.output_dict is not None

def test_capex_per_kg(pipe_storage):
    pipe_storage_capex_per_kg, _installed_capex, _compressor_capex = pipe_storage.pipe_storage_capex()
    assert pipe_storage_capex_per_kg == pytest.approx(512.689247292)

def test_capex(pipe_storage):
    _pipe_storage_capex_per_kg, installed_capex, _compressor_capex = pipe_storage.pipe_storage_capex()
    assert installed_capex == pytest.approx(508745483.851)

def test_compressor_capex(pipe_storage):
    _pipe_storage_capex_per_kg, _installed_capex, compressor_capex = pipe_storage.pipe_storage_capex()
    assert compressor_capex == pytest.approx(5907549.297)

def test_capex_output_dict(pipe_storage):
    _pipe_storage_capex_per_kg, _installed_capex, _compressor_capex = pipe_storage.pipe_storage_capex()
    assert pipe_storage.output_dict["pipe_storage_capex"] == pytest.approx(508745483.851)


def test_opex(pipe_storage):
    _pipe_storage_capex_per_kg, _installed_capex, _compressor_capex = pipe_storage.pipe_storage_capex()
    pipe_storage.pipe_storage_opex()
    assert pipe_storage.output_dict["pipe_storage_opex"] == pytest.approx(16439748)

