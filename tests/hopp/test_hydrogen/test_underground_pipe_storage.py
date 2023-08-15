import pytest

from hopp.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import Underground_Pipe_Storage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD

class TestUnderground_pipe_storage():
    in_dict = {
        "H2_storage_kg": 1000000,
        "system_flow_rate":100000,
        "model":'papadias',
        "compressor_output_pressure": 100}
    out_dict = dict()

    pipe_storage = Underground_Pipe_Storage(in_dict,out_dict)
    pipe_storage_capex_per_kg, installed_capex, compressor_capex = pipe_storage.pipe_storage_capex()
    pipe_storage.pipe_storage_opex()

    def test_capex_per_kg(self):
        assert self.pipe_storage_capex_per_kg == pytest.approx(512.689247292)
    def test_capex(self):
        assert self.out_dict["pipe_storage_capex"] == pytest.approx(512.689247292)
    def test_opex(self):
        assert self.out_dict["pipe_storage_opex"] == self.in_dict["H2_storage_kg"]*84.0
