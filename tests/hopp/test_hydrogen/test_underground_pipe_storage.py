import pytest

from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import Underground_Pipe_Storage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD

class TestRO_desal():
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000
    in_dict['compressor_output_pressure'] = 100

    pipe_storage = Underground_Pipe_Storage(in_dict,out_dict)
    pipe_storage.pipe_storage_costs()

    def test_capex(self):
        assert self.out_dict["pipe_storage_capex"] == self.in_dict["H2_storage_kg"]*560.0
    def test_opex(self):
        assert self.out_dict["pipe_storage_opex"] == self.in_dict["H2_storage_kg"]*84.0
