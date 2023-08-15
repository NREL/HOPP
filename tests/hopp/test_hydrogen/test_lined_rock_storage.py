import pytest

from hopp.hydrogen.h2_storage.lined_rock_cavern.lined_rock_cavern import Lined_Rock_Cavern_Storage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD

class TestLined_rock_cavern_storage():
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000000

    salt_cavern_storage = Lined_Rock_Cavern_Storage(in_dict,out_dict)
    salt_cavern_storage.lined_rock_cavern_storage_costs()

    def test_capex(self):
        assert self.out_dict["lined_rock_cavern_storage_capex"] == pytest.approx(51.532548895265045)
    def test_opex(self):
        assert self.out_dict["lined_rock_cavern_storage_opex"] == self.in_dict["H2_storage_kg"]*84.0
