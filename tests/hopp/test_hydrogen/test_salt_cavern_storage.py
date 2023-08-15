import pytest

from hopp.hydrogen.h2_storage.salt_cavern.salt_cavern import Salt_Cavern_Storage

# Test values are based on conclusions of Papadias 2021 and are in 2019 USD

class TestSalt_cavern_storage():
    in_dict = dict()
    out_dict = dict()
    in_dict['H2_storage_kg'] = 1000000

    salt_cavern_storage = Salt_Cavern_Storage(in_dict,out_dict)
    salt_cavern_storage.salt_cavern_storage_costs()

    def test_capex(self):
        assert self.out_dict["salt_cavern_storage_capex"] == pytest.approx(25.18622259358959)
    def test_opex(self):
        assert self.out_dict["salt_cavern_storage_opex"] == self.in_dict["H2_storage_kg"]*84.0
