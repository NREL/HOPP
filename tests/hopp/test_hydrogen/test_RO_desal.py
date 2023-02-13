import pytest

from hopp.hydrogen.desal.desal_model import RO_desal

# Test values are based on hand calculations

class TestRO_desal():
    rel_tol = 1E-2

    freshwater_needed = 10000   #[kg/hr]

    saltwater = RO_desal(freshwater_needed, "Seawater")
    brackish = RO_desal(freshwater_needed, "Brackish")

    def test_capacity_m3_per_hr(self):
        assert self.saltwater[0] == pytest.approx(10.03, rel=1E-5)
        assert self.brackish[0] == pytest.approx(10.03, rel=1E-5)

    def test_feedwater(self):
        assert self.saltwater[1] == pytest.approx(20.06, rel=1E-5)
        assert self.brackish[1] == pytest.approx(13.37, rel=1E-3)

    def test_power(self):
        assert self.saltwater[2] == pytest.approx(40.12, rel=1E-5)
        assert self.brackish[2] == pytest.approx(15.04, rel=1E-3)

    def test_capex(self):
        assert self.saltwater[3] == pytest.approx(91372, rel=1E-2)
        assert self.brackish[3] == pytest.approx(91372, rel=1E-2)
    
    def test_opex(self):
        assert self.saltwater[4] == pytest.approx(13447, rel=1E-2)
        assert self.brackish[4] == pytest.approx(13447, rel=1E-2)

if __name__ == "__main__":
    test_set = TestRO_desal()