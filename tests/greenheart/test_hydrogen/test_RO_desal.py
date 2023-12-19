from pytest import approx

from greenheart.simulation.technologies.hydrogen.desal.desal_model_eco import RO_desal_eco

# Test values are based on hand calculations

class TestRO_desal():
    rel_tol = 1E-2

    freshwater_needed = 10000   #[kg/hr]

    saltwater = RO_desal_eco(freshwater_needed, "Seawater")
    brackish = RO_desal_eco(freshwater_needed, "Brackish")

    def test_capacity_m3_per_hr(self):
        assert self.saltwater[0] == approx(10.03, rel=1E-5)
        assert self.brackish[0] == approx(10.03, rel=1E-5)

    def test_feedwater(self):
        assert self.saltwater[1] == approx(20.06, rel=1E-5)
        assert self.brackish[1] == approx(13.37, rel=1E-3)

    def test_power(self):
        assert self.saltwater[2] == approx(40.12, rel=1E-5)
        assert self.brackish[2] == approx(15.04, rel=1E-3)

    def test_capex(self):
        assert self.saltwater[3] == approx(91372, rel=1E-2)
        assert self.brackish[3] == approx(91372, rel=1E-2)
    
    def test_opex(self):
        assert self.saltwater[4] == approx(13447, rel=1E-2)
        assert self.brackish[4] == approx(13447, rel=1E-2)

    def test_RO_Desal_Seawater(self):
        '''Test Seawater RO Model'''
        outputs=RO_desal_eco(freshwater_kg_per_hr=997,salinity='Seawater')
        RO_desal_mass = outputs[5]
        RO_desal_footprint = outputs[6]
        assert approx(RO_desal_mass) == 346.7
        assert approx(RO_desal_footprint) == .467

    def test_RO_Desal_distributed(self):
        '''Test Seawater RO Model'''
        n_systems = 2
        total_freshwater_kg_per_hr_required = 997
        per_system_freshwater_kg_per_hr_required = total_freshwater_kg_per_hr_required/n_systems

        total_outputs=RO_desal_eco(freshwater_kg_per_hr=total_freshwater_kg_per_hr_required,salinity='Seawater')
        per_system_outputs = RO_desal_eco(per_system_freshwater_kg_per_hr_required, salinity="Seawater")

        for t, s in zip(total_outputs, per_system_outputs):
            assert t == approx(s*n_systems)

    
    def test_RO_Desal_Brackish(self):
        '''Test Brackish Model'''
        outputs=RO_desal_eco(freshwater_kg_per_hr=997,salinity='Brackish')
        RO_desal_mass = outputs[5]
        RO_desal_footprint = outputs[6]
        assert approx(RO_desal_mass) == 346.7
        assert approx(RO_desal_footprint) == .467
        
if __name__ == "__main__":
    test_set = TestRO_desal()