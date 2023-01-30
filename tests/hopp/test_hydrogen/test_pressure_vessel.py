from hopp.hydrogen.h2_storage.pressure_vessel import PressureVessel
from pytest import approx
import numpy as np

# test that we the results we got when the code was recieved
class TestPressureVessel():

    pressure_vessel_instance = PressureVessel()
    pressure_vessel_instance.run()
        
    def test_capacity_max(self):
        assert self.pressure_vessel_instance.capacity_max == 5585222.222222222
    
    def test_t_discharge_hr_max(self):
        assert self.pressure_vessel_instance.t_discharge_hr_max == 25.133499999999998

    def test_a_fit_capex(self):
        # assert self.pressure_vessel_instance.a_fit_capex == 9084.035219940572
        assert self.pressure_vessel_instance.a_fit_capex == 0.053925726563169414
    
    def test_b_fit_capex(self):
        # assert self.pressure_vessel_instance.b_fit_capex == -0.127478041731842
        assert self.pressure_vessel_instance.b_fit_capex == 1.6826965840450498
    
    def test_c_fit_capex(self):
        assert self.pressure_vessel_instance.c_fit_capex == 20.297862568544417

    def test_a_fit_opex(self):
        assert self.pressure_vessel_instance.a_fit_opex == 0.06772147747187984

    def test_b_fit_opex(self):
        assert self.pressure_vessel_instance.b_fit_opex == 3.0909992158528348

    def test_c_fit_opex(self):
        assert self.pressure_vessel_instance.c_fit_opex == 19.29450153559374

    def test_mass_footprint(self):
        """
        extension of gold standard test to new tank footprint outputs
        """
        cap_H2_tank_ref= 179.6322517351785
        capacity_req= 1.2e3
        Ntank_ref= np.ceil(capacity_req/cap_H2_tank_ref)
        Atank_ref= (2*53.7e-2)**2
        footprint_ref= Ntank_ref*Atank_ref
        mass_ref= 1865.0

        assert self.pressure_vessel_instance.get_tanks(capacity_req) == approx(Ntank_ref)
        assert self.pressure_vessel_instance.get_tank_footprint(capacity_req)[0] == approx(Atank_ref, rel= 0.01)
        assert self.pressure_vessel_instance.get_tank_footprint(capacity_req)[1] == approx(footprint_ref, rel= 0.01)

        assert self.pressure_vessel_instance.get_tank_mass(capacity_req)[0] == approx(mass_ref, rel= 0.01)


    def test_output_function(self):
        capacity = self.pressure_vessel_instance.compressed_gas_function.capacity_1[5]
        capex, opex, energy = self.pressure_vessel_instance.calculate_from_fit(capacity)
        tol = 1.0

        assert capex == approx(self.pressure_vessel_instance.compressed_gas_function.cost_kg[5]*capacity, tol)
        assert opex == approx(self.pressure_vessel_instance.compressed_gas_function.Op_c_Costs_kg[5]*capacity, tol)
        assert energy == approx(self.pressure_vessel_instance.compressed_gas_function.total_energy_used_kwh[5]*capacity, tol)

    def test_plots(self):
        self.pressure_vessel_instance.plot()

if __name__ == "__main__":
    test_set = test_pressure_vessel()
    