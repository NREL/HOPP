from hopp.hydrogen.h2_storage.pressure_vessel import PressureVessel
from pytest import approx

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

    def test_output_function(self):
        capex, opex, energy = self.pressure_vessel_instance.calculate_from_fit(self.pressure_vessel_instance.compressed_gas_function.capacity_1[5])
        capacity = self.pressure_vessel_instance.compressed_gas_function.capacity_1[5]
        tol = 1.0

        assert capex == approx(self.pressure_vessel_instance.compressed_gas_function.cost_kg[5]*capacity, tol)
        assert opex == approx(self.pressure_vessel_instance.compressed_gas_function.Op_c_Costs_kg[5]*capacity, tol)
        assert energy == approx(self.pressure_vessel_instance.compressed_gas_function.total_energy_used_kwh[5]*capacity, tol)

    def test_plots(self):
        self.pressure_vessel_instance.plot()

if __name__ == "__main__":
    test_set = test_pressure_vessel()
    