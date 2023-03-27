from hopp.hydrogen.h2_storage.pressure_vessel import PressureVessel
from pytest import approx
import numpy as np

# test that we the results we got when the code was recieved
class TestPressureVessel():


    # pressure_vessel_instance_no_cost = PressureVessel(Energy_cost=0.0)
    # pressure_vessel_instance_no_cost.run()
    pressure_vessel_instance = PressureVessel(Energy_cost=0.07)
    pressure_vessel_instance.run()


    pressure_vessel_instance_no_cost = PressureVessel(Energy_cost=0.0)
    pressure_vessel_instance_no_cost.run()


        
    def test_capacity_max(self):
        assert self.pressure_vessel_instance.capacity_max == 5585222.222222222
    
    def test_t_discharge_hr_max(self):
        assert self.pressure_vessel_instance.t_discharge_hr_max == 25.133499999999998

    def test_a_fit_capex(self):
        # assert self.pressure_vessel_instance.a_fit_capex == 9084.035219940572
        assert self.pressure_vessel_instance.a_fit_capex == approx(0.053925726563169414)
    
    def test_b_fit_capex(self):
        # assert self.pressure_vessel_instance.b_fit_capex == -0.127478041731842
        assert self.pressure_vessel_instance.b_fit_capex == approx(1.6826965840450498)
    
    def test_c_fit_capex(self):
        assert self.pressure_vessel_instance.c_fit_capex == approx(20.297862568544417)

    def test_a_fit_opex(self):
        assert self.pressure_vessel_instance.a_fit_opex == approx(0.05900068374896024)

    def test_b_fit_opex(self):
        assert self.pressure_vessel_instance.b_fit_opex == approx(1.8431485607717895)

    def test_c_fit_opex(self):
        assert self.pressure_vessel_instance.c_fit_opex == approx(17.538017086792006)

    def test_energy_fit(self):
        capacity = 1E6 # 1000 tonnes h2
        _, _, energy_per_kg = self.pressure_vessel_instance.calculate_from_fit(capacity_kg=capacity)
        assert energy_per_kg == approx(2.688696, 1E-5) # kWh/kg

    def test_compare_price_change_capex(self):
        capacity = 1E6 # 1000 tonnes h2
        capex_07, _, _ = self.pressure_vessel_instance.calculate_from_fit(capacity_kg=capacity)
        capex_00, _, _ = self.pressure_vessel_instance_no_cost.calculate_from_fit(capacity_kg=capacity)

        assert capex_00 == capex_07

    def test_compare_price_change_opex(self):
        capacity = 1E6 # 1000 tonnes h2
        _, opex_07, _ = self.pressure_vessel_instance.calculate_from_fit(capacity_kg=capacity)
        _, opex_00, _ = self.pressure_vessel_instance_no_cost.calculate_from_fit(capacity_kg=capacity)

        assert opex_00 < opex_07

    def test_compare_price_change_energy(self):
        capacity = 1E6 # 1000 tonnes h2
        _, _, energy_per_kg_07 = self.pressure_vessel_instance.calculate_from_fit(capacity_kg=capacity)
        _, _, energy_per_kg_00 = self.pressure_vessel_instance_no_cost.calculate_from_fit(capacity_kg=capacity)

        assert energy_per_kg_00 == energy_per_kg_07

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

    def test_distributed(self):
        capacity = self.pressure_vessel_instance.compressed_gas_function.capacity_1[5]
        capex, opex, energy = self.pressure_vessel_instance.calculate_from_fit(capacity)

        capex_dist, opex_dist, energy_kg_dist, area_footprint_site, mass_tank_empty_site, capacity_site= \
                self.pressure_vessel_instance.distributed_storage_vessels(capacity, 5)
        assert capex_dist == approx(6205232868.4722595)
        assert opex_dist == approx(113433768.86938927)
        assert energy_kg_dist == approx(9054713.963289429)
        assert area_footprint_site == approx(4866.189496204457)
        assert mass_tank_empty_site == approx(7870274.025926539)
        assert capacity_site == approx(757994.4444444444)

        capex_dist, opex_dist, energy_kg_dist, area_footprint_site, mass_tank_empty_site, capacity_site= \
                self.pressure_vessel_instance.distributed_storage_vessels(capacity, 10)
        assert capex_dist == approx(7430302244.729572)
        assert opex_dist == approx(138351814.3102437)
        assert energy_kg_dist == approx(9054713.963289421)
        assert area_footprint_site == approx(2433.0947481022286)
        assert mass_tank_empty_site == approx(3935137.0129632694)
        assert capacity_site == approx(378997.2222222222)

        capex_dist, opex_dist, energy_kg_dist, area_footprint_site, mass_tank_empty_site, capacity_site= \
                self.pressure_vessel_instance.distributed_storage_vessels(capacity, 20)
        assert capex_dist == approx(9370417735.496975)
        assert opex_dist == approx(178586780.2083488)
        assert energy_kg_dist == approx(9054713.963289410)
        assert area_footprint_site == approx(1216.5473740511143)
        assert mass_tank_empty_site == approx(1967568.5064816347)
        assert capacity_site == approx(189498.6111111111)

        # assert False

    # def test_plots(self):
    #     self.pressure_vessel_instance.plot()

if __name__ == "__main__":
    test_set = test_pressure_vessel()
    
# 0.0
# 6322420.744236805
# 1331189.5844818645
# 7363353.502353448

# 0.07
# 6322420.744236805
# 1331189.5844818645
# 7363353.502353448

# energy cost for both cases match as per above


# op costs - 0.07
# 442569.45209657634
# 345243.94167843653
# 0
# 93183.27091373052

# op costs - 0.0
# 0.0
# 0.0
# 0
# 0.0

# op c costs
# op_c_costs 0.07
# 880996.6646887433 
#  799322.4503233839 
#  0.03 
#  4262490675.039804 
#  25920

# op_c_costs 0.00
# 0.0 
#  0.0 
#  0.03 
#  4262490675.039804 
#  25920