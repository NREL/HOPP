
from hopp.hydrogen.h2_storage.pressure_vessel.tankinator import Tank
import hopp.hydrogen.h2_storage.pressure_vessel.von_mises as von_mises
import pytest
import numpy as np

# test that we the results we got when the code was recieved
class TestTankinator():

    def test_tank_type(self):
        tank1= Tank(1, '6061_T6_Aluminum')
        # tank3= Tank(3, '316SS')
        # tank4= Tank(4, 'steel')

        assert tank1.tank_type == 1
        # assert tank3.tank_type == 3
        # assert tank4.tank_type == 4

    def test_tank_type_exc(self):
        with pytest.raises(TypeError):
            Tank(2)
        with pytest.raises(TypeError):
            Tank("x")

    def test_hemicylindrical_static(self):
        """ a random sphere, and a random cylinder """

        # L = 2R; should reduce to a sphere!
        radius= 1.5
        length= 2*radius
        thickness= 0.19
        volume_exact= 4./3.*np.pi*radius**3

        assert Tank.compute_hemicylinder_outer_length(length, thickness) == pytest.approx(length + 2.0*thickness)
        assert Tank.compute_hemicylinder_outer_radius(radius, thickness) == pytest.approx(radius + thickness)
        assert Tank.compute_hemicylinder_volume(radius, length) == pytest.approx(volume_exact)

        # a lil bit between sphere halves
        radius= 2.1
        length= 6.4
        thickness= 0.02
        volume_exact= 4./3.*np.pi*radius**3 + (length - 2*radius)*(np.pi*radius**2)

        assert Tank.compute_hemicylinder_outer_length(length, thickness) == pytest.approx(length + 2.0*thickness)
        assert Tank.compute_hemicylinder_outer_radius(radius, thickness) == pytest.approx(radius + thickness)
        assert Tank.compute_hemicylinder_volume(radius, length) == pytest.approx(volume_exact)

    def test_tank_geometric(self):
        """ make sure geometric calls work correctly """

        # L = 2R; should reduce to a sphere!
        radius= 1.5
        length= 2*radius
        thickness= 0.19
        volume_exact= 4./3.*np.pi*radius**3

        # create tank, set dimensions, check dimensioning
        tank= Tank(1, "6061_T6_Aluminum")
        tank.set_length_radius(length, radius)
        tank.thickness = thickness # manual override
        tank.volume_inner = Tank.compute_hemicylinder_volume(tank.get_radius_inner(),
                                                             tank.get_length_inner())

        assert tank.get_length_outer() == pytest.approx(length + 2.0*thickness)
        assert tank.get_radius_outer() == pytest.approx(radius + thickness)
        assert tank.get_volume_inner() == pytest.approx(volume_exact)

        volume_outer_exact= 4./3.*np.pi*(radius + thickness)**3
        dvolume_exact= volume_outer_exact - volume_exact
        rho_ref= 0.002663
        cost_ref= 4.45
        mass_exact= dvolume_exact*rho_ref # mass of spherical vessel
        cost_exact= mass_exact*cost_ref

        assert tank.get_volume_outer() == pytest.approx(volume_outer_exact)
        assert tank.get_volume_metal() == pytest.approx(dvolume_exact)
        assert tank.get_mass_metal() == pytest.approx(mass_exact)
        assert tank.get_cost_metal() == pytest.approx(cost_exact)

        assert tank.get_gravimetric_tank_efficiency() == pytest.approx((volume_exact/1e3)/mass_exact)
    
    def test_tank_set_functions(self):
        """ make sure that the inverse geometry spec works """

        radius= 4.3
        length= 14.9
        volume_exact= 4./3.*np.pi*radius**3 + (length - 2*radius)*(np.pi*radius**2)

        tank= Tank(1, "316SS")
        tank.set_length_radius(length, radius)
        assert tank.get_volume_inner() == pytest.approx(volume_exact)
        
        tank.length_inner= tank.radius_inner= None # reset
        tank.set_length_volume(length, volume_exact)
        assert tank.get_radius_inner() == pytest.approx(radius)

        tank.length_inner= tank.radius_inner= None # reset
        tank.set_radius_volume(radius, volume_exact)
        assert tank.get_length_inner() == pytest.approx(length)

    def test_tankinator_comp(self):
        """ compare to the tankinator case """

        T_op= -50 # degC
        p_op= 170 # bar
        Ltank= 1000 # cm
        Vtank= 2994542 # ccm

        # reference values from the excel sheet default values
        R_ref= 31.2
        Sy_ref= 2953.475284
        Su_ref= 3327.58062
        density_ref= 0.002663
        costrate_ref= 4.45
        yield_thickness_ref= 2.69
        ultimate_thickness_ref= 3.586389441300914
        disp_vol_tw_ref= 7.462e5
        mass_tw_ref= 1987.08
        cost_tw_ref= 8842.51
        grav_eff_tw_ref= 1.51
        vmS1_0_ref= 1568.544508
        vmS2_0_ref= 699.2722538
        vmS3_0_ref= -170.
        vmSproof_0_ref= 2258.435564
        vmSburst_0_ref= 3387.653346
        WTAF_0_ref= 1.018052974
        thickness_1_ref= ultimate_thickness_ref*WTAF_0_ref
        vmS1_1_ref= 1542.397758
        vmS2_1_ref= 686.1988791
        vmS3_1_ref= -170.
        vmSproof_1_ref= 2224.46994
        vmSburst_1_ref= 3336.70491
        WTAF_1_ref= 1.002742019
        thickness_2_ref= thickness_1_ref*WTAF_1_ref
        thickness_f_ref= 3.6626944898294997
        vmS1_f_ref= 1537.826857
        vmS2_f_ref= 683.9134286
        vmS3_f_ref= -170.
        vmSproof_f_ref= 2218.532165
        vmSburst_f_ref= 3327.798248
        WTAF_f_ref= 1.000065401
        mass_f_ref= 2031.9
        cost_f_ref= 9041.80

        # set up w/ lookup shear approximation
        tank= Tank(1, "6061_T6_Aluminum", shear_approx= "lookup")
        tank.set_operating_temperature(T_op)
        tank.set_operating_pressure(p_op)
        tank.set_length_volume(Ltank, Vtank)

        # check agains reference values
        assert tank.get_radius_inner() == pytest.approx(R_ref)
        assert tank.material.ultimate_shear_fun(T_op) == pytest.approx(Su_ref)
        assert tank.material.yield_shear_fun(T_op) == pytest.approx(Sy_ref)
        assert tank.material.density == pytest.approx(density_ref)
        assert tank.material.cost_rate == pytest.approx(costrate_ref)

        # check the thinwall calculations
        assert tank.get_yield_thickness() == pytest.approx(yield_thickness_ref, abs= 0.01)
        assert tank.get_ultimate_thickness() == pytest.approx(ultimate_thickness_ref, abs= 0.001)
        assert tank.get_thickness_thinwall() == pytest.approx(ultimate_thickness_ref, abs= 0.001)

        # check the implied geometry if we set the thickness to the thinwall
        tank.set_thickness_thinwall()
        assert tank.get_volume_metal() == pytest.approx(disp_vol_tw_ref, rel= 0.001)
        assert tank.get_mass_metal() == pytest.approx(mass_tw_ref, rel= 0.001)
        assert tank.get_cost_metal() == pytest.approx(cost_tw_ref)
        assert tank.get_gravimetric_tank_efficiency() == pytest.approx(grav_eff_tw_ref, abs= 0.01)

        # check von Mises analysis variables
        assert von_mises.S1(p_op, R_ref + ultimate_thickness_ref, R_ref) == pytest.approx(vmS1_0_ref)
        assert von_mises.S2(p_op, R_ref + ultimate_thickness_ref, R_ref) == pytest.approx(vmS2_0_ref)
        assert von_mises.S3(p_op, R_ref + ultimate_thickness_ref, R_ref) == pytest.approx(vmS3_0_ref)
        vmSproof, vmSburst= von_mises.getPeakStresses(p_op, R_ref + ultimate_thickness_ref, R_ref)
        assert vmSproof == pytest.approx(vmSproof_0_ref)
        assert vmSburst == pytest.approx(vmSburst_0_ref)
        assert not Tank.check_thinwall(R_ref, ultimate_thickness_ref)
        assert von_mises.wallThicknessAdjustmentFactor(p_op, R_ref + ultimate_thickness_ref, R_ref,
                                                       Sy_ref, Su_ref) == pytest.approx(WTAF_0_ref)
        
        # check cycle iterations, through two
        WTAF_0, thickness_1= von_mises.iterate_thickness(p_op, R_ref, ultimate_thickness_ref,
                                                         Sy_ref, Su_ref)
        assert WTAF_0 == pytest.approx(WTAF_0_ref)
        assert thickness_1 == pytest.approx(thickness_1_ref)
        
        WTAF_1, thickness_2= von_mises.iterate_thickness(p_op, R_ref, thickness_1,
                                                         Sy_ref, Su_ref)
        assert WTAF_1 == pytest.approx(WTAF_1_ref)
        assert thickness_2 == pytest.approx(thickness_2_ref)
        
        # check final value: cycle three times (no tol) to match tankinator
        (thickness_cycle, WTAF_cycle, n_iter)= von_mises.cycle(p_op, R_ref, ultimate_thickness_ref,
                                                               Sy_ref, Su_ref,
                                                               max_iter= 3, WTAF_tol= 0)
        
        print(thickness_cycle, WTAF_cycle, n_iter) # DEBUG
        assert thickness_cycle == pytest.approx(thickness_f_ref)

        # make sure final calculations are correct
        tank.set_thickness_vonmises(p_op, T_op, max_cycle_iter= 3, adj_fac_tol= 0.0)
        assert tank.get_thickness() == pytest.approx(thickness_f_ref)
        assert tank.get_mass_metal() == pytest.approx(mass_f_ref, abs= 0.1)
        assert tank.get_cost_metal() == pytest.approx(cost_f_ref, abs= 0.01)    

if __name__ == "__main__":
    test_set = test_tankinator()
