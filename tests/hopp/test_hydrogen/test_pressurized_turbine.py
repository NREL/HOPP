
from hopp.hydrogen.h2_storage.on_turbine.on_turbine_hydrogen_storage import PressurizedTower
from pytest import approx
import numpy as np

class TestPressurizedTower():

    def test_frustum(self):
        """
        test static methods for geometry of a frustum
        """

        # try a cone
        D_ref= 15.
        h_ref= 2.1
        V_cone_ref= np.pi/3.*(D_ref/2)**2*h_ref

        V_cone= PressurizedTower.compute_frustum_volume(h_ref, D_ref, 0.0)

        assert V_cone == approx(V_cone_ref)

        # try a cylinder
        D_ref= 45.
        h_ref= 612.3
        V_cyl_ref= np.pi/4.*D_ref**2*h_ref

        V_cyl= PressurizedTower.compute_frustum_volume(h_ref, D_ref, D_ref)

        assert V_cyl == approx(V_cyl_ref)

        # try a frustum by delta of two cones
        D0_ref= 5.7
        h0_ref= 0.0 + 1.0
        D1_ref= 1.9
        h1_ref= 2.4 + 1.0
        D2_ref= 0.0
        h2_ref= 2.4 + 1.2 + 1.0
        V_2_ref= np.pi/3.*(D1_ref/2)**2*(h2_ref - h1_ref)
        V_12_ref= np.pi/3.*(D0_ref/2)**2*(h2_ref - h0_ref)
        V_1_ref= V_12_ref - V_2_ref

        V_1= PressurizedTower.compute_frustum_volume(h1_ref - h0_ref, D0_ref, D1_ref)
        V_12= PressurizedTower.compute_frustum_volume(h2_ref - h0_ref, D0_ref, D2_ref)
        V_2= PressurizedTower.compute_frustum_volume(h2_ref - h1_ref, D1_ref, D2_ref)

        assert V_1 == approx(V_1_ref)
        assert V_12 == approx(V_12_ref)
        assert V_2 == approx(V_2_ref)

    def test_crossover_pressure(self):

        # random plausible values
        E= 0.8
        Sut= 1200e3
        d_t_ratio= 321.

        p_ref= 4*E*Sut/(7*d_t_ratio*(1 - E/7.))

        p= PressurizedTower.get_crossover_pressure(E, Sut, d_t_ratio)

        assert p == approx(p_ref)

    if False: # paper values are untrustworthy
        def test_thickness_increment_const(self):

            # plot values
            p= 600.0e3
            Sut= 636.0e6
            
            alpha_dtp= PressurizedTower.get_thickness_increment_const(p, Sut)

            # values from graph seem to be off by a factor of ten... jacked up!!!
            # these are the values as I read them off the graph assuming that factor
            # is in fact erroneous
            assert 2*1.41*alpha_dtp == approx(0.675e-3, rel= 0.05)
            assert 2*2.12*alpha_dtp == approx(0.900e-3, abs= 0.05)
            assert 2*2.82*alpha_dtp == approx(1.250e-3, abs= 0.05)

    def test_cylinder(self):
        """
        a hypothetical (nonsensical) cylindical tower -> easy to compute
        """

        h_ref= 100.
        D_ref= 10.
        d_t_ratio_ref= 320.
        density_steel_ref= 7817 # kg/m^3
        costrate_steel_ref= 1.50

        thickness_wall_ref= D_ref/d_t_ratio_ref
        thickness_top_ref= 8.7e-3
        thickness_bot_ref= 17.4e-3
        surfacearea_wall_ref= np.pi*D_ref*h_ref
        surfacearea_cap_ref= np.pi/4.*D_ref**2
        volume_wall_trad_ref= surfacearea_wall_ref*thickness_wall_ref
        volume_inner_ref= h_ref*surfacearea_cap_ref
        volume_cap_top_trad_ref= surfacearea_cap_ref*thickness_top_ref
        volume_cap_bot_trad_ref= surfacearea_cap_ref*thickness_bot_ref

        mass_wall_trad_ref= density_steel_ref*volume_wall_trad_ref
        mass_cap_top_trad_ref= density_steel_ref*volume_cap_top_trad_ref
        mass_cap_bot_trad_ref= density_steel_ref*volume_cap_bot_trad_ref
        tower_cost_trad_ref= costrate_steel_ref*(mass_wall_trad_ref + mass_cap_top_trad_ref + mass_cap_bot_trad_ref)

        p_crossover_ref= 1098.e3


        turbine= {
            'tower_length': h_ref,
            'section_diameters': [D_ref, D_ref],
            'section_heights': [0., 0. + h_ref],
        }

        pressurized_cylinder= PressurizedTower(1992, turbine)
        pressurized_cylinder.run()

        assert pressurized_cylinder.get_tower_inner_volume() == approx(volume_inner_ref)
        assert pressurized_cylinder.get_operating_pressure() == approx(p_crossover_ref, rel= 0.01)
        assert pressurized_cylinder.get_tower_material_volume(pressure= 0)[0] == approx(volume_wall_trad_ref)
        assert pressurized_cylinder.get_tower_material_volume(pressure= 0)[1] == approx(volume_cap_bot_trad_ref)
        assert pressurized_cylinder.get_tower_material_volume(pressure= 0)[2] == approx(volume_cap_top_trad_ref)
        assert pressurized_cylinder.get_tower_material_cost(pressure= 0) == approx(tower_cost_trad_ref)

    if False:
        def test_paper(self):
        
            h_ref= 84.
            D_bot_ref= 5.66
            D_top_ref= 2.83
            t_bot= 17.3e-3
            t_top= 8.7e-3
            d_t_ratio_ref= 320.
            rho_density_ref= 7817
            costrate_steel= 1.50
            cost_tower_ref= 183828
            cost_nonwall_ref= 32.80*h_ref + 2000.

            turbine= {
                'tower_length': 84.,
                'section_diameters': [5.66, 4.9525, 4.245, 3.5375, 2.83],
                'section_heights': [0., 21., 42., 63., 84.]
            }
        
            pressurized_tower_instance= PressurizedTower(2004, turbine)
            pressurized_tower_instance.run()
          
            Vinner_balpark= PressurizedTower.compute_frustum_volume(h_ref, D_bot_ref, D_top_ref)


            # assert False

if __name__ == "__main__":
    test_set= test_pressurized_tower()
