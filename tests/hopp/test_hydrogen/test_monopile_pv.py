from hopp.hydrogen.h2_storage.monopile_pv_storage.monopile_pv_storage import MonopilePressureVesselJacket
from pytest import approx
import numpy as np

import pprint as pp

class TestMonopileJacket():
    
    monopile= {
        'diameter': 10.,
        'arcangle_reserved': np.pi/2,
        'height_available': 10.5,
    }
    monopile_pv_instance= MonopilePressureVesselJacket(monopile)

    def test_tankinator_ref(self):
        
        # operating conditions
        pressure= 170 # Bar
        Ri_tank= 0.312 # m
        Li_tank= 10.0 # m

        # reference values
        thickness_ref= 3.663 # cm
        mass_thinwall_ref= 1987.08 # kg
        Ncol_ref= int(36.08690959103532)
        Nrow_ref= 1
        install_factor_ref= 0.3

        # design using a fixed choice of tank
        self.monopile_pv_instance.design_fixed_tank(pressure, Ri_tank, Li_tank,
                                                    temp_ambient= -50,
                                                    material= '6061_T6_Aluminum')

        # print the resulting design
        pp.pprint(self.monopile_pv_instance.design)

        # get thickness for sanity-checking
        thickness= self.monopile_pv_instance.tank.get_thickness()
        mass_thinwall= self.monopile_pv_instance.tank.get_mass_metal()

        # make sure the thickness estimate is about right        
        assert thickness == approx(thickness_ref, rel= 0.05)
        assert mass_thinwall == approx(mass_thinwall_ref, rel= 0.05)

        # number of rows should be right
        assert self.monopile_pv_instance.design['Ncolumn'] == Ncol_ref
        assert self.monopile_pv_instance.design['Nrow'] == Nrow_ref

        # make sure the jacket masses are right
        assert self.monopile_pv_instance.design['mass_tanks_empty'] \
            == approx(Nrow_ref*Ncol_ref*mass_thinwall_ref, rel= 0.05)
        assert self.monopile_pv_instance.design['mass_jacket_installed_empty'] \
            == approx(Nrow_ref*Ncol_ref*mass_thinwall_ref*(1 + install_factor_ref), rel= 0.05)

if __name__ == "__main__":
    raise NotImplementedError("main not implemented! run with pytest")
