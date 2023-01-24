"""
Author: Cory Frontin
Date: 23 Jan 2023
Institution: National Renewable Energy Lab
Description: This file computes pressure vessel thickness, replacing Tankinator.xlsx
Sources:
    - Tankinator.xlsx
"""

import numpy as np
import scipy.optimize as opt

from typing import Union
from enum import Enum
import json
import os

import hopp.hydrogen.h2_storage.pressure_vessel.von_mises as von_mises

class MetalMaterial(object):
    """
    a class for the material properties for metals used in analysis

    :param metal_type: type of metal to use, must be defined in
            metalmaterial_properties.json
    :type metal_type: str, must be in list of known materials
    :param approx_method: method to approximate inexact table lookups
    :type approx_method: str, must be in list of known methods
    :raises NotImplementedError: if inputs are not found in known data
    """

    def __init__(self,
                 metal_type : str,
                 approx_method : str= "lookup"):
        # make sure settings are correct
        if approx_method not in ["nearest", "lookup", "interp"]:
            raise NotImplementedError("the requested approximation method " \
                                      + "(%s) is not implemented." % approx_method)
        
        with open(os.path.join(os.path.split(__file__)[0],
                  "metalmaterial_properties.json"), "r") as mmprop_file:
            mmprop= json.load(mmprop_file)
        if metal_type not in mmprop.keys():
            raise NotImplementedError("the requested metal/material (%s) has not" % metal_type \
                                      + "been implemented in metalmaterial_properties.json.\n" \
                                      + "available types:", mmprop.keys())
        self.mmprop= mmprop[metal_type] # select the relevant material property set

        # stash validated class data
        self.approx_method= approx_method
        self.metal_type= metal_type

        # density and cost per weight
        self.density= self.mmprop['density_kgccm'] # kg/ccm
        self.cost_rate= self.mmprop['costrate_$kg'] # cost per kg

    # nicely package an approximator function        
    def _get_approx_fun(approx_method):
        interp_fun= None
        if approx_method == "nearest":
            def nearest(xq, x, y):
                x= np.asarray(x)
                y= np.asarray(y)
                idx_y= np.argmin(np.abs(x - xq))
                return y[idx_y]
            return nearest
        elif approx_method == "lookup":
            def lookup(xq, x, y):
                x= np.asarray(x)
                y= np.asarray(y)
                idx_yplus= np.argmin(np.abs(x[x < xq] - xq))
                return y[x < xq][idx_yplus]
            return lookup
        elif approx_method == "interp":
            return lambda xq, x, y: np.interp(x, xq, y)
        else:
            raise LookupError("approx method (%s) not found." % approx_method)

    # stash functions to relate yield and ultimate shear (bar) to temp (degC)
    def yield_shear_fun(self, T):
        return MetalMaterial._get_approx_fun(self.approx_method)(T,
                self.mmprop['tables']['yield']['temp_degC'],
                self.mmprop['tables']['yield']['shear_bar'])
    def ultimate_shear_fun(self, T):
        return MetalMaterial._get_approx_fun(self.approx_method)(T,
                self.mmprop['tables']['ultimate']['temp_degC'],
                self.mmprop['tables']['ultimate']['shear_bar'])

class Tank(object):
    """
    a generalized class to size a pressurized gas tank
    assumed to be cylindrical with hemispherical ends

    :param tank_type: type of tank to be used, which can take values I, III, IV
            referring to all-metal, aluminum-lined carbon fiber, and HDPE-lined
            carbon fiber, respectively, as 
    :type tank_type: int, must be 1, 3, or 4
    :param material: material that the pressure vessel is made of
    :type material: str, must be in valid types
    """
    def __init__(self,
                 tank_type : int,
                 material : str,
                 yield_factor : float= 3./2.,
                 ultimate_factor : float= 2.25,
                 shear_approx = "interp"):
        # unpack the key variables
        if not tank_type in [1, 3, 4]:
            raise NotImplementedError("tank_type %d has not been implemented yet.\n" % tank_type)
        self.tank_type= tank_type

        if tank_type == 1:
            self.material= MetalMaterial(material, approx_method= shear_approx)
        else:
            raise NotImplementedError("haven't done other classes yet. -CVF")            

        # store fixed attributes
        self.yield_factor= yield_factor
        self.ultimate_factor= ultimate_factor

        # to start up: undefined geometry values
        self.length_inner= None # inner total length of tank (m)
        self.radius_inner= None # inner radius of tank (m)
        self.thickness= None # thickness of tank

        # operating conditions
        self.operating_temperature= None
        self.operating_pressure= None

        self.check_tol= 1e-10 # for validations

    # return functions for symmetry
    def get_length_inner(self):
        return self.length_inner
    def get_radius_inner(self):
        return self.radius_inner
    def get_volume_inner(self):
        """ computes the inner volume """
        return Tank.compute_hemicylinder_volume(self.radius_inner, self.length_inner)
    def get_operating_temperature(self):
        return self.operating_temperature
    def get_operating_pressure(self):
        return self.operating_pressure
    def get_thickness(self):
        return self.thickness

    # set functions: specify two of (length, radius, volume)
    def set_length_radius(self, length_in, radius_in):
        """
        set the pressure vessel dimensions by length and radius in cm and
        compute the volume in ccm
        """
        self.length_inner= length_in
        self.radius_inner= radius_in
        self.volume_inner= Tank.compute_hemicylinder_volume(radius_in, length_in)
    def set_length_volume(self, length_in, volume_in):
        """
        set pressure vessel dimensions by length in cm and volume in ccm

        sets the length and volume of the pressure volume, backsolves for the
        radius of the pressure volume
        """
        self.length_inner= length_in
        Rguess= length_in/3
        r_opt= opt.fsolve(lambda x: Tank.compute_hemicylinder_volume(x, length_in) - volume_in, Rguess)
        assert np.abs(Tank.compute_hemicylinder_volume(r_opt, length_in) - volume_in)/volume_in <= self.check_tol
        self.radius_inner= float(r_opt)
    def set_radius_volume(self, radius_in, volume_in):
        """
        set pressure vessel dimensions by radius in cm and volume in ccm

        sets the radius and volume of the pressure volume, backsolves for the
        length of the pressure volume
        """
        self.radius_inner= radius_in
        Lguess= 3*radius_in
        L_opt= opt.fsolve(lambda x: Tank.compute_hemicylinder_volume(radius_in, x) - volume_in, Lguess)
        assert np.abs(Tank.compute_hemicylinder_volume(radius_in, L_opt) - volume_in)/volume_in <= self.check_tol
        self.length_inner= float(L_opt)
    def set_operating_temperature(self, temperature_in):
        self.operating_temperature = temperature_in
    def set_operating_pressure(self, pressure_in):
        self.operating_pressure = pressure_in

    def get_length_outer(self):
        """ returns the outer length of the pressure vessel in cm """
        if None in [self.length_inner, self.thickness]:
            return None
        return Tank.compute_hemicylinder_outer_length(self.length_inner, self.thickness)
    def get_radius_outer(self):
        """ returns the outer radius of the pressure vessel in cm """
        if None in [self.radius_inner, self.thickness]:
            return None
        return Tank.compute_hemicylinder_outer_radius(self.radius_inner, self.thickness)
    def get_volume_outer(self):
        """
        returns the outer volume of the pressure vessel in ccm
        """
        if None in [self.length_inner, self.radius_inner, self.thickness]:
            return None
        return Tank.compute_hemicylinder_volume(self.get_radius_outer(), self.get_length_outer())

    def get_volume_metal(self):
        """
        returns the (unsealed) displacement volume of the pressure vessel in ccm
        """
        volume_inner= self.get_volume_inner()
        volume_outer= self.get_volume_outer()
        if None in [volume_inner, volume_outer]: return None
        assert volume_outer >= volume_inner
        return (volume_outer - volume_inner)

    def get_mass_metal(self):
        """ returns the mass of the pressure vessel in kg """
        volume_metal= self.get_volume_metal()
        if volume_metal is None: return None
        return self.material.density*volume_metal

    def get_cost_metal(self):
        """
        returns the cost of the metal in the pressure vessel in dollars
        """
        mass_metal= self.get_mass_metal()
        if mass_metal is None: return None
        return self.material.cost_rate*mass_metal
    
    def get_gravimetric_tank_efficiency(self):
        """
        returns the gravimetric tank efficiency:
        $$ \frac{m_{metal}}{V_{inner}} $$
        in L/kg
        """
        mass_metal= self.get_mass_metal()
        volume_inner= self.get_volume_inner()
        return (volume_inner/1e3)/mass_metal

    def get_yield_thickness(self,
                            pressure : float = None,
                            temperature : float = None):
        """
        gets the yield thickness

        returns the yield thickness given by:
        $$
        t_y= \frac{p R_0}{S_y} \times SF_{yield}
        $$
        with yield safety factor $SF_{yield}= 3/2$ by default

        temperature and pressure must be set in the class, or specified in this
        function

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        """

        if (temperature is None) and (self.operating_temperature is None):
            raise LookupError("you must specify an operating temperature.")
        elif temperature is None:
            temperature= self.operating_temperature

        if (pressure is None) and (self.operating_pressure is None):
            raise LookupError("you must specify an operating pressure.")
        elif pressure is None:
            pressure= self.operating_pressure

        Sy= self.material.yield_shear_fun(temperature)

        thickness_yield= pressure*self.radius_inner/Sy*self.yield_factor

        return thickness_yield

    def get_ultimate_thickness(self,
                               pressure : float = None,
                               temperature : float = None):
        """
        get the ultimate thickness

        returns the ultimate thicnkess given by:
        $$
        t_u= \frac{p R_0}{S_u} \times SF_{ultimate}
        $$
        with ultimate safety factor $SF_{yield}= 2.25$ by default

        temperature and pressure must be set in the class, or specified in this
        function

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        """

        if (temperature is None) and (self.operating_temperature is None):
            raise LookupError("you must specify an operating temperature.")
        elif temperature is None:
            temperature= self.operating_temperature

        if (pressure is None) and (self.operating_pressure is None):
            raise LookupError("you must specify an operating pressure.")
        elif pressure is None:
            pressure= self.operating_pressure

        Su= self.material.ultimate_shear_fun(temperature)

        thickness_ultimate= pressure*self.radius_inner/Su*self.ultimate_factor

        return thickness_ultimate
    
    def get_thickness_thinwall(self,
                               pressure : float = None,
                               temperature : float = None):
        """
        get the thickness based on thinwall assumptions

        maximum between yield and ultimate thickness

        temperature and pressure must be set in the class, or specified in this
        function

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        """

        t_y= self.get_yield_thickness(pressure, temperature)
        t_u= self.get_ultimate_thickness(pressure, temperature)

        thickness= max(t_y, t_u)

        return thickness
    
    def set_thickness_thinwall(self,
                               pressure : float = None,
                               temperature : float = None):
        """
        set the thickness based on thinwall assumptions

        maximum between yield and ultimate thickness

        temperature and pressure must be set in the class, or specified in this
        function

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        """

        self.thickness= self.get_thickness_thinwall(pressure, temperature)

    def get_thickness_vonmises(self,
                               pressure : float = None,
                               temperature : float = None,
                               max_cycle_iter : int = 10,
                               adj_fac_tol : float = 1e-6):
        """
        get the thickness based on a von Mises cycle

        temperature and pressure must be set in the class, or specified here

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        :param max_cycle_iter: maximum iterations for von Mises cycle
        :type max_cycle_iter: int
        :param adj_fac_tol: tolerance for close enough wall thickness adjustment
                factor
        """

        if (temperature is None) and (self.operating_temperature is None):
            raise LookupError("you must specify an operating temperature.")
        elif temperature is None:
            temperature= self.operating_temperature

        # get the limit shears
        Sy= self.material.yield_shear_fun(temperature)
        Su= self.material.ultimate_shear_fun(temperature)

        # start from the thinwall analysis
        thickness_init= self.get_thickness_thinwall(pressure, temperature)

        # check to see if von Mises analysis is even needed
        if (Tank.check_thinwall(self.radius_inner, thickness_init)) \
                and (von_mises.wallThicknessAdjustmentFactor(pressure,
                                                             self.radius_inner + thickness_init,
                                                             self.radius_inner,
                                                             Sy, Su) == 1.0):
            thickness_cycle= thickness_init # trivially satisfied
            iter_cycle= -1
            print("trivially satisfied")
        else:
            print("running von mises cycle")
            print(pressure, self.radius_inner, thickness_init, Sy, Su)
            (thickness_cycle, WTAF_cycle, iter_cycle)= \
                    von_mises.cycle(pressure, self.radius_inner, thickness_init,
                                    Sy, Su,
                                    max_iter= max_cycle_iter,
                                    WTAF_tol= adj_fac_tol)
        return thickness_cycle, iter_cycle


    def set_thickness_vonmises(self,
                               pressure : float = None,
                               temperature : float = None,
                               max_cycle_iter : int = 10,
                               adj_fac_tol : float = 1e-6):
        """
        set the thickness based on a von Mises cycle

        temperature and pressure must be set in the class, or specified here

        :param pressure: operating pressure, in bar
        :type pressure: float
        :param temperature: operating temperature, in degrees C
        :type temperature: float
        :param max_cycle_iter: maximum iterations for von Mises cycle
        :type max_cycle_iter: int
        :param adj_fac_tol: tolerance for close enough wall thickness adjustment
                factor
        """

        thickness, iter= self.get_thickness_vonmises(pressure, temperature,
                                                     max_cycle_iter, adj_fac_tol)
        self.thickness= thickness

    # useful static methods
    def compute_hemicylinder_volume(R : float, L : float) -> float:
        assert (L >= 2*R) # cylindrical tank with hemispherical ends
        return np.pi*R**2*(L - 2.*R/3.)
    def compute_hemicylinder_outer_length(L : float, t : float) -> float:
        return L + 2*t
    def compute_hemicylinder_outer_radius(R : float, t : float) -> float:
        return R + t
    def check_thinwall(Rinner : float, t : float, thinwallratio_crit= 10) -> bool:
        return (Rinner/t >= thinwallratio_crit)






# class TypeITank(Tank):
#     """
#     a class I tank: metal shell tank
#     """
#    
#     def __init__(self,
#                  material : str):
#         super().__init__(1, material)
