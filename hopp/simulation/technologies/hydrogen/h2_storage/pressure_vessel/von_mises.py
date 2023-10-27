"""
Author: Cory Frontin
Date: 23 Jan 2023
Institution: National Renewable Energy Lab
Description: This file computes von Mises quantities for hemicylindrical tanks,
        replacing Tankinator.xlsx
Sources:
    - Tankinator.xlsx
"""

import numpy as np

def S1(p, Re, R0): # von Mises hoop stress
    return p*(Re**2 + R0**2)/(Re**2 - R0**2)
def S2(p, Re, R0): # von Mises axial stress
    return p*R0**2/(Re**2 - R0**2)
def S3(p, Re, R0): # von Mises radial stress
    return -p

def getPeakStresses(p, Re, R0,
                    proof_factor= 3./2.,
                    burst_factor= 2.25):
    aVM= np.sqrt(2)/2
    bVM= (S2(p, Re, R0) - S1(p, Re, R0))**2
    cVM= (S3(p, Re, R0) - S1(p, Re, R0))**2
    dVM= (S3(p, Re, R0) - S2(p, Re, R0))**2
    eVM= np.sqrt(bVM + cVM + dVM)
    Sproof= proof_factor*aVM*eVM
    Sburst= burst_factor*aVM*eVM
    return (Sproof, Sburst)

def wallThicknessAdjustmentFactor(p, Re, R0, Syield, Sultimate,
                                  proof_factor= 3./2.,
                                  burst_factor= 2.25):
    """
    get factor by which to increase thickness when von Mises stresses exceed
    material yield safety margins
    """
    Sproof, Sburst= getPeakStresses(p, Re, R0,
                                    proof_factor, burst_factor)
    WTAF_proof= Sproof/Syield
    WTAF_burst= Sburst/Sultimate
    WTAF= max(WTAF_proof, WTAF_burst)
    return WTAF

def iterate_thickness(p, R0, thickness_in,
                      Syield, Sultimate,
                      proof_factor= 3./2.,
                      burst_factor= 2.25):
    """
    apply the wall thickness adjustment factor, return it w/ new thickness
    """

    Router= R0 + thickness_in
    WTAF= wallThicknessAdjustmentFactor(p, Router, R0,
                                        Syield, Sultimate,
                                        proof_factor, burst_factor)
    
    return max(1.0, WTAF), max(1.0, WTAF)*thickness_in

def cycle(p, R0, thickness_init,
          Syield, Sultimate,
          proof_factor= 3./2.,
          burst_factor= 2.25,
          max_iter= 10,
          WTAF_tol= 1e-6):
    """
    cycle to find a thickness that satisfies the von Mises criteria
    """

    # compute initial thickness, WTAF
    thickness= thickness_init
    WTAF= wallThicknessAdjustmentFactor(p, R0 + thickness, R0,
                                        Syield, Sultimate,
                                        proof_factor, burst_factor)

    # iterate while WTAF is greater than zero
    n_iter= 0
    while (WTAF - 1.0 > WTAF_tol) and (n_iter < max_iter):

        n_iter += 1 # this cycle iteration number

        # get the next thickness
        WTAF, thickness= iterate_thickness(p, R0, thickness,
                                           Syield, Sultimate,
                                           proof_factor, burst_factor)
    
    return (thickness, WTAF, n_iter)


