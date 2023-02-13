import pytest
import os
import sys
sys.path.append("c:/Users/CKIEFER/HOPP-KIEFER/")
from hopp.hydrogen.desal.desal_model import RO_desal


def test_RO_Desal_Seawater():
    '''Test Seawater RO Model'''
    outputs=RO_desal(freshwater_kg_per_hr=997,salinity='Seawater')
    RO_desal_mass = outputs[5]
    RO_desal_footprint = outputs[6]
    assert pytest.approx(RO_desal_mass) == 346.7
    assert pytest.approx(RO_desal_footprint) == .467
    assert pytest.approx(outputs) == (1.0, 1.3, 1.5, 9109.8, 1340.68, 346.7, 0.467)
    

def test_RO_Desal_Brackish():
    '''Test Brackish Model'''
    outputs=RO_desal(freshwater_kg_per_hr=997,salinity='Brackish')
    RO_desal_mass = outputs[5]
    RO_desal_footprint = outputs[6]
    assert pytest.approx(RO_desal_mass) == 346.7
    assert pytest.approx(RO_desal_footprint) == .467
    assert pytest.approx(outputs) == (1.0, 1.3, 1.5, 9109.8, 1340.68, 346.7, 0.467)
    


