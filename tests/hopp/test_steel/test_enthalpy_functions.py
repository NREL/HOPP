import pytest
import hopp.simulation.technologies.steel.enthalpy_functions as ep_f

def test_h2_enthalpy_upper_function():

    temp = 1500

    assert pytest.approx(ep_f.h2_enthalpy(temp),.01) == 18.002

def test_h2_enthalpy_lower_function():

    temp = 500

    assert pytest.approx(ep_f.h2_enthalpy(temp),.01) == 2.91

def test_h2_enthalpy_error_function():

    temp = 0

    with pytest.raises(ValueError):ep_f.h2_enthalpy(temp)


def test_h2o_enthalpy_function():
    
    temp = 1000
    
    assert pytest.approx(ep_f.h2o_enthalpy(temp),.01) == 1.443

def test_fe_enthalpy_upper_function():
    
    temp = 2000

    assert pytest.approx(ep_f.fe_enthalpy(temp),.01) == 1.23

def test_fe_enthalpy_lower_function():
    
    temp = 1000

    assert pytest.approx(ep_f.fe_enthalpy(temp),.01) == .369


def test_fe_enthalpy_error_function():

    temp = 0

    with pytest.raises(ValueError):ep_f.fe_enthalpy(temp)

def test_feo_enthalpy_function():
    
    temp = 1000

    assert pytest.approx(ep_f.feo_enthalpy(temp),.01) == .5394

def test_al2o3_enthalpy_function():
    
    temp = 1000

    assert pytest.approx(ep_f.al2o3_enthalpy(temp),.01) == .781

def test_sio2_enthalpy_function():
    
    temp = 1000

    assert pytest.approx(ep_f.sio2_enthalpy(temp),.01) == .754

def test_mgo_enthalpy_function():
    
    temp = 1000

    assert pytest.approx(ep_f.mgo_enthalpy(temp),.01) == .818

def test_cao_enthalpy_function():
    
    temp = 1000

    assert pytest.approx(ep_f.cao_enthalpy(temp),.01) == .628

def test_ch4_enthalpy_upper_function():

    temp = 1500

    assert pytest.approx(ep_f.ch4_enthalpy(temp),.01) == 4.87

def test_ch4_enthalpy_lower_function():

    temp = 1000

    assert pytest.approx(ep_f.ch4_enthalpy(temp),.01) == 2.38

def test_ch4_enthalpy_error_function():

    temp = 0

    with pytest.raises(ValueError):ep_f.ch4_enthalpy(temp)

def test_c_enthalpy_lower_function():

    temp = 900

    assert pytest.approx(ep_f.ch4_enthalpy(temp),.01) == 1.945