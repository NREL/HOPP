from pytest import raises
from greenheart.tools.eco.utilities import ceildiv, visualize_plant


def test_visualize_plant(subtests):
    
    with subtests.test("'visualize_plant()' only works with the 'floris' wind model"):
        hopp_config ={"technologies": {"wind": {"model_name": "pysam"}}}
        with raises(NotImplementedError, match="only works with the 'floris' wind model"):
            visualize_plant(hopp_config, None, None, None, None, None, None, None, None, None, None, None)

def test_ceildiv(subtests):
    
    with subtests.test("ceildiv"):
        a = 8
        b = 3
        
        assert ceildiv(a, b) == 3
    
    with subtests.test("ceildiv with one negative value"):
        a = 8
        b = -3
        
        assert ceildiv(a, b) == -2

    with subtests.test("ceildiv with two negative values"):
        a = -8
        b = -3
        
        assert ceildiv(a, b) == 3