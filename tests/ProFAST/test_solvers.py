import ProFAST

import pytest

"""
Test that ProFAST solvers function as expected
Author: Jared J. Thomas
"""

class TestPriceSolver():
    pf = ProFAST.ProFAST('only_variables')
    sol = pf.solve_price()
    abs_tol = 1E-4

class TestInitial(TestPriceSolver):
    def test_npv_zero(self):
        assert self.sol['NPV'] == pytest.approx(0, abs=self.abs_tol)
    
    def test_irr_zero(self):
        assert self.sol['irr'][1] == pytest.approx(self.pf.vals["leverage after tax nominal discount rate"] , abs=self.abs_tol)
