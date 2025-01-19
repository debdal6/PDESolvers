import unittest
import numpy as np

import pdesolvers.pdes.black_scholes as bse
import pdesolvers.solvers.black_scholes_solvers as solver


class TestBlackScholesEquation(unittest.TestCase):
    def test_get_option_type(self):
        pass

class TestBlackScholesSolver(unittest.TestCase):

    def setUp(self):
        self.equation = bse.BlackScholesEquation('call', 300, 1, 0.2, 0.05, 100, 200, 10000)

    def test_check_lower_boundary_for_call_explicit(self):
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()
        self.assertTrue(np.all(result[0,:]) == 0)

    def test_check_terminal_condition_call_explicit(self):
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.get_strike_price()
        expected_payoff = np.maximum(test_asset_grid - test_strike_price, 0)

        self.assertTrue(np.array_equal(result[:, -1], expected_payoff))

    def test_check_terminal_condition_put_explicit(self):
        self.equation.set_option_type('put')
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.get_strike_price()
        expected_payoff = np.maximum(test_strike_price - test_asset_grid, 0)

        self.assertTrue(np.array_equal(result[:,-1], expected_payoff))

    def test_check_absolute_difference_between_two_solvers(self):
        result1 = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()
        result2 = solver.BlackScholesCNSolver(self.equation).solve().get_result()
        diff = np.abs(result1, result2)

        self.assertTrue(np.all(diff) < 1e-4)

if __name__ == '__main__':
    unittest.main()
