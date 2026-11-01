import sys
import copy
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data import generate
from model import load_data, solve, status
from verify import check


class PortfolioTests(unittest.TestCase):
    def test_data_reproducible(self):
        self.assertEqual(load_data(), generate())

    def test_optimum_and_independent_check(self):
        data = load_data()
        res = solve(data)
        self.assertEqual(status(res), 'optimal')
        check_result = check(data, res.x)
        self.assertTrue(check_result['ok'])
        self.assertAlmostEqual(check_result['expected_yield_pct'], -res.fun, places=7)

    def test_reject_invalid_allocation(self):
        data = load_data()
        self.assertFalse(check(data, [1.0, 0, 0, 0, 0, 0])['ok'])

    def test_infeasible_risk_is_reported(self):
        data = copy.deepcopy(load_data())
        data['risk_limits'][0] = -0.1
        self.assertEqual(status(solve(data)), 'infeasible')

    def test_stricter_risk_cannot_increase_return(self):
        data = load_data()
        base = solve(data)
        tighter = copy.deepcopy(data)
        tighter['risk_limits'][0] *= 0.8
        result = solve(tighter)
        self.assertTrue(result.success)
        self.assertLessEqual(-result.fun, -base.fun + 1e-7)

    def test_no_negative_risk_slack(self):
        data = load_data()
        result = solve(data)
        self.assertTrue(all(v >= -1e-7 for v in result.ineqlin.residual))


if __name__ == '__main__':
    unittest.main()
