"""Test functions for stats module"""

import unittest
import numpy as np

import dyad
import data_test

from parameterized import parameterized, parameterized_class


class TestMassRatioFunctions(unittest.TestCase):
    @parameterized.expand(data_test.mass_ratio_func)
    def test_func(self, x):
        self.assertRaises(ValueError, dyad.stats.mass_ratio.func(*x))
        
    @parameterized.expand(data_test.mass_ratio_moe2017_norm)
    def test_moe2017_norm(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_norm(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.mass_ratio_moe2017_twin_excess_constant)
    def test_(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_twin_excess_constant(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.mass_ratio_moe2017_twin_excess_ratio)
    def test_moe2017_twin_excess_ratio(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_twin_excess_ratio(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(
        data_test.mass_ratio_moe2017_log10_excess_twin_period
    )
    def test_moe2017_log10_excess_twin_period(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_log10_excess_twin_period(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.mass_ratio_moe2017_gamma)
    def test_moe2017_gamma(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_gamma(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.mass_ratio_moe2017_delta)
    def test_moe2017_delta(self, x, target):
        result = dyad.stats.mass_ratio._moe2017_delta(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)


class TestPeriodFunctions(unittest.TestCase):
    @parameterized.expand(data_test.period_moe2017_norm)
    def test_moe2017_norm(self, x, target):
        result = dyad.stats.period._moe2017_norm(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_1)
    def test_moe2017_c_1(self, x, target):
        result = dyad.stats.period._moe2017_c_1(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_2)
    def test_moe2017_c_2(self, x, target):
        result = dyad.stats.period._moe2017_c_2(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_3)
    def test_moe2017_c_3(self, x, target):
        result = dyad.stats.period._moe2017_c_3(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_4)
    def test_moe2017_c_4(self, x, target):
        result = dyad.stats.period._moe2017_c_4(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_5)
    def test_moe2017_c_5(self, x, target):
        result = dyad.stats.period._moe2017_c_5(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_6)
    def test_moe2017_c_6(self, x, target):
        result = dyad.stats.period._moe2017_c_6(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_7)
    def test_moe2017_c_7(self, x, target):
        result = dyad.stats.period._moe2017_c_7(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.period_moe2017_c_8)
    def test_moe2017_c_8(self, x, target):
        result = dyad.stats.period._moe2017_c_8(*x)
        if __debug__:
            print(result)

        self.assertAlmostEqual(result, target)

    # @parameterized.expand(data_test.period_func)
    # def test_func(self, x, target):
    #     result = dyad.stats.period._func(*x)
    #     if __debug__:
    #         print(result)

    #     self.assertAlmostEqual(result, target)


class TestMoe2017(unittest.TestCase):
    def setUp(self):
        self.x = 0.

    # @parameterized.expand(data_test.initialization)
    # def test_initialization(self, x):
    #     self.assertRaises(ValueError, dyad.Orbit, *x)

    @parameterized.expand(data_test_stats._moe2017_pdf)
    def test_pdf(self, x, period, primary_mass):
        result = dyad.stats.mass_ratio.moe2017.pdf(x, kappa)
        
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test_stats._moe2017_cdf)
    def test_cdf(self, x, period, primary_mass):
        result = dyad.stats.mass_ratio.moe2017.cdf(x, kappa)
        
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test_stats._moe2017_ppf)
    def test_ppf(self, x, period, primary_mass):
        result = dyad.stats.mass_ratio.moe2017.ppf(x, kappa)
        
        self.assertAlmostEqual(result, target)

    def test_rvs(self):
        pass
    
        
if __name__ == "__main__":
    unittest.main()
