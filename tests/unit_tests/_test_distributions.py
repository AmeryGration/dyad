"""
Test functions for dyad.stats module

"""

import unittest
import numpy as np
import dyad
import data_test

from parameterized import parameterized, parameterized_class


class TestDistribution:
    @parameterized.expand(data_distributions.moments)
    def test_moments(self, x):
        result = distribution.stats(*x, moments="mvsk")
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_distributions.pdf)
    def test_pdf(self, x, target):
        result = distribution.pdf(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_distributions.cdf)
    def test_cdf(self, x, target):
        result = distribution.cdf(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_distributions.ppf)
    def test_ppf(self, x, target):
        result = distribution.ppf(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_distributions.sf)
    def test_sf(self, x, target):
        result = distribution.sf(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_distributions.isf)
    def test_isf(self, x, target):
        result = distribution.isf(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_distributions.rvs)
    def test_rvs(self, x, target):
        result = distribution.rvs(*x)
        self.assertAlmostEqual(result, target)


# class TestTrueAnomaly:
#     pass


# class TestLongitudeOfAscendingNode:
#     pass


# class TestInclination:
#     pass


# class TestArgumentOfPericentre:
#     pass


# class TestEccentricityUniform:
#     pass


# class TestEccentricityPowerlaw:
#     pass


# class TestEccentricityThermal:
#     pass


# class TestEccentricityDuquennoy1991:
#     pass


# class TestEccentricityMoe2017:
#     pass


# class TestLogPeriodDuquennoy1991:
#     pass


# class TestLogPeriodMoe2017:
#     pass


# class TestMassSplitpowerlaw:
#     pass


# class TestMassKroupa2002:
#     pass


# class TestMassSalpeter1955:
#     pass


# class TestMassMillerScalo1979:
#     pass


# class TestMassChabrier2003:
#     pass


# class TestMassratioDuquennoy1991:
#     pass


# class TestMassratioMoe2017:
#     pass


# class TestPeriodTrunclognorm:
#     pass


# class TestPeriodDuquennoy1991:
#     pass


# class TestPeriodMoe2017:
#     pass


# class TestSemimajoraxisOpik1924:
#     pass
