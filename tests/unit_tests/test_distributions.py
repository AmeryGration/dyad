"""
Test functions for dyad.stats module

"""

import unittest
import numpy as np
import dyad
import data_test

from parameterized import parameterized, parameterized_class

# # distributions to skip while testing the fix for the support method
# # introduced in gh-13294. These distributions are skipped as they
# # always return a non-nan support for every parametrization.
# skip_test_support_gh13294_regression = ['tukeylambda', 'pearson3']


# class TestTrueAnomaly:
#     @parameterized.expand(stats_test.moments)
#     def test_moments(self, x):
#         result = dyad.stats.true_anomaly.stats(*x, moments="mvsk")
#         self.assertAlmostEqual(result, target)

#     @parameterized.expand(stats_test.pdf)
#     "np.nan, -np.inf, np.inf"
#     def test_nonnumerical_arguments(self, x):
#         self.assertRaises(ValueError, dyad._core._check_eccentricity, x)

#     @parameterized.expand(stats_test.pdf)
#     def test_pdf(self, x, target):
#         result = dyad.true_anomaly.pdf(*x)
#         self.assertAlmostEqual(result, target)

#     @parameterized.expand(stats_test.cdf)
#     def test_cdf(self, x, target):
#         result = dyad.true_anomaly.cdf(*x)
#         self.assertAlmostEqual(result, target)

#     @parameterized.expand(stats_test.ppf)
#     def test_ppf(self, x, target):
#         result = dyad.true_anomaly.ppf(*x)
#         self.assertAlmostEqual(result, target)

#     @parameterized.expand(stats_test.sf)
#     def test_sf(self, x, target):
#         result = dyad.true_anomaly.ppf(*x)
#         self.assertAlmostEqual(result, target)

#     @parameterized.expand(stats_test.isf)
#     def test_isf(self, x, target):
#         result = dyad.true_anomaly.ppf(*x)
#         self.assertAlmostEqual(result, target)
        
#     @parameterized.expand(stats_test.rvs)
#     def test_rvs(self, x, target):
#         result = dyad.true_anomaly.rvs(*x)
#         self.assertAlmostEqual(result, target)


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
