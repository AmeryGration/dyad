#!/usr/bin/env python3

import unittest
import warnings
import numpy as np

from parameterized import parameterized

def test_factory(rv, data):
    class TestContinuousRandomVariable(unittest.TestCase):
        def setUp(self):
            self.rv = rv
            self.data = data

        @parameterized.expand(data[0])
        def test_pdf(self, x, target):
            res = self.rv.pdf(*x)
            np.testing.assert_almost_equal(res, target)

        @parameterized.expand(data[1])
        def test_cdf(self, x, target):
            res = self.rv.cdf(*x)
            np.testing.assert_almost_equal(res, target)

        @parameterized.expand(data[2])
        def test_ppf(self, x, target):
            res = self.rv.ppf(*x)
            np.testing.assert_almost_equal(res, target)

        @parameterized.expand(data[3])
        def test_rvs(self, x, target):
            with warnings.catch_warnings():
                # RV mass_ratio.moe2017 throws error with data set A
                warnings.simplefilter('ignore', category=DeprecationWarning)
                res = self.rv.rvs(*x[:-1], random_state=x[-1])
            np.testing.assert_almost_equal(res, target)

    return TestContinuousRandomVariable

