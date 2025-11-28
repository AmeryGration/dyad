#!/usr/bin/env python3

"""
Test functions for stats.inclination module

"""

import unittest
import data_test_inclination

from dyad.stats import inclination
from test_stats_infrastructure import test_factory

data = [
    data_test_inclination.inclination_pdf,
    data_test_inclination.inclination_cdf,
    data_test_inclination.inclination_ppf,
    data_test_inclination.inclination_rvs
]


class TestInclination(test_factory(inclination, data)):
    pass


if __name__ == "__main__":
    unittest.main()
