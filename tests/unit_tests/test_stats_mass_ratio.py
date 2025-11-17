#!/usr/bin/env python3

"""
Test functions for stats.mass_ratio module

"""

import unittest
import dyad.stats.mass_ratio as mass_ratio
import data_test_mass_ratio

from test_stats_infrastructure import test_factory

duquennoy1991_data = [
    data_test_mass_ratio.duquennoy1991_pdf,
    data_test_mass_ratio.duquennoy1991_cdf,
    data_test_mass_ratio.duquennoy1991_ppf,
    data_test_mass_ratio.duquennoy1991_rvs
]
moe2017_data = [
    data_test_mass_ratio.moe2017_pdf,
    data_test_mass_ratio.moe2017_cdf,
    data_test_mass_ratio.moe2017_ppf,
    data_test_mass_ratio.moe2017_rvs
]


class TestDuquennoy1991(
        test_factory(mass_ratio.duquennoy1991, duquennoy1991_data)):
    pass


class TestMoe2017(test_factory(mass_ratio.moe2017, moe2017_data)):
    pass


if __name__ == "__main__":
    unittest.main()
