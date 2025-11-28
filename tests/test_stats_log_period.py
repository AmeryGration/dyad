#!/usr/bin/env python3

"""
Test functions for stats.log_period module

"""

import unittest
import dyad.stats.log_period as log_period
import data_test_log_period

from test_stats_infrastructure import test_factory

duquennoy1991_data = [
    data_test_log_period.duquennoy1991_pdf,
    data_test_log_period.duquennoy1991_cdf,
    data_test_log_period.duquennoy1991_ppf,
    data_test_log_period.duquennoy1991_rvs
]
moe2017_data = [
    data_test_log_period.moe2017_pdf,
    data_test_log_period.moe2017_cdf,
    data_test_log_period.moe2017_ppf,
    data_test_log_period.moe2017_rvs
]


class TestDuquennoy1991(
        test_factory(log_period.duquennoy1991, duquennoy1991_data)):
    pass


class TestMoe2017(test_factory(log_period.moe2017, moe2017_data)):
    pass


if __name__ == "__main__":
    unittest.main()
