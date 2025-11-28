#!/usr/bin/env python3

"""
Test functions for stats.semimajor_axis module

"""

import unittest
import dyad.stats.semimajor_axis as semimajor_axis
import data_test_semimajor_axis

from test_stats_infrastructure import test_factory

opik1924_data = [
    data_test_semimajor_axis.opik1924_pdf,
    data_test_semimajor_axis.opik1924_cdf,
    data_test_semimajor_axis.opik1924_ppf,
    data_test_semimajor_axis.opik1924_rvs
]


class TestOpik1924(test_factory(semimajor_axis.opik1924, opik1924_data)):
    pass


if __name__ == "__main__":
    unittest.main()
