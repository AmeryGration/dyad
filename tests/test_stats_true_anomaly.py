#!/usr/bin/env python3

"""
Test functions for stats.true_anomaly module

"""

import unittest
import data_test_true_anomaly

from dyad.stats import true_anomaly
from test_stats_infrastructure import test_factory

true_anomaly_data = [
    data_test_true_anomaly.true_anomaly_pdf,
    data_test_true_anomaly.true_anomaly_cdf,
    data_test_true_anomaly.true_anomaly_ppf,
    data_test_true_anomaly.true_anomaly_rvs
]


class TestTrue_Anomaly(test_factory(true_anomaly, true_anomaly_data)):
    pass


if __name__ == "__main__":
    unittest.main()
 
