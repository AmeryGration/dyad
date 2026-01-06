#!/usr/bin/env python3

"""
Test functions for stats.longitude_of_ascending_node module

"""

import unittest
import data_test_longitude_of_ascending_node

from dyad.stats import longitude_of_ascending_node
from test_stats_infrastructure import test_factory

data = [
    data_test_longitude_of_ascending_node.longitude_of_ascending_node_pdf,
    data_test_longitude_of_ascending_node.longitude_of_ascending_node_cdf,
    data_test_longitude_of_ascending_node.longitude_of_ascending_node_ppf,
    data_test_longitude_of_ascending_node.longitude_of_ascending_node_rvs
]


class TestLongitude_Of_Ascending_Node(
        test_factory(longitude_of_ascending_node, data)):
    pass


if __name__ == "__main__":
    unittest.main()
