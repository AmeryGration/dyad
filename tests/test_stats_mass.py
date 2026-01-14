#!/usr/bin/env python3

"""
Test functions for stats.mass module

"""

import unittest
import dyad.stats.mass as mass
import data_test_mass

from test_stats_infrastructure import test_factory

salpeter1955_data = [
    data_test_mass.salpeter1955_pdf,
    data_test_mass.salpeter1955_cdf,
    data_test_mass.salpeter1955_ppf,
    data_test_mass.salpeter1955_rvs
]
kroupa2001_data = [
    data_test_mass.kroupa2001_pdf,
    data_test_mass.kroupa2001_cdf,
    data_test_mass.kroupa2001_ppf,
    data_test_mass.kroupa2001_rvs
]
splitpowerlaw_data = [
    data_test_mass.splitpowerlaw_pdf,
    data_test_mass.splitpowerlaw_cdf,
    data_test_mass.splitpowerlaw_ppf,
    data_test_mass.splitpowerlaw_rvs
]


class TestSalpeter1955(test_factory(mass.salpeter1955, salpeter1955_data)):
    pass


class TestKroupa2001(test_factory(mass.kroupa2001, kroupa2001_data)):
    pass


class TestSplitpowerlaw(test_factory(mass.splitpowerlaw, splitpowerlaw_data)):
    pass


if __name__ == "__main__":
    unittest.main()
