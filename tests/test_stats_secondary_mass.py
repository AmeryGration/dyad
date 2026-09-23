#!/usr/bin/env python3

"""
Test functions for stats.secondary_mass module

"""

import unittest
import dyad.stats.secondary_mass as secondary_mass
import data_test_secondary_mass

from test_stats_infrastructure import test_factory

random_salpeter1955_data = [
    data_test_secondary_mass.random_salpeter1955_pdf,
    data_test_secondary_mass.random_salpeter1955_cdf,
    data_test_secondary_mass.random_salpeter1955_ppf,
    data_test_secondary_mass.random_salpeter1955_rvs
]
random_kroupa2001_data = [
    data_test_secondary_mass.random_kroupa2001_pdf,
    data_test_secondary_mass.random_kroupa2001_cdf,
    data_test_secondary_mass.random_kroupa2001_ppf,
    data_test_secondary_mass.random_kroupa2001_rvs
]
random_splitpowerlaw_data = [
    data_test_secondary_mass.random_splitpowerlaw_pdf,
    data_test_secondary_mass.random_splitpowerlaw_cdf,
    data_test_secondary_mass.random_splitpowerlaw_ppf,
    data_test_secondary_mass.random_splitpowerlaw_rvs
]
uniform_data = [
    data_test_secondary_mass.uniform_pdf,
    data_test_secondary_mass.uniform_cdf,
    data_test_secondary_mass.uniform_ppf,
    data_test_secondary_mass.uniform_rvs
]


class TestRandomSalpeter1955(
        test_factory(secondary_mass.random.salpeter1955,
                     random_salpeter1955_data)):
    pass


class TestRandomKroupa2001(
        test_factory(secondary_mass.random.kroupa2001, random_kroupa2001_data)):
    pass


class TestRandomSplitpowerlaw(
        test_factory(secondary_mass.random.splitpowerlaw,
                     random_splitpowerlaw_data)):
    pass


class TestUniform(
        test_factory(secondary_mass.uniform,
                     uniform_data)):
    pass


if __name__ == "__main__":
    unittest.main()
