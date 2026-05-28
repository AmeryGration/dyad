#!/usr/bin/env python3

"""
Test functions for stats.primary_mass module

"""

import unittest
import dyad.stats.primary_mass as primary_mass
import data_test_primary_mass

from test_stats_infrastructure import test_factory

random_salpeter1955_data = [
    data_test_primary_mass.random_salpeter1955_pdf,
    data_test_primary_mass.random_salpeter1955_cdf,
    data_test_primary_mass.random_salpeter1955_ppf,
    data_test_primary_mass.random_salpeter1955_rvs
]
random_kroupa2001_data = [
    data_test_primary_mass.random_kroupa2001_pdf,
    data_test_primary_mass.random_kroupa2001_cdf,
    data_test_primary_mass.random_kroupa2001_ppf,
    data_test_primary_mass.random_kroupa2001_rvs
]
random_splitpowerlaw_data = [
    data_test_primary_mass.random_splitpowerlaw_pdf,
    data_test_primary_mass.random_splitpowerlaw_cdf,
    data_test_primary_mass.random_splitpowerlaw_ppf,
    data_test_primary_mass.random_splitpowerlaw_rvs
]
uniform_salpeter1955_data = [
    data_test_primary_mass.uniform_salpeter1955_pdf,
    data_test_primary_mass.uniform_salpeter1955_cdf,
    data_test_primary_mass.uniform_salpeter1955_ppf,
    data_test_primary_mass.uniform_salpeter1955_rvs
]
uniform_kroupa2001_data = [
    data_test_primary_mass.uniform_kroupa2001_pdf,
    data_test_primary_mass.uniform_kroupa2001_cdf,
    data_test_primary_mass.uniform_kroupa2001_ppf,
    data_test_primary_mass.uniform_kroupa2001_rvs
]
uniform_splitpowerlaw_data = [
    data_test_primary_mass.uniform_splitpowerlaw_pdf,
    data_test_primary_mass.uniform_splitpowerlaw_cdf,
    data_test_primary_mass.uniform_splitpowerlaw_ppf,
    data_test_primary_mass.uniform_splitpowerlaw_rvs
]


class TestRandomSalpeter1955(
        test_factory(primary_mass.random.salpeter1955,
                     random_salpeter1955_data)):
    pass


class TestRandomKroupa2001(
        test_factory(primary_mass.random.kroupa2001, random_kroupa2001_data)):
    pass


class TestRandomSplitpowerlaw(
        test_factory(primary_mass.random.splitpowerlaw,
                     random_splitpowerlaw_data)):
    pass


# class TestUniformSalpeter1955(
#         test_factory(primary_mass.uniform.salpeter1955,
#                      uniform_salpeter1955_data)):
#     pass


# class TestUniformKroupa2001(
#         test_factory(primary_mass.uniform.kroupa2001,
#                      uniform_kroupa2001_data)):
#     pass


# class TestUniformSplitpowerlaw(
#         test_factory(primary_mass.uniform.splitpowerlaw,
#                      uniform_splitpowerlaw_data)):
#     pass


if __name__ == "__main__":
    unittest.main()
