#!/usr/bin/env python3

"""
Test functions for stats.mass_ratio module

"""
import unittest
import dyad.stats.secondary_mass as secondary_mass
import data_test_secondary_mass

from test_stats_infrastructure import test_factory

# moe2017_data = [
#     data_test_secondary_mass.moe2017_pdf,
#     data_test_secondary_mass.moe2017_cdf,
#     data_test_secondary_mass.moe2017_ppf,
#     data_test_secondary_mass.moe2017_rvs
# ]
uniform_data = [
    data_test_secondary_mass.uniform_pdf,
    data_test_secondary_mass.uniform_cdf,
    data_test_secondary_mass.uniform_ppf,
    data_test_secondary_mass.uniform_rvs
]


# class TestMoe2017(test_factory(secondary_mass.moe2017, moe2017_data)):
#     pass


class TestUniform(test_factory(secondary_mass.uniform, uniform_data)):
    pass


if __name__ == "__main__":
    unittest.main()
