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
moe2017_data_a = [
    data_test_mass_ratio.moe2017_pdf_a,
    data_test_mass_ratio.moe2017_cdf_a,
    data_test_mass_ratio.moe2017_ppf_a,
    data_test_mass_ratio.moe2017_rvs_a
]
moe2017_data_b = [
    data_test_mass_ratio.moe2017_pdf_b,
    data_test_mass_ratio.moe2017_cdf_b,
    data_test_mass_ratio.moe2017_ppf_b,
    data_test_mass_ratio.moe2017_rvs_b
]
uniform_data = [
    data_test_mass_ratio.uniform_pdf,
    data_test_mass_ratio.uniform_cdf,
    data_test_mass_ratio.uniform_ppf,
    data_test_mass_ratio.uniform_rvs
]


class TestDuquennoy1991(
        test_factory(mass_ratio.duquennoy1991, duquennoy1991_data)):
    pass


class TestMoe2017A(test_factory(mass_ratio.moe2017, moe2017_data_a)):
    pass


class TestMoe2017B(test_factory(mass_ratio.moe2017, moe2017_data_b)):
    pass


class TestUniform(test_factory(mass_ratio.uniform, uniform_data)):
    pass


if __name__ == "__main__":
    unittest.main()
