#!/usr/bin/env python3

"""
Test functions for stats.eccentricity module

"""
import unittest
import dyad.stats.eccentricity as eccentricity
import data_test_eccentricity

from test_stats_infrastructure import test_factory

thermal_data = [
    data_test_eccentricity.thermal_pdf,
    data_test_eccentricity.thermal_cdf,
    data_test_eccentricity.thermal_ppf,
    data_test_eccentricity.thermal_rvs
]
duquennoy1991_data = [
    data_test_eccentricity.duquennoy1991_pdf,
    data_test_eccentricity.duquennoy1991_cdf,
    data_test_eccentricity.duquennoy1991_ppf,
    data_test_eccentricity.duquennoy1991_rvs
]
moe2017_data = [
    data_test_eccentricity.moe2017_pdf,
    data_test_eccentricity.moe2017_cdf,
    data_test_eccentricity.moe2017_ppf,
    data_test_eccentricity.moe2017_rvs
]
uniform_data = [
    data_test_eccentricity.uniform_pdf,
    data_test_eccentricity.uniform_cdf,
    data_test_eccentricity.uniform_ppf,
    data_test_eccentricity.uniform_rvs
]
powerlaw_data = [
    data_test_eccentricity.powerlaw_pdf,
    data_test_eccentricity.powerlaw_cdf,
    data_test_eccentricity.powerlaw_ppf,
    data_test_eccentricity.powerlaw_rvs
]


class TestThermal(
        test_factory(eccentricity.thermal, thermal_data)):
    pass


class TestDuquennoy1991(
        test_factory(eccentricity.duquennoy1991, duquennoy1991_data)):
    pass


class TestMoe2017(
        test_factory(eccentricity.moe2017, moe2017_data)):
    pass


class TestUniform(
        test_factory(eccentricity.uniform, uniform_data)):
    pass


class TestPowerlaw(
        test_factory(eccentricity.powerlaw, powerlaw_data)):
    pass


if __name__ == "__main__":
    unittest.main()
