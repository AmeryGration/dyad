#!/usr/bin/env python

r"""Sample of the log-period distribution given by Moe and Stefano (2017)

Neither the probability density function (PDF) nor cumulative
distribution function (CDF) for the log-period random variable
defined by Moe and Stefano (2017) have closed-form expression. Dyad
evaluates these functions by interpolating between values pre-computed
on a regular lattice of arguments. The CDF is computed by integrating
the PDF using the trapezium rule with nodes placed on the points of
this lattice. Moe and Stefano give the observed frequency of
log-period rather than its PDF. To compute the latter we must
normalize the former. This script performs the required sampling,
integration, and normalization, saving the results to file. Those
files are read by `dyad/stats/log_period.py`, which implements the
random variable `dyad.stats.log_period.moe2017` by performing the
interpolation.

"""
import numpy as np

from scipy.integrate import cumulative_trapezoid
from dyad.stats import mass_ratio

#############################################################################
# Define the frequency function
#############################################################################
def moe2017_c_1(log10_primary_mass):
    res = (
        0.07*log10_primary_mass**2.
        + 0.04*log10_primary_mass
        + 0.020
    )

    return res

def moe2017_c_2(log10_primary_mass):
    res = (
        - 0.06*log10_primary_mass**2.
        + 0.03*log10_primary_mass
        + 0.0064
    )

    return res

def moe2017_c_3(log10_primary_mass):
    res = (
        0.13*log10_primary_mass**2.
        + 0.01*log10_primary_mass
        + 0.0136
    )

    return res

def moe2017_c_4(log10_primary_mass):
    res = 0.018

    return res

def moe2017_c_5(log10_primary_mass):
    res = (
        0.01*log10_primary_mass**2.
        + 0.07*log10_primary_mass
        - 0.0096
    )

    return res

def moe2017_c_6(log10_primary_mass):
    res = (
        0.03*log10_primary_mass**2./2.1
        - 0.12*log10_primary_mass/2.1
        + 0.0264/2.1
    )

    return res

def moe2017_c_7(log10_primary_mass):
    res = (
        - 0.081*log10_primary_mass**2./2.1
        + 0.555*log10_primary_mass/2.1
        + 0.0186/2.1
    )

    return res

def moe2017_c_8(log10_primary_mass):
    res = (
        np.exp(1.65)
        *(
            0.04*log10_primary_mass**2.
            - 0.05*log10_primary_mass
            + 0.078
        )
    )

    return res

def pdf(x, primary_mass):
    def f_1(x, log10_primary_mass):
        res = moe2017_c_1(log10_primary_mass)

        return res

    def f_2(x, log10_primary_mass):
        res = (
            moe2017_c_2(log10_primary_mass)*x
            + moe2017_c_3(log10_primary_mass)
        )

        return res

    def f_3(x, log10_primary_mass):
        res = (
            moe2017_c_4(log10_primary_mass)*x
            + moe2017_c_5(log10_primary_mass)
        )

        return res

    def f_4(x, log10_primary_mass):
        res = (
            moe2017_c_6(log10_primary_mass)*x
            + moe2017_c_7(log10_primary_mass)
        )

        return res

    def f_5(x, log10_primary_mass):
        res = (
            moe2017_c_8(log10_primary_mass)
            *np.exp(-0.3*x)
        )

        return res

    x = np.asarray(x)
    primary_mass = np.asarray(primary_mass)
    log10_primary_mass = np.log10(primary_mass)

    rv_mass_ratio = mass_ratio.moe2017(x, 10.**log10_primary_mass)
    correction_factor = 1./(1. - rv_mass_ratio.cdf(0.3))

    condition = [
        (0.2 <= x) & (x <= 1.),
        (1. < x) & (x <= 2.),
        (2. < x) & (x <= 3.4),
        (3.4 < x) & (x <= 5.5),
        (5.5 < x) & (x <= 8.),
    ]
    value = [
        f_1(x, log10_primary_mass),
        f_2(x, log10_primary_mass),
        f_3(x, log10_primary_mass),
        f_4(x, log10_primary_mass),
        f_5(x, log10_primary_mass),
    ]
    res = (
        correction_factor
        *np.select(condition, value)
    )

    return res

#############################################################################
# Create grid of sample points
#############################################################################
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 40.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass_sample = np.hstack(
    [
        np.linspace(0.8, 1.2, n),
        np.linspace(1.2, 3.5, n)[1:],
        np.linspace(3.5, 6., n)[1:],
        np.linspace(6., 40., n)[1:],
    ]
)
log10_period_sample = np.hstack(
    [
        np.linspace(0.2, 1., n),
        np.linspace(1., 1.3, n)[1:],
        np.linspace(1.3, 2., n)[1:],
        np.linspace(2., 2.5, n)[1:],
        np.linspace(2.5, 3.4, n)[1:],
        np.linspace(3.4, 3.5, n)[1:],
        np.linspace(3.5, 4., n)[1:],
        np.linspace(4., 4.5, n)[1:],
        np.linspace(4.5, 5.5, n)[1:],
        np.linspace(5.5, 6., n)[1:],
        np.linspace(6., 6.5, n)[1:],
        np.linspace(6.5, 8., n)[1:],
    ]
)

#############################################################################
# Sample the frequency function using a rectilinear lattice 
#############################################################################
frequency_sample = pdf(
    log10_period_sample, primary_mass_sample.reshape([-1, 1])
)

#############################################################################
# Sample the cumulative frequency function using a rectilinear lattice 
#############################################################################
cumulative_frequency_sample = cumulative_trapezoid(
    frequency_sample, log10_period_sample, initial=0.
)

#############################################################################
# Compute equivalent sample of the PDF and CDF
#############################################################################
frequency_sample = frequency_sample/cumulative_frequency_sample[:,-1:]
cumulative_frequency_sample = (
    cumulative_frequency_sample/cumulative_frequency_sample[:,-1:]
)

#############################################################################
# Save data
#############################################################################
np.savetxt(
    "./primary_mass_sample.dat", primary_mass_sample
)
np.savetxt(
    "./log10_period_sample.dat", log10_period_sample
)
np.savetxt(
    "./frequency_sample.dat", frequency_sample
)
np.savetxt(
    "./cumulative_frequency_sample.dat", cumulative_frequency_sample
)
