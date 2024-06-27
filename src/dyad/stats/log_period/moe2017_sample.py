#!/usr/bin/env python

r"""Generate a sample of the log-period distribution given by Moe2017

The cumulative distribution function (CDF) for the log-period
distribution given by Moe & Stefano (2017) does not have closed
form. Instead we must compute it by numerically integrating the
frequency and then normalizing the result. This script performs the
required numerical integration, saving the result as a text file. That
text file is read by the ~distributions.py~, which implements the
random variable `moe2017`. Note that the result of the integration is
the /cumulative frequency/. It must be normalized to give the CDF.
"""

import numpy as np

from scipy.integrate import cumulative_trapezoid
from dyad.stats import mass_ratio

#############################################################################
# Definte the frequency function
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
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass_sample = np.hstack(
    [
        np.linspace(0.8, 1.2, n),
        np.linspace(1.2, 3.5, n)[1:],
        np.linspace(3.5, 6., n)[1:],
        np.linspace(6., 60., n)[1:],
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
np.savetxt("../../data/primary_mass_sample.dat",
           primary_mass_sample)
np.savetxt("../../data/log10_period_sample.dat",
           log10_period_sample)
np.savetxt("../../data/frequency_sample.dat",
           frequency_sample)
np.savetxt("../../data/cumulative_frequency_sample.dat",
           cumulative_frequency_sample)

# #############################################################################
# # Create interpolating functions
# #############################################################################
# from scipy.interpolate import RegularGridInterpolator
# from scipy.interpolate import LinearNDInterpolator

# frequency_interp = RegularGridInterpolator(
#     (log10_period_sample, primary_mass_sample),
#     frequency_sample.T
# )

# cumulative_frequency_interp = RegularGridInterpolator(
#     (log10_period_sample, primary_mass_sample),
#     cumulative_frequency_sample.T
# )
# # Suppose that we have an invertible function, :math:`f`, of some variable, :math:`t`, represented by arrays `f` and `t`. We can interpolate between function values as follows.
# # >>> interp = RegularGridInterpolator((t,), f)
# # To interpolate between equivalent points of the inverse function, :math:`f^{-1}`, of some variable, :math:`q \in [0, 1]`, we may reverse the arguments.
# # >>> interp_inv = RegularGridInterpolator(f[::-1], (t[::-1],))
# # Note that in doing this we are not sampling :math:`q` uniformly on :math:`[0, 1]`. Instead we may take advantage of points of interest in :math:`x := log(P)`
# # To find the inverse of the conditional cumulative distribution function we may extend this trick to two dimensions. In this two-dimensional case the values `f` are no longer regularly spaced so we must use a interpolator that accepts irregularly space data, such as Scipy's ~LinearNDInterpolator~.
# # See https://kitchingroup.cheme.cmu.edu/blog/category/interpolation/.
# xx = cumulative_frequency_sample[:,::-1]
# yy = np.tile(primary_mass_sample, (log10_period_sample.size, 1)).T
# points = np.vstack([xx.ravel(), yy.ravel()])
# values = np.tile(log10_period_sample[::-1], primary_mass_sample.size)
# inverse_cumulative_frequency_interp = LinearNDInterpolator(points.T, values)

# #############################################################################
# # Plot results
# #############################################################################
# import matplotlib.pyplot as plt

# idx = 98
# primary_mass = primary_mass_sample[idx]
# q_sample = np.linspace(0., 1.)

# plt.plot(log10_period_sample,
#          frequency_sample[idx])
# plt.plot(log10_period_sample,
#          frequency_interp((log10_period_sample, primary_mass)),
#          color="red")
# plt.show()

# plt.plot(log10_period_sample,
#          cumulative_frequency_sample[idx])
# plt.plot(log10_period_sample,
#          cumulative_frequency_interp((log10_period_sample, primary_mass)),
#          color="red")
# plt.plot(cumulative_frequency_sample[idx][::-1],
#          log10_period_sample[::-1])
# plt.plot(q_sample,
#          inverse_cumulative_frequency_interp((q_sample, primary_mass)),
#          color="red")
# plt.show()

# # idx = 108
# # mass = primary_mass_sample[idx]
# # inverse_cumulative_frequency_interp = RegularGridInterpolator(
# #     (cumulative_frequency_sample[idx][::-1], primary_mass_sample),
# #     np.tile(log10_period_sample[::-1], (primary_mass_sample.size, 1)).T,
# #     bounds_error=True
# # )

