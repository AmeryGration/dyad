#!/usr/bin/env python

r"""Sample of the primary-mass distribution

Sample of the primary-mass distribution given the initial-mass function of Kroupa (2002) and the pairing function of Moe and Stefano (2017).

"""
import numpy as np
import vie_solver as vie

from scipy.integrate import cumulative_trapezoid
from dyad.stats import mass
from dyad.stats import secondary_mass

def f(x):
    """The known function"""
    res = 2.*pdf_mass(x)

    return res

def k(x, y):
    """The kernel"""
    # x = np.atleast_2d(x)
    # y = np.atleast_2d(y)
    x = x.reshape([-1, 1])
    # res = secondary_mass.moe2017(y).pdf(x)
    res = secondary_mass.random(y).pdf(x)

    return res

rv_mass = mass.salpeter1955
pdf_mass = rv_mass.pdf

primary_mass_sample, frequency_sample = vie.solve(
    f, k, rv_mass.a, rv_mass.b, -1., (), 2**12
)
cumulative_frequency_sample = cumulative_trapezoid(
    frequency_sample, primary_mass_sample, initial=0.
)
np.savetxt("primary_mass_sample.dat", primary_mass_sample)
np.savetxt("frequency_sample.dat", frequency_sample)
np.savetxt("cumulative_frequency_sample.dat", cumulative_frequency_sample)
