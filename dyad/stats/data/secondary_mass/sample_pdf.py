#!/usr/bin/env python

r"""Sample of the secondary-mass distribution implied by Moe and Stefano (2017)

We require the PDF for secondary mass (conditional on primary
mass). This is not given by Moe and Stefano (2017) but can be computed
as follows:

First, 

.. math::

   f_{M_{\secondary}|M_{\primary} = m_{\primary}}(m_{\secondary}|m_{\primary}) = \dfrac{1}{m_{\primary}}f_{Q}(m_{\secondary}/m_{\primary}).

where

.. math::

   f_{Q|M_{\primary} = m_{\primary}}(q|m_{\primary}) = \int_{p_{\min}}^{p_{\max}}f_{(Q, P)|M_{\primary} = m_{\primary}}(q, p|m_{\primary})\diff{}p

and, by the chain rule for probability,

.. math::

   f_{(Q, P)|M_{\primary} = m_{\primary}}(q, p|m_{\primary}) = f_{Q|(P, M_{\primary}) = (p, m_{\primary})}(q|p, m_{\primary})f_{P|M_{\primary} = m_{\primary}}(p|m_{\primary}).

The two factors of this last formula are in fact given by Moe and
Stefano.

Neither the probability density function (PDF) nor cumulative
distribution function (CDF) for the primary mass random variable
implied by Moe and Stefano (2017) have closed-form expression. Dyad
evaluates these functions by interpolating between values pre-computed
on a regular lattice of arguments. The CDF is computed by integrating
the PDF using the trapezium rule with nodes placed on the points of
this lattice. Moe and Stefano give the observed frequency of
log-period rather than its PDF. To compute the latter we must
normalize the former. This script performs the required sampling,
integration, and normalization, saving the results to file. Those
files are read by `dyad/stats/secondary_mass.py`, which implements the
random variable `dyad.stats.secondary_mass.moe2017` by performing the
interpolation.

"""
import numpy as np

from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from dyad.stats import mass_ratio
from dyad.stats import period as period

#############################################################################
# Define the integrand
#############################################################################
def f(p, q, m):
    """Return the joint probability density for given mass ratio and period"""
    res = (
        mass_ratio.moe2017(np.log10(p), m).pdf(q)
        *period.moe2017(m).pdf(p)
        /m
    )

    return res

#############################################################################
# Create regular lattice of sample points
#############################################################################
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 40.)
mass_ratio_boundary = (0.1, 0.3, 0.95, 1.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass_sample = np.hstack([
    np.linspace(0.8, 1.2, n),
    np.linspace(1.2, 3.5, n)[1:],
    np.linspace(3.5, 6., n)[1:],
    np.linspace(6., 40., n)[1:],
])
mass_ratio_sample = np.hstack([
    np.linspace(0.1, 0.3, n),
    np.linspace(0.3, 0.95, n)[1:],
    np.linspace(0.95, 1., n)[1:],
])
log10_period_sample = np.hstack([
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
])
period_sample = 10.**log10_period_sample

#############################################################################
# Sample the PDF and CDF
#############################################################################
pp, qq, m1m1 = np.meshgrid(
    period_sample, mass_ratio_sample, primary_mass_sample
)
f_sample = f(pp, qq, m1m1)
pdf_sample = trapezoid(f_sample, period_sample, axis=1).T
cdf_sample = cumulative_trapezoid(
    pdf_sample, mass_ratio_sample, axis=1, initial=0.
)

#############################################################################
# Compute equivalent sample of the PDF and CDF
#############################################################################
pdf_sample = pdf_sample/cdf_sample[:,-1:]
cdf_sample = cdf_sample/cdf_sample[:,-1:]

#############################################################################
# Save data
#############################################################################
np.savetxt("./primary_mass_sample.dat", primary_mass_sample)
np.savetxt("./mass_ratio_sample.dat", mass_ratio_sample)
np.savetxt("./frequency_sample.dat", pdf_sample)
np.savetxt("./cumulative_frequency_sample.dat", cdf_sample)
