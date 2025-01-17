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
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
mass_ratio_boundary = (0.1, 0.3, 0.95, 1.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass_sample = np.hstack([
    np.linspace(0.8, 1.2, n),
    np.linspace(1.2, 3.5, n)[1:],
    np.linspace(3.5, 6., n)[1:],
    np.linspace(6., 60., n)[1:],
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

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# mpl.style.use("sm")

# # fig, ax = plt.subplots()
# # ax.pcolormesh(mass_ratio_sample, primary_mass_sample, np.log10(pdf_sample))
# # ax.vlines(mass_ratio_boundary, 0.8, 60.)
# # ax.hlines(primary_mass_boundary, 0.1, 1.)
# # ax.set_xlim(0.1, 1.)
# # ax.set_ylim(0.8, 60.)
# # ax.set_yscale("log")
# # plt.show()

# # fig, ax = plt.subplots()
# # ax.pcolormesh(mass_ratio_sample, primary_mass_sample, np.log10(cdf_sample))
# # ax.vlines(mass_ratio_boundary, 0.8, 60.)
# # ax.hlines(primary_mass_boundary, 0.1, 1.)
# # ax.set_yscale("log")
# # plt.show()

# ########################################################################
# # Interpolate the pairing function: PDF and CDF
# ########################################################################
# pdf_interp = RegularGridInterpolator(
#     (mass_ratio_sample, primary_mass_sample),
#     pdf_sample.T,
#     bounds_error=False,
#     fill_value=0.
# )
# cdf_interp = RegularGridInterpolator(
#     (mass_ratio_sample, primary_mass_sample),
#     cdf_sample.T,
#     bounds_error=False,
#     fill_value=0.
# )

# m = np.logspace(np.log10(0.8), np.log10(60), 2**9)
# m1m1, m2m2 = np.meshgrid(m, m)
# z = pdf_interp((m2m2/m1m1, m1m1))
# Z = cdf_interp((m2m2/m1m1, m1m1))

# fig, ax = plt.subplots()
# ax.pcolormesh(m, m, np.log10(z), cmap="Greys")
# ax.contour(m, m, np.log10(z), colors="k")
# ax.plot(m, 0.1*m, color="k", ls="solid")
# ax.plot(m, 0.3*m, color="k", ls="dashed")
# ax.plot(m, m, color="k", ls="solid")
# ax.vlines(primary_mass_boundary, 0.8, 60., ls="dashed")
# ax.set_xlim(0.8, 60.)
# ax.set_ylim(0.8, 60.)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$m_{2}$")
# fig.savefig("f_M2givenM2_moe2017_pairing.pdf", dpi=300)
# fig.savefig("f_M2givenM2_moe2017_pairing.jpg", dpi=300)
# plt.show()

# fig, ax = plt.subplots()
# ax.pcolormesh(m, m, np.log10(Z), cmap="Greys")
# # ax.contour(m, m, np.log10(Z), colors="k")
# ax.plot(m, 0.1*m, color="k", ls="solid")
# ax.plot(m, 0.3*m, color="k", ls="dashed")
# ax.plot(m, m, color="k", ls="solid")
# ax.vlines(primary_mass_boundary, 0.8, 60., ls="dashed")
# ax.set_xlim(0.8, 60.)
# ax.set_ylim(0.8, 60.)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$m_{2}$")
# fig.savefig("F_M2givenM2_moe2017_pairing.pdf", dpi=300)
# fig.savefig("F_M2givenM2_moe2017_pairing.jpg", dpi=300)
# plt.show()

# m1 = np.linspace(0.8, 60., 5_000)
# m2 = 1.
# za = pdf_interp((m2/m1, m1))
# m2 = 2.
# zb = pdf_interp((m2/m1, m1))
# m2 = 4.
# zc = pdf_interp((m2/m1, m1))
# m2 = 8.
# zd = pdf_interp((m2/m1, m1))
# m2 = 16.
# ze = pdf_interp((m2/m1, m1))

# fig, ax = plt.subplots()
# ax.plot(m1, za, ls="solid")
# ax.plot(m1, zb, ls="solid")
# ax.plot(m1, zc, ls="solid")
# ax.plot(m1, zd, ls="solid")
# ax.plot(m1, ze, ls="solid")
# ax.set_xlim(0.8, 60.)
# ax.set_ylim(0.1, 10.)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# # ax.set_ylabel(r"$m_{2}$")
# plt.show()
