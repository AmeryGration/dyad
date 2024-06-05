#!/usr/bin/env python

"""Generate sample of the cumulative frequency for Moe2017

The CDF for the log-period distribution given by Moe & Stefano (2017) does not have closed form. Instead we must compute it by numerically integrating the frequency and then normalizing the result. This script performs the required numerical integration, saving the result as a text file. That text file is read by the ~distributions.py~, which implements the random variable `moe2017`. Note that the result of the integration is the /cumulative frequency/, not the CDF. The normalization is performed by the random variable subclass in ~distributions.py~.

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid

from dyad import stats

# Sample the domain using a rectilinear lattice 
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

rv_period = stats.log_period.moe2017(primary_mass_sample.reshape([-1, 1]))
freq_sample = rv_period.pdf(log10_period_sample)
cum_freq_sample = cumulative_trapezoid(
    freq_sample, log10_period_sample, initial=0.
)

np.savetxt("primary_mass_sample.dat", primary_mass_sample)
np.savetxt("log10_period_sample.dat", log10_period_sample)
np.savetxt("cumulative_frequency_sample.dat", cum_freq_sample)

# # Period: PDF
# fig, ax = plt.subplots()
# im = ax.pcolormesh(log10_period_sample, primary_mass_sample, freq_sample,
#                    rasterized=True)
# ax.contour(log10_period_sample, primary_mass_sample, freq_sample, colors="k",
#            levels=10)
# ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
# ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
# ax.set_yscale("log")
# ax.set_xlim(0.2, 8.)
# ax.set_ylim(0.8, 60.)
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$M_{1}$")
# # fig.savefig("moe2017_logperiod_pdf_2d.pdf")
# plt.show()

# # Period: CDF
# fig, ax = plt.subplots()
# im = ax.pcolormesh(log10_period_sample, primary_mass_sample, cum_freq_sample,
#                    rasterized=True)
# ax.contour(log10_period_sample, primary_mass_sample, cum_freq_sample,
#            colors="k", levels=10)
# ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
# ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
# ax.set_yscale("log")
# ax.set_xlim(0.2, 8.)

# ax.set_ylim(0.8, 60.)
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$M_{1}$")
# # fig.savefig("moe2017_logperiod_cdf.pdf")
# plt.show()


