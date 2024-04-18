#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import plot

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

from dyad import stats

mpl.style.use("sm")

# Sample the domain using a rectilinear lattice 
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass = np.hstack(
    [
        np.linspace(0.8, 1.2, n),
        np.linspace(1.2, 3.5, n)[1:],
        np.linspace(3.5, 6., n)[1:],
        np.linspace(6., 60., n)[1:],
    ]
)
log10_period = np.hstack(
    [
        np.linspace(0.2 + 1.e-6, 1., n),
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

period = 10.**log10_period
rv_period = stats.period.moe2017(primary_mass.reshape([-1, 1]))
rv_mass_ratio = stats.mass_ratio.moe2017(period, primary_mass.reshape([-1, 1]))
pdf_period = rv_period.pdf(log10_period)
pdf_mass_ratio = rv_mass_ratio.pdf(0.3)
pdf_period = pdf_period/pdf_mass_ratio

cdf_period = cumulative_trapezoid(pdf_period, log10_period, initial=0)
pdf_period = pdf_period/cdf_period[:, [-1]]
cdf_period = cdf_period/cdf_period[:, [-1]]

# Period: PDF
fig, ax, cbar = plot.plot(cbar=True)
im = ax.pcolormesh(log10_period, primary_mass, pdf_period, rasterized=True)
ax.contour(log10_period, primary_mass, pdf_period, colors="k", levels=10)
ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
ax.set_yscale("log")
ax.set_xlim(0.2, 8.)
ax.set_ylim(0.8, 60.)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$M_{1}$")
cbar = fig.colorbar(im, cax=cbar)
cbar.set_label(r"$f_{X|M_{1}}$")
fig.savefig("moe2017_logperiod_pdf_2d.pdf")
plt.show()

# Period: CDF
fig, ax, cbar = plot.plot(cbar=True)
im = ax.pcolormesh(log10_period, primary_mass, cdf_period, rasterized=True)
ax.contour(log10_period, primary_mass, cdf_period, colors="k", levels=10)
ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
ax.set_yscale("log")
ax.set_xlim(0.2, 8.)
ax.set_ylim(0.8, 60.)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$M_{1}$")
cbar = fig.colorbar(im, cax=cbar)
cbar.set_label(r"$f_{X|M_{1}}$")
fig.savefig("moe2017_logperiod_cdf.pdf")
plt.show()

# # Interpolator
# point = np.array([
#     [[primary_mass_i, log10_period_j] for primary_mass_i in primary_mass
#      for log10_period_j in log10_period]
# ])
# point = point.squeeze()
# cdf = RegularGridInterpolator(
#     (primary_mass, log10_period), cdf_period, bounds_error=True
# )

#############################################################################
primary_mass_test = (1., 3.5, 7., 12.5, 25.)

rv_a = stats.period.moe2017(primary_mass_test[0])
rv_b = stats.period.moe2017(primary_mass_test[1])
rv_c = stats.period.moe2017(primary_mass_test[2])
rv_d = stats.period.moe2017(primary_mass_test[3])
rv_e = stats.period.moe2017(primary_mass_test[4])

x_1 = np.linspace(0., 0.2, 500, endpoint=False)
x_2 = np.linspace(0.2, 8., 500)

rv_period = stats.period.moe2017(np.array(primary_mass_test).reshape([-1, 1]))
rv_mass_ratio = stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[0])
pdf = rv_period.pdf(0.3)

pdf_1a = (rv_a.pdf(x_1))
pdf_2a = (
    rv_a.pdf(x_2)
    /(1. - stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[0]).cdf(0.3))
)
pdf_1b = rv_b.pdf(x_1)
pdf_2b = (
    rv_b.pdf(x_2)
    /(1. - stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[1]).cdf(0.3))
)
pdf_1c = rv_c.pdf(x_1)
pdf_2c = (
    rv_c.pdf(x_2)
    /(1. - stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[2]).cdf(0.3))
)
pdf_1d = rv_d.pdf(x_1)
pdf_2d = (
    rv_d.pdf(x_2)
    /(1. - stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[3]).cdf(0.3))
)
pdf_1e = rv_e.pdf(x_1)
pdf_2e = (
    rv_e.pdf(x_2)
    /(1. - stats.mass_ratio.moe2017(10.**x_2, primary_mass_test[4]).cdf(0.3))
)

fig, ax = plt.subplots()
ax.plot(
    x_1, pdf_1a, color="red", ls="solid",
    label=r"$M_{{1}} = {}$".format(primary_mass_test[0])
)
ax.plot(
    x_2, pdf_2a, color="red", ls="solid",
)
ax.plot(
    x_1, pdf_1b, color="orange", ls="solid",
    label=r"$M_{{1}} = {}$".format(primary_mass_test[1])
)
ax.plot(
    x_2, pdf_2b, color="orange", ls="solid"
)
ax.plot(
    x_1, pdf_1c, color="green", ls="solid",
    label=r"$M_{{1}} = {}$".format(primary_mass_test[2])
)
ax.plot(
    x_2, pdf_2c, color="green", ls="solid")
ax.plot(
    x_1, pdf_1d, color="blue", ls="solid",
    label=r"$M_{{1}} = {}$".format(primary_mass_test[3])
)
ax.plot(
    x_2, pdf_2d, color="blue", ls="solid"
)
ax.plot(
    x_1, pdf_1e, color="magenta", ls="solid",
    label=r"$M_{{1}} = {}$".format(primary_mass_test[4])
)
ax.plot(
    x_2, pdf_2e, color="magenta", ls="solid"
)
ax.scatter(0.2, pdf_2a[0], color="red", s=2., zorder=100.)
ax.scatter(0.2, pdf_2b[0], color="orange", s=2., zorder=100.)
ax.scatter(0.2, pdf_2c[0], color="green", s=2., zorder=100.)
ax.scatter(0.2, pdf_2d[0], color="blue", s=2., zorder=100.)
ax.scatter(0.2, pdf_2e[0], color="magenta", s=2., zorder=100.)
ax.scatter(0.2, 0., s=2., facecolors="white", edgecolors="magenta",
           zorder=np.inf)
ax.legend(frameon=False)
ax.set_xlim(0., 8.)
ax.set_ylim(-0.05, 0.4)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f_{X|M_{1}}$")
plt.savefig("moe2017_logperiod_pdf.pdf")
plt.show()
