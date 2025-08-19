#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator

primary_mass_sample = np.loadtxt("./primary_mass_sample.dat")
mass_ratio_sample = np.loadtxt("./mass_ratio_sample.dat")
pdf_sample = np.loadtxt("./frequency_sample.dat")
cdf_sample = np.loadtxt("./cumulative_frequency_sample.dat")

# primary_mass_boundary = (0.8, 1.2, 3.5, 6., 40.)
primary_mass_boundary = (0.08, 1.2, 3.5, 6., 150.)
mass_ratio_boundary = (0.1, 0.3, 0.95, 1.)

mpl.style.use("sm")

fig, ax = plt.subplots()
ax.pcolormesh(
    mass_ratio_sample, primary_mass_sample, np.log10(pdf_sample),
    rasterized=True
)
ax.contour(
    mass_ratio_sample, primary_mass_sample, pdf_sample, colors="k", norm="log"
)
ax.vlines(mass_ratio_boundary, 0.8, 40.)
ax.hlines(primary_mass_boundary, 0.1, 1.)
ax.set_xlim(0.1, 1.)
# ax.set_ylim(0.8, 40.)
ax.set_yscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/f_QgivenM1_moe2017.pdf", dpi=300)
plt.savefig("./Figures/f_QgivenM1_moe2017.jpg", dpi=300)
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(
    mass_ratio_sample, primary_mass_sample, np.log10(cdf_sample),
    rasterized=True
)
ax.contour(
    mass_ratio_sample, primary_mass_sample, cdf_sample, colors="k", norm="log"
)
ax.vlines(mass_ratio_boundary, 0.8, 40.)
ax.hlines(primary_mass_boundary, 0.1, 1.)
ax.set_yscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/F_QgivenM1_moe2017.pdf", dpi=300)
plt.savefig("./Figures/F_QgivenM1_moe2017.jpg", dpi=300)
plt.show()

########################################################################
# Interpolate the pairing function: PDF and CDF
########################################################################
pdf_interp = LinearNDInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    pdf_sample.T,
    # bounds_error=False,
    # fill_value=0.
)
cdf_interp = LinearNDInterpolator(
    (mass_ratio_sample, primary_mass_sample),
    cdf_sample.T,
    # bounds_error=False,
    # fill_value=0.
)

# m = np.logspace(np.log10(0.8), np.log10(40.), 2**9)
m = np.logspace(np.log10(0.08), np.log10(150.), 2**9)
m1m1, m2m2 = np.meshgrid(m, m)
z = pdf_interp((m2m2/m1m1, m1m1))
Z = cdf_interp((m2m2/m1m1, m1m1))

fig, ax = plt.subplots()
ax.pcolormesh(m, m, np.log10(z), cmap="Greys")
ax.contour(m, m, np.log10(z), colors="k")
ax.plot(m, 0.1*m, color="k", ls="solid")
ax.plot(m, 0.3*m, color="k", ls="dashed")
ax.plot(m, m, color="k", ls="solid")
ax.vlines(primary_mass_boundary, 0.8, 40., ls="dashed")
ax.set_xlim(0.8, 40.)
ax.set_ylim(0.8, 40.)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
ax.set_ylabel(r"$m_{2}$")
# fig.savefig("./Figures/f_M2givenM2_moe2017_pairing.pdf", dpi=300)
# fig.savefig("./Figures/f_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(m, m, np.log10(Z), cmap="Greys")
# ax.contour(m, m, np.log10(Z), colors="k")
ax.plot(m, 0.1*m, color="k", ls="solid")
ax.plot(m, 0.3*m, color="k", ls="dashed")
ax.plot(m, m, color="k", ls="solid")
ax.vlines(primary_mass_boundary, 0.8, 40., ls="dashed")
# ax.set_xlim(0.8, 40.)
# ax.set_ylim(0.8, 40.)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
ax.set_ylabel(r"$m_{2}$")
# fig.savefig("./Figures/F_M2givenM2_moe2017_pairing.pdf", dpi=300)
# fig.savefig("./Figures/F_M2givenM2_moe2017_pairing.jpg", dpi=300)
plt.show()

# m1 = np.linspace(0.8, 40., 5_000)
m1 = np.linspace(0.08, 150., 5_000)
m2 = 1.
za = pdf_interp((m2/m1, m1))
m2 = 2.
zb = pdf_interp((m2/m1, m1))
m2 = 4.
zc = pdf_interp((m2/m1, m1))
m2 = 8.
zd = pdf_interp((m2/m1, m1))
m2 = 16.
ze = pdf_interp((m2/m1, m1))

fig, ax = plt.subplots()
ax.plot(m1, za, ls="solid")
ax.plot(m1, zb, ls="solid")
ax.plot(m1, zc, ls="solid")
ax.plot(m1, zd, ls="solid")
ax.plot(m1, ze, ls="solid")
# ax.set_xlim(0.8, 40.)
ax.set_ylim(0.1, 10.)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$m_{2}$")
plt.show()
