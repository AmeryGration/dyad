#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator

primary_mass_sample = np.loadtxt("./primary_mass_sample.dat")
mass_ratio_sample = np.loadtxt("./mass_ratio_sample.dat")
pdf_sample = np.loadtxt("./frequency_sample.dat")
pdf_sample[pdf_sample == np.max(pdf_sample)] = 0.
cdf_sample = np.loadtxt("./cumulative_frequency_sample.dat")

primary_mass_boundary = (0.08, 0.8, 1.2, 3.5, 6., 40.)
mass_ratio_boundary = (0.1, 0.3, 0.95, 1.)

mpl.style.use("sm")

fig, ax = plt.subplots()
ax.pcolormesh(
    mass_ratio_sample, primary_mass_sample, pdf_sample, norm="log",
    rasterized=True
)
ax.contour(
    mass_ratio_sample, primary_mass_sample, np.log10(pdf_sample), levels=10,
    colors="k"
)
ax.plot(mass_ratio_sample, 0.08/mass_ratio_sample)
ax.vlines(mass_ratio_boundary, 0.08, 150.)
ax.hlines(primary_mass_boundary, 0.1, 1.)
ax.set_xlim(0., 1.)
ax.set_ylim(0.08, 150.)
ax.set_yscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/f_QgivenM1_moe2017.pdf", dpi=300)
plt.savefig("./Figures/f_QgivenM1_moe2017.jpg", dpi=300)
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(
    mass_ratio_sample, primary_mass_sample, cdf_sample, norm="log",
    rasterized=True, 
)
ax.contour(
    mass_ratio_sample, primary_mass_sample, np.log10(cdf_sample), levels=10,
    colors="k",
)
ax.plot(mass_ratio_sample, 0.08/mass_ratio_sample)
ax.vlines(mass_ratio_boundary, 0.08, 150.)
ax.hlines(primary_mass_boundary, 0.1, 1.)
ax.set_xlim(0., 1.)
ax.set_ylim(0.08, 150.)
ax.set_yscale("log")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/F_QgivenM1_moe2017.pdf", dpi=300)
plt.savefig("./Figures/F_QgivenM1_moe2017.jpg", dpi=300)
plt.show()

