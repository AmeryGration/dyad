#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

########################################################################
# Plot log-period
########################################################################
primary_mass_sample = np.loadtxt("./primary_mass_sample.dat")
period_sample= np.loadtxt("./period_sample.dat")
frequency_sample = np.loadtxt("./frequency_sample.dat")
cumulative_frequency_sample = np.loadtxt("./cumulative_frequency_sample.dat")

fig, ax = plt.subplots()
ax.pcolormesh(
    period_sample,
    primary_mass_sample,
    frequency_sample,
    rasterized=True
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\log(p)$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/pdf_log10_period.pdf")
plt.savefig("./Figures/pdf_log10_period.jpg")
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(
    period_sample,
    primary_mass_sample,
    cumulative_frequency_sample,
    rasterized=True
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\log(p)$")
ax.set_ylabel(r"$m_{1}/\mathrm{M}_{\odot}$")
plt.savefig("./Figures/cdf_log10_period.pdf")
plt.savefig("./Figures/cdf_log10_period.jpg")
plt.show()
