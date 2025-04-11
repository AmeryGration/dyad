#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

m_primary = np.loadtxt("./primary_mass_sample.dat")
p = np.loadtxt("./period_sample.dat")
pp, mm = np.meshgrid(p, m_primary)
f = np.loadtxt("./frequency_sample.dat")
F = np.loadtxt("./cumulative_frequency_sample.dat")

fig, ax = plt.subplots()
ax.pcolormesh(pp, mm, f, cmap="Grays", rasterized=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("./Figures/pdf_period.jpg")
plt.savefig("./Figures/pdf_period.pdf")
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(pp, mm, F, cmap="Grays", rasterized=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("./Figures/cdf_period.jpg")
plt.savefig("./Figures/cdf_period.pdf")
plt.show()

