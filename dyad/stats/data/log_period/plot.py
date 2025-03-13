#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

x = np.loadtxt("./log10_period_sample.dat")
m_primary = np.loadtxt("./primary_mass_sample.dat")
xx, mm = np.meshgrid(x, m_primary)
f = np.loadtxt("./frequency_sample.dat")
F = np.loadtxt("./cumulative_frequency_sample.dat")

fig, ax = plt.subplots()
# ax.pcolormesh(xx, mm, f, rasterized=True)
ax.pcolormesh(xx, mm, f, rasterized=True)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("./Figures/pdf_log10_period.jpg")
plt.savefig("./Figures/pdf_log10_period.pdf")
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(xx, mm, F, rasterized=True)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("./Figures/cdf_log10_period.jpg")
plt.savefig("./Figures/cdf_log10_period.pdf")
plt.show()
