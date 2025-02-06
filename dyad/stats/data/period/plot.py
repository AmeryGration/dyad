#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

m_primary = np.loadtxt("./primary_mass_sample.dat")
p = np.loadtxt("./period_sample.dat")
pp, mm = np.meshgrid(p, m_primary)
z = np.loadtxt("./frequency_sample.dat")

fig, ax = plt.subplots()
ax.pcolormesh(pp, mm, z, rasterized=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("test.pdf")
plt.show()

