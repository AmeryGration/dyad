#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("sm")

x = np.loadtxt("./log10_period_sample.dat")
m_primary = np.loadtxt("./primary_mass_sample.dat")
xx, mm = np.meshgrid(x, m_primary)
z = np.loadtxt("./frequency_sample.dat")

fig, ax = plt.subplots()
ax.pcolormesh(xx, mm, z, rasterized=True)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$m_{1}$")
plt.savefig("test.pdf")
plt.show()

