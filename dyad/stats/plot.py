#!/usr/bin/env python

import numpy as np
import dyad
import matplotlib.pyplot as plt

m_1 = np.logspace(np.log10(0.08), np.log10(150.), 250)
m_2 = np.logspace(np.log10(0.08), np.log10(150.), 250)
m1m1, m2m2 = np.meshgrid(m_1, m_2)
f_M2givenM1 = dyad.stats.secondary_mass.moe2017(m_1[:,None]).pdf(m_2)

fig, ax = plt.subplots()
ax.pcolormesh(m1m1, m2m2, f_M2givenM1, norm="log")
ax.contour(m1m1, m2m2, np.log10(f_M2givenM1), colors="k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.08, 150.)
ax.set_ylim(0.08, 150.)
plt.show()
