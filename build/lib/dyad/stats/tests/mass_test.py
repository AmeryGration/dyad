#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distributions import *

mpl.style.use("sm")

# Plot Duquennoy 1991
masses = kroupa2002.rvs(size=100_000)
counts, edges = np.histogram(masses, bins=np.logspace(-2., 2.),
                             density=True)
print(np.min(masses), np.max(masses))

fig, ax = plt.subplots()
ax.stairs(counts, edges)
ax.set_xlabel(r"$M/\mathrm{M}_{\odot}$")
ax.set_ylabel(r"$\hat{f}$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
