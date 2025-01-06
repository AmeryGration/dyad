#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distributions import *

mpl.style.use("sm")

# Plot loguniform semimajor axis
semimajor_axis = opik1924(1.e3, 3.e3).rvs(size=10_000)
counts, edges = np.histogram(
    np.log10(semimajor_axis),
    bins=25,
    density=True
)
print(np.min(semimajor_axis), np.max(semimajor_axis))

fig, ax = plt.subplots()
ax.stairs(counts, edges)
ax.set_xlabel(r"$\log_{10}(a/\mathrm{AU})$")
ax.set_ylabel(r"$\hat{f}$")
plt.show()

