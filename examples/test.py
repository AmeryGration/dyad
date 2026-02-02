import numpy as np
import matplotlib.pyplot as plt
from dyad.stats import primary_mass
from dyad.stats import secondary_mass
from dyad.stats import mass_ratio

x = np.logspace(np.log10(0.08), np.log10(150.), 1_000)
y_kroupa2001 = primary_mass.random.kroupa2001(0.08, 150.).pdf(x)
y_salpeter1955 = primary_mass.random.salpeter1955(0.08, 150.).pdf(x)
y_splitpowerlaw = primary_mass.random.splitpowerlaw(10., 0.08, 150., -1.3, -2.3).pdf(x)
rvs_kroupa2001 = primary_mass.random.kroupa2001(0.08, 150.).rvs(size=100_000)
rvs_salpeter1955 = primary_mass.random.salpeter1955(0.08, 150.).rvs(size=100_000)
rvs_splitpowerlaw = primary_mass.random.splitpowerlaw(10., 0.08, 150., -1.3, -2.3).rvs(size=100_000)
bins = np.logspace(np.log10(0.08), np.log10(150.))

fig, ax = plt.subplots()
ax.hist(rvs_kroupa2001, bins=bins, density=True, alpha=0.5)
ax.plot(x, y_kroupa2001)
ax.hist(rvs_salpeter1955, bins=bins, density=True, alpha=0.5)
ax.plot(x, y_salpeter1955)
ax.hist(rvs_splitpowerlaw, bins=bins, density=True, alpha=0.5)
ax.plot(x, y_splitpowerlaw)
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()

Y_kroupa2001 = primary_mass.random.kroupa2001(0.08, 150.).cdf(x)
Y_salpeter1955 = primary_mass.random.salpeter1955(0.08, 150.).cdf(x)
Y_splitpowerlaw = primary_mass.random.splitpowerlaw(5., 0.08, 150., -1.3, -2.3).cdf(x)

fig, ax = plt.subplots()
ax.plot(x, Y_kroupa2001)
ax.plot(x, Y_salpeter1955)
ax.plot(x, Y_splitpowerlaw)
ax.set_xscale("log")
# ax.set_yscale("log")
plt.show()

# mass_1 = primary_mass.random.kroupa2001(0.08, 150.).rvs(size=1_000)
# mass_2 = secondary_mass.


x = np.linspace(0., 1, 1_000)
y_kroupa2001 = mass_ratio.random.kroupa2001(0.08, 150., 1.).pdf(x)

import scipy as sp

# I = [sp.integrate.quad(
#     mass_ratio.random.kroupa2001(0.08, 150., 1.).pdf, 0., i)[0] for i in [0.1, 0.5, 0.9]
#      ]

y_kroupa2001 = mass_ratio.random.kroupa2001(1., 0.08, 150.).pdf(x)
rvs_kroupa2001 = mass_ratio.random.kroupa2001(1., 0.08, 150.).rvs(size=100_000)

Y_kroupa2001 = mass_ratio.random.kroupa2001(1., 0.08, 150.).cdf(x)

fig, ax = plt.subplots()
ax.hist(rvs_kroupa2001, bins=100, density=True, alpha=0.5)
ax.plot(x, y_kroupa2001)
ax.plot(x, Y_kroupa2001)
# ax.set_yscale("log")
# plt.show()

y_salpeter1955 = mass_ratio.random.salpeter1955(1., 0.08, 150.).pdf(x)
rvs_salpeter1955 = mass_ratio.random.salpeter1955(1., 0.08, 150.).rvs(size=100_000)

Y_salpeter1955 = mass_ratio.random.salpeter1955(1., 0.08, 150.).cdf(x)

# fig, ax = plt.subplots()
ax.hist(rvs_salpeter1955, bins=100, density=True, alpha=0.5)
ax.plot(x, y_salpeter1955)
ax.plot(x, Y_salpeter1955)
ax.set_yscale("log")
plt.show()

