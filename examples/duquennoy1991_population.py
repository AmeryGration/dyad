#!/usr/bin/env python3

r"""Synthesize a population of binary stars (Duquennoy & Mayor, 1991)

Realize a sample of orbital elements using the distributions published
by Duquennoy & Mayor (1991) for sun-like stars in the solar
neighbourhood. Use this sample to compute the kinematic properties of
a population of binary stars. Plot histograms of the line-of-sight
velocities of the primary and secondary stars.

"""

import numpy as np
import matplotlib.pyplot as plt
import dyad

from scipy import constants

n = 10_000

# Sample orbital elements
m_1 = np.full((n,), 0.8)
q = dyad.stats.mass_ratio.duquennoy1991.rvs(size=n)
p = 10.**dyad.stats.log_period.duquennoy1991.rvs(size=n)
e = np.zeros(n)
idx = (p > 11.6)
e[idx] = dyad.stats.eccentricity.duquennoy1991(p[idx]).rvs()
theta = dyad.stats.true_anomaly(e).rvs()
Omega = dyad.stats.longitude_of_ascending_node.rvs(size=n)
i = dyad.stats.inclination.rvs(size=n)
omega = dyad.stats.argument_of_pericentre().rvs(size=n)
a = dyad.semimajor_axis_from_period(p, m_1, m_1*q)

# Create population
binary = dyad.TwoBody(m_1, m_1*q, a, e, Omega, i, omega)
xv_1 = binary.primary.state(theta)
xv_2 = binary.secondary.state(theta)
xv_1[:,3:] *= constants.astronomical_unit/constants.day/1.e3
xv_2[:,3:] *= constants.astronomical_unit/constants.day/1.e3

header = (
    "x (AU), "
    + "y (AU), "
    + "v (AU), "
    + "v_{x} (km s^{-1}), "
    + "v_{y} (km s^{-1}), "
    + "v_{z} (km s^{-1})"
)
np.savetxt(
    "./Data/duquennoy1991_primary_stars.dat", xv_1, delimiter=", ",
    header=header
)
np.savetxt(
    "./Data/duquennoy1991_secondary_stars.dat", xv_2, delimiter=", ",
    header=header
)

# Plot LOS velocity
fig, ax = plt.subplots()
ax.hist(
    xv_1[:,-1], bins="auto", range=(-1.e0, 1.e0), alpha=0.2, label="primary"
)
ax.hist(
    xv_2[:,-1], bins="auto", range=(-1.e0, 1.e0), alpha=0.2, label="secondary"
)
ax.legend(frameon=False)
ax.set_xlabel(r"$v_{z}/\text{km}~\text{s}^{-1}$")
ax.set_ylabel(r"$\nu$")
fig.savefig("./Figures/duquennoy1991_los_velocity.pdf")
plt.show()
