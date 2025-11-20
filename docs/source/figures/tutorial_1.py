#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dyad

# Earth-sun system
orbit = dyad.Orbit(1., 1., 0.0167, 1., 1., 1.)
mu = np.linspace(0., 2.*np.pi)
theta = dyad.true_anomaly_from_mean_anomaly(mu, 0.0167)
r = orbit.radius(theta)

fig, ax = plt.subplots()
ax.plot(theta, r)
ax.set_xticks([0., np.pi, 2.*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$r/\mathrm{au}$")
plt.savefig("evolution_of_radius.jpg")
plt.savefig("evolution_of_radius.pdf")
plt.show()
