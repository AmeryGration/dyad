#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dyad

mpl.style.use("sm")

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

# PDF for inclination
rv = dyad.stats.inclination
x = np.linspace(0., np.pi)
pdf = rv.pdf(x)
sample = rv.rvs(size=10_000)

fig, ax = plt.subplots()
ax.hist(sample, bins=25, density=True, histtype="step")
ax.set_xticks([0., 0.5*np.pi, np.pi], [r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_xlabel(r"$i$")
ax.set_ylabel(r"$f_{I}$")
ax.plot(x, pdf)
plt.savefig("pdf_inclination.jpg")
plt.savefig("pdf_inclination.pdf")
plt.show()

# PDF for true anomaly
rv = dyad.stats.true_anomaly(e=0.5)
x = np.linspace(0., 2.*np.pi)
pdf = rv.pdf(x)
sample = rv.rvs(size=10_000)

fig, ax = plt.subplots()
ax.hist(sample, bins=25, density=True, histtype="step")
ax.plot(x, pdf)
ax.set_xticks([0., np.pi, 2.*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$f_{\Theta}$")
plt.savefig("pdf_true_anomaly.jpg")
plt.savefig("pdf_true_anomaly.pdf")
plt.show()

