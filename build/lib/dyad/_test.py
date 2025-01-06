#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import dyad

# import importlib

# importlib.reload(dyad)

# from kinematics import *
# from properties import *

# m1 = 0.5488135039273248*np.ones(3)
# a1 = 0.7151893663724195*np.ones(3)
# e1 = 0.6027633760716439*np.ones(3)
# theta1 = 0.5448831829968969*np.ones(3)
# Omega1 = 0.4236547993389047*np.ones(3)
# i1 = 0.6458941130666561*np.ones(3)
# omega1 = 0.4375872112626925*np.ones(3)
# q = 0.8070353890880327*np.ones(3)

# bin = Binary(m1, q, a1, e1, theta1, Omega1, i1, omega1)
# print(bin.primary.period)
# print(bin.secondary.period)
# print(bin.primary.speed)
# print(bin.secondary.speed)

# m2 = q*m1
# M = m2**3./(m1 + m2)**2.
# orb = Orbit(M, [a1, e1, theta1, Omega1, i1, omega1])
# print(orb.speed)
# print(np.sqrt(np.sum(orb._velocity**2., axis=1)))

# #########################################################################
# # How to find the luminosity-weighted velocity?
# #########################################################################
# T1 = 0.30526230045514713*np.ones(3)
# T2 = 0.46441980639616787*np.ones(3)
# body_1 = Body(m1)#, T1)
# body_2 = Body(m2)#, T2)

# binary = Binary(body_1.mass, body_2.mass/body_1.mass, a1, e1,
#                 theta1, Omega1, i1, omega1)

# a = [binary.primary.state[:,-1], binary.secondary.state[:,-1]]
# weights = [body_1.luminosity, body_2.luminosity]
# lv = np.average(a, axis=0, weights=weights)

e = 0.5

theta = np.linspace(0., 2.*np.pi)
q = np.linspace(0., 1.)

pdf_theta = dyad.stats.true_anomaly(e).pdf(theta)
cdf_theta = dyad.stats.true_anomaly(e).cdf(theta)
ppf_theta = dyad.stats.true_anomaly(e).ppf(q)

plt.plot(theta, pdf_theta)
plt.plot(theta, cdf_theta)
plt.plot(cdf_theta, theta)
plt.plot(q, ppf_theta, color="red")
plt.show()

theta = dyad.stats.true_anomaly(e).rvs(size=10_000)
hist = np.histogram(theta/(2.*np.pi), np.linspace(0., 1), density=True)
plt.stairs(*hist)
plt.show()
