#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distributions import *

mpl.style.use("sm")

N_SAMPLE = 10

def test_single_shape_param(rv):
    #####################################################################
    # Single argument
    #####################################################################
    x = 0.5
    f = rv.pdf(x)
    print(f)
    F = rv.cdf(x)
    print(F)
    q = 0.5
    F_inv = rv.ppf(q)
    print(F_inv)
    sample = rv.rvs()
    print(sample)
    sample = rv.rvs(size=1)
    print(sample)

    #####################################################################
    # Multiple argument
    #####################################################################
    x = np.linspace(0., 2.*np.pi, 50)
    f = rv.pdf(x)
    print(f)
    F = rv.cdf(x)
    print(F)
    q = np.linspace(0., 1., 50)
    F_inv = rv.ppf(q)
    print(F_inv)

    fig, ax = plt.subplots()
    ax.plot(x, f)
    ax.plot(x, F)
    ax.plot(q, F_inv)
    plt.show()

    res = rv.rvs(size=100_000)
    print(res)

    counts, edges = np.histogram(res, bins=50, density=True)

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    plt.show()

def test_multiple_shape_params(rv):
    #####################################################################
    # Single argument
    #####################################################################
    x = 0.5
    f = rv.pdf(x)
    print(f)
    F = rv.cdf(x)
    print(F)
    q = 0.5
    F_inv = rv.ppf(q)
    print(F_inv)
    sample = rv.rvs()
    print(sample)

    #####################################################################
    # Multiple argument
    #####################################################################
    x = np.linspace(0., 2.*np.pi, 50)
    f = rv.pdf(x)
    print(f)
    F = rv.cdf(x)
    print(F)
    q = np.linspace(0., 1., 50)
    F_inv = rv.ppf(q)
    print(F_inv)

    fig, ax = plt.subplots()
    ax.plot(x, f)
    ax.plot(x, F)
    ax.plot(q, F_inv)
    plt.show()

    res = rv.rvs()
    print(res)

    counts, edges = np.histogram(res, bins=50, density=True)

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    plt.show()


#########################################################################
#########################################################################
#########################################################################
# TO DO:
# Check broadcasting of shape params and method args
#########################################################################
#########################################################################
#########################################################################

#########################################################################
# Single shape parameter
#########################################################################
test_single_shape_param(sp.stats.norm())
test_single_shape_param(longitude_of_ascending_node)
test_single_shape_param(inclination)
test_single_shape_param(argument_of_pericentre)
test_single_shape_param(true_anomaly(0.5))

#########################################################################
# Multiple shape parameter
#########################################################################
test_multiple_shape_params(
    sp.stats.norm(np.linspace(0., 2.), np.linspace(1., 2.))
)
test_multiple_shape_params(
    true_anomaly(np.linspace(0, 1., endpoint=False))
)

x = np.linspace(0., 1.)
ppf = [true_anomaly(0.5).ppf(x_i) for x_i in x]
x = np.linspace(0., 2.*np.pi)
cdf = [true_anomaly(0.5).cdf(x_i) for x_i in x]
print(ppf)
plt.plot(ppf)
plt.plot(cdf)
plt.show()

rv = true_anomaly(0.)

