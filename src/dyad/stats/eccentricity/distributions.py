__all__ = [
    "uniform",
    "powerlaw",
    "thermal",
    "duquennoy1991",
]

import numpy as np
import scipy as sp


uniform = sp.stats._continuous_distns.uniform_gen(a=0., b=1., name="uniform")

powerlaw = sp.stats._continuous_distns.powerlaw_gen(a=0., b=1.,
                                                    name="powerlaw")


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The Duquennoy and Mayor (1991) eccentricity random variable

    %(before_notes)s

    Notes
    -----

    """
    # Accept shape parameter
    # Can call a tuple!
    # >>> sp.stats.norm(loc=(1., 2.)).pdf(1.)
    # array([0.39894228, 0.24197072])
    # Or
    # >>> sp.stats.norm(loc=(1., 2.)).pdf((1., 2.))
    # array([0.39894228, 0.39894228])
    # This speeds things up!

    # Inherit the rv_continuous class but override its methods with
    # those of a frozen rv.
    def _argcheck(self, s):
        return 11. < s
    
    def _pdf(self, x, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.pdf(x),
            _duquennoy1991_f2.pdf(x)
        )

    def _cdf(self, x, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.cdf(x),
            _duquennoy1991_f2.cdf(x)
        )

    def _ppf(self, q, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.ppf(q),
            _duquennoy1991_f2.ppf(q)
        )

        
# Duquennoy and Mayor (1991) tight binaries: truncated normal
loc = 0.27
scale = 0.13
a = (0. - loc)/scale
b = (np.inf - loc)/scale
_duquennoy1991_f1 = sp.stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
# Duquennoy and Mayor (1991) wide binaries: thermal 
_duquennoy1991_f2 = sp.stats.powerlaw(2.)
# Duquennoy and Mayor (1991) all binaries: conditional
duquennoy1991 = _duquennoy1991_gen(a=0., b=1., name="duquennoy1991")


class _thermal_gen(sp.stats.rv_continuous):
    r"""The thermal eccentricity random variable

    %(before_notes)s

    Notes
    -----

    """
    def _pdf(self, x):
        return _thermal.pdf(x)

    def _cdf(self, x):
        return _thermal.cdf(x)

    def _ppf(self, q):
        return _thermal.ppf(q)
    

_thermal = sp.stats.powerlaw(2.)
thermal = _thermal_gen(a=0., b=1., name="thermal")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import dyad
    
    # Plot Duquennoy 1991: tight binaries
    eccentricities = duquennoy1991(500).rvs(size=10_000)
    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

    # Plot Duquennoy 1991: wide binaries
    eccentricities = duquennoy1991(5000).rvs(size=10_000)
    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

    # Plot Duquennoy 1991: conditional random variable
    periods = dyad.stats.period.duquennoy1991.rvs(size=10_000)
    #2000*np.random.rand(10_000)
    eccentricities = np.zeros_like(periods)
    eccentricities[periods>11.] = duquennoy1991(periods[periods>11.]).rvs()

    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()
 
