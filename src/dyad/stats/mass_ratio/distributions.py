__all__ = [
    "truncnorm",
    "duquennoy1991",
]

import numpy as np
import scipy as sp

truncnorm = sp.stats.truncnorm


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The Duquennoy and Mayor (1991) mass-ratio random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `duquennoy1991` is:

    .. math::

        f(x) = 

    where

    .. math::

        A := 
    
    :math:`x > 0` [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Reference

    %(example)s

    """
    # Check 0 < a < b.
    def _pdf(self, x):
        return _duquennoy1991.pdf(x)

    def _cdf(self, x):
        return _duquennoy1991.cdf(x)

    def _ppf(self, x):
        return _duquennoy1991.ppf(x)
    

loc = 0.23
scale = 0.42
a = (0. - loc)/scale
b = (np.inf - loc)/scale
_duquennoy1991 = truncnorm(a=a, b=b, loc=loc, scale=scale)
duquennoy1991 = _duquennoy1991_gen(a=0., b=np.inf, name="duquennoy1991")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Plot Duquennoy 1991
    mass_ratios = duquennoy1991.rvs(size=100_000)
    counts, edges = np.histogram(mass_ratios, bins=25,
                                 density=True)
    print(np.min(mass_ratios), np.max(mass_ratios))
    
    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

