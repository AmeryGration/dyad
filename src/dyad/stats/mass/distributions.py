__all__ = [
    "splitpowerlaw",
    "kroupa2002",
    # "salpeter1955",
    # "millerscalo1979",
    # "chabrier2003",
]

import numpy as np
import scipy as sp


class _splitpowerlaw_gen(sp.stats.rv_continuous):
    r"""The two-piece split power law initial stellar mass random variable

    """
    def _argcheck(self, s, a, b, c, d):
        return (0. < a) & (a < b) & (a < s) & (s < b) & (c < 0.) & (d < 0.)

    def _shape_info(self):
        is_ = _ShapeInfo("s", False, (0, np.inf), (False, False))
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))

        return [is_, ia, ib]
    
    def _get_support(self, s, a, b, c, d):
        return a, b

    def _pdf(self, x, s, a, b, c, d):
        def f1(x):
            return x**c

        def f2(x):
            return s**(c - d)*x**d

        A = (
            (s**(c + 1.) - a**(c + 1.))/(c + 1.)
            + s**(c - d)*(b**(d + 1.) - s**(d + 1.))/(d + 1.)
        )

        return np.where(x < s, f1(x), f2(x))/A

    def _cdf(self, x, s, a, b, c, d):
        def f1(x):
            return (x**(c + 1.) - a**(c + 1.))/(c + 1.)

        def f2(x):
            return f1(s) + s**(c - d)*(x**(d + 1.) - s**(d + 1.))/(d + 1.)

        A = (
            (s**(c + 1.) - a**(c + 1.))/(c + 1.)
            + s**(c - d)*(b**(d + 1.) - s**(d + 1.))/(d + 1.)
        )

        return np.where(x < s, f1(x), f2(x))/A
    
    def _ppf(self, q, s, a, b, c, d):
        def f1(q):
            return (A*(c + 1.)*q + a**(c + 1.))**(1./(c + 1.))

        def f2(q):
            res = (
                A*(d + 1.)*(q - self._cdf(s, s, a, b, c, d))/s**(c - d)
                + s**(d + 1.)
            )

            return res**(1./(d + 1.))

        A = (
            (s**(c + 1.) - a**(c + 1.))/(c + 1.)
            + s**(c - d)*(b**(d + 1.) - s**(d + 1.))/(d + 1.)
        )

        return np.where(q < self._cdf(s, s, a, b, c, d), f1(q), f2(q))

    
splitpowerlaw = _splitpowerlaw_gen(name="splitpowerlaw")


class _kroupa2002_gen(sp.stats.rv_continuous):
    r"""The Kroupa (2002) initial stellar mass random variable

    """
    # Check 0 < a < b.
    def _pdf(self, x):
        return _kroupa2002.pdf(x)

    def _cdf(self, x):
        return _kroupa2002.cdf(x)
    
    def _ppf(self, q):
        return _kroupa2002.ppf(q)

    
_kroupa2002 = splitpowerlaw(s=0.5, a=0.1, b=60., c=-1.3, d=-2.3)
kroupa2002 = _kroupa2002_gen(a=0.1, b=60., name="kroupa2002")


# class _salpeter1955_gen(sp.stats.rv_continuous):
#     r"""The Salpeter (1955) initial stellar mass random variable

#     """
#     # Check 0 < a < b.
#     def _pdf(self, x):
#         return _salpeter1955.pdf(x)

#     def _cdf(self, x):
#         return _salpeter1955.cdf(x)
    
#     def _ppf(self, q):
#         return _salpeter1955.ppf(q)


# _salpeter1955 = sp.stats.pareto(1.3) # Placeholder
# salpeter1955 = _salpeter1955_gen(a=0.4, b=50., name="salpeter1955")


# class _millerscalo1979_gen(sp.stats.rv_continuous):
#     r"""The Miller-Scalo (1979) initial stellar mass random variable

#     """
#     # Check 0 < a < b.
#     def _pdf(self, x):
#         return _millerscalo1979.pdf(x)

#     def _cdf(self, x):
#         return _millerscalo1979.cdf(x)
    
#     def _ppf(self, q):
#         return _millerscalo1979.ppf(q)


# mu = 0. # Placeholder
# sigma = 1. # Placeholder
# s = sigma
# loc = 0.
# scale = np.exp(mu)
# _millerscalo1979 = sp.stats.lognorm(s=s, loc=loc, scale=scale)
# millerscalo1979 = _millerscalo1979_gen(a=0., b=np.inf, name="millerscalo1979")


# class _chabrier2003_gen(sp.stats.rv_continuous):
#     r"""The Chabrier (2003) initial stellar mass random variable

#     """
#     # Check 0 < a < b.
#     def _pdf(self, x):
#         return _chabrier2003.pdf(x)

#     def _cdf(self, x):
#         return _chabrier2003.cdf(x)
    
#     def _ppf(self, q):
#         return _chabrier2003.ppf(q)


# _chabrier2003 = xxx
# chabrier2003 = _chabrier2003_gen(a=0.4, b=10., name="chabrier2003")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
   
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
