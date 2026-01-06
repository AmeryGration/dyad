"""
=============================
Mass (:mod:`dyad.stats.mass`)
=============================

.. currentmodule:: dyad.stats.mass

This module contains probability distributions for the masses of a
population of stars. In its documentation the random variable is
denoted :math:`M` and a realization of that random variable is denoted
:math:`m`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   splitpowerlaw
   kroupa2002
   salpeter1955

"""

__all__ = [
    "splitpowerlaw",
    "salpeter1955",
    # "millerscalo1979",
    "kroupa2002",
    # "chabrier2003",
    # "maschberger2013"
]

import numpy as np
import scipy as sp

from . import _distn_infrastructure


class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
    r"""The two-piece split-power-law mass random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `splitpower` is:

    .. math::
       f_{M}(m)
       =
       A
       \begin{cases}
       m^{-\alpha}&\text{ if $m \in [a, b)$,}\\
       Bm^{-\beta}&\text{ if $m \in [b, c]$}
       \end{cases}

    for

    .. math::
       A = \dfrac{1}{1 - \alpha}(b^{1 - \alpha} - a^{1 - \alpha}) + \dfrac{B}{1 - \beta}(c^{1 - \beta} - b^{1 - \beta}),

    and

    .. math::
       B = b^{\beta - \alpha}.

    and :math:`m \in [a, c]`.
    
    %(after_notes)s

    %(example)s

    """
    def _argcheck(self, s, a, b, c, d):
        return (0. < a) & (a < b) & (a < s) & (s < b) & (c < 0.) & (d < 0.)

    def _shape_info(self):
        is_ = _ShapeInfo("s", False, (0., np.inf), (True, False))
        ia = _ShapeInfo("a", False, (0., np.inf), (True, False))
        ib = _ShapeInfo("b", False, (0., np.inf), (True, False))
        ic = _ShapeInfo("c", False, (-np.inf, np.inf), (False, False))
        id_ = _ShapeInfo("d", False, (-np.inf, np.inf), (False, False))

        return [is_, ia, ib, ic, id_]

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


splitpowerlaw = splitpowerlaw_gen(name="mass.splitpowerlaw")


class kroupa2002_gen(_distn_infrastructure.rv_continuous):
    r"""The mass random variable of Kroupa (2002)

    %(before_notes)s

    Notes
    -----
    The probability density function for `kroupa2002` is:

    .. math::
       f_{M}(m)
       =
       A_{M}
       \begin{cases}
       m^{-1.3} &\text{ if $m \in [a, 0.5)$}\\
       0.5m^{-2.3} &\text{ if $m \in [0.5, b]$}\\
       \end{cases}

    for

    .. math::
       A_{M} := \dfrac{0.5(0.5^{-1.3} - b^{-1.3})}{1.3} +
       \dfrac{a^{-0.3} - 0.5^{-0.3}}{0.3}

    and :math:`m \in [a, b]` where :math:`a < 0.5` and :math:`b > 0.5`.
    
    %(after_notes)s

    References
    ----------
    Kroupa, P. 2002. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.
    
    %(example)s

    """
    # Check 0 < a < b.
    def _get_support(self, a, b):
        res = a, b

        return res
    
    def _pdf(self, x, a, b):
        res = splitpowerlaw(0.5, a, b, -1.3, -2.3).pdf(x)

        return res

    def _cdf(self, x, a, b):
        res = splitpowerlaw(0.5, a, b, -1.3, -2.3).cdf(x)

        return res

    def _ppf(self, q, a, b):
        res = splitpowerlaw(0.5, a, b, -1.3, -2.3).ppf(q)

        return res


kroupa2002 = kroupa2002_gen(name="mass.kroupa2002")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""The mass random variable Salpeter (1955)

    %(before_notes)s

    Notes
    -----
    The probability density function for `salpeter1955` is:

    .. math::
       f_{M}(m) = \dfrac{c - 1}{a^{1 - c} - b^{1 - c}}\dfrac{1}{m^{c}}

    for :math:`m \in [a, b]` and :math:`c = 2.35`.
    
    %(after_notes)s

    References
    ----------
    Salpeter, Edwin E. 1955. \'The luminosity function and stellar
    evolution.\' *The Astrophysical Journal* 121 (January): 161.

    %(example)s

    """
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0., np.inf), (False, True))
        ib = _ShapeInfo("b", False, (0., np.inf), (False, True))

        return [ia, ib]

    def _argcheck(self, a, b):
        res = (0. < a) & (0. < b) & (a < b)

        return res

    def _get_support(self, a, b):
        res = a, b

        return res

    def _pdf(self, x, a, b):
        res =  (2.35 - 1.)/(a**(1. - 2.35) - b**(1. - 2.35))/x**2.35

        return res

    def _cdf(self, x, a, b):
        res = (
            (x**(1. - 2.35) - a**(1. - 2.35))
            /(b**(1. - 2.35) - a**(1. - 2.35))
        )

        return res
    
    def _ppf(self, q, a, b):
        base = (b**(1. - 2.35) - a**(1. - 2.35))*q + a**(1. - 2.35)
        exp = 1./(1. - 2.35)
        res = base**exp

        return res


salpeter1955 = salpeter1955_gen(name="mass.salpeter1955")
salpeter1955._support = ("a", "b")


# class _millerscalo1979_gen(_distn_infrastructure.rv_continuous):
#     r"""The Miller--Scalo (1979) mass random variable

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
# millerscalo1979 = _millerscalo1979_gen(a=0., b=np.inf, name="mass.millerscalo1979")


# class _chabrier2003_gen(_distn_infrastructure.rv_continuous):
#     r"""The Chabrier (2003) mass random variable

#     """
#     # Check 0 < a < b.
#     def _pdf(self, x):
#         return _chabrier2003.pdf(x)

#     def _cdf(self, x):
#         return _chabrier2003.cdf(x)

#     def _ppf(self, q):
#         return _chabrier2003.ppf(q)


# _chabrier2003 = xxx
# chabrier2003 = _chabrier2003_gen(a=0.4, b=10., name="mass.chabrier2003")


# class _maschberger2013_gen(_distn_infrastructure.rv_continuous):
#     r"""The Maschberger (2013) mass random variable

#     """
#     def _pdf(self, x):
#         return _maschberger2013.pdf(x)

#     def _cdf(self, x):
#         return _maschberger2013.cdf(x)

#     def _ppf(self, q):
#         return _maschberger2013.ppf(q)


# _maschberger2013 = xxx
# maschberger2013 = _maschberger2013_gen(a=0.4, b=10., name="mass.maschberger2013")
