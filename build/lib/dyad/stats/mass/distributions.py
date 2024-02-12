__all__ = [
    "kroupa"
]

import numpy as np
import scipy as sp


class _kroupa_gen(sp.stats.rv_continuous):
    r"""The Kroupa initial stellar mass random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `kroupa` is:

    .. math::

        f(x, s, a, b) =
        \dfrac{1}{A}
        \begin{cases}
        x^{c}  &\text{if $x < s$,}
        sx^{c - d} &\text{otherwise}
        \end{cases}

    where

    .. math::

        A := \dfrac{s^{c + 1} - a^{c + 1}}{c + 1}
        + \dfrac{s^{c - d}(b^{d + 1} - s^{d + 1})}{d + 1},
    
    :math:`x > 0`, :math:`a > 0`, :math:`a < s < b`, :math:`c < 0`, and
    :math:`d < 0` [1]_.

    `kroupa` takes :math:`s`, :math:`a`, :math:`b`, :math:`c`, and :math`d`
    as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Reference

    %(example)s

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

    
kroupa = _kroupa_gen(name="kroupa")
