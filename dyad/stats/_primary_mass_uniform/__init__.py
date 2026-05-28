"""
=========================================================================
Primary mass for uniform pairing (:mod:`dyad.stats.primary_mass.uniform`)
=========================================================================

.. currentmodule:: dyad.stats.primary_mass.uniform

This module contains probability distributions for the primary
masses of a population of binary stars formed by uniform pairing. In
its documentation the random variable is denoted :math:`M_{1}` and a
realization of that random variable is denoted :math:`m_{1}`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   kroupa2001
   salpeter1955

"""

__all__ = [
    "kroupa2001",
    "salpeter1955",
    "splitpowerlaw",
]

import numpy as np
import scipy as sp
import flourmill as flr

from dyad.stats import mass
from .. import _distn_infrastructure

def solve_fie2(rv_mass, q_min=0.1):
    """Return primary-mass function as instance of `flr.IeResult`"""
    def g(m):
        """Return the known function in log space"""
        x = 10.**np.array(m)
        res = 2.*rv_mass.pdf(x)

        return res

    def kernel_1(x, y, m_min=rv_mass.a, q_min=q_min):
        """Return CSMF for uniform pairing in log space"""
        m_2 = 10.**np.array(x)
        m_1 = 10.**np.array(y)
        res = (
            np.log(10.)*np.ones_like(m_2/m_1)
            /(1. - np.maximum(q_min, m_min/m_1))
        )
        res *= (q_min <= m_2/m_1) & (m_2/m_1 <= 1.)

        return res

    # def kernel_2(x, y, q_min=q_min):
    #     m_2 = 10.**np.array(x)
    #     m_1 = 10.**np.array(y)
    #     res = (q_min <= m_2/m_1) & (m_2/m_1 <= 1.)

    #     return res

    eps = 1.e-12
    sol = flr.solve_fie2(g, kernel_1, np.log10(rv_mass.a) + eps,
                         np.log10(rv_mass.b) + eps, -1., dense_output=True,
                         args_ker=(rv_mass.a, q_min))
    # sol = flr.solve_fie2(g, kernel_1, np.log10(rv_mass.a) + eps,
    #                      np.log10(rv_mass.b) + eps, -1., dense_output=True,
    #                      method="ProdTrapRule", ker_singular=kernel_2,
    #                      rtol=1.e-02, atol=1.e-06,
    #                      # n_max=2**12 + 1,
    #                      args_ker=(rv_mass.a, q_min),
    #                      )

    # Check the solution is good
    # Code here

    return sol

def interp(rv_mass, q_min):
    """Return interpolation functions for the primary-mass PDF, CDF, and PPF"""
    sol = solve_fie2(rv_mass, q_min)
    pdf = sp.interpolate.interp1d(10.**sol.x, sol.g, kind="linear",
                                  bounds_error=False, fill_value=0.)
    G_nodes = sp.integrate.cumulative_trapezoid(sol.g, 10.**sol.x, initial=0.)
    G_nodes /= G_nodes[-1]
    cdf = sp.interpolate.interp1d(10.**sol.x, G_nodes, kind="quadratic",
                                  bounds_error=False, fill_value=(0, 1))
    ppf = sp.interpolate.interp1d(G_nodes, 10.**sol.x, kind="quadratic",
                                  bounds_error=False, fill_value=np.nan)

    return pdf, cdf, ppf


class kroupa2001_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for uniform pairing

    %(before_notes)s

    Notes
    -----

    The probability density function for `uniform.kroupa2001` is the
    solution to the integral equation

    .. math::
       xxx

    where :math:`f_{M}` is the probability density function for the
    mass random variable of Kroupa (2001) and :math:`f_{M_{2}|M_{2}}`
    is the conditional secondary mass function for uniform pairing,
    which is given by

    .. math::
       f_{M_{2}|M_{2}}(m_{2}|m_{1})
       = \dfrac{1}{m_{1}}f_{Q|M_{1}}(m_{2}/m_{1})|m_{2})

    where :math:`f_{Q|M_{1}}` is the conditional mass-ratio function
    for uniform pairing.
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.kroupa2001
    dyad.stats.mass_ratio.uniform
    
    References
    ----------
    Kroupa, P. 2001. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp = None
        
    def pdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.kroupa2001(*a, *b), *q_min)

        return self._interp[0]

    def cdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.kroupa2001(*a, *b), *q_min)

        return self._interp[1]

    def ppf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.kroupa2001(*a, *b), *q_min)

        return self._interp[2]

    def _argcheck(self, a, b, q_min):
        return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

    def _get_support(self, a, b, q_min):
        res = (a, b)

        return res
        
    def _pdf(self, x, a, b, q_min):
        res = self.pdf_interp(a, b, q_min)(x)

        return res

    def _cdf(self, x, a, b, q_min):
        res = self.cdf_interp(a, b, q_min)(x)

        return res

    def _ppf(self, x, a, b, q_min):
        res = self.ppf_interp(a, b, q_min)(x)

        return res


kroupa2001 = kroupa2001_gen(name="primary_mass.uniform.kroupa2001")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for uniform pairing

    %(before_notes)s

    Notes
    -----

    The probability density function for `uniform.salpeter1955` is the
    solution to the integral equation

    .. math::
       xxx

    where :math:`f_{M}` is the probability density function for the
    mass random variable of Salpeter (1955) and
    :math:`f_{M_{2}|M_{2}}` is the conditional secondary mass function
    for uniform pairing, which is given by

    .. math::
       f_{M_{2}|M_{2}}(m_{2}|m_{1})
       = \dfrac{1}{m_{1}}f_{Q|M_{1}}(m_{2}/m_{1})|m_{2})

    where :math:`f_{Q|M_{1}}` is the conditional mass-ratio function
    for uniform pairing.
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.salpeter1955
    dyad.stats.mass_ratio.uniform
    
    References
    ----------
    Kroupa, P. 2001. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp = None
        
    def pdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

        return self._interp[0]

    def cdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

        return self._interp[1]

    def ppf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

        return self._interp[2]

    def _argcheck(self, a, b, q_min):
        return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

    def _get_support(self, a, b, q_min):
        res = (a, b)

        return res
        
    def _pdf(self, x, a, b, q_min):
        res = self.pdf_interp(a, b, q_min)(x)

        return res

    def _cdf(self, x, a, b, q_min):
        res = self.cdf_interp(a, b, q_min)(x)

        return res

    def _ppf(self, x, a, b, q_min):
        res = self.ppf_interp(a, b, q_min)(x)

        return res


salpeter1955 = salpeter1955_gen(name="primary_mass.uniform.salpeter1955")


class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for uniform pairing

    %(before_notes)s

    Notes
    -----

    The probability density function for `uniform.splitpowerlaw` is the
    solution to the integral equation

    .. math::
       xxx

    where :math:`f_{M}` is the probability density function for the
    two-piece power-function mass random variable and
    :math:`f_{M_{2}|M_{2}}` is the conditional secondary mass function
    for uniform pairing, which is given by

    .. math::
       f_{M_{2}|M_{2}}(m_{2}|m_{1})
       = \dfrac{1}{m_{1}}f_{Q|M_{1}}(m_{2}/m_{1})|m_{2})

    where :math:`f_{Q|M_{1}}` is the conditional mass-ratio function
    for uniform pairing.
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.splitpowerlaw
    dyad.stats.mass_ratio.uniform
    
    References
    ----------
    Kroupa, P. 2001. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp = None
        
    def pdf_interp(self, s, a, b, c, d, q_min):
        if self._interp is None:
            self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
                                  *q_min)

        return self._interp[0]

    def cdf_interp(self, s, a, b, c, d, q_min):
        if self._interp is None:
            self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
                                  *q_min)

        return self._interp[1]

    def ppf_interp(self, s, a, b, c, d, q_min):
        if self._interp is None:
            self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
                                  *q_min)

        return self._interp[2]

    def _argcheck(self, s, a, b, c, d, q_min):
        return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

    def _get_support(self, s, a, b, c, d, q_min):
        res = (a, b)

        return res
        
    def _pdf(self, x, s, a, b, c, d, q_min):
        res = self.pdf_interp(s, a, b, c, d, q_min)(x)

        return res

    def _cdf(self, x, s, a, b, c, d, q_min):
        res = self.cdf_interp(s, a, b, c, d, q_min)(x)

        return res

    def _ppf(self, x, s, a, b, c, d, q_min):
        res = self.ppf_interp(s, a, b, c, d, q_min)(x)

        return res


splitpowerlaw = splitpowerlaw_gen(name="primary_mass.uniform.splitpowerlaw")
