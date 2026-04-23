"""
=======================================================================
Primary mass for random pairing (:mod:`dyad.stats.primary_mass.random`)
=======================================================================

.. currentmodule:: dyad.stats.primary_mass.random

This module contains probability distributions for the primary
masses of a population of binary stars formed by random pairing. In
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

from dyad.stats import mass
from .. import _distn_infrastructure


class kroupa2001_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.kroupa2001` is:

    .. math::
       f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1}),

    where :math:`f_{M}` and :math:`F_{M}` are the probability density
    function and cumulative distribution function for the mass random
    variable of Kroupa (2001).
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.kroupa2001
    
    References
    ----------
    Kroupa, P. 2001. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

    Malkov, O., and H. Zinnecker. 2001. \'Binary Stars and the
    Fundamental Initial Mass Function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149--54.

    %(example)s

    """
    def _get_support(self, a, b):
        res = (a, b)

        return res
        
    def _pdf(self, x, a, b):
        rv_mass = mass.kroupa2001(a, b)
        res = 2.*rv_mass.pdf(x)*rv_mass.cdf(x)
        
        return res

    def _cdf(self, x, a, b):
        rv_mass = mass.kroupa2001(a, b)
        res = rv_mass.cdf(x)**2.
                
        return res

    def _ppf(self, q, a, b):
        rv_mass = mass.kroupa2001(a, b)
        res = rv_mass.ppf(np.sqrt(q))
        
        return res
    
kroupa2001 = kroupa2001_gen(name="primary_mass.kroupa2001")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.salpeter1955` is:

    .. math::
       f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1}),

    where :math:`f_{M}` and :math:`F_{M}` are the probability density
    function and cumulative distribution function for the mass random
    variable of Salpeter (1955).
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.salpeter1955
    
    References
    ----------
    Salpeter, Edwin E. 1955. \'The luminosity function and stellar
    evolution.\' *The Astrophysical Journal* 121 (January): 161.

    Malkov, O., and H. Zinnecker. 2001. \'Binary Stars and the
    Fundamental Initial Mass Function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149--54.

    %(example)s

    """
    def _get_support(self, a, b):
        res = (a, b)

        return res
        
    def _pdf(self, x, a, b):
        rv_mass = mass.salpeter1955(a, b)
        res = 2.*rv_mass.pdf(x)*rv_mass.cdf(x)
        
        return res

    def _cdf(self, x, a, b):
        rv_mass = mass.salpeter1955(a, b)
        res = rv_mass.cdf(x)**2.
                
        return res

    def _ppf(self, q, a, b):
        rv_mass = mass.salpeter1955(a, b)
        res = rv_mass.ppf(np.sqrt(q))
        
        return res
    
salpeter1955 = salpeter1955_gen(name="primary_mass.salpeter1955")


class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.splitpowerlaw` is:

    .. math::
       f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1}),

    where :math:`f_{M}` and :math:`F_{M}` are the probability density
    function and cumulative distribution function for the mass random
    variable of Salpeter (1955).
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass.splitpowerlaw
    
    References
    Malkov, O., and H. Zinnecker. 2001. \'Binary Stars and the
    Fundamental Initial Mass Function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149--54.

    %(example)s

    """
    def _argcheck(self, s, a, b, c, d):
        return (0. < a) & (a < b) & (a < s) & (s < b) & (c < 0.) & (d < 0.)

    def _get_support(self, s, a, b, c, d):
        res = (a, b)

        return res
        
    def _pdf(self, x, s, a, b, c, d):
        rv_mass = mass.splitpowerlaw(s, a, b, c, d)
        res = 2.*rv_mass.pdf(x)*rv_mass.cdf(x)
        
        return res

    def _cdf(self, x, s, a, b, c, d):
        rv_mass = mass.splitpowerlaw(s, a, b, c, d)
        res = rv_mass.cdf(x)**2.
                
        return res

    def _ppf(self, q, s, a, b, c, d):
        rv_mass = mass.splitpowerlaw(s, a, b, c, d)
        res = rv_mass.ppf(np.sqrt(q))
        
        return res
    
splitpowerlaw = splitpowerlaw_gen(name="primary_mass.splitpowerlaw")
