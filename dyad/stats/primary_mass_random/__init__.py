"""
========================================================
Secondary mass (:mod:`dyad.stats.secondary_mass.random`)
========================================================

.. currentmodule:: dyad.stats.secondary_mass.random

This module contains probability distributions for the secondary
masses of a population of binary stars formed by random pairing. In
its documentation the random variable is denoted :math:`M_{2}` and a
realization of that random variable is denoted :math:`m_{2}`. The
random variable for secondary mass is conditional on primary mass. The
PDF for secondary mass is known as the `pairing function'

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   kroupa2002

"""

__all__ = [
    "kroupa2002",
]

import numpy as np
import scipy as sp

from .. import _distn_infrastructure
from .. import mass


# class kroupa2002_gen(_distn_infrastructure.rv_continuous):
#     r"""The primary-star mass random variable for random pairing

#     %(before_notes)s

#     Notes
#     -----
#     The probability density function for `random.kroupa2002` is:

#     .. math::
#        f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1}),

#     where :math:`f_{M}` and :math:`F_{M}` are the probability density
#     function and cumulative distribution function for the mass random
#     variable of Kroupa (2002).
    
#     %(after_notes)s

#     See also
#     --------
#     dyad.stats.mass.kroupa2002
    
#     References
#     ----------
#     Kroupa, P. 2002. \'The initial mass function and its variation
#     (review)\'. *ASP conference series* 285 (January): 86.

#     Malkov, O., and H. Zinnecker. 2001. \'Binary Stars and the
#     Fundamental Initial Mass Function\'. *Monthly Notices of the Royal
#     Astronomical Society* 321 (1): 149--54.

#     %(example)s

#     """
#     def _pdf(self, x):
#         res = (
#             2.*mass.kroupa2002.pdf(x)*mass.kroupa2002.cdf(x)
#         )
        
#         return res

#     def _cdf(self, x):
#         res = mass.kroupa2002.cdf(x)**2.
                
#         return res

#     def _ppf(self, q):
#         res = mass.kroupa2002.ppf(np.sqrt(q))
        
#         return res
    

# kroupa2002 = kroupa2002_gen(
#     a=0.08, b=150., name="primary_mass.random.kroupa2002"
# )


# class salpeter1955_gen(_distn_infrastructure.rv_continuous):
#     r"""The primary-star mass random variable for random pairing

#     %(before_notes)s

#     Notes
#     -----
#     The probability density function for `random.kroupa2002` is:

#     .. math::
#        f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1}),

#     where :math:`f_{M}` and :math:`F_{M}` are the probability density
#     function and cumulative distribution function for the mass random
#     variable of Salpeter (1955).
    
#     %(after_notes)s

#     See also
#     --------
#     dyad.stats.mass.kroupa2002
    
#     References
#     ----------
#     Salpeter, Edwin E. 1955. \'The luminosity function and stellar
#     evolution.\' *The Astrophysical Journal* 121 (January): 161.

#     Malkov, O., and H. Zinnecker. 2001. \'Binary Stars and the
#     Fundamental Initial Mass Function\'. *Monthly Notices of the Royal
#     Astronomical Society* 321 (1): 149--54.

#     %(example)s

#     """
#     def _pdf(self, x):
#         res = (
#             2.*mass.salpeter1955.pdf(x)*mass.salpeter1955.cdf(x)
#         )
        
#         return res

#     def _cdf(self, x):
#         res = mass.salpeter1955.cdf(x)**2.
                
#         return res

#     def _ppf(self, q):
#         res = mass.salpeter1955.ppf(np.sqrt(q))
        
#         return res
    

# salpeter1955 = salpeter1955_gen(
#     a=0.08, b=150., name="primary_mass.random.salpeter1955"
# )


class primary_mass_gen(_distn_infrastructure.rv_continuous):
    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None,
                 name=None, longname=None, shapes=None, seed=None,
                 rv_mass=None):
        super().__init__(
            momtype=1, a=None, b=None, xtol=1e-14, badvalue=None,
            name=None, longname=None, shapes=None, seed=None
        )
        self._rv_mass = rv_mass
        
    def _pdf(self, x):
        res = (
            2.*self._rv_mass.pdf(x)*self._rv_mass.cdf(x)
        )
        
        return res

    def _cdf(self, x):
        res = self._rv_mass.cdf(x)**2.
                
        return res

    def _ppf(self, q):
        res = self._rv_mass.ppf(np.sqrt(q))
        
        return res
    

kroupa2002 = primary_mass_gen(
    a=0.08, b=150., rv_mass=mass.kroupa2002
)

salpeter1995 = primary_mass_gen(
    a=0.08, b=150., rv_mass=mass.salpeter1955
)


