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


class kroupa2002_gen(_distn_infrastructure.rv_continuous):
    r"""The secondary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random` is:

    .. math::

    %(after_notes)s

    See also
    --------

    References
    ----------

    %(example)s

    """
    def _argcheck(self, primary_mass):
        return (0.08 <= primary_mass) & (primary_mass <= 150.)

    def _get_support(self, primary_mass):
        return (0.08, primary_mass)
    
    def _pdf(self, x, primary_mass):
        res = mass.kroupa2002.pdf(x)/mass.kroupa2002.cdf(primary_mass)
        
        return res

    def _cdf(self, x, primary_mass):
        res = mass.kroupa2002.cdf(x)/mass.kroupa2002.cdf(primary_mass)
                
        return res

    def _ppf(self, q, primary_mass):
        res = mass.kroupa2002.ppf(q*mass.kroupa2002.cdf(primary_mass))
        
        return res
    

kroupa2002 = kroupa2002_gen(a=0.08, b=150., name="random.kroupa2002")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""The secondary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random` is:

    .. math::

    %(after_notes)s

    See also
    --------

    References
    ----------

    %(example)s

    """
    def _argcheck(self, primary_mass):
        return (0.08 <= primary_mass) & (primary_mass <= 150.)

    def _get_support(self, primary_mass):
        return (0.08, primary_mass)
    
    def _pdf(self, x, primary_mass):
        res = mass.salpeter1955.pdf(x)/mass.salpeter1955.cdf(primary_mass)
        
        return res

    def _cdf(self, x, primary_mass):
        res = mass.salpeter1955.cdf(x)/mass.salpeter1955.cdf(primary_mass)
                
        return res

    def _ppf(self, q, primary_mass):
        res = mass.salpeter1955.ppf(q*mass.salpeter1955.cdf(primary_mass))
        
        return res
    

salpeter1955 = kroupa2002_gen(a=0.08, b=150., name="random.salpeter1955")
