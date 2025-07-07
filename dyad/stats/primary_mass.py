"""
=============================================
Primary mass (:mod:`dyad.stats.primary_mass`)
=============================================

.. currentmodule:: dyad.stats.primary_mass

This module contains probability distributions for the primary
masses of a population of binary stars. In its documentation the
random variable is denoted :math:`M_{1}` and a realization of that
random variable is denoted :math:`m_{1}`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   random

"""

__all__ = [
    "random",
    "moe2017",
]

import os
import numpy as np
import scipy as sp

from importlib.resources import files
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from . import _distn_infrastructure
from . import mass

class random_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

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
    def _pdf(self, x):
        alpha = 2.3
        x = np.asarray(x)
        res = 2.*_rv_mass.pdf(x)*_rv_mass.cdf(x)
        
        return res

    def _cdf(self, x):
        alpha = 2.3
        x = np.asarray(x)
        A = (1. - alpha)/(0.1**(1. - alpha) - 100.**(1. - alpha))
        B = 2.*A**2./(1. - alpha)**2.
        res = B*(
            x**(2.*(1. - alpha))/2.
            - 0.1**(1. - alpha)*x**(1. - alpha)
            + 0.1**(2.*(1. - alpha))/2.
        )
                
        return res


_rv_mass = sp.stats.truncpareto(1.3, (100. - 0.)/0.1, scale=0.1)
random = random_gen(a=0.1, b=100., name="secondary.mass.random")
