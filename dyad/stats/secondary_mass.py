"""
=================================================
Secondary mass (:mod:`dyad.stats.secondary_mass`)
=================================================

.. currentmodule:: dyad.stats.secondary_mass

This module contains probability distributions for the secondary
masses of a population of binary stars. In its documentation the
variable is denoted :math:`M_{2}` and a realization of that variable
is denoted :math:`m_{2}`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   kroupa2001
   salpeter1955

"""

__all__ = [
    "uniform",
]

import numpy as np
import scipy as sp

from . import _distn_infrastructure
from . import secondary_mass_random as random


class uniform_gen(_distn_infrastructure.rv_continuous):
    r"""The secondary-star mass variable for uniform pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `uniform` is:

       f_{M_{2}|M_{1}}(m_{2}|m_{1}) = \dfrac{1}{m_{1}(1 - \max(q_{\min},
       m_{\min}/m_{1}))}

    where :math:`q_{\min} \in (0, 1)` is the minimum allowed mass
    ratio, :math:`m_{1} \in (0, \infty)` is the primary-star mass, and
    :math:`m_{\min} \in (0, \infty)` is the minimum allowed stellar
    mass such that :math:`m_{\min} < m_{1}`.

    ``uniform`` takes ``m_1`` as a shape parameter for :math:`m_{1}`,
    the primary mass, ``m_min`` as a shape parameter for
    :math:`m_{\text{min}}`, and ``q_min`` as a shape parameter for

    :math:`q_{\text{min}}`.
    
    %(after_notes)s

    See also
    --------
    dyad.stats.mass_ratio.uniform
    
    References
    ----------

    %(example)s

    """
    def _get_support(self, m_1, m_min, q_min):
        res = (np.maximum(q_min*m_1, m_min/m_1), m_1)
        
        return res
    
    def _pdf(self, x, m_1, m_min, q_min):
        num = np.ones_like(x)
        denom = m_1*(1. - np.maximum(q_min, m_min/m_1))
        res = num/denom
        
        return res

    def _cdf(self, x, m_1, m_min, q_min):
        num = x/m_1 - np.maximum(q_min, m_min/m_1)
        denom = 1. - np.maximum(q_min, m_min/m_1)
        res = num/denom
        
        return res

    def _ppf(self, q, m_1, m_min, q_min):
        res = (
            (1. - np.maximum(q_min, m_min/m_1))*m_1*q
            + np.maximum(q_min*m_1, m_min/m_1)
        )
        
        return res


uniform = uniform_gen(name="secondary_mass.uniform")
