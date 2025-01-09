"""
=================================================
Semimajor-axis (:mod:`dyad.stats.semimajor_axis`)
=================================================

.. currentmodule:: dyad.stats.semimajor_axis

This module contains probability distributions for the orbital semimajor axes of a population of binary stars. In its documentation the random variable is denoted :math:`A` and a realization of that random variable is denoted :math:`a`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   opik1924

"""

__all__ = [
    "opik1924"
]

import numpy as np
import scipy as sp

from . import _distn_infrastructure

# class opik1924_gen(sp.stats._continuous_distns.reciprocal_gen):
#     r"""The semimajor-axis random variable of Öpik (1924)

#     %(before_notes)s

#     Notes
#     -----
#     The probability density function for `opik1924` is:

#     .. math::
#        f_{A}(a; b, c) = \dfrac{1}{a\log_{10}(c/b)}

#     for :math:`a \in [b, c]`, :math:`b, c \in (0, \infty)`, and
#     :math:`b < c`. The probability density function `opik1924` is
#     identical to `scipy.stats.reciprocal`.

#     %(after_notes)s

#     References
#     ----------
#     Öpik, E. 1924. \'Statistical studies of double stars: on the
#     distribution of relative luminosities and distances of double
#     stars in the Harvard Revised Photometry North of
#     Declination---31°\'. *Publications of the Tartu Astrofizica
#     Observatory* 25 (January):1.
    
#     %(example)s

#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(args, kwargs)

# opik1924 = opik1924_gen(a=0., b=1., name="powerlaw")

opik1924 = sp.stats._continuous_distns.powerlaw_gen(
    a=0., b=1., name="powerlaw"
)
