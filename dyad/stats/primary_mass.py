"""
=============================================
Primary mass (:mod:`dyad.stats.primary_mass`)
=============================================

.. currentmodule:: dyad.stats.primary_mass

This module contains probability distributions for the primary-star
masses of a population of binary-star systems. In its documentation
the random variable is denoted :math:`M_{1}` and a realization of that
random variable is denoted :math:`m_{1}`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   primary_mass

"""

import numpy as np
import scipy as sp

from . import _distn_infrastructure
from . import primary_mass_random as random
