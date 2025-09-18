"""
=====================================
Mass (:mod:`dyad.stats.primary_mass`)
=====================================

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

# __all__ = [
#     "uniform"
#     "from_functions",
# ]

import numpy as np
import scipy as sp

from . import _distn_infrastructure
from . import primary_mass_random as random

# rv_primary_mass = primary_mass(
#     dyad.stats.mass.kroupa2002,
#     dyad.stats.secondary_mass.random
# )

# # primary_mass.random.kroupa2002
# # primary_mass.random.salpeter1955
# # primary_mass.uniform.kroupa2002
# # primary_mass.uniform.salpeter1955

# class primary_mass_gen(_distn_infrastructure.rv_continuous):
#     pass


# def from_functions(
#         f, # MF
#         k, # PF
#         a, # 
#         b, # 
#         f_args=None, # arguments for MF
#         k_args=None,  # arguments for PF
#         name=None # name of random variable (e.g. "uniform.kroupa2002").
#         ):
#     r"""The primary-mass random variable
    
#     A factory method for constructing the primary-mass random variable
#     given a mass random variable and a condition secondary-mass random
#     variable.

#     Notes
#     -----

#     References
#     ----------

#     """
#     res = primary_mass_gen(a, b, name=name)

#     return res

