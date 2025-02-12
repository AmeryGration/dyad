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

   moe2017
   random

"""

__all__ = [
    "moe2017_kroupa2002",
    "random_kroupa2002",
]

import numpy as np
import scipy as sp

from . import _distn_infrastructure

# class primary_mass():
#     def __init__(self, secondary_mass_function, mass_function):
#         pass


class moe2017_kroupa2002_gen(_distn_infrastructure.rv_continuous):
    def _pdf(self, x):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _moe2017_pdf_interp((x/primary_mass, primary_mass))/primary_mass
        
        return res

    def _cdf(self, x, primary_mass):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _moe2017_cdf_interp((x/primary_mass, primary_mass))/primary_mass
        
        return res

    def _ppf(self, q, primary_mass):
        q = np.asarray(q)
        primary_mass = np.asarray(primary_mass)

        res = _moe2017_ppf_interp((q, primary_mass))*primary_mass
        
        return res


# For guidance on the use of data files see:
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html
# (section `Accessing Data Files at Runtime')
path = "dyad.stats.data.moe2017.primary_mass.kroupa2002"
path = "dyad.stats.data.primary_mass.moe2017.kroupa2002"
with files(path).joinpath("primary_mass_sample.dat") as f_name:
    _moe2017_primary_mass_sample = np.loadtxt(f_name)
with files(path).joinpath("frequency_sample.dat") as f_name:
    _moe2017_frequency_sample = np.loadtxt(f_name)
with files(path).joinpath("cumulative_frequency_sample.dat") as f_name:
    _moe2017_cumulative_frequency_sample = np.loadtxt(f_name)
    
_moe2017_pdf_interp = RegularGridInterpolator(
    _moe2017_primary_mass_sample,
    _moe2017_frequency_sample.T,
    bounds_error=False,
    fill_value=0.
)
_moe2017_cdf_interp = RegularGridInterpolator(
    _moe2017_primary_mass_sample,
    _moe2017_cumulative_frequency_sample.T,
    bounds_error=False,
    fill_value=0.
)

# Suppose that we have an invertible function, :math:`f`, of some
# variable, :math:`t`, represented by arrays `f` and `t`. We can
# interpolate between function values as follows.
# >>> interp = RegularGridInterpolator((t,), f)
# To interpolate between equivalent points of the inverse function,
# :math:`f^{-1}`, of some variable, :math:`q \in [0, 1]`, we may
# reverse the arguments.
# >>> interp_inv = RegularGridInterpolator(f[::-1], (t[::-1],))
# Note that in doing this we are not sampling :math:`q` uniformly on
# :math:`[0, 1]`. Instead we may take advantage of points of interest
# in :math:`x := log(P)`
# To find the inverse of the conditional cumulative distribution
# function we may extend this trick to two dimensions. In this
# two-dimensional case the values `f` are no longer regularly spaced
# so we must use a interpolator that accepts irregularly space data,
# such as Scipy's ~LinearNDInterpolator~.

# See https://kitchingroup.cheme.cmu.edu/blog/category/interpolation/.
_moe2017_ppf_interp = LinearNDInterpolator(
    _moe2017_cumulative_frequency_sample[::-1],
    _moe2017_primary_mass_sample[::-1]
)

moe2017 = moe2017_gen(a=0.08, b=60., name="primary_mass.moe2017")
