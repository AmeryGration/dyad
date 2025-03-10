"""
=================================================
Secondary mass (:mod:`dyad.stats.secondary_mass`)
=================================================

.. currentmodule:: dyad.stats.secondary_mass

This module contains probability distributions for the secondary
masses of a population of binary stars. In its documentation the
random variable is denoted :math:`M_{2}` and a realization of that
random variable is denoted :math:`m_{2}`. The random variable for
secondary mass is conditional on primary mass. The PDF for secondary
mass is known as the `pairing function'

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   moe2017

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


class moe2017_gen(_distn_infrastructure.rv_continuous):
    r"""The secondary-star mass random variable of Moe and Stefano (2017)

    %(before_notes)s

    Notes
    -----
    The probability density function for `moe2017` is:

    .. math::
       f_{M_{2}|M_{1} = m_{1}}(m_{2}|m_{1}) = \dfrac{1}{m_{1}}f_{Q}(m_{2}/m_{1}).

    where

    .. math::
       f_{Q|M_{1} = m_{1}}(q|m_{1}) = \int_{p_{\min}}^{p_{\max}}f_{(Q, P)|M_{1} = m_{1}}(q, p|m_{1})\mathrm{d}\,p

    and, by the chain rule for probability,

    .. math::
       f_{(Q, P)|M_{1} = m_{1}}(q, p|m_{1}) = f_{Q|(P, M_{1}) = (p, m_{1})}(q|p, m_{1})f_{P|M_{1} = m_{1}}(p|m_{1}).

    for primary-star mass :math:`m_{1} \in (m_{1}, 60)`,
    secondary-star mass :math:`m_{2} \in (0.1m_{1}, 60)`, mass ratio
    :math:`q \in [0.1, 1]`, and period :math:`p \in [10^{0.2}, 10^{8}]`. The
    functions :math:`f_{Q|(P, M_{1}) = (p, m_{1})}` and
    :math:`f_{P|M_{1} = m_{1}}` are the probability density functions
    for random variables :class:`dyad.stats.mass_ratio.moe2017` and
    :class:`dyad.stats.period.moe2017`.

    ``moe2017`` takes ``primary_mass`` as a shape parameter for
    :math:`m_{1}`, the primary mass.

    %(after_notes)s

    See also
    --------

    References
    ----------
    Moe, Maxwell, and Rosanne Di Stefano. 2017. \'Mind your Ps and Qs:
    the interrelation between period (P) and mass-ratio (Q)
    distributions of binary stars.\' *The Astrophysical Journal
    Supplement Series* 230 (2): 15.

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("primary_mass", False, (0, np.inf), (False, False))]

    def _argcheck(self, primary_mass):
        return (0.8 <= primary_mass) & (primary_mass <= 60.)
        # return (0. <= primary_mass) & (primary_mass < np.inf)
    
    def _pdf(self, x, primary_mass):
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
path = "dyad.stats.data.secondary_mass"
with files(path).joinpath("mass_ratio_sample.dat") as f_name:
    _moe2017_mass_ratio_sample = np.loadtxt(f_name)
with files(path).joinpath("primary_mass_sample.dat") as f_name:
    _moe2017_primary_mass_sample = np.loadtxt(f_name)
with files(path).joinpath("frequency_sample.dat") as f_name:
    _moe2017_frequency_sample = np.loadtxt(f_name)
with files(path).joinpath("cumulative_frequency_sample.dat") as f_name:
    _moe2017_cumulative_frequency_sample = np.loadtxt(f_name)
    
_moe2017_pdf_interp = RegularGridInterpolator(
    (_moe2017_mass_ratio_sample, _moe2017_primary_mass_sample),
    _moe2017_frequency_sample.T,
    bounds_error=False,
    fill_value=0.
)
_moe2017_cdf_interp = RegularGridInterpolator(
    (_moe2017_mass_ratio_sample, _moe2017_primary_mass_sample),
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
_moe2017_xx = _moe2017_cumulative_frequency_sample[:,::-1]
_moe2017_yy = np.tile(
    _moe2017_primary_mass_sample, (_moe2017_mass_ratio_sample.size, 1)
)
_moe2017_points = np.vstack([_moe2017_xx.ravel(), _moe2017_yy.T.ravel()])
_moe2017_values = np.tile(
    _moe2017_mass_ratio_sample[::-1], _moe2017_primary_mass_sample.size
)
_moe2017_ppf_interp = LinearNDInterpolator(_moe2017_points.T, _moe2017_values)

moe2017 = moe2017_gen(a=0.1, b=60., name="secondary_mass.moe2017")


class random_gen(_distn_infrastructure.rv_continuous):
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
    # def _shape_info(self):
    #     return [
    #         _ShapeInfo("primary_mass", False, (0.1, 60.), (False, False))
    #     ]

    def _argcheck(self, primary_mass):
        return (0.1 <= primary_mass) & (primary_mass <= 60.)

    def _get_support(self, primary_mass):
        return (0.1, primary_mass)
    
    def _pdf(self, x, primary_mass):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _kroupa2002.pdf(x)/_kroupa2002.cdf(primary_mass)
        
        return res

    def _cdf(self, x, primary_mass):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _kroupa2002.cdf(x)/_kroupa2002.cdf(primary_mass)
                
        return res

    def _ppf(self, q, primary_mass):
        q = np.asarray(q)
        primary_mass = np.asarray(primary_mass)
        res = _kroupa2002.ppf(q*_kroupa2002.cdf(primary_mass))
        
        return res
    

_kroupa2002 = mass.kroupa2002

random = random_gen(a=0.1, b=60., name="secondary.mass.random")
