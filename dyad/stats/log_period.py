"""
=========================================
Log-period (:mod:`dyad.stats.log_period`)
=========================================

.. currentmodule:: dyad.stats.log_period

This module contains probability distributions for the orbital
log-periods of a population of binary stars. In its documentation the
random variable is denoted :math:`X` and a realization of that random
variable is denoted :math:`x`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   duquennoy1991
   moe2017

"""

__all__ = [
    "duquennoy1991",
    "moe2017",
]

import os
import numpy as np
import scipy as sp

from importlib.resources import files
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from dyad.stats import mass_ratio
from . import _distn_infrastructure

_truncnorm = sp.stats.truncnorm


class duquennoy1991_gen(_distn_infrastructure.rv_continuous):
    r"""The log-period random variable of Duquennoy and Mayor (1991)

    %(before_notes)s

    Notes
    -----
    The probability density function for `duquennoy1991` is:

    .. math::
       f_{X}(x) = \dfrac{\phi(\log(x); \mu, \sigma^{2})}{\Phi(\log(b);
       \mu, \sigma^{2}) - \Phi(\log(a); \mu, \sigma^{2})}

    for log-period :math:`x \in [a, b]`, :math:`a = -2.3`, and
    :math:`b = 12`, where :math:`\phi(\cdot; \mu, \sigma^{2})` and
    :math:`\Phi(\cdot; \mu, \sigma^{2})` are the probability density
    function and the cumulative distribution function for a Gaussian
    random variable with mean :math:`\mu` and variance
    :math:`\sigma^{2}` and where :math:`\mu = 4.8` and :math:`\sigma =
    2.3`.

    %(after_notes)s

    See also
    --------
    dyad.stats.period.moe2017

    References
    ----------
    Duquennoy, A., and M. Mayor. 1991. \'Multiplicity among
    solar-type stars in the solar neighbourhood---II. Distribution of
    the orbital elements in an unbiased Sample\'. *Astronomy and
    Astrophysics* 248 (August): 485.

    %(example)s

    """
    def _pdf(self, x):
        res = _duquennoy1991.pdf(x)

        return res

    def _cdf(self, x):
        res = _duquennoy1991.cdf(x)

        return res

    def _ppf(self, x):
        res = _duquennoy1991.ppf(x)

        return res


_duquennoy1991_loc = 4.8
_duquennoy1991_scale = 2.3
_duquennoy1991_a = (-2.3 - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991_b = (12. - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991 = _truncnorm(
    a=_duquennoy1991_a, b=_duquennoy1991_b, loc=_duquennoy1991_loc,
    scale=_duquennoy1991_scale
)
duquennoy1991 = duquennoy1991_gen(
    a=-2.3, b=12., name="log_period.duquennoy1991"
)


class moe2017_gen(_distn_infrastructure.rv_continuous):
    r"""The log-period random variable of Moe and Stefano (2017)

    %(before_notes)s

    Notes
    -----
    The probability density function for `moe2017` is:

    .. math::
       f_{X|M_{1}}(x|m_{1})
       &= \dfrac{A_{X}(m_{1})}{1 - F_{Q|P, m_{1}}(0.3|10^{x}, m_{1})}
       \begin{cases}
       c_{1}(m_{1})
       &\text{if $x \in [0.2, 1]$}\\
       c_{2}(m_{1})x + c_{3}(m_{1})
       &\text{if $x \in (1, 2]$}\\
       c_{4}(m_{1})x + c_{5}(m_{1})
       &\text{if $x \in (2, 3.4]$}\\
       c_{6}(m_{1})x + c_{7}(m_{1})
       &\text{if $x \in (3.4, 5.5]$}\\
       c_{8}(m_{1})\exp(-0.3x)
       &\text{if $x \in (5.5, 8]$}
       \end{cases}

    for log-period :math:`x \in [0.2, 8]` and primary mass
    :math:`m_{1} \in (0, \infty)` where the normalization constant,
    :math:`A_{X}(m_{1})`, is such that

    .. math::
       \int_{0.2}^{8}f_{X|m_{1}}(x|m_{1})\mathrm{d}\,X = 1

    where :math:`F_{Q|P, m_{1}}` is the cumulative distribution
    function for the mass ratio
    (:func:`dyad.stats.mass_ratio.moe2017`) and where

    .. math::
       c_{1}(m_{1})
       &= 0.07\log_{10}(m_{1})^{2} + 0.04\log_{10}(m_{1}) + 0.02,\\
       c_{2}(m_{1})
       &= -0.06\log_{10}(m_{1})^{2} + 0.03\log_{10}(m_{1}) + 0.0064\\
       c_{3}(m_{1})
       &= 0.13\log_{10}(m_{1})^{2} + 0.01\log_{10}(m_{1}) + 0.0136\\
       c_{4}(m_{1})
       &= 0.018\\
       c_{5}(m_{1})
       &= 0.01\log_{10}(m_{1})^{2} + 0.07\log_{10}(m_{1}) - 0.0096\\
       c_{6}(m_{1})
       &= \dfrac{0.03}{2.1}\log_{10}(m_{1})^{2} -
       \dfrac{0.12}{2.1}\log_{10}(m_{1}) - \dfrac{0.0264}{2.1}\\
       c_{7}(m_{1})
       &= - \dfrac{0.081}{2.1}\log_{10}(m_{1})^{2} +
       \dfrac{0.555}{2.1}\log_{10}(m_{1}) + \dfrac{0.0186}{2.1}\\
       c_{8}(m_{1})
       &= \exp(1.65)\left(0.04\log_{10}(m_{1})^{2} -
       0.05\log_{10}(m_{1}) + 0.078\right).

    %(after_notes)s

    See also
    --------
    dyad.stats.period.moe2017

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
        return (0. <= primary_mass) & (primary_mass < np.inf)

    def _pdf(self, x, primary_mass):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _moe2017_pdf_interp((x, primary_mass))
        
        return res

    def _cdf(self, x, primary_mass):
        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        res = _moe2017_cdf_interp((x, primary_mass))
        
        return res

    def _ppf(self, q, primary_mass):
        q = np.asarray(q)
        primary_mass = np.asarray(primary_mass)        
        res = _moe2017_ppf_interp((q, primary_mass))
        
        return res

# For guidance on the use of data files see:
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html
# (section `Accessing Data Files at Runtime')
path = "dyad.stats.data.log_period"
with files(path).joinpath("log10_period_sample.dat") as f_name:
    _moe2017_log10_period_sample = np.loadtxt(f_name)
with files(path).joinpath("primary_mass_sample.dat") as f_name:
    _moe2017_primary_mass_sample = np.loadtxt(f_name)
with files(path).joinpath("frequency_sample.dat") as f_name:
    _moe2017_frequency_sample = np.loadtxt(f_name)
with files(path).joinpath("cumulative_frequency_sample.dat") as f_name:
    _moe2017_cumulative_frequency_sample = np.loadtxt(f_name)
    
_moe2017_pdf_interp = RegularGridInterpolator(
    (_moe2017_log10_period_sample, _moe2017_primary_mass_sample),
    _moe2017_frequency_sample.T
)
_moe2017_cdf_interp = RegularGridInterpolator(
    (_moe2017_log10_period_sample, _moe2017_primary_mass_sample),
    _moe2017_cumulative_frequency_sample.T
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
    _moe2017_primary_mass_sample, (_moe2017_log10_period_sample.size, 1)
)
_moe2017_points = np.vstack([_moe2017_xx.ravel(), _moe2017_yy.T.ravel()])
_moe2017_values = np.tile(
    _moe2017_log10_period_sample[::-1], _moe2017_primary_mass_sample.size
)
_moe2017_ppf_interp = LinearNDInterpolator(_moe2017_points.T, _moe2017_values)

moe2017 = moe2017_gen(a=0.2, b=8., name="log_period.moe2017")
