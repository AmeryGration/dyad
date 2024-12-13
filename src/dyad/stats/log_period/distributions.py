"""Distributions

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
# from scipy.interpolate import interp2d
from dyad.stats import mass_ratio

_truncnorm = sp.stats.truncnorm


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The log-period random variable of Duquennoy and Mayor (1991)

    %(before_notes)s

    Notes
    -----
    The probability density function for `duquennoy1991` is:

    .. math::

        f(x) = 

    where

    .. math::

        A := 

    :math:`x > 0` [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Duquennoy, A., and M. Mayor. 1991. `Multiplicity among
    solar-type stars in the solar neighbourhood---II. Distribution of
    the orbital elements in an unbiased Sample'. /Astronomy and
    Astrophysics/ 248 (August): 485.

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
duquennoy1991 = _duquennoy1991_gen(a=-2.3, b=12., name="duquennoy1991")


class _moe2017_gen(sp.stats.rv_continuous):
    r"""The log-period random variable of Moe and Stefano (2017)

    %(before_notes)s

    Notes
    -----
    The probability density function for `moe2017` is:

    .. math::

        f(x) =

    where

    .. math::

        A :=

    :math:`x > 0` [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Moe, Maxwell, and Rosanne Di Stefano. 2017. `Mind your Ps and Qs:
    the interrelation between period (P) and mass-ratio (Q) distributions of
    binary stars.' /The Astrophysical Journal Supplement Series/ 230 (2): 15.

    %(example)s

    """
    def _argcheck(self, primary_mass):
        return (0. <= primary_mass) & (primary_mass < np.inf)

    def _pdf(self, x, primary_mass):
        x = np.asarray(x)

        # res = _moe2017_pdf_interp((x, primary_mass))
        res = _moe2017_pdf_interp(x, primary_mass)
        
        return res

    def _cdf(self, x, primary_mass):
        x = np.asarray(x)

        # res = _moe2017_cdf_interp((x, primary_mass))
        res = _moe2017_cdf_interp(x, primary_mass)
        
        return res

    def _ppf(self, q, primary_mass):
        q = np.asarray(q)
        
        # res = _moe2017_ppf_interp((q, primary_mass))
        res = _moe2017_ppf_interp(q, primary_mass)
        
        return res

# For guidance on the use of data files see:
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html
# (section `Accessing Data Files at Runtime')
with files("dyad.data").joinpath("log10_period_sample.dat") as f_name:
    _moe2017_log10_period_sample = np.loadtxt(f_name)
with files("dyad.data").joinpath("primary_mass_sample.dat") as f_name:
    _moe2017_primary_mass_sample = np.loadtxt(f_name)
with files("dyad.data").joinpath("frequency_sample.dat") as f_name:
    _moe2017_frequency_sample = np.loadtxt(f_name)
with files("dyad.data").joinpath("cumulative_frequency_sample.dat") as f_name:
    _moe2017_cumulative_frequency_sample = np.loadtxt(f_name)

_moe2017_pdf_interp = RegularGridInterpolator(
    (_moe2017_log10_period_sample, _moe2017_primary_mass_sample),
    _moe2017_frequency_sample.T
)
_moe2017_cdf_interp = RegularGridInterpolator(
    (_moe2017_log10_period_sample, _moe2017_primary_mass_sample),
    _moe2017_cumulative_frequency_sample.T
)
# Suppose that we have an invertible function, :math:`f`, of some variable, :math:`t`, represented by arrays `f` and `t`. We can interpolate between function values as follows.
# >>> interp = RegularGridInterpolator((t,), f)
# To interpolate between equivalent points of the inverse function, :math:`f^{-1}`, of some variable, :math:`q \in [0, 1]`, we may reverse the arguments.
# >>> interp_inv = RegularGridInterpolator(f[::-1], (t[::-1],))
# Note that in doing this we are not sampling :math:`q` uniformly on :math:`[0, 1]`. Instead we may take advantage of points of interest in :math:`x := log(P)`
# To find the inverse of the conditional cumulative distribution function we may extend this trick to two dimensions. In this two-dimensional case the values `f` are no longer regularly spaced so we must use a interpolator that accepts irregularly space data, such as Scipy's ~LinearNDInterpolator~.
# See https://kitchingroup.cheme.cmu.edu/blog/category/interpolation/.
_xx = _moe2017_cumulative_frequency_sample[:,::-1]
_yy = np.tile(
    _moe2017_primary_mass_sample, (_moe2017_log10_period_sample.size, 1)
)
_points = np.vstack([_xx.ravel(), _yy.T.ravel()])
_values = np.tile(
    _moe2017_log10_period_sample[::-1], _moe2017_primary_mass_sample.size
)
_moe2017_ppf_interp = LinearNDInterpolator(_points.T, _values)
# _moe2017_ppf_interp = interp2d(
#     _moe2017_log10_period_sample, _moe2017_primary_mass_sample,
#     _moe2017_cumulative_frequency_sample
# )

moe2017 = _moe2017_gen(a=0.2, b=8., name="moe2017")

# class _moe2017_gen(sp.stats.rv_continuous):
#     r"""The Moe and Stefano (2017) log-period random variable

#     %(before_notes)s

#     Notes
#     -----
#     The probability density function for `moe1991` is:

#     .. math::

#         f(x) =

#     where

#     .. math::

#         A :=

#     :math:`x > 0` [1]_.

#     %(after_notes)s

#     References
#     ----------
#     .. [1] Reference

#     %(example)s

#     """
#     def _argcheck(self, primary_mass):
#         return (0. <= primary_mass) & (primary_mass < np.inf)

#     def _pdf(self, x, primary_mass):
#         def f_1(x, log10_primary_mass):
#             res = _moe2017_c_1(log10_primary_mass)

#             return res

#         def f_2(x, log10_primary_mass):
#             res = (
#                 _moe2017_c_2(log10_primary_mass)*x
#                 + _moe2017_c_3(log10_primary_mass)
#             )

#             return res

#         def f_3(x, log10_primary_mass):
#             res = (
#                 _moe2017_c_4(log10_primary_mass)*x
#                 + _moe2017_c_5(log10_primary_mass)
#             )

#             return res

#         def f_4(x, log10_primary_mass):
#             res = (
#                 _moe2017_c_6(log10_primary_mass)*x
#                 + _moe2017_c_7(log10_primary_mass)
#             )

#             return res

#         def f_5(x, log10_primary_mass):
#             res = (
#                 _moe2017_c_8(log10_primary_mass)
#                 *np.exp(-0.3*x)
#             )

#             return res

#         x = np.asarray(x)
#         primary_mass = np.asarray(primary_mass)
#         log10_primary_mass = np.log10(primary_mass)

#         rv_mass_ratio = mass_ratio.moe2017(x, 10.**log10_primary_mass)
#         correction_factor = 1./(1. - rv_mass_ratio.cdf(0.3))

#         condition = [
#             (0.2 <= x) & (x <= 1.),
#             (1. < x) & (x <= 2.),
#             (2. < x) & (x <= 3.4),
#             (3.4 < x) & (x <= 5.5),
#             (5.5 < x) & (x <= 8.),
#         ]
#         value = [
#             f_1(x, log10_primary_mass),
#             f_2(x, log10_primary_mass),
#             f_3(x, log10_primary_mass),
#             f_4(x, log10_primary_mass),
#             f_5(x, log10_primary_mass),
#         ]
#         res = (
#             correction_factor
#             *np.select(condition, value)
#             /_moe2017_cumulative_frequency((8., primary_mass)).T
#         )

#         return res

#     def _cdf(self, x, primary_mass):
#         x = np.asarray(x)
#         primary_mass = np.asarray(primary_mass)
#         # x = np.unique(x)
#         # primary_mass = np.unique(primary_mass)

#         res = (
#             _moe2017_cumulative_frequency((x, primary_mass))
#             /_moe2017_cumulative_frequency((8., primary_mass))
#         )            

#         return res

#     # def _ppf(self, q, primary_mass):
#     #     res = 0.

#     #     return res


# def _moe2017_c_1(log10_primary_mass):
#     res = (
#         0.07*log10_primary_mass**2.
#         + 0.04*log10_primary_mass
#         + 0.020
#     )

#     return res

# def _moe2017_c_2(log10_primary_mass):
#     res = (
#         - 0.06*log10_primary_mass**2.
#         + 0.03*log10_primary_mass
#         + 0.0064
#     )

#     return res

# def _moe2017_c_3(log10_primary_mass):
#     res = (
#         0.13*log10_primary_mass**2.
#         + 0.01*log10_primary_mass
#         + 0.0136
#     )

#     return res

# def _moe2017_c_4(log10_primary_mass):
#     res = 0.018

#     return res

# def _moe2017_c_5(log10_primary_mass):
#     res = (
#         0.01*log10_primary_mass**2.
#         + 0.07*log10_primary_mass
#         - 0.0096
#     )

#     return res

# def _moe2017_c_6(log10_primary_mass):
#     res = (
#         0.03*log10_primary_mass**2./2.1
#         - 0.12*log10_primary_mass/2.1
#         + 0.0264/2.1
#     )

#     return res

# def _moe2017_c_7(log10_primary_mass):
#     res = (
#         - 0.081*log10_primary_mass**2./2.1
#         + 0.555*log10_primary_mass/2.1
#         + 0.0186/2.1
#     )

#     return res

# def _moe2017_c_8(log10_primary_mass):
#     res = (
#         np.exp(1.65)
#         *(
#             0.04*log10_primary_mass**2.
#             - 0.05*log10_primary_mass
#             + 0.078
#         )
#     )

#     return res

# moe2017 = _moe2017_gen(a=0.2, b=8., name="moe2017")
