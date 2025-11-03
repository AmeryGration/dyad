"""
=================================
Period (:mod:`dyad.stats.period`)
=================================

.. currentmodule:: dyad.stats.period

This module contains probability distributions for the orbital periods
of a population of binary stars. In its documentation the random
variable is denoted :math:`P` and a realization of that random
variable is denoted :math:`p`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   trunclognorm
   duquennoy1991
   moe2017

"""

__all__ = [
    "trunclognorm",
    "duquennoy1991",
    "moe2017",
]

import numpy as np
import scipy as sp

from importlib.resources import files
from scipy._lib._util import _lazywhere
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from dyad.stats import mass_ratio
from . import _distn_infrastructure

def _lognorm_logpdf(x, s):
    # Replica of ~scipy.stats._continuous_distns.cd._lognorm_logpdf~,
    # which is called by ~scipy.stats.lognorm~.
    return _lazywhere(
        x != 0,
        (x, s),
        lambda x, s: -np.log(x)**2./(2.*s**2.) - np.log(s*x*np.sqrt(2*np.pi)),
        -np.inf
    )


class trunclognorm_gen(_distn_infrastructure.rv_continuous):
    r"""A truncated lognormal random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `trunclognorm` is:

    .. math::
       f_{P}(p, a, b, s) =
       \dfrac{\exp\left(-\dfrac{\log(p)^{2}}{2s^{2}}\right)}
       {\sqrt{2\pi}sx\left(\operatorname{erf}\left(\dfrac{\log(b)}{s}\right)
       - \operatorname{erf}\left(\dfrac{\log(a)}{s}\right)\right)}

    where :math:`p \in (a, b)`, :math:`a, b \in (0, \infty)`, and
    :math:`s \in (0, \infty)`.

    The probability density function `trunclognorm` takes ``a``,
    ``b``, and ``s`` as shape parameters for $a$, $b$, and $s$.

    Notice that the truncation values, $a$ and $b$, are defined in
    standardized form:

    .. math::
       a = (u_{\mathrm{l}} - \mathrm{loc})/\mathrm{scale},
       b = (u_{\mathrm{r}} - \mathrm{loc})/\mathrm{scale}

    where :math:`u_{\mathrm{l}}` and :math:`u_{\text{r}}` are the
    specific left and right truncation values, respectively. In other
    words, the support of the distribution becomes
    :math:`(a*\mathrm{scale} + \mathrm{loc}) < x <= (b*\mathrm{scale}
    + \mathrm{loc})` when :math:`\mathrm{loc}` and/or
    :math:`\mathrm{scale}` are provided.

    Suppose a normally distributed random variable ``X`` has mean
    ``mu`` and standard deviation ``sigma``. Then ``Y = exp(X)`` is
    lognormally distributed with ``s = sigma`` and ``scale =
    exp(mu)``. To change the base of the lognormal distribution from
    ``e`` to base ``b`` multiply ``mu`` and ``sigma`` by ``ln(b)``.
    
    %(after_notes)s

    References
    ----------

    \'Truncated Normal Distribution\'. 2024. In /Wikipedia/. https://en.wikipedia.org/w/index.php?title=Truncated_normal_distribution&oldid=1217498083.

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("s", False, (0., np.inf), (False, False))]

    def _argcheck(self, s, a, b):
        return (a >= 0.) & (b > a) & (s > 0.)

    def _shape_info(self):
        is_ = _ShapeInfo("s", False, (0, np.inf), (False, False))
        ia = _ShapeInfo("a", False, (0, np.inf), (True, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))

        return [is_, ia, ib]

    # def _fitstart(self, data):
    #     pass
    #     # # Arbitrary, but default a=b=c=1 is not valid
    #     # return super()._fitstart(data, args=(1, 0, 1)) # ???

    def _get_support(self, s, a, b):
        return a, b

    def _pdf(self, x, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)

        return np.exp(_lognorm_logpdf(x, s))/A

    def _cdf(self, x, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)

        return (sp.special.ndtr(np.log(x)/s) - sp.special.ndtr(np.log(a)/s))/A

    def _ppf(self, q, s, a, b):
        A = sp.special.ndtr(np.log(b)/s) - sp.special.ndtr(np.log(a)/s)

        return np.exp(
            s*sp.stats.norm.ppf(A*q + sp.special.ndtr(np.log(a)/s))
        )


trunclognorm = trunclognorm_gen(name="period.trunclognorm")


class duquennoy1991_gen(_distn_infrastructure.rv_continuous):
    r"""The period random variable of Duquennoy and Mayor (1991)

    %(before_notes)s

    Notes
    -----
    The probability density function for `duquennoy1991` is:

    .. math::
       f_{P}(p) = \dfrac{\exp\left(-\dfrac{(\log(p) -
       \mu)^{2}}{2\sigma^{2}}\right)}
       {\sqrt{2\pi}\sigma{}p\left(\operatorname{erf}\left(\dfrac{\log(b)
       - \mu}{\sigma}\right) - \operatorname{erf}\left(\dfrac{\log(a)
       - \mu}{\sigma}\right)\right)}

    for :math:`p \in (a, b]`, :math:`a = 10^{-2.3}`, :math:`b =
    10^{12}`, :math:`\mu = 4.8`, and :math:`\sigma = 2.3`. It is
    implemented as an instance of `trunclognorm`.
    
    %(after_notes)s

    See also
    --------
    dyad.stats.log_period.moe2017

    References
    ----------
    Duquennoy, A., and M. Mayor. 1991. \'Multiplicity among
    solar-type stars in the solar neighbourhood---II. Distribution of
    the orbital elements in an unbiased Sample\'. *Astronomy and
    Astrophysics* 248 (August): 485.

    %(example)s

    """
    # Check 0 < a < b.
    def _pdf(self, x):
        return _duquennoy1991.pdf(x)

    def _cdf(self, x):
        return _duquennoy1991.cdf(x)

    def _ppf(self, q):
        return _duquennoy1991.ppf(q)


# Duquennoy and Mayor (1991) period: truncated lognormal
_duquennoy1991_mu = np.exp(np.log(10.)*4.8)
_duquennoy1991_sigma = np.log(10.)*2.3
_duquennoy1991_loc = 0.
_duquennoy1991_scale = _duquennoy1991_mu
_duquennoy1991_s = _duquennoy1991_sigma
_duquennoy1991_a = (10.**-2. - 0.)/_duquennoy1991_scale
_duquennoy1991_b = (10.**12. - 0.)/_duquennoy1991_scale
_duquennoy1991 = trunclognorm(
    s=_duquennoy1991_s, a=_duquennoy1991_a, b=_duquennoy1991_b,
    scale=_duquennoy1991_scale
)
duquennoy1991 = duquennoy1991_gen(
    a=10.**-2.3, b=10.**12., name="period.duquennoy1991"
)


class moe2017_gen(_distn_infrastructure.rv_continuous):
    r"""The period random variable of Moe and Stefano (2017)

    %(before_notes)s

    Notes
    -----
    The probability density function for `moe2017` is:

    .. math::
       f_{P|M_{1}}(p|m_{1}) =
       \left|\dfrac{1}{\ln(10)p}\right|f_{X|M_{1}}(\log_{10}(p)|m_{1}).

    for :math:`p \in [10^{0.2}, 10^{8}]` and where :math:`f_{X|M_{1}}` is the
    probability density function for log-period given by Moe and
    Stefano (2017).

    See also
    --------
    dyad.stats.log_period.moe2017

    References
    ----------
    Moe, Maxwell, and Rosanne Di Stefano. 2017. \'Mind your Ps and Qs:
    the interrelation between period (P) and mass-ratio (Q)
    distributions of binary stars.\' *The Astrophysical Journal
    Supplement Series* 230 (2): 15.

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("p", False, (0.8, 40.), (True, False))]

    def _argcheck(self, primary_mass):
        return (0.8 <= primary_mass) & (primary_mass < 40.)

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
path = "dyad.stats.data.period"
with files(path).joinpath("period_sample.dat") as f_name:
    _moe2017_period_sample = np.loadtxt(f_name)
with files(path).joinpath("primary_mass_sample.dat") as f_name:
    _moe2017_primary_mass_sample = np.loadtxt(f_name)
with files(path).joinpath("frequency_sample.dat") as f_name:
    _moe2017_frequency_sample = np.loadtxt(f_name)
with files(path).joinpath("cumulative_frequency_sample.dat") as f_name:
    _moe2017_cumulative_frequency_sample = np.loadtxt(f_name)
    
_moe2017_pdf_interp = RegularGridInterpolator(
    (_moe2017_period_sample, _moe2017_primary_mass_sample),
    _moe2017_frequency_sample.T,
    bounds_error=False,
    fill_value=0.
)
_moe2017_cdf_interp = RegularGridInterpolator(
    (_moe2017_period_sample, _moe2017_primary_mass_sample),
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
    _moe2017_primary_mass_sample, (_moe2017_period_sample.size, 1)
)
_moe2017_points = np.vstack([_moe2017_xx.ravel(), _moe2017_yy.T.ravel()])
_moe2017_values = np.tile(
    _moe2017_period_sample[::-1], _moe2017_primary_mass_sample.size
)
_moe2017_ppf_interp = LinearNDInterpolator(_moe2017_points.T, _moe2017_values)

moe2017 = moe2017_gen(a=10.**0.2, b=1.e8, name="period.moe2017")
