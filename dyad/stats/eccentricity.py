"""
=============================================
Eccentricity (:mod:`dyad.stats.eccentricity`)
=============================================

.. currentmodule:: dyad.stats.eccentricity

This module contains probability distributions for the orbital
eccentricities of a population of binary stars. In its documentation
the random variable is denoted :math:`E` and a realization of that
random variable is denoted :math:`e`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   duquennoy1991
   moe2017
   powerlaw
   thermal
   uniform

"""

__all__ = [
    "thermal",
    "duquennoy1991",
    "moe2017",
]

import numpy as np
import scipy as sp

from scipy.stats._distn_infrastructure import _ShapeInfo
from . import _distn_infrastructure


class duquennoy1991_gen(_distn_infrastructure.rv_continuous):
    r"""The eccentricity random variable of Duquennoy and Mayor (1991)

    %(before_notes)s

    Notes
    -----
    The probability density function for `duquennoy1991` is:

    .. math::
       f_{E|P = p}(e|p)
       =
       \begin{cases}
       \dfrac{\phi(e; \mu, \sigma^{2})}{\sigma^{2}{}(\Phi(1; \mu,
       \sigma^{2}) - \Phi(0; \mu, \sigma^{2}))} &\text{ if
       $p \in (11.6, 1000]$}\\
       2e &\text{ if $p \in (1000, 1 \times 10^{12})$}
       \end{cases}

    for eccentricity :math:`e \in [0, 1)` and period
    :math:`p \in (11.6, 1 \times 10^{12})` where :math:`\phi(\cdot;
    \mu, \sigma^{2})` and :math:`\Phi(\cdot; \mu, \sigma^{2})` are the
    probability density function and the cumulative distribution
    function for a Gaussian random variable with mean :math:`\mu` and
    variance :math:`\sigma^{2}` and where :math:`\mu = 0.27` and
    :math:`\sigma = 0.13`. Duquennoy and Mayor (1991) do not give a
    closed-form expression for the probability density function. They
    describe it only as \'bell shaped\' with mean :math:`0.31`. This
    expression has been derived by fitting a normal distribution to
    their data under the assumption that the errors on this data are
    due to Poisson noise.

    `duquennoy1991` takes ``p`` as a shape parameter for :math:`p`,
    the period.

    %(after_notes)s

    References
    ----------
    Duquennoy, A., and M. Mayor. 1991. \'Multiplicity among
    solar-type stars in the solar neighbourhood---II. Distribution of
    the orbital elements in an unbiased Sample\'. *Astronomy and
    Astrophysics* 248 (August): 485.

    %(example)s

    """
    # def _shape_info(self):
    #     return [_ShapeInfo("p", False, (11.6, 1.e12), (False, True))]

    def _argcheck(self, p):
        return (11.6 <= p) & (p < 1.e12)

    def _pdf(self, x, p):
        return np.where(
            (11.6 <= p) & (p <= 1_000.),
            _duquennoy1991_f1.pdf(x),
            _duquennoy1991_f2.pdf(x)
        )

    def _cdf(self, x, p):
        return np.where(
            (11.6 <= p) & (p <= 1_000.),
            _duquennoy1991_f1.cdf(x),
            _duquennoy1991_f2.cdf(x)
        )

    def _ppf(self, q, p):
        return np.where(
            (11.6 <= p) & (p <= 1_000.),
            _duquennoy1991_f1.ppf(q),
            _duquennoy1991_f2.ppf(q)
        )


# Duquennoy and Mayor (1991) tight binaries: truncated normal
_duquennoy1991_loc = 0.27
_duquennoy1991_scale = 0.13
_duquennoy1991_a = (0. - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991_b = (np.inf - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991_f1 = sp.stats.truncnorm(
    a=_duquennoy1991_a, b=_duquennoy1991_b, loc=_duquennoy1991_loc,
    scale=_duquennoy1991_scale
)
# Duquennoy and Mayor (1991) wide binaries: thermal 
_duquennoy1991_f2 = sp.stats.powerlaw(2.)
# Duquennoy and Mayor (1991) all binaries: conditional
duquennoy1991 = duquennoy1991_gen(
    a=0., b=1., name="eccentricity.duquennoy1991"
)


class _thermal_gen(_distn_infrastructure.rv_continuous):
    r"""The thermal eccentricity random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `thermal` is:

    .. math::

        f_{E}(e) = 2e

    for :math:`e \in [0, 1)`.

    %(after_notes)s

    References
    ----------
    Jeans, J. H. 1919. \'The Origin of Binary Systems\'. *Monthly
    Notices of the Royal Astronomical Society* 79 (April):408.
    
    Ambartsumian, V. A. 1937. \'On the Statistics of Double Stars\'.
    *Astronomicheskii Zhurnal* 14 (January):207–19.
    
    %(example)s

    """
    def _pdf(self, x):
        return _thermal.pdf(x)

    def _cdf(self, x):
        return _thermal.cdf(x)

    def _ppf(self, q):
        return _thermal.ppf(q)


_thermal = sp.stats.powerlaw(2.)
thermal = _thermal_gen(a=0., b=1., name="eccentricity.thermal")


class moe2017_gen(_distn_infrastructure.rv_continuous):
    r"""The eccentricity random variable of Moe and Stefano (2017)

    %(before_notes)s

    Notes
    -----
    The probability density function for `moe2017` is:

    .. math::
        f_{E|X, M_{1}}(e|x, m_{1}) = \dfrac{\eta(x, m_{1}) +
        1}{e_{\text{max}}(10^{x})^{\eta(x, m_{1}) + 1}}e^{\eta(x,
        m_{1})}

    where

    .. math::
        e_{\text{max}}(p) &= 1 - \left(\dfrac{1}{2}p\right)^{-2/3}\\
        \eta(x, m_{1})
        &=
        \begin{cases}
        \eta_{1}(x)
        &\text{if $m_{1} \in [0.8, 3]$}\\
        \eta_{2}(x)
        &\text{if $m_{1} \in (3, 7]$}\\
        \eta_{3}(x)
        &\text{if $m_{1} \in (7, 40)$}
        \end{cases}\\
        \eta_{1}(x, m_{1})
        &=
        0.6 - \dfrac{0.7}{x - 0.5}\\
        \eta_{2}(x, m_{1})
        &= \eta_{1}(x) + \dfrac{1}{4}(\eta_{3}(x) - \eta_{1}(x))(m_{1} - 3)\\
        \eta_{3}(x, m_{1})
        &=
        0.9 - \dfrac{0.2}{x - 0.5}.

    for :math:`e \in [0, e_{\text{max}}(p)]` and where log-period
    :math:`x \in (0.9735, 8]` and primary mass :math:`m_{1} \in [0.8, 40]`.

    Note that this function differs from that proposed by Moe and
    Stefano (2017) who used a minimum log-period of :math:`0.5`. Here
    the mimimum log-period is :math:`0.9375` so that :math:`f_{E|X,
    M_{1}}` always has a convergent integral.

    `moe2017` takes ``log10_period`` as a shape parameter for
    :math:`x` and ``primary_mass`` as a shape parameer for
    :math:`m_{1}`.

    %(after_notes)s

    References
    ----------
    Moe, Maxwell, and Rosanne Di Stefano. 2017. \'Mind your Ps and Qs:
    the interrelation between period (P) and mass-ratio (Q)
    distributions of binary stars.\' *The Astrophysical Journal
    Supplement Series* 230 (2): 15.

    %(example)s

    """
    def _shape_info(self):
        ia = _ShapeInfo("log10_period", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("primary_mass", False, (0, np.inf), (False, False))
        
        return [ia, ib]
    
    def _argcheck(self, log10_period, primary_mass):
        # res = _moe2017_eta(log10_period, primary_mass) > -1.

        # return res

        res = (
            (0.9375 < log10_period) & (log10_period <= 8.)
            & (0.8 <= primary_mass) & (primary_mass <= 40.)
        )

        return res

    def _get_support(self, log10_period, primary_mass):
        log10_period = np.asarray(log10_period)
        e_max = 1 - (0.5*10.**log10_period)**(-2./3.)
        res = (np.zeros_like(e_max), e_max)

        return res
        
    def _pdf(self, x, log10_period, primary_mass):
        res = (
            _moe2017_norm(log10_period, primary_mass)
            *x**_moe2017_eta(log10_period, primary_mass)
        )

        return res

    def _cdf(self, x, log10_period, primary_mass):
        res = (
            _moe2017_norm(log10_period, primary_mass)
            *x**(_moe2017_eta(log10_period, primary_mass) + 1.)
            /(_moe2017_eta(log10_period, primary_mass) + 1.)
        )

        return res

    def _ppf(self, q, log10_period, primary_mass):
        num = q*(_moe2017_eta(log10_period, primary_mass) + 1.)
        denom = _moe2017_norm(log10_period, primary_mass)
        res = (num/denom)**(1./(_moe2017_eta(log10_period, primary_mass) + 1.))

        return res


def _moe2017_norm(log10_period, primary_mass):
    """Return the normalization constant"""
    e_max = 1. - (0.5*10.**log10_period)**(-2./3.)
    num = _moe2017_eta(log10_period, primary_mass) + 1.
    denom = e_max**(_moe2017_eta(log10_period, primary_mass) + 1.)
    res = num/denom

    return res

def _moe2017_eta_1(log10_period, primary_mass):
    def f_1(log10_period, primary_mass):
        res = 0.6 - 0.7/(log10_period - 0.5)

        return res

    def f_2(log10_period, primary_mass):
        """Provisionally the same as f_1"""
        # res = 0.6 - 0.7/(log10_period - 0.5)
        res = f_1(log10_period, primary_mass)

        return res

    condition = [
        (0.5 <= log10_period) & (log10_period <= 6.),
        (6. < log10_period) & (log10_period <= 8.),
    ]
    value = [
        f_1(log10_period, primary_mass),
        f_2(log10_period, primary_mass),
    ]
    res = np.select(condition, value, default=np.nan)

    return res

def _moe2017_eta_2(log10_period, primary_mass):
    res = (
        _moe2017_eta_1(log10_period, primary_mass)
        + 0.25
        *(primary_mass - 3.)
        *(
            _moe2017_eta_3(log10_period, primary_mass)
            - _moe2017_eta_1(log10_period, primary_mass)
        )
    )

    return res

def _moe2017_eta_3(log10_period, primary_mass):
    def f_1(log10_period, primary_mass):
        res = 0.9 - 0.2/(log10_period - 0.5)

        return res

    def f_2(log10_period, primary_mass):
        """Provisionally the same as f_1"""
        # res = 0.9 - 0.2/(log10_period - 0.5)
        res = f_1(log10_period, primary_mass)
        
        return res

    condition = [
        (0.5 <= log10_period) & (log10_period <= 5.),        
        (5. < log10_period) & (log10_period <= 8.),
    ]
    value = [
        f_1(log10_period, primary_mass),
        f_2(log10_period, primary_mass),
    ]
    res = np.select(condition, value, default=np.nan)

    return res

def _moe2017_eta(log10_period, primary_mass):
    condition = [
        (0.8 <= primary_mass) & (primary_mass <= 3.),
        (3. < primary_mass) & (primary_mass <= 7.),
        (7. < primary_mass) & (primary_mass < 40.),
    ]
    value = [
        _moe2017_eta_1(log10_period, primary_mass),
        _moe2017_eta_2(log10_period, primary_mass),
        _moe2017_eta_3(log10_period, primary_mass),
    ]
    res = np.select(condition, value)

    return res

moe2017 = moe2017_gen(a=0., b=1., name="eccentricity.moe2017")
