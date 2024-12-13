"""Distributions

"""

__all__ = [
    "duquennoy1991",
    "moe2017"
]

import numpy as np
import scipy as sp

from scipy._lib._util import _lazyselect
from scipy._lib._util import _lazywhere

_truncnorm = sp.stats.truncnorm


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The mass-ratio random variable of Duquennoy and Mayor (1991)

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


_duquennoy1991_loc = 0.23
_duquennoy1991_scale = 0.42
_duquennoy1991_a = (0. - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991_b = (np.inf - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991 = _truncnorm(
    a=_duquennoy1991_a, b=_duquennoy1991_b, loc=_duquennoy1991_loc,
    scale=_duquennoy1991_scale
)
duquennoy1991 = _duquennoy1991_gen(a=0., b=np.inf, name="duquennoy1991")


class _moe2017_gen(sp.stats.rv_continuous):
    r"""The mass-ratio random variable of Moe and Stefano (2017)

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
    def _argcheck(self, log10_period, primary_mass):
        res = (
            (0.2 <= log10_period) & (log10_period <= 8.)
            & (0. <= primary_mass) & (primary_mass < np.inf)
        )

        return res

    def _pdf(self, x, log10_period, primary_mass):
        def f_1(x, norm, gamma, delta, log10_period, primary_mass):
            res = norm*0.3**(delta - gamma)*x**gamma

            return res

        def f_2(x, norm, gamma, delta, log10_period, primary_mass):
            res = norm*x**delta

            return res

        def f_3(x, norm, gamma, delta, log10_period, primary_mass):
            # res = norm*(
            #     x**delta
            #     + _moe2017_twin_excess_constant(delta, period, primary_mass)
            # )
            res = (
                norm
                *(
                    x**delta
                    + _moe2017_twin_excess_constant(
                        delta, log10_period, primary_mass
                    )
                )
            )

            return res

        x = np.asarray(x)
        log10_period = np.asarray(log10_period)
        primary_mass = np.asarray(primary_mass)

        gamma = _moe2017_gamma(log10_period, primary_mass)
        delta = _moe2017_delta(log10_period, primary_mass)
        norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)

        condition = (
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        )
        choice = (
            f_1,
            f_2,
            f_3
        )
        res = _lazyselect(
            condition,
            choice,
            (x, norm, gamma, delta, log10_period, primary_mass)
        )
        
        return res

    def _cdf(self, x, log10_period, primary_mass):
        def g_1(x, norm, gamma, delta, log10_period, primary_mass):
            def g_1a(x, norm, gamma, delta):
                res = norm*0.3**(delta - gamma)*(np.log(x) - np.log(0.1))

                return res

            def g_1b(x, norm, gamma, delta):
                res = norm*(
                    0.3**(delta - gamma)
                    *(x**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )

                return res

            condition = np.isclose(gamma, -1.)
            choice = (
                g_1a,
                g_1b
            )
            res = _lazywhere(
                condition, (x, norm, gamma, delta), f=choice[0], f2=choice[1]
            )

            return res            
        
        def g_2(x, norm, gamma, delta, log10_period, primary_mass):
            def g_2a(x, norm, gamma, delta, log10_period, primary_mass):
                res = norm*(np.log(x) - np.log(0.3))

                return res

            def g_2b(x, norm, gamma, delta, log10_period, primary_mass):
                res = (
                    g_1(0.3, norm, gamma, delta, log10_period, primary_mass)
                    + norm*((x**(delta + 1.) - 0.3**(delta + 1.))/(delta + 1.))
                )

                return res

            condition = (delta == -1.)
            choice = (
                g_2a,
                g_2b
            )
            res = _lazywhere(
                condition,
                (x, norm, gamma, delta, log10_period, primary_mass),
                f=choice[0],
                f2=choice[1]
            )

            return res
            
        def g_3(x, norm, gamma, delta, log10_period, primary_mass):
            def g_3a(x, norm, gamma, delta, log10_period, primary_mass):
                res = np.log(x) - np.log(0.95)

                return res

            def g_3b(x, norm, gamma, delta, log10_period, primary_mass):
                res = (
                    g_2(0.95, norm, gamma, delta, log10_period, primary_mass)
                    + norm*(
                        _moe2017_twin_excess_constant(
                            delta, log10_period, primary_mass
                        )
                        *(x - 0.95)
                        + (x**(delta + 1.) - 0.95**(delta + 1.))/(delta + 1.)
                    )
                )

                return res

            condition = (delta == -1.)
            choice = (
                g_3a,
                g_3b
            )
            res = _lazywhere(
                condition,
                (x, norm, gamma, delta, log10_period, primary_mass),
                f=choice[0],
                f2=choice[1]
            )

            return res

        x = np.asarray(x)
        log10_period = np.asarray(log10_period)
        primary_mass = np.asarray(primary_mass)
        
        gamma = _moe2017_gamma(log10_period, primary_mass)
        delta = _moe2017_delta(log10_period, primary_mass)
        norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)

        condition = (
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        )
        choice = (
            g_1,
            g_2,
            g_3
        )
        res = _lazyselect(
            condition,
            choice,
            (x, norm, gamma, delta, log10_period, primary_mass)
        )

        return res

    def _ppf(self, q, log10_period, primary_mass):
        def f_1(q, gamma, delta, norm, log10_period, primary_mass):
            def f_1a(q, gamma, delta, norm):
                q = q/norm
                res = 0.1*np.exp(q/0.3**(delta - gamma))

                return res
            
            def f_1b(q, gamma, delta, norm):
                q = q/norm
                base = (gamma + 1.)*q/0.3**(delta - gamma) + 0.1**(gamma + 1.)
                res = base**(1./(gamma + 1.))

                return res

            condition = np.isclose(gamma, -1.)
            choice = (
                f_1a,
                f_1b
            )
            res = _lazywhere(
                condition,
                (q, gamma, delta, norm),
                f=choice[0],
                f2=choice[1]
            )

            return res

        def f_2(q, gamma, delta, norm, log10_period, primary_mass):
            def f_2a(q, gamma, delta, norm):
                q = q/norm
                a = (
                    0.3**(delta - gamma)
                    *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )
                res = 0.3*np.exp(q - a)
                
                return res
            
            def f_2b(q, gamma, delta, norm):
                q = q/norm
                a = (
                    0.3**(delta - gamma)
                    *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )
                base = (q - a)*(delta + 1.) + 0.3**(delta + 1.)
                res = base**(1./(delta + 1.))

                return res

            condition = (delta == -1.)
            choice = (
                f_2a,
                f_2b
            )
            res = _lazywhere(
                condition, (q, gamma, delta, norm), f=choice[0], f2=choice[1]
            )

            return res

        def f_3(q, gamma, delta, norm, log10_period, primary_mass):
            #################################################################
            #################################################################
            #################################################################
            # TO DO
            # This is very slow. I have vectorized it in the most naive way.
            # Redo it properly.
            #################################################################
            #################################################################
            #################################################################
            def f(x, q, log10_period, primary_mass):
                gamma = _moe2017_gamma(log10_period, primary_mass)
                delta = _moe2017_delta(log10_period, primary_mass)
                norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)
                moe2017_twin_excess_constant = _moe2017_twin_excess_constant(
                    delta, log10_period, primary_mass
                )
                a = (
                    0.3**(delta - gamma)
                    *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )
                b = a + (0.95**(delta + 1.) - 0.3**(delta + 1.))/(delta + 1.)
                res = (
                    norm*b
                    + norm*(x**(delta + 1.) - 0.95**(delta + 1.))/(delta + 1.)
                    + norm*(x - 0.95)*moe2017_twin_excess_constant
                    - q
                )

                return res

            def fsolve(q, log10_period, primary_mass):
                res = sp.optimize.fsolve(
                    f, 0.95, (q, log10_period, primary_mass)
                )

                return res

            # The following is a hack to allow _lazyselecte to
            # work. It passes x = [] when condition is False. Empty
            # arrays cause a vectorized function to throw a
            # ValueError.
            try:
                res = np.vectorize(fsolve)(
                    q, log10_period, primary_mass
                ).squeeze()
            except:
                res = q
                
            return res

        q = np.atleast_1d(q)
        log10_period = np.atleast_1d(log10_period)
        primary_mass = np.atleast_1d(primary_mass)
        
        gamma = _moe2017_gamma(log10_period, primary_mass)
        delta = _moe2017_delta(log10_period, primary_mass)
        norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)
        moe2017_twin_excess_constant = _moe2017_twin_excess_constant(
            delta, log10_period, primary_mass
        )

        condition = (
            (0. < q)
            & (q <= self.cdf(0.3, log10_period, primary_mass)),
            (self.cdf(0.3, log10_period, primary_mass) < q)
            & (q <= self.cdf(0.95, log10_period, primary_mass)),
            (self.cdf(0.95, log10_period, primary_mass) < q)
            & (q <= 1.)
        )
        choice = (
            f_1,
            f_2,
            f_3
        )
        res = _lazyselect(
            condition,
            choice,
            (q, gamma, delta, norm, log10_period, primary_mass)
        )

        return res


def _moe2017_norm(gamma, delta, log10_period, primary_mass):
    """Return the normalization constant"""
    def f_a(gamma, delta):
        # NB: natural logarithm not common logarithm
        res = 0.3**(delta - gamma)*(np.log(0.3) - np.log(0.1))

        return res

    def f_b(gamma, delta):
        res = (
            0.3**(delta - gamma)
            *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
            /(gamma + 1.)
        )

        return res

    def g_a(gamma, delta):
        # NB: natural logarithm not common logarithm
        res = - np.log(0.3)*np.ones_like(gamma)

        return res

    def g_b(gamma, delta):
        res = (1. - 0.3**(delta + 1.))/(delta + 1.)

        return res

    gamma = np.asarray(gamma)
    delta = np.asarray(delta)
    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    # Handle division by zero
    # The formula for the norm is piecewise: specifically, the formula
    # for the case gamma == 1 is distinct from that for the case gamma
    # != 1. For values of gamma close to 1 the formula for the second
    # case is numerically unstable, so I use the formula for the
    # first. The use of np.select does not violate the principle
    # `better to ask forgiveness than permission' since try-except
    # blocks would not handle the numerical instability.
    condition = np.isclose(gamma, -1.)
    choice = (
        f_a,
        f_b
    )
    f = _lazywhere(condition, (gamma, delta), f=choice[0], f2=choice[1])

    # Handle division by zero
    # The formula for the norm is piecewise: specifically, the formula
    # for the case delta == 1 is distinct from that for the case delta
    # != 1. For values of delta close to 1 the formula for the second
    # case is numerically unstable, so I use the formula for the
    # first. The use of np.select does not violate the principle
    # `better to ask forgiveness than permission' since try-except
    # blocks would not handle the numerical instability.
    condition = np.isclose(delta, -1.)
    choice = (
        g_a,
        g_b
    )
    g = _lazywhere(condition, (gamma, delta), f=choice[0], f2=choice[1])

    num = 1.
    denom = (
        f
        + g
        + 0.05*_moe2017_twin_excess_constant(delta, log10_period, primary_mass)
    )
    res = num/denom
    
    return res

def _moe2017_twin_excess_constant(delta, log10_period, primary_mass):
    """Return the twin excess constant"""
    def f_1(delta, log10_period, primary_mass):
        # NB: natural logarithm not common logarithm
        num = (
            _moe2017_twin_excess_fraction(log10_period, primary_mass)
            *np.log(1./0.3)
        )
        denom = (
            0.05
            *(1. - _moe2017_twin_excess_fraction(log10_period, primary_mass))
        )
        res = num/denom
        
        return res

    def f_2(delta, log10_period, primary_mass):
        num = (
            _moe2017_twin_excess_fraction(log10_period, primary_mass)
            *(1. - 0.3**(delta + 1.))
        )
        denom = (
            0.05
            *(delta + 1.)
            *(1. - _moe2017_twin_excess_fraction(log10_period, primary_mass))
        )
        res = num/denom

        return res

    delta = np.asarray(delta)
    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    # Handle division by zero
    # The formula for the norm is piecewise: specifically, the formula
    # for the case delta == 1 is distinct from that for the case delta
    # != 1. This would be better done using try-except blocks (better
    # to ask forgiveness than permission), but I use np.select for
    # consistency with the function `_moe2017_norm`.
    condition = (delta == -1.)
    choice = (
        f_1,
        f_2
    )
    res = _lazywhere(
        condition,
        (delta, log10_period, primary_mass),
        f=choice[0],
        f2=choice[1]
    )

    return res

def _moe2017_twin_excess_fraction(log10_period, primary_mass):
    """Return the twin excess fraction"""
    def f_1(log10_period, primary_mass):
        res = 0.3 - 0.15*np.log10(primary_mass)

        return res

    def f_2(log10_period, primary_mass):
        res = (
            f_1(log10_period, primary_mass)
            *(
                1.
                - (
                    (log10_period - 1.)
                    /(_moe2017_log10_excess_twin_period(primary_mass) - 1.)
                )
            )
        )

        return res

    def f_3(log10_period, primary_mass):
        res = np.zeros_like(primary_mass)

        return res

    log10_period = np.array(log10_period)
    primary_mass = np.asarray(primary_mass)
    
    condition = (
        (0.2 <= log10_period)
        & (log10_period <= 1.),
        (1. < log10_period)
        & (log10_period <= _moe2017_log10_excess_twin_period(primary_mass)),
        (_moe2017_log10_excess_twin_period(primary_mass) < log10_period)
        & (log10_period <= 8.)
    )
    choice = (
        f_1,
        f_2,
        f_3
    )
    res = _lazyselect(condition, choice, (log10_period, primary_mass))

    return res

def _moe2017_log10_excess_twin_period(primary_mass):
    """Return the twin-excess period"""
    def f_1(primary_mass):
        res = 8. - primary_mass

        return res

    def f_2(primary_mass):
        res = 1.5*np.ones_like(primary_mass)

        return res

    primary_mass = np.asarray(primary_mass)
    
    condition = (
            (0. < primary_mass) & (primary_mass <= 6.5),
            (6.5 < primary_mass) & (primary_mass < np.inf)
    )
    choice = (
        f_1,
        f_2
    )
    res = _lazyselect(condition, choice, (primary_mass,))

    return res

def _moe2017_gamma(log10_period, primary_mass):
    """Return the power-law index gamma"""
    def gamma_1(log10_period, primary_mass):
        res  = 0.3*np.ones_like(log10_period)

        return res

    def gamma_2(log10_period, primary_mass):
        res = (
            gamma_1(log10_period, 1.2)
            + (primary_mass - 1.2)
            *(
                gamma_3(log10_period, 3.5) - gamma_1(log10_period, 1.2)
            )
            /2.3
        )

        return res
        
    def gamma_3(log10_period, primary_mass):
        def f_1(log10_period, primary_mass):
            res = 0.2*np.ones_like(log10_period)

            return res

        def f_2(log10_period, primary_mass):
            res = 0.2 - 0.3*(log10_period - 2.5)

            return res

        def f_3(log10_period, primary_mass):
            res = -0.7 - 0.2*(log10_period - 5.5)
            return res
        
        condition = (
            (0.2 <= log10_period) & (log10_period <= 2.5),
            (2.5 < log10_period) & (log10_period <= 5.5),
            (5.5 < log10_period) & (log10_period <= 8.)
        )
        choice = (
            f_1,
            f_2,
            f_3
        )
        res = _lazyselect(
            condition, choice, (log10_period, primary_mass)
        ).squeeze()

        return res

    def gamma_4(log10_period, primary_mass):
        res = (
            gamma_3(log10_period, 3.5)
            + (
                (gamma_5(log10_period, 6.) - gamma_3(log10_period, 3.5))
                *(primary_mass - 3.5)
            )
            /2.5
        )

        return res

    def gamma_5(log10_period, primary_mass):
        def f_1(log10_period, primary_mass):
            res = 0.1*np.ones_like(log10_period)

            return res

        def f_2(log10_period, primary_mass):
            res = 0.1 - 0.15*(log10_period - 1.)

            return res

        def f_3(log10_period, primary_mass):
            res = -0.2 - 0.5*(log10_period - 3.)

            return res

        def f_4(log10_period, primary_mass):
            res = -1.5*np.ones_like(log10_period)

            return res

        condition = (
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 3.),
            (3. < log10_period) & (log10_period <= 5.6),
            (5.6 < log10_period) & (log10_period <= 8.)
        )
        choice = (
            f_1,
            f_2,
            f_3,
            f_4
        )
        res = _lazyselect(condition, choice, (log10_period, primary_mass))

        return res

    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    condition = (
        (0.8 <= primary_mass) & (primary_mass <= 1.2),
        (1.2 < primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass <= 6.),
        (6. < primary_mass) & (primary_mass < np.inf)
    )
    choice = (
        gamma_1,
        gamma_2,
        gamma_3,
        gamma_4,
        gamma_5
    )
    gamma = _lazyselect(condition, choice, (log10_period, primary_mass))

    return gamma

def _moe2017_delta(log10_period, primary_mass):
    """Return the power-law index delta"""
    def delta_1(log10_period, primary_mass):
        def f_1(log10_period, primary_mass):
            res = -0.5*np.ones_like(log10_period)

            return res

        def f_2(log10_period, primary_mass):
            res = -0.5 - 0.3*(log10_period - 5.)

            return res

        condition = (
            (0.2 <= log10_period) & (log10_period <= 5.),
            (5. < log10_period) & (log10_period <= 8.)
        )
        choice = (
            f_1,
            f_2
        )
        res = _lazyselect(condition, choice, (log10_period, primary_mass))

        return res

    def delta_2(log10_period, primary_mass):
        res = (
            delta_1(log10_period, 1.2)
            + (primary_mass - 1.2)
            *(
                delta_3(log10_period, 3.5) - delta_1(log10_period, 1.2)
            )
            /2.3
        )

        return res

    def delta_3(log10_period, primary_mass):
        def f_1(log10_period, primary_mass):
            res = -0.5*np.ones_like(log10_period)

            return res

        def f_2(log10_period, primary_mass):
            res = -0.5 - 0.2*(log10_period - 1.)

            return res

        def f_3(log10_period, primary_mass):
            res = -1.2 - 0.4*(log10_period - 4.5)

            return res

        def f_4(log10_period, primary_mass):
            res = -2.*np.ones_like(log10_period)

            return res

        condition = (
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 4.5),
            (4.5 < log10_period) & (log10_period <= 6.5),
            (6.5 < log10_period) & (log10_period <= 8.)
        )
        choice = (
            f_1,
            f_2,
            f_3,
            f_4
        )
        res = _lazyselect(condition, choice, (log10_period, primary_mass))

        return res

    def delta_4(log10_period, primary_mass):
        #####################################################################
        # Floating point error here!
        # See https://www.geeksforgeeks.org/floating-point-error-in-python/
        #####################################################################
        res = (
            delta_3(log10_period, 3.5)
            + (
                (delta_5(log10_period, 6.) - delta_3(log10_period, 3.5))
                *(primary_mass - 3.5)
            )
            /2.5
        )

        return res

    def delta_5(log10_period, primary_mass):
        def f_1(log10_period, primary_mass):
            res = -0.5*np.ones_like(log10_period)

            return res

        def f_2(log10_period, primary_mass):
            res = -0.5 - 0.9*(log10_period - 1.)

            return res

        def f_3(log10_period, primary_mass):
            res = -1.4 - 0.3*(log10_period - 2.)

            return res

        def f_4(log10_period, primary_mass):
            res = -2.*np.ones_like(log10_period)

            return res

        condition = (
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 2.),
            (2. < log10_period) & (log10_period <= 4.),
            (4. < log10_period) & (log10_period <= 8.)
        )
        choice = (
            f_1,
            f_2,
            f_3,
            f_4
        )
        res = _lazyselect(condition, choice, (log10_period, primary_mass))

        return res

    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    condition = (
        (0.8 <= primary_mass) & (primary_mass <= 1.2),
        (1.2 < primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass <= 6.),
        (6. < primary_mass) & (primary_mass < np.inf)
    )
    choice = (
        delta_1,
        delta_2,
        delta_3,
        delta_4,
        delta_5
    )
    delta = _lazyselect(condition, choice, (log10_period, primary_mass))

    return delta

moe2017 = _moe2017_gen(a=0.1, b=1., name="moe2017")
