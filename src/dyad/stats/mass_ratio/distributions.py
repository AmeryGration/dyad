"""Module providing random variables for mass ratio"""

__all__ = [
    "truncnorm",
    "duquennoy1991",
    "moe2017"
]

import numpy as np
import scipy as sp

import plot

truncnorm = sp.stats.truncnorm


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The Duquennoy and Mayor (1991) mass-ratio random variable

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
    .. [1] Reference

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


loc = 0.23
scale = 0.42
a = (0. - loc)/scale
b = (np.inf - loc)/scale
_duquennoy1991 = truncnorm(a=a, b=b, loc=loc, scale=scale)
duquennoy1991 = _duquennoy1991_gen(a=0., b=np.inf, name="duquennoy1991")

def _moe2017_norm(gamma, delta, period, primary_mass):
    """Return the normalization constant"""
    # gamma = np.asarray(gamma)
    # delta = np.asarray(delta)
    # period = np.asarray(period)
    # primary_mass = np.asarray(primary_mass)

    # num = 1. 
    # denom = (
    #     0.3**(delta - gamma)
    #     *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
    #     /(gamma + 1.)
    #     + (1. - 0.3**(delta + 1.))
    #     /(delta + 1.)
    #     # + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    # )
    # mask = (denom == np.inf)
    # denom[mask] = (
    #     0.3**(delta[mask] - gamma[mask])
    #     *(0.3**(gamma[mask] + 1.) - 0.1**(gamma[mask] + 1.))
    #     /(gamma[mask] + 1.)
    #     # + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    # )

    # res = num/denom

    # return res

    # def f(gamma, delta):
    #     with np.errstate(invalid="raise"):
    #         # Handle division by zero
    #         try:
    #             res = (
    #                 0.3**(delta - gamma)
    #                 *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
    #                 /(gamma + 1.)
    #             )
    #         except FloatingPointError:
    #             # NB: natural log not common logarithm
    #             res = 0.3**(delta - gamma)*(np.log(0.3) - np.log(0.1)) 

    #     return res

    # def g(gamma, delta, period, primary_mass):
    #     with np.errstate(invalid="raise"):
    #         # Handle division by zero
    #         try:
    #             res = (
    #                 (1. - 0.3**(delta + 1.))
    #                 /(delta + 1.)
    #                 + 0.05*_moe2017_twin_excess_constant(
    #                     delta, period, primary_mass
    #                 )
    #             )
    #         except FloatingPointError:
    #             # NB: natural log not common logarithm
    #             res = - np.log(0.3)

    #     return res

    # gamma = np.asarray(gamma)
    # delta = np.asarray(delta)
    # period = np.asarray(period)
    # primary_mass = np.asarray(primary_mass)

    # num = 1.
    # denom = (
    #     f(gamma, delta)
    #     + g(gamma, delta, period, primary_mass)
    #     + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    # )
    # res = num/denom

    # return res

    def f_a(gamma, delta):
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
        res = - np.log(0.3)*np.ones_like(gamma)

        return res

    def g_b(gamma, delta):
        res = (1. - 0.3**(delta + 1.))/(delta + 1.)

        return res

    gamma = np.asarray(gamma)
    delta = np.asarray(delta)
    period = np.asarray(period)
    primary_mass = np.asarray(primary_mass)

    # Handle division by zero

    # The formula for the norm is piecewise: specifically, the formula
    # for the case gamma == 1 is distinct from that for the case gamma
    # != 1. For values of gamma close to 1 the formula for the second
    # case is numerically unstable, so I use the formula for the
    # first. The use of np.select does not violate the principle that
    # `better to ask forgiveness than permission' since try-except
    # blocks would not handle the numerical instability.
    condition = [np.isclose(gamma, -1.), ~np.isclose(gamma, -1.)]
    value = [f_a(gamma, delta), f_b(gamma, delta)]
    f = np.select(condition, value)

    # The formula for the norm is piecewise: specifically, the formula
    # for the case delta == 1 is distinct from that for the case delta
    # != 1. For values of delta close to 1 the formula for the second
    # case is numerically unstable, so I use the formula for the
    # first. The use of np.select does not violate the principle that
    # `better to ask forgiveness than permission' since try-except
    # blocks would not handle the numerical instability.
    condition = [np.isclose(delta, -1.), ~np.isclose(delta, -1.)]
    value = [g_a(gamma, delta), g_b(gamma, delta)]
    g = np.select(condition, value)    

    num = 1.
    denom = (
        f + g + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    )
    res = num/denom
    
    return res

def _moe2017_twin_excess_constant(delta, period, primary_mass):
    """Return the twin excess constant"""
    # delta = np.asarray(delta)
    # period = np.asarray(period)
    # primary_mass = np.array(primary_mass)

    # num = (
    #     _moe2017_twin_excess_fraction(period, primary_mass)
    #     *(1. - 0.3**(delta + 1.))
    # )
    # denom = (
    #     0.05
    #     *(delta + 1.)
    #     *(1. - _moe2017_twin_excess_fraction(period, primary_mass))
    # )
    # res = num/denom

    # return res

    # delta = np.asarray(delta)
    # period = np.asarray(period)
    # primary_mass = np.array(primary_mass)
    
    # with np.errstate(invalid="raise"):
    #     # Handle division by zero
    #     try:
    #         num = (
    #             _moe2017_twin_excess_fraction(period, primary_mass)
    #             *(1. - 0.3**(delta + 1.))
    #         )
    #         denom = (
    #             0.05
    #             *(delta + 1.)
    #             *(1. - _moe2017_twin_excess_fraction(period, primary_mass))
    #         )
    #         res = num/denom
    #     except FloatingPointError:
    #         # NB: natural log not common logarithm
    #         num = (
    #             - _moe2017_twin_excess_fraction(period, primary_mass)
    #             *np.log(0.3)
    #         )
    #         denom = (
    #             0.05
    #             *(1. - _moe2017_twin_excess_fraction(period, primary_mass))
    #         )
    #         res = num/denom
            
    # return res

    def f_1(delta, period, primary_mass):
        # NB: natural log not common logarithm
        num = (
            _moe2017_twin_excess_fraction(period, primary_mass)
            *np.log(1./0.3)
        )
        denom = (
            0.05
            *(1. - _moe2017_twin_excess_fraction(period, primary_mass))
        )
        res = num/denom
        
        return res

    def f_2(delta, period, primary_mass):
        num = (
            _moe2017_twin_excess_fraction(period, primary_mass)
            *(1. - 0.3**(delta + 1.))
        )
        denom = (
            0.05
            *(delta + 1.)
            *(1. - _moe2017_twin_excess_fraction(period, primary_mass))
        )
        res = num/denom

        return res

    delta = np.asarray(delta)
    period = np.asarray(period)
    primary_mass = np.asarray(primary_mass)

    # Handle division by zero
    # The formula for the norm is piecewise: specifically, the formula
    # for the case delta == 1 is distinct from that for the case delta
    # != 1. This would be better done using try-except blocks (better
    # to ask forgiveness than permission), but I use np.select for
    # consistency with the function `_moe2017_norm`.
    condition = [delta == -1., delta != -1.]
    value = [
        f_1(delta, period, primary_mass), f_2(delta, period, primary_mass)
    ]
    res = np.select(condition, value)

    return res

def _moe2017_twin_excess_fraction(period, primary_mass):
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

    log10_period = np.log10(period)
    primary_mass = np.asarray(primary_mass)
    
    condition = [
        (0.2 <= log10_period)
        & (log10_period <= 1.),
        (1. < log10_period)
        & (log10_period <= _moe2017_log10_excess_twin_period(primary_mass)),
        (_moe2017_log10_excess_twin_period(primary_mass) < log10_period)
        & (log10_period <= 8.)
    ]
    value = [
        f_1(log10_period, primary_mass),
        f_2(log10_period, primary_mass),
        f_3(log10_period, primary_mass)
    ]
    res = np.select(condition, value)

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
    
    condition = [
        (0. < primary_mass) & (primary_mass <= 6.5),
        (6.5 < primary_mass) & (primary_mass < np.inf)
    ]
    value = [
        f_1(primary_mass),
        f_2(primary_mass)
    ]
    res = np.select(condition, value)

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
        
        condition = [
            (0.2 <= log10_period) & (log10_period <= 2.5),
            (2.5 < log10_period) & (log10_period <= 5.5),
            (5.5 < log10_period) & (log10_period <= 8.)
        ]
        value = [
            f_1(log10_period, primary_mass),
            f_2(log10_period, primary_mass),
            f_3(log10_period, primary_mass)
        ]
        res = np.select(condition, value).squeeze()

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

        condition = [
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 3.),
            (3. < log10_period) & (log10_period <= 5.6),
            (5.6 < log10_period) & (log10_period <= 8.)
        ]
        value = [
            f_1(log10_period, primary_mass),
            f_2(log10_period, primary_mass),
            f_3(log10_period, primary_mass),
            f_4(log10_period, primary_mass)
        ]
        res = np.select(condition, value)

        return res

    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    condition = [
        (0.8 <= primary_mass) & (primary_mass <= 1.2),
        (1.2 < primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass <= 6.),
        (6. < primary_mass) & (primary_mass < np.inf)
    ]
    value = [
        gamma_1(log10_period, primary_mass),
        gamma_2(log10_period, primary_mass),
        gamma_3(log10_period, primary_mass),
        gamma_4(log10_period, primary_mass),
        gamma_5(log10_period, primary_mass)
    ]
    gamma = np.select(condition, value)

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

        condition = [
            (0.2 <= log10_period) & (log10_period <= 5.),
            (5. < log10_period) & (log10_period <= 8.)
        ]
        value = [
            f_1(log10_period, primary_mass),
            f_2(log10_period, primary_mass)
        ]
        res = np.select(condition, value)

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

        condition = [
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 4.5),
            (4.5 < log10_period) & (log10_period <= 6.5),
            (6.5 < log10_period) & (log10_period <= 8.)
        ]
        value = [
            f_1(log10_period, primary_mass),
            f_2(log10_period, primary_mass),
            f_3(log10_period, primary_mass),
            f_4(log10_period, primary_mass)
        ]
        res = np.select(condition, value)

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

        condition = [
            (0.2 <= log10_period) & (log10_period <= 1.),
            (1. < log10_period) & (log10_period <= 2.),
            (2. < log10_period) & (log10_period <= 4.),
            (4. < log10_period) & (log10_period <= 8.)
        ]
        value = [
            f_1(log10_period, primary_mass),
            f_2(log10_period, primary_mass),
            f_3(log10_period, primary_mass),
            f_4(log10_period, primary_mass)
        ]
        res = np.select(condition, value)

        return res

    log10_period = np.asarray(log10_period)
    primary_mass = np.asarray(primary_mass)

    condition = [
        (0.8 <= primary_mass) & (primary_mass <= 1.2),
        (1.2 < primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass <= 6.),
        (6. < primary_mass) & (primary_mass < np.inf)
    ]
    value = [
        delta_1(log10_period, primary_mass),
        delta_2(log10_period, primary_mass),
        delta_3(log10_period, primary_mass),
        delta_4(log10_period, primary_mass),
        delta_5(log10_period, primary_mass)
    ]
    delta = np.select(condition, value)

    return delta


class _moe2017_gen(sp.stats.rv_continuous):
    r"""The Moe and Stefano (2017) mass-ratio random variable

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
    .. [1] Reference

    %(example)s

    """
    #########################################################################
    #########################################################################
    #########################################################################
    # TO DO
    # Write lazy version of np.select.
    #########################################################################
    #########################################################################
    #########################################################################
    def _argcheck(self, period, primary_mass):
        log10_period = np.log10(period)
        return (
            (0.2 <= log10_period) & (log10_period <= 8.)
            & (0. <= primary_mass) & (primary_mass < np.inf)
        )

    def _pdf(self, x, period, primary_mass):
        def f_1(x, gamma, delta):
            res = norm*0.3**(delta - gamma)*x**gamma

            return res

        def f_2(x, delta):
            res = norm*x**delta

            return res

        def f_3(x, delta):
            res = norm*(
                x**delta
                + _moe2017_twin_excess_constant(delta, period, primary_mass)
            )

            return res

        period = np.asarray(period)
        primary_mass = np.asarray(primary_mass)

        gamma = _moe2017_gamma(np.log10(period), primary_mass)
        delta = _moe2017_delta(np.log10(period), primary_mass)
        norm = _moe2017_norm(gamma, delta, period, primary_mass)

        condition = [
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        ]
        value = [
            f_1(x, gamma, delta),
            f_2(x, delta),
            f_3(x, delta)
        ]
        res = np.select(condition, value)
        
        return res

    def _cdf(self, x, period, primary_mass):
        def g_1(x, gamma, delta):
            def g_1a(x, gamma, delta):
                res = 0.3**(delta - gamma)*(np.log(x) - np.log(0.1))

                return res

            def g_1b(x, gamma, delta):
                res = norm*(
                    0.3**(delta - gamma)
                    *(x**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )

                return res

            condition = [np.isclose(gamma, -1.), ~np.isclose(gamma, -1.)]
            value = [
                g_1a(x, gamma, delta),
                g_1b(x, gamma, delta)
            ]
            
            res = np.select(condition, value)

            return res

        def g_2(x, gamma, delta):
            def g_2a(x):
                res = np.log(x) - np.log(0.3)

                return res
            
            def g_2b(x, gamma, delta):
                res = (
                    g_1(0.3, gamma, delta)
                    + norm*((x**(delta + 1.) - 0.3**(delta + 1.))/(delta + 1.))
                )

                return res

            condition = [delta == -1., delta != -1.]
            value = [
                g_2a(x),
                g_2b(x, gamma, delta)
            ]
            res = np.select(condition, value)

            return res

        def g_3(x, gamma, delta, period, primary_mass):
            def g_3a(x):
                res = np.log(x) - np.log(0.95)

                return res
            
            def g_3b(x, gamma, delta, primary_mass):
                res = g_2(0.95, gamma, delta) + norm*(
                    _moe2017_twin_excess_constant(delta, period, primary_mass)
                    *(x - 0.95)
                    + (x**(delta + 1.) - 0.95**(delta + 1.))/(delta + 1.)
                )

                return res

            condition = [delta == -1., delta != -1.]
            value = [
                g_3a(x),
                g_3b(x, gamma, delta, primary_mass)
            ]
            res = np.select(condition, value)

            return res

        gamma = _moe2017_gamma(np.log10(period), primary_mass)
        delta = _moe2017_delta(np.log10(period), primary_mass)
        norm = _moe2017_norm(gamma, delta, period, primary_mass)

        condition = [
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        ]
        value = [
            g_1(x, gamma, delta),
            g_2(x, gamma, delta),
            g_3(x, gamma, delta, period, primary_mass)
        ]
        res = np.select(condition, value)

        return res

    def _ppf(self, q, period, primary_mass):
        def f_1(q, gamma, delta, norm):
            def f_1a(q, gamma, delta, norm):
                q = q/norm
                res = 0.1*np.exp(q/0.3**(delta - gamma))

                return res
            
            def f_1b(q, gamma, delta, norm):
                q = q/norm
                base = (gamma + 1.)*q/0.3**(delta - gamma) + 0.1**(gamma + 1.)
                res = base**(1./(gamma + 1.))

                return res

            condition = [np.isclose(gamma, -1.), ~np.isclose(gamma, -1.)]
            value = [
                f_1a(q, gamma, delta, norm),
                f_1b(q, gamma, delta, norm)
            ]
            res = np.select(condition, value)

            return res

        def f_2(q, gamma, delta, norm):
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

            condition = [delta == -1., delta != -1.]
            value = [
                f_2a(q, gamma, delta, norm),
                f_2b(q, gamma, delta, norm)
            ]
            res = np.select(condition, value)

            return res

        def f_3(q, period, primary_mass):
            #################################################################
            #################################################################
            #################################################################
            # TO DO
            # This is very slow. I have vectorized it in the most naive way.
            # Redo it properly.
            #################################################################
            #################################################################
            #################################################################
            def f(x, q, period, primary_mass):
                gamma = _moe2017_gamma(np.log10(period), primary_mass)
                delta = _moe2017_delta(np.log10(period), primary_mass)
                norm = _moe2017_norm(gamma, delta, period, primary_mass)
                moe2017_twin_excess_constant = _moe2017_twin_excess_constant(
                    delta, period, primary_mass
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

            def fsolve(q, period, primary_mass):
                res = sp.optimize.fsolve(f, 0.95, (q, period, primary_mass))

                return res

            res = np.vectorize(fsolve)(q, period, primary_mass)

            return res

        q = np.atleast_1d(q)
        period = np.atleast_1d(period)
        primary_mass = np.atleast_1d(primary_mass)
        
        gamma = _moe2017_gamma(np.log10(period), primary_mass)
        delta = _moe2017_delta(np.log10(period), primary_mass)
        norm = _moe2017_norm(gamma, delta, period, primary_mass)
        moe2017_twin_excess_constant = _moe2017_twin_excess_constant(
            delta, period, primary_mass
        )

        condition = [
            (0. < q)
            & (q <= self.cdf(0.3, period, primary_mass)),
            (self.cdf(0.3, period, primary_mass) < q)
            & (q <= self.cdf(0.95, period, primary_mass)),
            (self.cdf(0.95, period, primary_mass) < q)
            & (q <= 1.)
        ]
        value = [
            f_1(q, gamma, delta, norm),
            f_2(q, gamma, delta, norm),
            f_3(q, period, primary_mass),
        ]
        res = np.select(condition, value)

        return res


moe2017 = _moe2017_gen(a=0.1, b=1., name="moe2017")

def main():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from scipy.integrate import quad

    mpl.style.use("sm")

    N_SAMPLE = 1_000

    #########################################################################
    # Plot Duquennoy 1991
    #########################################################################
    # Test methods: pdf, cdf, ppf
    rv_mass_ratio = duquennoy1991

    q = np.linspace(0., 2., 500)
    pdf = rv_mass_ratio.pdf(q)
    cdf = rv_mass_ratio.cdf(q)
    ppf = rv_mass_ratio.ppf(q)

    mass_ratio = rv_mass_ratio.rvs(size=N_SAMPLE)
    counts, edges = np.histogram(
        mass_ratio, bins=np.linspace(0., 2., 25), density=True
    )

    fig, ax = plot.plot()
    ax.plot(q, pdf)
    # ax.stairs(counts, edges)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$f_{q}$")
    ax.set_ylim(0., 1.5)
    fig.savefig("duquennoy1991_mass_ratio_pdf.pdf")
    fig.savefig("duquennoy1991_mass_ratio_pdf.jpg")
    fig.show()

    fig, ax = plot.plot()
    ax.plot(q, cdf, ls="solid", label=r"$y = F_{q}$")
    ax.plot(q, ppf, ls="dashed", label=r"$y = F_{q}^{-1}$")
    ax.legend(frameon=False, loc=2)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$y$")
    fig.savefig("duquennoy1991_mass_ratio_cdf.pdf")
    fig.savefig("duquennoy1991_mass_ratio_cdf.jpg")
    fig.show()

    #########################################################################
    # Plot Moe 2017
    #########################################################################
    # Test utility functions: F_twin, c_twin
    primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
    log10_period_boundary = (
        0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
    )

    n = 50
    primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
    log10_period = np.hstack(
        [
            np.linspace(0.2 + 1.e-6, 1., n),
            np.linspace(1., 1.3, n)[1:],
            np.linspace(1.3, 2., n)[1:],
            np.linspace(2., 2.5, n)[1:],
            np.linspace(2.5, 3.4, n)[1:],
            np.linspace(3.4, 3.5, n)[1:],
            np.linspace(3.5, 4., n)[1:],
            np.linspace(4., 4.5, n)[1:],
            np.linspace(4.5, 5.5, n)[1:],
            np.linspace(5.5, 6., n)[1:],
            np.linspace(6., 6.5, n)[1:],
            np.linspace(6.5, 8., n)[1:],
        ]
    )
    period = 10.**log10_period

    log10_twin_excess_period = _moe2017_log10_excess_twin_period(primary_mass)

    F_twin = _moe2017_twin_excess_fraction(
        period, primary_mass.reshape([-1, 1])
    )
    F_twin_closed_dots_x = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
    )
    F_twin_closed_dots_x += 1.e-6
    F_twin_closed_dots_y = _moe2017_twin_excess_fraction(
        10.**F_twin_closed_dots_x, np.tile(primary_mass, 2)
    )

    delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
    c_twin = _moe2017_twin_excess_constant(
        delta, period, primary_mass.reshape([-1, 1])
    )
    c_twin_closed_dots_x = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
    )
    c_twin_closed_dots_x += 1.e-6
    c_twin_closed_dots_y = _moe2017_twin_excess_constant(
        np.hstack([np.tile(delta[0, 0], 5), np.tile(delta[0, -1], 5)]),
        10.**c_twin_closed_dots_x,
        np.tile(primary_mass, 2)
    )

    fig, ax = plot.array(2, 1, sharex=True)
    ax[0].plot(log10_period, F_twin[0], color="red", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[0].plot(log10_period, F_twin[1], color="orange", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[0].plot(log10_period, F_twin[2], color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[0].plot(log10_period, F_twin[3], color="blue", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[0].plot(log10_period, F_twin[4], color="magenta", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax[0].vlines(log10_twin_excess_period, -0.05, 0.45, ls="dashed")
    ax[0].text(
        log10_twin_excess_period[2] + 0.15,
        0. + 0.015,
        (r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[4])
         + "\n"
         + r"$= \log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[3])
         + "\n"
         + r"$= \log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[2])
         + "\n"),
        horizontalalignment="left",
        verticalalignment="bottom",
        rotation=90.,
    )
    ax[0].text(
       log10_twin_excess_period[1] + 0.15,
       0. + 0.015,
       r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[1]),
       horizontalalignment="left",
       verticalalignment="bottom",
       rotation=90.,
    )
    ax[0].text(
       log10_twin_excess_period[0] + 0.15,
       0. + 0.015,
       r"$\log_{{10}}(P_{{\text{{twin}}}}({}))$".format(primary_mass[0]),
       horizontalalignment="left",
       verticalalignment="bottom",
       rotation=90.,
    )
    ax[0].scatter(F_twin_closed_dots_x[0::5], F_twin_closed_dots_y[0::5],
                  s=2., color="red")
    ax[0].scatter(F_twin_closed_dots_x[1::5], F_twin_closed_dots_y[1::5],
                  s=2., color="orange")
    ax[0].scatter(F_twin_closed_dots_x[2::5], F_twin_closed_dots_y[2::5],
                  s=2., color="green")
    ax[0].scatter(F_twin_closed_dots_x[3::5], F_twin_closed_dots_y[3::5],
                  s=2., color="blue")
    ax[0].scatter(F_twin_closed_dots_x[4::5], F_twin_closed_dots_y[4::5],
                  s=2., color="magenta")
    ax[0].set_xlim(-1., 9.)
    ax[0].set_ylim(-0.05, 0.45)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel(r"$F_{\text{twin}}$")
    ax[1].plot(log10_period, c_twin[0], color="red", ls="solid",
               label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[1].plot(log10_period, c_twin[1], color="orange", ls="solid",
               label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[1].plot(log10_period, c_twin[2], color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[1].plot(log10_period, c_twin[3], color="blue", ls="solid",
               label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[1].plot(log10_period, c_twin[4], color="magenta", ls="solid",
               label=r"$M_{{1}} = {}$".format(primary_mass[4]))

    ax[1].vlines(log10_twin_excess_period, -2., 10., ls="dashed")
    ax[1].scatter(c_twin_closed_dots_x[0::5], c_twin_closed_dots_y[0::5],
                  s=2., color="red")
    ax[1].scatter(c_twin_closed_dots_x[1::5], c_twin_closed_dots_y[1::5],
                  s=2., color="orange")
    ax[1].scatter(c_twin_closed_dots_x[2::5], c_twin_closed_dots_y[2::5],
                  s=2., color="green")
    ax[1].scatter(c_twin_closed_dots_x[3::5], c_twin_closed_dots_y[3::5],
                  s=2., color="blue")
    ax[1].scatter(c_twin_closed_dots_x[4::5], c_twin_closed_dots_y[4::5],
                  s=2., color="magenta")
    ax[1].set_xlim(-1., 9.)
    ax[1].set_ylim(-2., 10.)
    ax[1].set_xlabel(r"$\log(P)$")
    ax[1].set_ylabel(r"$c_{\text{twin}}$")
    fig.savefig("moe2017_twin_excess.pdf")
    fig.savefig("moe2017_twin_excess.jpg")
    fig.show()

    # Test utility functions: gamma, delta
    gamma = _moe2017_gamma(log10_period, primary_mass.reshape([-1, 1]))
    delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
    gamma_closed_dots_x = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
    )
    gamma_closed_dots_y = _moe2017_gamma(
        gamma_closed_dots_x, np.tile(primary_mass, 2)
    )
    delta_closed_dots_x = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2, 8., 8., 8., 8., 8.]
    )
    delta_closed_dots_y = _moe2017_delta(
        delta_closed_dots_x, np.tile(primary_mass, 2)
    )

    fig, ax = plot.array(2, 1, sharex=True)
    ax[0].plot(log10_period, delta[0], color="red", ls="solid")
    ax[0].plot(log10_period, delta[1], color="orange", ls="solid")
    ax[0].plot(log10_period, delta[2], color="green", ls="solid")
    ax[0].plot(log10_period, delta[3], color="blue", ls="solid")
    ax[0].plot(log10_period, delta[4], color="magenta", ls="solid")
    ax[0].scatter(delta_closed_dots_x[0::5], delta_closed_dots_y[0::5],
                  s=2., color="red")
    ax[0].scatter(delta_closed_dots_x[1::5], delta_closed_dots_y[1::5],
                  s=2., color="orange")
    ax[0].scatter(delta_closed_dots_x[2::5], delta_closed_dots_y[2::5],
                  s=2., color="green")
    ax[0].scatter(delta_closed_dots_x[3::5], delta_closed_dots_y[3::5],
                  s=2., color="blue")
    ax[0].scatter(delta_closed_dots_x[4::5], delta_closed_dots_y[4::5],
                  s=2., color="magenta")
    ax[0].set_xlim(0., 8.)
    ax[0].set_ylim(-3., 0.5)
    ax[0].set_ylabel(r"$\delta$")
    ax[1].plot(log10_period, gamma[0], color="red", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[1].plot(log10_period, gamma[1], color="orange", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[1].plot(log10_period, gamma[2], color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[1].plot(log10_period, gamma[3], color="blue", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[1].plot(log10_period, gamma[4], color="magenta", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax[1].scatter(gamma_closed_dots_x[0::5], gamma_closed_dots_y[0::5],
                  s=2., color="red")
    ax[1].scatter(gamma_closed_dots_x[1::5], gamma_closed_dots_y[1::5],
                  s=2., color="orange")
    ax[1].scatter(gamma_closed_dots_x[2::5], gamma_closed_dots_y[2::5],
                  s=2., color="green")
    ax[1].scatter(gamma_closed_dots_x[3::5], gamma_closed_dots_y[3::5],
                  s=2., color="blue")
    ax[1].scatter(gamma_closed_dots_x[4::5], gamma_closed_dots_y[4::5],
                  s=2., color="magenta")
    ax[1].legend(frameon=False)
    ax[1].set_xlim(-1., 9.)
    ax[1].set_ylim(-2.5, 1.5)
    ax[1].set_xlabel(r"$\log(P)$")
    ax[1].set_ylabel(r"$\gamma$")
    fig.savefig("moe2017_gamma_delta.pdf")
    fig.savefig("moe2017_gamma_delta.jpg")
    fig.show()
 
    # Test utility functions: twin period
    primary_mass = np.linspace(0., 10., 250)
    log10_period_twin = _moe2017_log10_excess_twin_period(primary_mass)

    fig, ax = plot.plot()
    ax.plot(primary_mass[1:], log10_period_twin[1:])
    ax.scatter(0., 8., s=2., color="k", facecolor="white", zorder=np.inf)
    ax.set_xlim(-1., 10.)
    ax.set_ylim(0., 10.)
    ax.set_xlabel(r"$M_{1}$")
    ax.set_ylabel(r"$\log_{10}(P_{\text{twin}})$")
    fig.savefig("moe2017_twin_excess_period.pdf")
    fig.savefig("moe2017_twin_excess_period.jpg")
    fig.show()

    # Test utility functions: norm
    primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
    log10_period_boundary = (
        0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
    )

    n = 50
    primary_mass = np.hstack(
        [
            np.linspace(0.8, 1.2, n),
            np.linspace(1.2, 3.5, n)[1:],
            np.linspace(3.5, 6., n)[1:],
            np.linspace(6., 60., n)[1:],
        ]
    )
    log10_period = np.hstack(
        [
            np.linspace(0.2 + 1.e-6, 1., n),
            np.linspace(1., 1.3, n)[1:],
            np.linspace(1.3, 2., n)[1:],
            np.linspace(2., 2.5, n)[1:],
            np.linspace(2.5, 3.4, n)[1:],
            np.linspace(3.4, 3.5, n)[1:],
            np.linspace(3.5, 4., n)[1:],
            np.linspace(4., 4.5, n)[1:],
            np.linspace(4.5, 5.5, n)[1:],
            np.linspace(5.5, 6., n)[1:],
            np.linspace(6., 6.5, n)[1:],
            np.linspace(6.5, 8., n)[1:],
        ]
    )
    period = 10.**log10_period

    gamma = _moe2017_gamma(log10_period, primary_mass.reshape([-1, 1]))
    delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))
    norm =_moe2017_norm(
        gamma, delta, period, primary_mass.reshape([-1, 1])
    )

    fig, ax, cbar = plot.plot(cbar=True)
    im = ax.pcolormesh(log10_period, primary_mass, norm, rasterized=True)
    ax.contour(log10_period, primary_mass, norm, colors="k")
    ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
    ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
    ax.set_yscale("log")
    ax.set_xlim(0.2, 8.)
    ax.set_ylim(0.8, 60.)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$M_{1}$")
    cbar = fig.colorbar(im, cax=cbar)
    cbar.set_label(r"$A_{q}$")
    plt.savefig("norm.pdf")
    plt.savefig("norm.jpg")
    plt.show()

    # Test class methods: PDF and CDF (no twin excess)
    primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
    print("primary_mass =", primary_mass)
    log10_excess_twin_period = _moe2017_log10_excess_twin_period(primary_mass)
    print("log10_excess_twin_period =", log10_excess_twin_period)
    
    log10_period = 8.
    # log10_period = log10_excess_twin_period + 1.
    rv = moe2017(10.**log10_period, primary_mass.reshape([-1, 1]))

    n = 100
    q_0 = np.linspace(0., 0.1, n)[:-1]
    q_1 = np.linspace(0.1, 1., n)
    q_2 = np.linspace(1., 1.1, n)[1:]
    pdf_0 = rv.pdf(q_0)
    pdf_1 = rv.pdf(q_1)
    pdf_2 = rv.pdf(q_2)

    q_0 = np.linspace(0., 0.1, n)[:-1]
    q_1 = np.linspace(0.1, 1., n)
    q_2 = np.linspace(1., 1.1, n)[1:]
    cdf_0 = rv.cdf(q_0)
    cdf_1 = rv.cdf(q_1)
    cdf_2 = rv.cdf(q_2)

    pdf_open_dots_x = (0.1, 0.1, 0.1, 0.1, 0.1, 1., 1., 1., 1., 1.)
    pdf_open_dots_y = (0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
    pdf_closed_dots_x = (0.1, 0.1, 0.1, 0.1, 0.1, 1., 1., 1., 1., 1.)
    pdf_closed_dots_y = (*pdf_1[:, 0], *pdf_1[:, -1])
        
    fig, ax_1 = plot.plot()
    ax_2 = ax_1.twinx()
    ax_1.plot(q_0, pdf_0[0], ls="solid", color="red")
    ax_1.plot(q_1, pdf_1[0], ls="solid", color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_1.plot(q_2, pdf_2[0], ls="solid", color="red")
    ax_1.plot(q_0, pdf_0[1], ls="solid", color="orange")
    ax_1.plot(q_1, pdf_1[1], ls="solid", color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_1.plot(q_2, pdf_2[1], ls="solid", color="orange")
    ax_1.plot(q_0, pdf_0[2], ls="solid", color="green")
    ax_1.plot(q_1, pdf_1[2], ls="solid", color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_1.plot(q_2, pdf_2[2], ls="solid", color="green")
    ax_1.plot(q_0, pdf_0[3], ls="solid", color="blue")
    ax_1.plot(q_1, pdf_1[3], ls="solid", color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_1.plot(q_2, pdf_2[3], ls="solid", color="blue")
    ax_1.plot(q_0, pdf_0[4], ls="solid", color="magenta")
    ax_1.plot(q_1, pdf_1[4], ls="solid", color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_1.plot(q_2, pdf_2[4], ls="solid", color="magenta")
    ax_1.scatter(pdf_closed_dots_x[0::5], pdf_closed_dots_y[0::5],
                 s=2., color="red", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[1::5], pdf_closed_dots_y[1::5],
                 s=2., color="orange", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[2::5], pdf_closed_dots_y[2::5],
                 s=2., color="green", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[3::5], pdf_closed_dots_y[3::5],
                 s=2., color="blue", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[4::5], pdf_closed_dots_y[4::5],
                 s=2., color="magenta", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[4::5], pdf_open_dots_y[4::5],
                 s=2., color="magenta", facecolor="white", zorder=np.inf)
    ax_1.legend(frameon=False)
    ax_1.set_xlabel(r"$q$")
    ax_1.set_ylabel(r"$f_{q|P, M_{1}}$")
    ax_1.set_ylim(-1., 11)
    # ax_2.plot(q_0, cdf_0[0], ls="dashed", color="red")
    ax_2.plot(q_1, cdf_1[0], ls="dashed", color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_2.plot(q_2, cdf_2[0], ls="dashed", color="red")
    # ax_2.plot(q_0, cdf_0[1], ls="dashed", color="orange")
    ax_2.plot(q_1, cdf_1[1], ls="dashed", color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_2.plot(q_2, cdf_2[1], ls="dashed", color="orange")
    # ax_2.plot(q_0, cdf_0[2], ls="dashed", color="green")
    ax_2.plot(q_1, cdf_1[2], ls="dashed", color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_2.plot(q_2, cdf_2[2], ls="dashed", color="green")
    # ax_2.plot(q_0, cdf_0[3], ls="dashed", color="blue")
    ax_2.plot(q_1, cdf_1[3], ls="dashed", color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_2.plot(q_2, cdf_2[3], ls="dashed", color="blue")
    # ax_2.plot(q_0, cdf_0[4], ls="dashed", color="magenta")
    ax_2.plot(q_1, cdf_1[4], ls="dashed", color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_2.plot(q_2, cdf_2[4], ls="dashed", color="magenta")
    ax_2.set_ylim(-0.1, 1.1)
    ax_2.set_ylabel(r"$F_{q|P, M_{1}}$")
    fig.savefig("moe2017_mass_ratio_pdf.pdf")
    fig.savefig("moe2017_mass_ratio_pdf.jpg")
    plt.show()

    # Test class methods: PPF (twin excess)
    p = np.linspace(0., 1., n)
    ppf = rv.ppf(p)

    fig, ax = plot.plot()
    ax.plot(p, ppf[0], color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax.plot(p, ppf[1], color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax.plot(p, ppf[2], color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax.plot(p, ppf[3], color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax.plot(p, ppf[4], color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax.legend(frameon=False)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
    fig.savefig("moe2017_mass_ratio_ppf.pdf")
    fig.savefig("moe2017_mass_ratio_ppf.jpg")
    plt.show()

    # Test class methods: PDF and CDF (no twin excess)
    primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
    print("primary_mass =", primary_mass)
    log10_excess_twin_period = _moe2017_log10_excess_twin_period(primary_mass)
    print("log10_excess_twin_period =", log10_excess_twin_period)
    
    log10_period = 0.2    
    # log10_period = log10_excess_twin_period + 1.
    rv = moe2017(10.**log10_period, primary_mass.reshape([-1, 1]))

    n = 100
    q_0_pdf = np.linspace(0., 0.1, n)[:-1]
    q_1_pdf = np.linspace(0.1, 0.95, n)
    q_2_pdf = np.linspace(0.95, 1., n)[1:]
    q_3_pdf = np.linspace(1., 1.1, n)[1:]    
    pdf_0 = rv.pdf(q_0_pdf)
    pdf_1 = rv.pdf(q_1_pdf)
    pdf_2 = rv.pdf(q_2_pdf)
    pdf_3 = rv.pdf(q_3_pdf)
    
    q_0_cdf = np.linspace(0., 0.1, n)[:-1]
    q_1_cdf = np.linspace(0.1, 1., n)
    q_2_cdf = np.linspace(1., 1.1, n)[1:]
    cdf_0 = rv.cdf(q_0_cdf)
    cdf_1 = rv.cdf(q_1_cdf)
    cdf_2 = rv.cdf(q_2_cdf)

    pdf_open_dots_x = (
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.95, 0.95, 0.95, 0.95, 0.95,
        1., 1., 1., 1., 1.
    )
    pdf_open_dots_y = (
        0., 0., 0., 0., 0.,
        *pdf_2[:, 0],
        0., 0., 0., 0., 0.
    )
    pdf_closed_dots_x = (
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.95, 0.95, 0.95, 0.95, 0.95,
        1., 1., 1., 1., 1.
    )
    pdf_closed_dots_y = (*pdf_1[:, 0], *pdf_1[:, -1], *pdf_2[:,-1])
        
    fig, ax_1 = plot.plot()
    ax_2 = ax_1.twinx()
    ax_1.plot(q_0_pdf, pdf_0[0], ls="solid", color="red")
    ax_1.plot(q_1_pdf, pdf_1[0], ls="solid", color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_1.plot(q_2_pdf, pdf_2[0], ls="solid", color="red")
    ax_1.plot(q_3_pdf, pdf_3[0], ls="solid", color="red")
    ax_1.plot(q_0_pdf, pdf_0[1], ls="solid", color="orange")
    ax_1.plot(q_1_pdf, pdf_1[1], ls="solid", color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_1.plot(q_2_pdf, pdf_2[1], ls="solid", color="orange")
    ax_1.plot(q_3_pdf, pdf_3[1], ls="solid", color="orange")
    ax_1.plot(q_0_pdf, pdf_0[2], ls="solid", color="green")
    ax_1.plot(q_1_pdf, pdf_1[2], ls="solid", color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_1.plot(q_2_pdf, pdf_2[2], ls="solid", color="green")
    ax_1.plot(q_3_pdf, pdf_3[2], ls="solid", color="green")
    ax_1.plot(q_0_pdf, pdf_0[3], ls="solid", color="blue")
    ax_1.plot(q_1_pdf, pdf_1[3], ls="solid", color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_1.plot(q_2_pdf, pdf_2[3], ls="solid", color="blue")
    ax_1.plot(q_3_pdf, pdf_3[3], ls="solid", color="blue")
    ax_1.plot(q_0_pdf, pdf_0[4], ls="solid", color="magenta")
    ax_1.plot(q_1_pdf, pdf_1[4], ls="solid", color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_1.plot(q_2_pdf, pdf_2[4], ls="solid", color="magenta")
    ax_1.plot(q_3_pdf, pdf_3[4], ls="solid", color="magenta")
    ax_1.scatter(pdf_closed_dots_x[0::5], pdf_closed_dots_y[0::5],
                 s=2., color="red", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[1::5], pdf_closed_dots_y[1::5],
                 s=2., color="orange", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[2::5], pdf_closed_dots_y[2::5],
                 s=2., color="green", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[3::5], pdf_closed_dots_y[3::5],
                 s=2., color="blue", zorder=np.inf)
    ax_1.scatter(pdf_closed_dots_x[4::5], pdf_closed_dots_y[4::5],
                 s=2., color="magenta", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[0::5], pdf_open_dots_y[0::5],
                 s=2., color="red", facecolor="white", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[1::5], pdf_open_dots_y[1::5],
                 s=2., color="orange", facecolor="white", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[2::5], pdf_open_dots_y[2::5],
                 s=2., color="green", facecolor="white", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[3::5], pdf_open_dots_y[3::5],
                 s=2., color="blue", facecolor="white", zorder=np.inf)
    ax_1.scatter(pdf_open_dots_x[4::5], pdf_open_dots_y[4::5],
                 s=2., color="magenta", facecolor="white", zorder=np.inf)
    ax_1.legend(frameon=False, loc=2)
    ax_1.set_xlabel(r"$q$")
    ax_1.set_ylabel(r"$f_{q|P, M_{1}}$")
    ax_1.set_ylim(-1., 11)
    # ax_2.plot(q_0, cdf_0[0], ls="dashed", color="red")
    ax_2.plot(q_1_cdf, cdf_1[0], ls="dashed", color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_2.plot(q_2_cdf, cdf_2[0], ls="dashed", color="red")
    # ax_2.plot(q_0, cdf_0[1], ls="dashed", color="orange")
    ax_2.plot(q_1_cdf, cdf_1[1], ls="dashed", color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_2.plot(q_2_cdf, cdf_2[1], ls="dashed", color="orange")
    # ax_2.plot(q_0, cdf_0[2], ls="dashed", color="green")
    ax_2.plot(q_1_cdf, cdf_1[2], ls="dashed", color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_2.plot(q_2_cdf, cdf_2[2], ls="dashed", color="green")
    # ax_2.plot(q_0, cdf_0[3], ls="dashed", color="blue")
    ax_2.plot(q_1_cdf, cdf_1[3], ls="dashed", color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_2.plot(q_2_cdf, cdf_2[3], ls="dashed", color="blue")
    # ax_2.plot(q_0, cdf_0[4], ls="dashed", color="magenta")
    ax_2.plot(q_1_cdf, cdf_1[4], ls="dashed", color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_2.plot(q_2_cdf, cdf_2[4], ls="dashed", color="magenta")
    ax_2.set_ylim(-0.1, 1.1)
    ax_2.set_ylabel(r"$F_{q|P, M_{1}}$")
    fig.savefig("moe2017_mass_ratio_pdf_excess.pdf")
    fig.savefig("moe2017_mass_ratio_pdf_excess.jpg")
    plt.show()

    # Test class methods: PPF (twin excess)
    p = np.linspace(0., 1., n)
    ppf = rv.ppf(p)

    fig, ax = plot.plot()
    ax.plot(p, ppf[0], color="red",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax.plot(p, ppf[1], color="orange",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax.plot(p, ppf[2], color="green",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax.plot(p, ppf[3], color="blue",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax.plot(p, ppf[4], color="magenta",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax.legend(frameon=False)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
    fig.savefig("moe2017_mass_ratio_ppf_excess.pdf")
    fig.savefig("moe2017_mass_ratio_ppf_excess.jpg")
    plt.show()

if __name__ == "__main__":
    main()
