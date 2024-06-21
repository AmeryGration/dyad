"""Module providing random variables for mass ratio"""

__all__ = [
    "duquennoy1991",
    "moe2017"
]

import numpy as np
import scipy as sp

_truncnorm = sp.stats.truncnorm


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


_duquennoy1991_loc = 0.23
_duquennoy1991_scale = 0.42
_duquennoy1991_a = (0. - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991_b = (np.inf - _duquennoy1991_loc)/_duquennoy1991_scale
_duquennoy1991 = _truncnorm(
    a=_duquennoy1991_a, b=_duquennoy1991_b, loc=_duquennoy1991_loc,
    scale=_duquennoy1991_scale
)
duquennoy1991 = _duquennoy1991_gen(a=0., b=np.inf, name="duquennoy1991")

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
    condition = [np.isclose(gamma, -1.), ~np.isclose(gamma, -1.)]
    value = [f_a(gamma, delta), f_b(gamma, delta)]
    f = np.select(condition, value)

    # Handle division by zero
    # The formula for the norm is piecewise: specifically, the formula
    # for the case delta == 1 is distinct from that for the case delta
    # != 1. For values of delta close to 1 the formula for the second
    # case is numerically unstable, so I use the formula for the
    # first. The use of np.select does not violate the principle
    # `better to ask forgiveness than permission' since try-except
    # blocks would not handle the numerical instability.
    condition = [np.isclose(delta, -1.), ~np.isclose(delta, -1.)]
    # condition = [delta == -1., delta != -1.]    
    value = [g_a(gamma, delta), g_b(gamma, delta)]
    g = np.select(condition, value)    

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
    condition = [delta == -1., delta != -1.]
    # condition = [np.isclose(delta, -1.), ~np.isclose(delta, -1.)]
    value = [
        f_1(delta, log10_period, primary_mass),
        f_2(delta, log10_period, primary_mass)
    ]
    res = np.select(condition, value)

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

    # print("log10_period")
    # print(log10_period)
    # print("primary_mass")
    # print(primary_mass)

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
    def _argcheck(self, log10_period, primary_mass):
        res = (
            (0.2 <= log10_period) & (log10_period <= 8.)
            & (0. <= primary_mass) & (primary_mass < np.inf)
        )

        return res

    def _pdf(self, x, log10_period, primary_mass):
        def f_1(x, norm, gamma, delta):
            res = norm*0.3**(delta - gamma)*x**gamma

            return res

        def f_2(x, norm, delta):
            res = norm*x**delta

            return res

        def f_3(x, norm, delta):
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

        log10_period = np.asarray(log10_period)
        primary_mass = np.asarray(primary_mass)

        gamma = _moe2017_gamma(log10_period, primary_mass)
        delta = _moe2017_delta(log10_period, primary_mass)
        norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)

        condition = [
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        ]
        value = [
            f_1(x, norm, gamma, delta),
            f_2(x, norm, delta),
            f_3(x, norm, delta)
        ]
        res = np.select(condition, value)
        
        return res

    def _cdf(self, x, log10_period, primary_mass):
        def g_1(x, gamma, delta):
            def g_1a(x, gamma, delta):
                res = norm*0.3**(delta - gamma)*(np.log(x) - np.log(0.1))

                return res

            def g_1b(x, gamma, delta):
                res = norm*(
                    0.3**(delta - gamma)
                    *(x**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )

                return res

            condition = [
                np.isclose(gamma, -1.),
                ~np.isclose(gamma, -1.)
            ]
            value = [
                g_1a(x, gamma, delta),
                g_1b(x, gamma, delta)
            ]
            
            res = np.select(condition, value)

            return res
        
        def g_2(x, gamma, delta):
            def g_2a(x):
                res = norm*(np.log(x) - np.log(0.3))

                return res

            def g_2b(x, gamma, delta):
                res = (
                    g_1(0.3, gamma, delta)
                    + norm*((x**(delta + 1.) - 0.3**(delta + 1.))/(delta + 1.))
                )

                return res

            condition = [delta == -1., delta != -1.]
            # condition = [
            #     np.isclose(delta, -1.),
            #     np.isclose(delta, -1.)
            # ]
            value = [
                g_2a(x),
                g_2b(x, gamma, delta)
            ]
            res = np.select(condition, value)

            return res

        def g_3(x, gamma, delta, log10_period, primary_mass):
            def g_3a(x):
                res = np.log(x) - np.log(0.95)

                return res

            def g_3b(x, gamma, delta, primary_mass):
                res = (
                    g_2(0.95, gamma, delta)
                    + norm*(
                        _moe2017_twin_excess_constant(
                            delta, log10_period, primary_mass
                        )
                        *(x - 0.95)
                        + (x**(delta + 1.) - 0.95**(delta + 1.))/(delta + 1.)
                    )
                )

                return res

            condition = [delta == -1., delta != -1.]
            # condition = [
            #     np.isclose(delta, -1.),
            #     np.isclose(delta, -1.)
            # ]
            value = [
                g_3a(x),
                g_3b(x, gamma, delta, primary_mass)
            ]
            res = np.select(condition, value)

            return res

        gamma = _moe2017_gamma(log10_period, primary_mass)
        delta = _moe2017_delta(log10_period, primary_mass)
        norm = _moe2017_norm(gamma, delta, log10_period, primary_mass)

        condition = [
            (0.1 <= x) & (x <= 0.3),
            (0.3 < x) & (x <= 0.95),
            (0.95 < x) & (x <= 1.)
        ]
        value = [
            g_1(x, gamma, delta),
            g_2(x, gamma, delta),
            g_3(x, gamma, delta, log10_period, primary_mass)
        ]
        res = np.select(condition, value)

        return res

    def _ppf(self, q, log10_period, primary_mass):
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

        def f_3(q, log10_period, primary_mass):
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

            res = np.vectorize(fsolve)(q, log10_period, primary_mass)

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

        condition = [
            (0. < q)
            & (q <= self.cdf(0.3, log10_period, primary_mass)),
            (self.cdf(0.3, log10_period, primary_mass) < q)
            & (q <= self.cdf(0.95, log10_period, primary_mass)),
            (self.cdf(0.95, log10_period, primary_mass) < q)
            & (q <= 1.)
        ]
        value = [
            f_1(q, gamma, delta, norm),
            f_2(q, gamma, delta, norm),
            f_3(q, log10_period, primary_mass),
        ]
        res = np.select(condition, value)

        return res


moe2017 = _moe2017_gen(a=0.1, b=1., name="moe2017")
