__all__ = [
    "uniform",
    "powerlaw",
    "thermal",
    "duquennoy1991",
]

import numpy as np
import scipy as sp

uniform = sp.stats._continuous_distns.uniform_gen(a=0., b=1., name="uniform")

powerlaw = sp.stats._continuous_distns.powerlaw_gen(
    a=0., b=1., name="powerlaw"
)


class _duquennoy1991_gen(sp.stats.rv_continuous):
    r"""The Duquennoy and Mayor (1991) eccentricity random variable

    %(before_notes)s

    Notes
    -----

    """
    # Accept shape parameter
    # Can call a tuple!
    # >>> sp.stats.norm(loc=(1., 2.)).pdf(1.)
    # array([0.39894228, 0.24197072])
    # Or
    # >>> sp.stats.norm(loc=(1., 2.)).pdf((1., 2.))
    # array([0.39894228, 0.39894228])
    # This speeds things up!

    # Inherit the rv_continuous class but override its methods with
    # those of a frozen rv.
    def _argcheck(self, s):
        return 11. < s

    def _pdf(self, x, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.pdf(x),
            _duquennoy1991_f2.pdf(x)
        )

    def _cdf(self, x, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.cdf(x),
            _duquennoy1991_f2.cdf(x)
        )

    def _ppf(self, q, s):
        return np.where(
            (11. < s) & (s <= 1000.),
            _duquennoy1991_f1.ppf(q),
            _duquennoy1991_f2.ppf(q)
        )


# Duquennoy and Mayor (1991) tight binaries: truncated normal
loc = 0.27
scale = 0.13
a = (0. - loc)/scale
b = (np.inf - loc)/scale
_duquennoy1991_f1 = sp.stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
# Duquennoy and Mayor (1991) wide binaries: thermal 
_duquennoy1991_f2 = sp.stats.powerlaw(2.)
# Duquennoy and Mayor (1991) all binaries: conditional
duquennoy1991 = _duquennoy1991_gen(a=0., b=1., name="duquennoy1991")


class _thermal_gen(sp.stats.rv_continuous):
    r"""The thermal eccentricity random variable

    %(before_notes)s

    Notes
    -----

    """
    def _pdf(self, x):
        return _thermal.pdf(x)

    def _cdf(self, x):
        return _thermal.cdf(x)

    def _ppf(self, q):
        return _thermal.ppf(q)


_thermal = sp.stats.powerlaw(2.)
thermal = _thermal_gen(a=0., b=1., name="thermal")

def _moe2017_norm(log10_period, primary_mass):
    """Return the normalization constant"""
    e_max = 1 - (0.5*10.**log10_period)**(-2./3.)
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
        res = 0.6 - 0.7/(log10_period - 0.5)

        return res

    condition = [
        (0.5 <= log10_period) & (log10_period <= 6.),
        (6. < log10_period) & (log10_period <= 8.),

    ]
    value = [
        f_1(log10_period, primary_mass),
        f_2(log10_period, primary_mass),
    ]
    res = np.select(condition, value)

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
        res = 0.9 - 0.2/(log10_period - 0.5)

        return res

    condition = [
        (0.5/0.9 <= log10_period) & (log10_period <= 5.),        
        (5. < log10_period) & (log10_period <= 8.),

    ]
    value = [
        f_1(log10_period, primary_mass),
        f_2(log10_period, primary_mass),
    ]
    res = np.select(condition, value)

    return res

def _moe2017_eta(log10_period, primary_mass):
    condition = [
        (0.8 <= primary_mass) & (primary_mass <= 3.),
        (3. <= primary_mass) & (primary_mass <= 7.),
        (7. <= primary_mass) & (primary_mass < np.inf),
    ]
    value = [
        _moe2017_eta_1(log10_period, primary_mass),
        _moe2017_eta_2(log10_period, primary_mass),
        _moe2017_eta_3(log10_period, primary_mass),
    ]
    res = np.select(condition, value)

    return res

class _moe2017_gen(sp.stats.rv_continuous):
    r"""The Moe and Stefano (2017) eccentricity random variable

    %(before_notes)s

    Notes
    -----

    """
    def _argcheck(self, log10_period, primary_mass):
        res = _moe2017_eta(log10_period, primary_mass) >= 0.

        return res

    def _get_support(self, log10_period, primary_mass):
        e_min = 0.
        e_max = 1 - (0.5*10.**log10_period)**(-2./3.)
        res = [e_min, e_max]

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


moe2017 = _moe2017_gen(a=0., b=1., name="moe2017")
