__all__ = [
    "truncnorm",
    "duquennoy1991",
    "moe2017"
]

import numpy as np
import scipy as sp

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
    num = 1. 
    denom = (
        0.3**(delta - gamma)
        *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
        /(gamma + 1.)
        + (1. - 0.3**(delta + 1.))
        /(delta + 1.)
        + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    )
    res = num/denom

    return res

def _moe2017_twin_excess_constant(delta, period, primary_mass):
    """Return the twin excess constant"""
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
        (0. < log10_period)
        & (log10_period < 1.),
        (1. <= log10_period)
        & (log10_period < _moe2017_log10_excess_twin_period(primary_mass)),
        (_moe2017_log10_excess_twin_period(primary_mass) <= log10_period)
        & (log10_period < np.inf)
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
        (0. < primary_mass) & (primary_mass < 6.5),
        (6.5 <= primary_mass) & (primary_mass < np.inf)
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
            (0.2 <= log10_period) & (log10_period < 2.5),
            (2.5 <= log10_period) & (log10_period < 5.5),
            (5.5 <= log10_period) & (log10_period <= 8)
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
            (0.1 <= log10_period) & (log10_period < 1.),
            (1. <= log10_period) & (log10_period < 3.),
            (3. <= log10_period) & (log10_period < 5.6),
            (5.6 <= log10_period) & (log10_period <= 8.)
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
        (0.8 <= primary_mass) & (primary_mass < 1.2),
        (1.2 <= primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass < 6.),
        (6. <= primary_mass) & (primary_mass < np.inf)
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
            (0.2 <= log10_period) & (log10_period < 5.),
            (5. <= log10_period) & (log10_period <= 8.)
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
            (0.2 <= log10_period) & (log10_period < 1.),
            (1 <= log10_period) & (log10_period < 4.5),
            (4.5 <= log10_period) & (log10_period < 6.5),
            (6.5 <= log10_period) & (log10_period <= 8)
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
            (0.1 <= log10_period) & (log10_period < 1.),
            (1. <= log10_period) & (log10_period < 2.),
            (2. <= log10_period) & (log10_period < 4.),
            (4. <= log10_period) & (log10_period <= 8.)
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
        (0.8 <= primary_mass) & (primary_mass < 1.2),
        (1.2 <= primary_mass) & (primary_mass < 3.5),
        primary_mass == 3.5,
        (3.5 < primary_mass) & (primary_mass < 6.),
        (6. <= primary_mass) & (primary_mass < np.inf)
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
    The probability density function for `moe1991` is:

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
    def _argcheck(self, period, primary_mass):
        return (0. <= period) & (0. <= primary_mass)

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
            (0.1 <= x) & (x < 0.3),
            (0.3 <= x) & (x < 0.95),
            (0.95 <= x) & (x <= 1.)
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
            res = norm*(
                0.3**(delta - gamma)
                *(x**(gamma + 1.) - 0.1**(gamma + 1.))
                /(gamma + 1.)
            )

            return res

        def g_2(x, gamma, delta):
            res = (
                g_1(0.3, gamma, delta)
                + norm*((x**(delta + 1.) - 0.3**(delta + 1.))/(delta + 1.))
            )

            return res

        def g_3(x, gamma, delta):
            res = g_2(0.95, gamma, delta) + norm*(
                _moe2017_twin_excess_constant(delta, period, primary_mass)
                *(x - 0.95)
                + (x**(delta + 1.) - 0.95**(delta + 1.))/(delta + 1.)
            )

            return res

        gamma = _moe2017_gamma(np.log10(period), primary_mass)
        delta = _moe2017_delta(np.log10(period), primary_mass)
        norm = _moe2017_norm(gamma, delta, period, primary_mass)

        condition = [
            (0.1 <= x) & (x < 0.3),
            (0.3 <= x) & (x < 0.95),
            (0.95 <= x) & (x <= 1.)
        ]
        value = [
            g_1(x, gamma, delta),
            g_2(x, gamma, delta),
            g_3(x, gamma, delta)
        ]
        res = np.select(condition, value)

        return res

    def _ppf(self, q, period, primary_mass):
        def f_1(q):#, gamma, delta, norm):
            q = q/norm
            base = (gamma + 1.)*q/0.3**(delta - gamma) + 0.1**(gamma + 1.)
            res = base**(1./(gamma + 1.))

            return res

        def f_2(q):#, gamma, delta, norm):
            q = q/norm
            a = (
                0.3**(delta - gamma)
                *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
                /(gamma + 1.)
            )
            base = (q - a)*(delta + 1.) + 0.3**(delta + 1.)
            res = base**(1./(delta + 1.))

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
            (0 <= q)
            & (q < self.cdf(0.3, period, primary_mass)),
            (self.cdf(0.3, period, primary_mass) <= q)
            & (q < self.cdf(0.95, period, primary_mass)),
            (self.cdf(0.95, period, primary_mass) <= q)
            & (q <= 1.)
        ]
        value = [
            f_1(q),#, gamma, delta, norm),
            f_2(q),#, gamma, delta, norm),
            f_3(q, period, primary_mass),#, gamma, delta, norm)
        ]
        res = np.select(condition, value)

        return res


moe2017 = _moe2017_gen(a=0.1, b=1., name="moe2017")

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from scipy.integrate import quad

    mpl.style.use("sm")

    N_SAMPLE = 10_000

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

    fig, ax = plt.subplots()
    ax.plot(q, pdf)
    # ax.stairs(counts, edges)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$f_{q}$")
    ax.set_ylim(0., 1.5)
    fig.savefig("duquennoy1991_mass_ratio_pdf.pdf")
    fig.savefig("duquennoy1991_mass_ratio_pdf.jpg")
    fig.show()

    fig, ax = plt.subplots()
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
    primary_mass = (1., 3.5, 7., 12.5, 25.)
    log10_twin_excess_period = _moe2017_log10_excess_twin_period(primary_mass)
    log10_period = np.linspace(0.2, 8., 250)
    period = 10.**log10_period

    F_twin_1 = _moe2017_twin_excess_fraction(period, primary_mass[0])
    F_twin_2 = _moe2017_twin_excess_fraction(period, primary_mass[1])
    F_twin_3 = _moe2017_twin_excess_fraction(period, primary_mass[2])
    F_twin_4 = _moe2017_twin_excess_fraction(period, primary_mass[3])
    F_twin_5 = _moe2017_twin_excess_fraction(period, primary_mass[4])

    delta = _moe2017_delta(log10_period, primary_mass[0])
    c_twin_1 = _moe2017_twin_excess_constant(delta, period, primary_mass[0])
    c_twin_2 = _moe2017_twin_excess_constant(delta, period, primary_mass[1])
    c_twin_3 = _moe2017_twin_excess_constant(delta, period, primary_mass[2])
    c_twin_4 = _moe2017_twin_excess_constant(delta, period, primary_mass[3])
    c_twin_5 = _moe2017_twin_excess_constant(delta, period, primary_mass[4])

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.4/2.54, 12./2.54))
    ax[0].plot(log10_period, F_twin_1, color="red", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[0].plot(log10_period, F_twin_2, color="orange", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[0].plot(log10_period, F_twin_3, color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[0].plot(log10_period, F_twin_4, color="blue", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[0].plot(log10_period, F_twin_5, color="magenta", ls="solid",
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
    ax[0].set_xlim(0., 8.)
    ax[0].set_ylim(-0.05, 0.45)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel(r"$F_{\text{twin}}$")
    ax[1].plot(log10_period, c_twin_1, color="red", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[1].plot(log10_period, c_twin_2, color="orange", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[1].plot(log10_period, c_twin_3, color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[1].plot(log10_period, c_twin_4, color="blue", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[1].plot(log10_period, c_twin_5, color="magenta", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax[1].vlines(log10_twin_excess_period, -2., 10., ls="dashed")
    ax[1].set_xlim(0., 8.)
    ax[1].set_ylim(-2., 10.)
    ax[1].set_xlabel(r"$\log(P)$")
    ax[1].set_ylabel(r"$c_{\text{twin}}$")
    fig.savefig("moe2017_twin_excess.pdf")
    fig.savefig("moe2017_twin_excess.jpg")
    fig.show()

    # Test utility functions: gamma, delta
    gamma_1 = _moe2017_gamma(log10_period, primary_mass[0])
    gamma_2 = _moe2017_gamma(log10_period, primary_mass[1])
    gamma_3 = _moe2017_gamma(log10_period, primary_mass[2])
    gamma_4 = _moe2017_gamma(log10_period, primary_mass[3])
    gamma_5 = _moe2017_gamma(log10_period, primary_mass[4])

    delta_1 = _moe2017_delta(log10_period, primary_mass[0])
    delta_2 = _moe2017_delta(log10_period, primary_mass[1])
    delta_3 = _moe2017_delta(log10_period, primary_mass[2])
    delta_4 = _moe2017_delta(log10_period, primary_mass[3])
    delta_5 = _moe2017_delta(log10_period, primary_mass[4])

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.4/2.54, 10./2.54))
    ax[0].plot(log10_period, delta_1, color="red", ls="solid")
    ax[0].plot(log10_period, delta_2, color="orange", ls="solid")
    ax[0].plot(log10_period, delta_3, color="green", ls="solid")
    ax[0].plot(log10_period, delta_4, color="blue", ls="solid")
    ax[0].plot(log10_period, delta_5, color="magenta", ls="solid")
    ax[0].set_xlim(0., 8.)
    ax[0].set_ylim(-3., 0.5)
    ax[0].set_ylabel(r"$\delta$")
    ax[1].plot(log10_period, gamma_1, color="red", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax[1].plot(log10_period, gamma_2, color="orange", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax[1].plot(log10_period, gamma_3, color="green", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax[1].plot(log10_period, gamma_4, color="blue", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax[1].plot(log10_period, gamma_5, color="magenta", ls="solid",
              label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax[1].legend(frameon=False)
    ax[1].set_xlim(0., 8.)
    ax[1].set_ylim(-2.5, 1.5)
    ax[1].set_xlabel(r"$\log(P)$")
    ax[1].set_ylabel(r"$\gamma$")
    fig.savefig("moe2017_gamma_delta.pdf")
    fig.savefig("moe2017_gamma_delta.jpg")
    fig.show()

    # Test utility functions: twin period
    primary_mass = np.linspace(0., 10., 500)
    log10_period_twin = _moe2017_log10_excess_twin_period(primary_mass)

    fig, ax = plt.subplots()
    ax.plot(primary_mass[1:], log10_period_twin[1:])
    ax.set_xlim(0., 10.)
    ax.set_ylim(0., 10.)
    ax.set_xlabel(r"$M_{1}$")
    ax.set_ylabel(r"$\log_{10}(P_{\text{twin}})$")
    fig.savefig("moe2017_twin_excess_period.pdf")
    fig.savefig("moe2017_twin_excess_period.jpg")
    fig.show()

    primary_mass = 1.
    print("primary_mass =", primary_mass)
    log10_excess_twin_period = _moe2017_log10_excess_twin_period(primary_mass)
    print("log10_excess_twin_period =", log10_excess_twin_period)

    # Ensure there is a twin excess
    log10_period_excess = log10_excess_twin_period - 1.
    print("log10_period =", log10_period_excess)
    period_excess = 10**log10_period_excess
    print("period =", period_excess)

    # Ensure there is no twin excess
    log10_period = log10_excess_twin_period + 1.
    print("log10_period =", log10_period)
    period = 10**log10_period
    print("period =", period)

    F_twin = _moe2017_twin_excess_fraction(period_excess, primary_mass)
    print("F_twin =", F_twin)

    delta = _moe2017_delta(log10_period_excess, primary_mass)
    c_twin = _moe2017_twin_excess_constant(delta, period_excess, primary_mass)
    print("c_twin =", c_twin)

    gamma = _moe2017_gamma(log10_period_excess, primary_mass)
    print("gamma =", gamma)
    delta = _moe2017_delta(log10_period_excess, primary_mass)
    print("delta =", delta)
    norm = _moe2017_norm(gamma, delta, period_excess, primary_mass)
    print("norm =", norm)

    # Test methods: pdf, cdf, ppf
    rv_excess = moe2017(period_excess, primary_mass)
    rv = moe2017(period, primary_mass)
    q_0 = np.linspace(0., 0.1, endpoint=False)
    q_1 = np.linspace(0.1, 0.95, 500, endpoint=False)
    q_2 = np.linspace(0.95, 1., endpoint=False)
    q_3 = np.linspace(1., 1.1)
    pdf_excess_0 = rv_excess.pdf(q_0)
    pdf_excess_1 = rv_excess.pdf(q_1)
    pdf_excess_2 = rv_excess.pdf(q_2)
    pdf_excess_3 = rv.pdf(q_3)
    pdf_0 = rv.pdf(q_0)
    pdf_1 = rv.pdf(q_1)
    pdf_2 = rv.pdf(q_2)
    pdf_3 = rv.pdf(q_3)
    q = np.linspace(0., 1.1, 50)
    cdf_excess = rv_excess.cdf(q)
    cdf = rv.cdf(q)
    p = np.linspace(0., 1.1, 50)
    ppf_excess = rv_excess.ppf(p)
    ppf = rv.ppf(p)

    # I = [quad(rv.pdf, 0., q_i)[0] for q_i in q]

    # I = quad(rv.pdf, 0., 0.95)[0]
    # J = quad(rv.pdf, 0.95, 1.)[0]
    # print("F(0.95) =", rv.cdf(0.95))
    # print("I(0.95) =", I)
    # print("F(1) =", rv.cdf(1.))
    # print("I(1) =", I + J)
    # print("F(1) - F(0.95) =", rv.cdf(1.) - rv.cdf(0.95))
    # print("I(1) - I(0.95) =", J)

    # mass_ratio_excess = rv_excess.rvs(size=N_SAMPLE)
    # mass_ratio = rv.rvs(size=N_SAMPLE)
    # counts_excess, edges_excess = np.histogram(
    #     mass_ratio_excess, bins=np.linspace(0.1, 1., 37), density=True
    # )
    # counts, edges = np.histogram(
    #     mass_ratio, bins=np.linspace(0.1, 1., 37), density=True
    # )

    open_dots_x = (0.1, 0.95, 1.)
    open_dots_y = (
       0.,
       rv_excess.pdf(0.95)
       - _moe2017_norm(gamma, delta, period_excess, primary_mass)
       *_moe2017_twin_excess_constant(delta, period_excess, primary_mass),
       0.
    )
    closed_dots_x = (
       0.1,
       0.1,
       0.95,
       0.95,
       1.,
       1.
    )
    closed_dots_y = (
       rv_excess.pdf(0.1),
       rv.pdf(0.1),
       rv_excess.pdf(0.95),
       rv.pdf(0.95),
       rv_excess.pdf(1.),
       rv.pdf(1.)
    )

    label_1 = r"$M_{1} = 1, \log_{10}(P) = \log_{10}(P_{\text{twin}}(1)) - 1$"
    label_2 = r"$M_{1} = 1, \log_{10}(P) = \log_{10}(P_{\text{twin}}(1)) + 1$"

    fig, ax = plt.subplots()
    ax.plot(q_0, pdf_excess_0, ls="solid", label=label_1, zorder=0)
    ax.plot(q_1, pdf_excess_1, ls="solid", zorder=0)
    ax.plot(q_2, pdf_excess_2, ls="solid", zorder=0)
    ax.plot(q_3[1:], pdf_excess_3[1:], ls="solid", zorder=0)
    ax.plot(q_1, pdf_1, ls="dashed", label=label_2, zorder=0)
    ax.plot(q_2, pdf_2, ls="dashed", zorder=0)
    ax.plot(q_3[1:], pdf_3[1:], ls="solid", zorder=0)
    ax.scatter(open_dots_x, open_dots_y, s=2, facecolors="white",
                 edgecolors="k", zorder=1)
    ax.scatter(closed_dots_x, closed_dots_y, s=2, facecolors="k",
                 edgecolors="k", zorder=1)
    # ax.stairs(counts_excess, edges_excess)
    # ax.stairs(counts, edges, ls="dotted")
    ax.set_xlim(0., 1.1)
    ax.set_ylim(-0.25, 2.5)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$f_{q|P, M_{1}}(q|P, M_{1})$")
    ax.legend(frameon=False)
    fig.savefig("moe2017_mass_ratio_pdf.pdf")
    fig.savefig("moe2017_mass_ratio_pdf.jpg")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(q, cdf_excess, ls="solid", label=r"$y = F_{q}$")
    ax.plot(q, ppf_excess, ls="dashed", label=r"$y = F_{q}^{-1}$")
    ax.legend(frameon=False, loc=2)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$y$")
    fig.savefig("moe2017_mass_ratio_cdf_excess.pdf")
    fig.savefig("moe2017_mass_ratio_cdf_excess.jpg")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(q, cdf, ls="solid", label=r"$y = F_{q}$")
    ax.plot(q, ppf, ls="dashed", label=r"$y = F_{q}^{-1}$")
    ax.legend(frameon=False, loc=2)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$y$")
    fig.savefig("moe2017_mass_ratio_cdf.pdf")
    fig.savefig("moe2017_mass_ratio_cdf.jpg")
    fig.show()

    def test_single_shape_param(rv):
       #####################################################################
       # Single argument
       #####################################################################
       x = 0.5
       f = rv.pdf(x)
       print(f)
       F = rv.cdf(x)
       print(F)
       q = 0.5
       F_inv = rv.ppf(q)
       print(F_inv)
       sample = rv.rvs()
       print(sample)
       sample = rv.rvs(size=1)
       print(sample)

       #####################################################################
       # Multiple argument
       #####################################################################
       x = np.linspace(0., 1., 500)
       f = rv.pdf(x)
       print(f)
       F = rv.cdf(x)
       print(F)
       q = np.linspace(0., 1., 500)
       F_inv = rv.ppf(q)
       print(F_inv)

       fig, ax = plt.subplots()
       ax.plot(x, f)
       ax.plot(x, F)
       ax.plot(q, F_inv)
       plt.show()

       res = rv.rvs(size=1_000)
       print(res)

       counts, edges = np.histogram(res, bins=50, density=True)

       fig, ax = plt.subplots()
       ax.stairs(counts, edges)
       plt.show()

    def test_multiple_shape_params(rv):
       #####################################################################
       # Single argument
       #####################################################################
       x = 0.5
       f = rv.pdf(x)
       print(f)
       F = rv.cdf(x)
       print(F)
       q = 0.5
       F_inv = rv.ppf(q)
       print(F_inv)
       sample = rv.rvs()
       print(sample)

       counts, edges = np.histogram(res, bins=50, density=True)

       fig, ax = plt.subplots()
       ax.stairs(counts, edges)
       plt.show()

    rv = moe2017(10., 10.)
    test_single_shape_param(rv)

    period = [10., 50.]
    primary_mass = [1., 10.]
    rv = moe2017([10., 50.], [1., 10.])
    test_multiple_shape_params(rv)
