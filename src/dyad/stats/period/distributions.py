"""Module providing random variables for period"""

__all__ = [
    "trunclognorm",
    "duquennoy1991",
    "moe2017",
]

import numpy as np
import scipy as sp

from dyad.stats import mass_ratio

def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    cond = np.asarray(cond)
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    args = np.broadcast_arrays(cond, *arrays)
    cond, arrays = args[0], args[1:]
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out

def _lognorm_logpdf(x, s):
    # Replica of ~scipy.stats._continuous_distns.cd._lognorm_logpdf~,
    # which is called by ~scipy.stats.lognorm~.
    return _lazywhere(
        x != 0,
        (x, s),
        lambda x, s: -np.log(x)**2./(2.*s**2.) - np.log(s*x*np.sqrt(2*np.pi)),
        -np.inf
    )


class _trunclognorm_gen(sp.stats.rv_continuous):
    r"""The truncated lognormal random variable

    %(before_notes)s

    See Also
    --------
    scipy.stats.lognorm, scipy.stats.truncnorm

    Notes
    -----
    The probability density function for `trunclognorm` is:

    .. math::

        f(x, a, b, s) =
        \dfrac{\exp\left(-\dfrac{\log^{2}(x)}{2s^{2}}\right)}
        {\erf\left(\dfrac{\log(b)}{s}\right)
        - \erf\left(\dfrac{\log(a)}{s}\right)}

    where :math:`a < x < 0`, :math:`0 < a < b`, and :math:`s > 0` [1]_.

    `trunclognorm` takes :math:`a`, :math:`b`, and :math:`c` as shape
    parameters.

    Notice that the truncation values, :math:`a` and :math:`b`, are
    defined in standardized form:

    .. math::

        a = (u_{\mathrm{l}} - \mathrm{loc})/\mathrm{scale},
        b = (u_{\mathrm{r}} - \mathrm{loc})/\mathrm{scale}

    where :math:`u_{\mathrm{l}}` and :math:`u_{mathrm{r}` are the
    specific left and right truncation values, respectively. In other
    words, the support of the distribution becomes
    :math:`(a*\mathrm{scale} + \mathrm{loc}) < x <= (b*\mathrm{scale}
    + \mathrm{loc})` when :math:`\mathrm{loc}` and/or
    :math:`\mathrm{scale}` are provided.

    %(after_notes)s

    Suppose a normally distributed random variable ``X`` has mean
    ``mu`` and standard deviation ``sigma``. Then ``Y = exp(X)`` is
    lognormally distributed with ``s = sigma`` and ``scale =
    exp(mu)``. To change the base of the lognormal distribution from
    ``e`` to base ``b`` multiply ``mu`` and ``sigma`` by ``ln(b)``.

    References
    ----------
    .. [1] Reference

    %(example)s

    """
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


trunclognorm = _trunclognorm_gen(name="trunclognorm")


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
    # Check 0 < a < b.
    def _pdf(self, x):
        return _duquennoy1991.pdf(x)

    def _cdf(self, x):
        return _duquennoy1991.cdf(x)

    def _ppf(self, q):
        return _duquennoy1991.ppf(q)


# Duquennoy and Mayor (1991) period: truncated lognormal
mu = np.exp(np.log(10.)*4.8)
sigma = np.log(10.)*2.3
loc = 0.
scale = mu
s = sigma
a = 10.**-2./scale
b = 10.**12./scale
_duquennoy1991 = trunclognorm(s=s, a=a, b=b, scale=scale)
duquennoy1991 = _duquennoy1991_gen(
    a=10.**-2.3, b=10.**12., name="duquennoy1991"
)

def _moe2017_norm(primary_mass):
    res = 1.

    return res

def _moe2017_c_1(log10_primary_mass):
    res = (
        0.07*log10_primary_mass**2.
        + 0.04*log10_primary_mass
        + 0.020
    )

    return res

def _moe2017_c_2(log10_primary_mass):
    res = (
        - 0.06*log10_primary_mass**2.
        + 0.03*log10_primary_mass
        + 0.0064
    )

    return res

def _moe2017_c_3(log10_primary_mass):
    res = (
        0.13*log10_primary_mass**2.
        + 0.01*log10_primary_mass
        + 0.0136
    )

    return res

def _moe2017_c_4(log10_primary_mass):
    res = 0.018

    return res

def _moe2017_c_5(log10_primary_mass):
    res = (
        0.01*log10_primary_mass**2.
        + 0.07*log10_primary_mass
        - 0.0096
    )

    return res

def _moe2017_c_6(log10_primary_mass):
    res = (
        0.03*log10_primary_mass**2./2.1
        - 0.12*log10_primary_mass/2.1
        + 0.0264/2.1
    )
    
    return res

def _moe2017_c_7(log10_primary_mass):
    res = (
        - 0.081*log10_primary_mass**2./2.1
        + 0.555*log10_primary_mass/2.1
        + 0.0186/2.1
    )
    
    return res

def _moe2017_c_8(log10_primary_mass):
    res = (
        np.exp(1.65)
        *(
            0.04*log10_primary_mass**2.
            - 0.05*log10_primary_mass
            + 0.078
        )
    )

    return res


class _moe2017_gen(sp.stats.rv_continuous):
    r"""The Moe and Stefano (2017) period random variable

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
    def _argcheck(self, primary_mass):
        return (0. <= primary_mass) & (primary_mass < np.inf)

    def _pdf(self, x, primary_mass):
        def f_1(x, log10_primary_mass):
            res = _moe2017_c_1(log10_primary_mass)

            return res

        def f_2(x, log10_primary_mass):
            res = (
                _moe2017_c_2(log10_primary_mass)*x
                + _moe2017_c_3(log10_primary_mass)
            )

            return res

        def f_3(x, log10_primary_mass):
            res = (
                _moe2017_c_4(log10_primary_mass)*x
                + _moe2017_c_5(log10_primary_mass)
            )

            return res

        def f_4(x, log10_primary_mass):
            res = (
                _moe2017_c_6(log10_primary_mass)*x
                + _moe2017_c_7(log10_primary_mass)
            )

            return res

        def f_5(x, log10_primary_mass):
            res = _moe2017_c_8(log10_primary_mass)*np.exp(-0.3*x)

            return res

        x = np.asarray(x)
        primary_mass = np.asarray(primary_mass)
        log10_primary_mass = np.log10(primary_mass)

        rv_mass_ratio = mass_ratio.moe2017(
            10.**x, primary_mass.reshape([-1, 1])
        )
        correction_factor = 1./(1. - rv_mass_ratio.cdf(0.3))
        
        condition = [
            (0.2 <= x) & (x <= 1.),
            (1. < x) & (x <= 2.),
            (2. < x) & (x <= 3.4),
            (3.4 < x) & (x <= 5.5),
            (5.5 < x) & (x <= 8.),
        ]
        value = [
            correction_factor*f_1(x, log10_primary_mass),
            correction_factor*f_2(x, log10_primary_mass),
            correction_factor*f_3(x, log10_primary_mass),
            correction_factor*f_4(x, log10_primary_mass),
            correction_factor*f_5(x, log10_primary_mass),
        ]
        res = np.select(condition, value)
        
        return res

    def _cdf(self, x, primary_mass):
        res = 0.

        return res

    def _ppf(self, q, primary_mass):
        res = 0.

        return res


moe2017 = _moe2017_gen(a=0.2, b=8., name="moe2017")

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use("sm")

    # #########################################################################
    # # Plot Duquennoy 1991
    # #########################################################################
    # periods = duquennoy1991.rvs(size=100_000)
    # counts, edges = np.histogram(periods, bins=np.linspace(0., 10_000., 25),
    #                              density=True)
    # print(np.min(periods), np.max(periods))

    # fig, ax = plt.subplots()
    # ax.stairs(counts, edges)
    # ax.set_xlabel(r"$P/\mathrm{day}$")
    # ax.set_ylabel(r"$\hat{f}$")
    # fig.savefig("duquennoy1991_period_pdf.pdf")
    # plt.show()

    # # Plot Duquennoy 1991: log period
    # periods = np.log10(duquennoy1991.rvs(size=100_000))
    # counts, edges = np.histogram(periods, bins=np.linspace(-3., 13., 17),
    #                              density=True)
    # print(np.min(periods), np.max(periods))

    # fig, ax = plt.subplots()
    # ax.stairs(counts, edges)
    # ax.set_xlabel(r"$P/\mathrm{day}$") 
    # ax.set_ylabel(r"$\hat{f}$")
    # fig.savefig("duquennoy1991_logperiod_pdf.pdf")
    # plt.show()

    #########################################################################
    # Plot Moe 2017
    #########################################################################
    primary_mass = (1., 3.5, 7., 12.5, 25.)
    rv_a = moe2017(primary_mass[0])
    rv_b = moe2017(primary_mass[1])
    rv_c = moe2017(primary_mass[2])
    rv_d = moe2017(primary_mass[3])
    rv_e = moe2017(primary_mass[4])

    x_1 = np.linspace(0., 0.2, 500, endpoint=False)
    x_2 = np.linspace(0.2, 8., 500)
    pdf_1a = rv_a.pdf(x_1)
    pdf_2a = rv_a.pdf(x_2)
    pdf_1b = rv_b.pdf(x_1)
    pdf_2b = rv_b.pdf(x_2)
    pdf_1c = rv_c.pdf(x_1)
    pdf_2c = rv_c.pdf(x_2)
    pdf_1d = rv_d.pdf(x_1)
    pdf_2d = rv_d.pdf(x_2)
    pdf_1e = rv_e.pdf(x_1)
    pdf_2e = rv_e.pdf(x_2)

    fig, ax = plt.subplots()
    ax.plot(
        x_1, pdf_1a, color="red", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[0])
    )
    ax.plot(
        x_2, pdf_2a, color="red", ls="solid",
    )
    ax.plot(
        x_1, pdf_1b, color="orange", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[1])
    )
    ax.plot(
        x_2, pdf_2b, color="orange", ls="solid"
    )
    ax.plot(
        x_1, pdf_1c, color="green", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[2])
    )
    ax.plot(
        x_2, pdf_2c, color="green", ls="solid")
    ax.plot(
        x_1, pdf_1d, color="blue", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[3])
    )
    ax.plot(
        x_2, pdf_2d, color="blue", ls="solid"
    )
    ax.plot(
        x_1, pdf_1e, color="magenta", ls="solid",
        label=r"$M_{{1}} = {}$".format(primary_mass[4])
    )
    ax.plot(
        x_2, pdf_2e, color="magenta", ls="solid"
    )
    ax.scatter(0.2, pdf_2a[0], color="red", s=2., zorder=100.)
    ax.scatter(0.2, pdf_2b[0], color="orange", s=2., zorder=100.)
    ax.scatter(0.2, pdf_2c[0], color="green", s=2., zorder=100.)
    ax.scatter(0.2, pdf_2d[0], color="blue", s=2., zorder=100.)
    ax.scatter(0.2, pdf_2e[0], color="magenta", s=2., zorder=100.)
    ax.scatter(0.2, 0., s=2., facecolors="white", edgecolors="magenta",
               zorder=np.inf)
    ax.legend(frameon=False)
    ax.set_xlim(0., 8.)
    ax.set_ylim(-0.05, 0.4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f_{X|M_{1}}$")
    plt.savefig("moe2017_logperiod_pdf.pdf")
    plt.show()
