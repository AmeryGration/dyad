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

def main():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use("sm")

    ######################################################################### 
    # Plot Moe 2017: eta
    #########################################################################
    log10_period = np.linspace(0.6, 8, 500)
    primary_mass = np.array([1., 3.5, 7., 12.5, 25.])

    eta = _moe2017_eta(log10_period, primary_mass.reshape([-1, 1]))

    fig, ax = plt.subplots()
    ax.plot(log10_period, eta[0], color="red", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax.plot(log10_period, eta[1], color="orange", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax.plot(log10_period, eta[2], color="green", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax.plot(log10_period, eta[3], color="blue", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax.plot(log10_period, eta[4], color="magenta", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax.legend(frameon=False, loc=4)
    ax.set_xlim(0.2, 8.)
    ax.set_ylim(-1., 1.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\eta$")
    plt.savefig("moe2017_eta.pdf")
    plt.savefig("moe2017_eta.jpg")
    plt.show()

    ######################################################################### 
    # Plot Moe 2017: PDF
    #########################################################################
    log10_period = 3.
    primary_mass = np.array([1., 3.5, 7., 12.5, 25.])
    rv = moe2017(log10_period, primary_mass.reshape([-1, 1]))
    e_max = 1 - (0.5*10.**log10_period)**(-2./3.)

    n = 500
    x_1 = np.linspace(-0.1, 0., n)[:-1]
    x_2 = np.linspace(0., e_max, n)
    x_3 = np.linspace(e_max, 1.1, n)[1:]

    pdf_1 = rv.pdf(x_1)
    pdf_2 = rv.pdf(x_2)
    pdf_3 = rv.pdf(x_3)

    cdf_1 = rv.cdf(x_1)
    cdf_2 = rv.cdf(x_2)
    cdf_3 = rv.cdf(x_3)

    pdf_closed_dots_x = 5*(e_max,)
    pdf_closed_dots_y = pdf_2[:, -1]
    pdf_open_dots_x = 5*(e_max,)
    pdf_open_dots_y = 5*(0.,)
    
    fig, ax_1 = plt.subplots()
    ax_2 = ax_1.twinx()
    ax_1.plot(x_1, pdf_1[0], color="red", ls="solid" )
    ax_1.plot(x_2, pdf_2[0], color="red", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_1.plot(x_3, pdf_3[0], color="red", ls="solid")
    ax_1.plot(x_1, pdf_1[1], color="orange", ls="solid" )
    ax_1.plot(x_2, pdf_2[1], color="orange", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_1.plot(x_3, pdf_3[1], color="orange", ls="solid")
    ax_1.plot(x_1, pdf_1[2], color="green", ls="solid" )
    ax_1.plot(x_2, pdf_2[2], color="green", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_1.plot(x_3, pdf_3[2], color="green", ls="solid")
    ax_1.plot(x_1, pdf_1[3], color="blue", ls="solid" )
    ax_1.plot(x_2, pdf_2[3], color="blue", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_1.plot(x_3, pdf_3[3], color="blue", ls="solid")
    ax_1.plot(x_1, pdf_1[4], color="magenta", ls="solid" )
    ax_1.plot(x_2, pdf_2[4], color="magenta", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_1.plot(x_3, pdf_3[4], color="magenta", ls="solid")
    ax_1.legend(frameon=False, loc=2)
    ax_1.set_xlim(-0.1, 1.1)
    ax_1.set_ylim(-0.5, 2.5)
    ax_1.set_xlabel(r"$x$")
    ax_1.set_ylabel(r"$f_{X|M_{1}}$")
    ax_2.plot(x_1, cdf_1[0], color="red", ls="dashed" )
    ax_2.plot(x_2, cdf_2[0], color="red", ls="dashed",
              label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax_2.plot(x_3, cdf_3[0], color="red", ls="dashed")
    ax_2.plot(x_1, cdf_1[1], color="orange", ls="dashed" )
    ax_2.plot(x_2, cdf_2[1], color="orange", ls="dashed",
              label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax_2.plot(x_3, cdf_3[1], color="orange", ls="dashed")
    ax_2.plot(x_1, cdf_1[2], color="green", ls="dashed" )
    ax_2.plot(x_2, cdf_2[2], color="green", ls="dashed",
              label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax_2.plot(x_3, cdf_3[2], color="green", ls="dashed")
    ax_2.plot(x_1, cdf_1[3], color="blue", ls="dashed" )
    ax_2.plot(x_2, cdf_2[3], color="blue", ls="dashed",
              label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax_2.plot(x_3, cdf_3[3], color="blue", ls="dashed")
    ax_2.plot(x_1, cdf_1[4], color="magenta", ls="dashed" )
    ax_2.plot(x_2, cdf_2[4], color="magenta", ls="dashed",
              label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax_2.plot(x_3, cdf_3[4], color="magenta", ls="dashed")
    ax_2.set_ylim(-0.25, 1.25)
    ax_2.set_ylabel(r"$F_{X|M_{1}}$")

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

    plt.savefig("moe2017_eccentricity_pdf.pdf")
    plt.savefig("moe2017_eccentricity_pdf.jpg")
    plt.show()

    # Test class methods: PPF (no twin excess)
    p = np.linspace(0., 1., n)
    ppf = rv.ppf(p)

    fig, ax = plt.subplots()
    ax.plot(p, ppf[0], color="red", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[0]))
    ax.plot(p, ppf[1], color="orange", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[1]))
    ax.plot(p, ppf[2], color="green", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[2]))
    ax.plot(p, ppf[3], color="blue", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[3]))
    ax.plot(p, ppf[4], color="magenta", ls="solid",
            label=r"$M_{{1}} = {}$".format(primary_mass[4]))
    ax.legend(frameon=False, loc=2)
    ax.set_ylim(0., 1.)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$F^{-1}_{q|P, M_{1}}$")
    fig.savefig("moe2017_eccentricity_ppf.pdf")
    fig.savefig("moe2017_eccentricity_ppf.jpg")
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import dyad

    # Plot Duquennoy 1991: tight binaries
    eccentricities = duquennoy1991(500).rvs(size=10_000)
    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

    # Plot Duquennoy 1991: wide binaries
    eccentricities = duquennoy1991(5000).rvs(size=10_000)
    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

    # Plot Duquennoy 1991: conditional random variable
    periods = dyad.stats.period.duquennoy1991.rvs(size=10_000)
    #2000*np.random.rand(10_000)
    eccentricities = np.zeros_like(periods)
    eccentricities[periods>11.] = duquennoy1991(periods[periods>11.]).rvs()

    counts, edges = np.histogram(
        eccentricities, bins=np.linspace(0., 1., 13), density=True
    )
    print(np.min(eccentricities), np.max(eccentricities))

    fig, ax = plt.subplots()
    ax.stairs(counts, edges)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\hat{f}$")
    plt.show()

