#!usr/bin/env python

import inspect
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import dyad
import flourmill as flr
import plot

mpl.style.use("sm")

def uniform(rv_mass, q_min=0.1):
    """Return primary-mass function as instance of `flr.IeResult`"""
    def g(m):
        """Return the known function in log space"""
        x = 10.**np.array(m)
        res = 2.*rv_mass.pdf(x)

        return res

    def kernel_1(x, y, m_min=rv_mass.a, q_min=q_min):
        """Return CSMF for uniform pairing in log space"""
        m_2 = 10.**np.array(x)
        m_1 = 10.**np.array(y)
        res = (
            np.log(10.)*np.ones_like(m_2/m_1)
            /(1. - np.maximum(q_min, m_min/m_1))
        )
        res *= (q_min <= m_2/m_1) & (m_2/m_1 <= 1.)

        return res

    # def kernel_2(x, y, q_min=q_min):
    #     m_2 = 10.**np.array(x)
    #     m_1 = 10.**np.array(y)
    #     res = (q_min <= m_2/m_1) & (m_2/m_1 <= 1.)

    #     return res

    eps = 1.e-12
    sol = flr.solve_fie2(g, kernel_1, np.log10(rv_mass.a) + eps,
                         np.log10(rv_mass.b) + eps, -1., dense_output=True,
                         args_ker=(rv_mass.a, q_min))
    # sol = flr.solve_fie2(g, kernel_1, np.log10(rv_mass.a) + eps,
    #                      np.log10(rv_mass.b) + eps, -1., dense_output=True,
    #                      method="ProdTrapRule", ker_singular=kernel_2,
    #                      rtol=1.e-02, atol=1.e-06,
    #                      # n_max=2**12 + 1,
    #                      args_ker=(rv_mass.a, q_min),
    #                      )

    # Check the solution is good
    # Code here

    return sol

def interp(rv_mass, q_min):
    """Return interpolation functions for the primary-mass PDF, CDF, and PPF"""
    sol = uniform(rv_mass, q_min)
    pdf = sp.interpolate.interp1d(10.**sol.x, sol.g, kind="linear",
                                  bounds_error=False, fill_value=0.)
    G_nodes = sp.integrate.cumulative_trapezoid(sol.g, 10.**sol.x, initial=0.)
    G_nodes /= G_nodes[-1]
    cdf = sp.interpolate.interp1d(10.**sol.x, G_nodes, kind="quadratic",
                                  bounds_error=False, fill_value=(0, 1))
    ppf = sp.interpolate.interp1d(G_nodes, 10.**sol.x, kind="quadratic",
                                  bounds_error=False, fill_value=np.nan)

    return pdf, cdf, ppf

rv_mass = dyad.stats.mass.kroupa2001(0.08, 150.)
rv_mass = dyad.stats.mass.salpeter1955(0.08, 150.)
rv_mass = dyad.stats.mass.splitpowerlaw(0.5, 0.08, 150., -0.3, -2.3)

pdf, cdf, ppf = interp(rv_mass, 0.1)

m = np.logspace(np.log10(0.08), np.log10(150.), 500)
q = np.linspace(0., 1.)

fig, ax = plot.plot()
ax.plot(m, pdf(m))
ax.plot(m, cdf(m))
ax.plot(q, ppf(q))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1.e-02, 1.e+03)
ax.set_ylim(1.e-09, 1.e03)
ax.set_xlabel(r"$m/\mathrm{M}_{\odot}$")
plt.show()


class kroupa2001_gen(dyad.stats._distn_infrastructure.rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp = None
        
    def pdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(dyad.stats.mass.kroupa2001(*a, *b), *q_min)

        return self._interp[0]

    def cdf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(dyad.stats.mass.kroupa2001(*a, *b), *q_min)

        return self._interp[1]

    def ppf_interp(self, a, b, q_min):
        if self._interp is None:
            self._interp = interp(dyad.stats.mass.kroupa2001(*a, *b), *q_min)

        return self._interp[2]

    def _argcheck(self, a, b, q_min):
        return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

    def _get_support(self, a, b, q_min):
        res = (a, b)

        return res
        
    def _pdf(self, x, a, b, q_min):
        res = self.pdf_interp(a, b, q_min)(x)

        return res

    def _cdf(self, x, a, b, q_min):
        res = self.cdf_interp(a, b, q_min)(x)

        return res

    def _ppf(self, x, a, b, q_min):
        res = self.ppf_interp(a, b, q_min)(x)

        return res


kroupa2001 = kroupa2001_gen(name="primary_mass.uniform.kroupa2001")

m = np.logspace(-2., 3., 500)
q = np.linspace(0., 1.)

rv_primary_mass = kroupa2001(0.08, 150., 0.1)
fig, ax = plot.plot()
ax.plot(m, rv_primary_mass.pdf(m))
ax.plot(m, rv_primary_mass.cdf(m))
ax.plot(q, rv_primary_mass.ppf(q))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1.e-02, 1.e+03)
ax.set_ylim(1.e-09, 1.e03)
ax.set_xlabel(r"$m/\mathrm{M}_{\odot}$")
plt.show()

