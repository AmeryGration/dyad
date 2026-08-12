"""
=========================================================================
Primary mass for uniform pairing (:mod:`dyad.stats.primary_mass.uniform`)
=========================================================================

.. currentmodule:: dyad.stats.primary_mass.uniform

This module contains probability distributions for the primary
masses of a population of binary stars formed by uniform pairing. In
its documentation the random variable is denoted :math:`M_{1}` and a
realization of that random variable is denoted :math:`m_{1}`.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   kroupa2001
   salpeter1955

"""

__all__ = [
    "kroupa2001",
    "salpeter1955",
    "splitpowerlaw",
]

import numpy as np
import scipy as sp

from dyad.stats import mass
from .. import _distn_infrastructure

class _FuncWrapper:
    r"""Wrap function so it can take additional arguments"""
    def __init__(self, f, args):
        self.f = f
        if args is None:
            self.args = ()
        elif np.isscalar(args):
            self.args = (args,)
        else:
            self.args = args
        
    def __call__(self, t):
        return self.f(t, *self.args)


class _CovfuncWrapper:
    r"""Wrap positive-definite kernel so it can take additional arguments"""
    def __init__(self, f, args):
        self.f = f
        if args is None:
            self.args = ()
        elif np.isscalar(args):
            self.args = (args,)
        else:
            self.args = args
        
    def __call__(self, s, t):
        return self.f(s, t, *self.args)

    
class BackwardVolterraNystromSolver:
    def __init__(self, g, k, a, b, c, g_args=(), k_args=()):
        r"""Solve a backwards Volterra equation of the second kind

        A backwards Volterra integral equation of the second kind has
        the form

        .. math::

           f(x) = g(x) + c\int_{x}^{b}k(x, y)f(y)\,\mathrm{d}y

        where :math:`x, y \in [a, b]`.

        Parameters
        ----------

        g : callable

            The known function, :math:`g`, as a callable object,::

                g(x, *g_args) -> float

            where ``x`` is a 1-D array with shape (n,) and ``g_args``
            is a tuple of the fixed parameters needed to completely
            specify the function.

        k : callable
        
            The kernel, :math:`k` as a callable object,::

                k(x, y, *k_args) -> float

            where ``x`` and ``y`` are 1-D arrays with shapes (n,) and
            ``k_args`` is a tuple of the fixed parameters needed to
            completely specify the function.

        a : float
        
            The lower bound of the interval :math:`[a, b]`.

        b : float
        
            The upper bound of the interval :math:`[a, b]`.

        c : float

            The scalar coefficient $c$.

        k_args : tuple, optional

            Extra arguments passed to the objective function.

        """
        self.g = _FuncWrapper(g, g_args)
        self.k = _CovfuncWrapper(k, k_args)
        self.a = a
        self.b = b
        self.c = c

    def _x_nodes(self, n_nodes):
        res = np.linspace(self.a, self.b, n_nodes).squeeze()
        
        return res

    def weight(self, x_nodes):
        """
        Trapezoidal weights W_ij for ∫_{x_i}^{b} φ(y) dy.

        Uniform grid: x_j = a + j*h, h = (b - a)/(n-1).

        For each row i:
            W[i, j] = 0     for j < i
            W[i, i] = h/2   (left endpoint)
            W[i, j] = h     for i < j < n-1
            W[i, n-1] = h/2 (right endpoint b)
        Row n-1 remains all zeros (integral from b to b is 0).
        """
        n = len(x_nodes)
        h = (self.b - self.a) / (n - 1)
        W = np.zeros((n, n), dtype=float)

        for i in range(n - 1):  # last row (i = n-1) stays zero
            W[i, i] = 0.5*h
            if i + 1 < n - 1:
                W[i, i + 1:n - 1] = h
            W[i, n - 1] = 0.5*h

        return W  # shape (n, n), upper-triangular by construction

    def solve_system(self, n_nodes):
        x_nodes = self._x_nodes(n_nodes)

        g_nodes = self.g(x_nodes)
        assert g_nodes.shape == (n_nodes,)

        X = np.atleast_2d(x_nodes)
        K = self.k(X, X.T)
        assert K.shape == (n_nodes, n_nodes)

        W = self.weight(x_nodes)
        assert W.shape == (n_nodes, n_nodes)

        A = W*K
        assert A.shape == (n_nodes, n_nodes)

        M = np.eye(n_nodes) - self.c*A
        f_nodes = sp.linalg.solve_triangular(M, g_nodes, lower=False)
        assert f_nodes.shape == (n_nodes,)

        return x_nodes, f_nodes

    
def g_kroupa2001(x, m_min, m_max):
    """Return value of known function in log space"""
    m = 10.**np.array(x)
    res = 2.*mass.kroupa2001(m_min, m_max).pdf(m)

    return res

def g_salpeter1955(x, m_min, m_max):
    """Return value of known function in log space"""
    m = 10.**np.array(x)
    res = 2.*mass.salpeter1955(m_min, m_max).pdf(m)

    return res

# def k_random(x_1, x_2):
#     """Return CSMF for random pairing in log space"""
#     m_1 = 10.**np.array(x_1)
#     m_2 = 10.**np.array(x_2)
#     res = rv_mass.pdf(m_2)/rv_mass.cdf(m_1)
#     res *= np.log(10.)*m_1

#     return res

def k_uniform(x_1, x_2, m_min=0.08, q_min=0.1):
    """Return value of CSMF for uniform pairing in log space"""
    m_1 = 10.**np.array(x_1)
    m_2 = 10.**np.array(x_2)
    num = np.ones_like(m_2/m_1)
    denom = 1. - np.maximum(q_min, m_min/m_1)
    res = num/denom
    res *= np.log(10.)

    return res

# q_min = 0.1
# m_min = 0.08
# m_max = 150.

# eps = 1.e-06
# a = np.log10(m_min) + eps
# b = np.log10(m_max) - eps
# c = -1.
# # sol_random = BackwardVolterraNystromSolver(g_kroupa2001, k_random, a, b, c)
# # x_nodes, y_random = sol_random.solve_system(n_nodes)
# g = _FuncWrapper(g_kroupa2001, (m_min, m_max))
# # g = _FuncWrapper(g_salpeter1955, (m_min, m_max))
# k = _CovfuncWrapper(k_uniform, (m_min, q_min))
# n_nodes = 2**9

# sol_uniform = BackwardVolterraNystromSolver(g, k, a, b, c)
# x_nodes, y_nodes = sol_uniform.solve_system(n_nodes)
# x_nodes = 10.**x_nodes
# x_nodes[0] = m_min
# x_nodes[-1] = m_max

# G_nodes = sp.integrate.cumulative_trapezoid(y_nodes, x_nodes, initial=0.)
# G_nodes /= G_nodes[-1]
# _pdf_uniform = sp.interpolate.interp1d(x_nodes, y_nodes, kind="linear",
#                                        bounds_error=False, fill_value=0.)
# _cdf_uniform = sp.interpolate.interp1d(x_nodes, G_nodes, kind="quadratic",
#                                      bounds_error=False, fill_value=(0, 1))
# _ppf_uniform = sp.interpolate.interp1d(G_nodes, x_nodes, kind="quadratic",
#                                      bounds_error=False, fill_value=np.nan)

# import plot

# fig, ax = plot.plot()
# # ax.plot(x_nodes, y_random, label=r"random")
# ax.plot(x_nodes, y_nodes, label=r"uniform")
# # ax.plot(x_nodes, _pdf_uniform(x_nodes), label=r"uniform")
# # ax.plot(x_nodes, _cdf_uniform(x_nodes), label=r"uniform")
# # ax.plot(np.linspace(0., 1.), _ppf_uniform(np.linspace(0., 1.)),
# #         label=r"uniform")
# ax.legend(frameon=False)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$f_{M_{1}}$")
# plt.savefig("./Figures/vie_ii_primary_mass.jpg")
# plt.savefig("./Figures/vie_ii_primary_mass.pdf")
# plt.show()

# Create interpolator
def interp(g, k, m_min, m_max, q_min, n_nodes=2**9):
    g = _FuncWrapper(g, (m_min, m_max))
    k = _CovfuncWrapper(k, (m_min, q_min))
    eps = 1.e-06
    a = np.log10(m_min) + eps
    b = np.log10(m_max) - eps
    c = -1.
    
    sol_uniform = BackwardVolterraNystromSolver(g, k, a, b, c)
    x_nodes, y_nodes = sol_uniform.solve_system(n_nodes)
    x_nodes = 10.**x_nodes
    x_nodes[0] = m_min
    x_nodes[-1] = m_max
    
    G_nodes = sp.integrate.cumulative_trapezoid(y_nodes, x_nodes, initial=0.)
    G_nodes /= G_nodes[-1]
    pdf = sp.interpolate.interp1d(x_nodes, y_nodes, kind="linear",
                                  bounds_error=False, fill_value=0.)
    cdf = sp.interpolate.interp1d(x_nodes, G_nodes, kind="quadratic",
                                  bounds_error=False,
                                  fill_value=(0, 1))
    ppf = sp.interpolate.interp1d(G_nodes, x_nodes, kind="quadratic",
                                  bounds_error=False,
                                  fill_value=np.nan)

    return pdf, cdf, ppf


class kroupa2001_gen(sp.stats.rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interp = None

    def pdf_interp(self, a, b, q_min):
        if self._interp is None:
            print(111111)
            self._pdf_interp, self._cdf_interp, self._ppf_interp = interp(
                g_kroupa2001, k_uniform, a, b, q_min
            )
            
        return self._pdf_interp

    def cdf_interp(self, a, b, q_min):
        if self._interp is None:
            print(222222)
            self._pdf_interp, self._cdf_interp, self._ppf_interp = interp(
                g_kroupa2001, k_uniform, a, b, q_min
            )

        return self._cdf_interp

    def ppf_interp(self, a, b, q_min):
        print(3333333)
        if self._interp is None:
            self._pdf_interp, self._cdf_interp, self._ppf_interp = interp(
                g_kroupa2001, k_uniform, a, b, q_min
            )

        return self._ppf_interp

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


# q_min = 0.1
# m_min = 0.08
# m_max = 150.
# pdf_interp, _, _ = interp(g_kroupa2001, k_uniform, m_min, m_max, q_min)
kroupa2001 = kroupa2001_gen(name="primary_mass.uniform.kroupa2001")
# kroupa2001 = kroupa2001(m_min, m_max, q_min)
# # kroupa2001 = dyad.stats.primary_mass.uniform.kroupa2001(m_min, m_max, q_min)

# x = np.logspace(np.log10(m_min), np.log10(m_max), 50)
# f = kroupa2001.pdf(x)
# F = kroupa2001.cdf(x)

# import plot

# fig, ax = plot.plot()
# ax.plot(x, f, label=r"uniform")
# ax.plot(x, F, label=r"uniform")
# ax.legend(frameon=False)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$m_{1}$")
# ax.set_ylabel(r"$f_{M_{1}}$")
# plt.savefig("./Figures/vie_ii_primary_mass.jpg")
# plt.savefig("./Figures/vie_ii_primary_mass.pdf")
# plt.show()

# f, _, _ = interp(g_kroupa2001, k_uniform, m_min, m_max, q_min)
# f(x)

########################################################################

# def g_salpeter1955(x, m_min, m_max):
#     """Return value of known function in log space"""
#     m = 10.**np.array(x)
#     res = 2.*dyad.stats.mass.salpeter1955(m_min, m_max).pdf(m)

#     return res

# class salpeter1955_gen(_distn_infrastructure.rv_continuous):
#     r"""The primary-star mass random variable for uniform pairing

#     %(before_notes)s

#     Notes
#     -----

#     The probability density function for `uniform.salpeter1955` is the
#     solution to the integral equation

#     .. math::
#        xxx

#     where :math:`f_{M}` is the probability density function for the
#     mass random variable of Salpeter (1955) and
#     :math:`f_{M_{2}|M_{2}}` is the conditional secondary mass function
#     for uniform pairing, which is given by

#     .. math::
#        f_{M_{2}|M_{2}}(m_{2}|m_{1})
#        = \dfrac{1}{m_{1}}f_{Q|M_{1}}(m_{2}/m_{1})|m_{2})

#     where :math:`f_{Q|M_{1}}` is the conditional mass-ratio function
#     for uniform pairing.
    
#     %(after_notes)s

#     See also
#     --------
#     dyad.stats.mass.salpeter1955
#     dyad.stats.mass_ratio.uniform
    
#     References
#     ----------
#     Kroupa, P. 2001. \'The initial mass function and its variation
#     (review)\'. *ASP conference series* 285 (January): 86.

#     %(example)s

#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._interp = None
        
#     def pdf_interp(self, a, b, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

#         return self._interp[0]

#     def cdf_interp(self, a, b, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

#         return self._interp[1]

#     def ppf_interp(self, a, b, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.salpeter1955(*a, *b), *q_min)

#         return self._interp[2]

#     def _argcheck(self, a, b, q_min):
#         return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

#     def _get_support(self, a, b, q_min):
#         res = (a, b)

#         return res
        
#     def _pdf(self, x, a, b, q_min):
#         res = self.pdf_interp(a, b, q_min)(x)

#         return res

#     def _cdf(self, x, a, b, q_min):
#         res = self.cdf_interp(a, b, q_min)(x)

#         return res

#     def _ppf(self, x, a, b, q_min):
#         res = self.ppf_interp(a, b, q_min)(x)

#         return res


# salpeter1955 = salpeter1955_gen(name="primary_mass.uniform.salpeter1955")


# class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
#     r"""The primary-star mass random variable for uniform pairing

#     %(before_notes)s

#     Notes
#     -----

#     The probability density function for `uniform.splitpowerlaw` is the
#     solution to the integral equation

#     .. math::
#        xxx

#     where :math:`f_{M}` is the probability density function for the
#     two-piece power-function mass random variable and
#     :math:`f_{M_{2}|M_{2}}` is the conditional secondary mass function
#     for uniform pairing, which is given by

#     .. math::
#        f_{M_{2}|M_{2}}(m_{2}|m_{1})
#        = \dfrac{1}{m_{1}}f_{Q|M_{1}}(m_{2}/m_{1})|m_{2})

#     where :math:`f_{Q|M_{1}}` is the conditional mass-ratio function
#     for uniform pairing.
    
#     %(after_notes)s

#     See also
#     --------
#     dyad.stats.mass.splitpowerlaw
#     dyad.stats.mass_ratio.uniform
    
#     References
#     ----------
#     Kroupa, P. 2001. \'The initial mass function and its variation
#     (review)\'. *ASP conference series* 285 (January): 86.

#     %(example)s

#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._interp = None
        
#     def pdf_interp(self, s, a, b, c, d, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
#                                   *q_min)

#         return self._interp[0]

#     def cdf_interp(self, s, a, b, c, d, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
#                                   *q_min)

#         return self._interp[1]

#     def ppf_interp(self, s, a, b, c, d, q_min):
#         if self._interp is None:
#             self._interp = interp(mass.splitpowerlaw(*s, *a, *b, *c, *d),
#                                   *q_min)

#         return self._interp[2]

#     def _argcheck(self, s, a, b, c, d, q_min):
#         return (0. < a) & (a < b) & (a < 0.5) & (0. < q_min ) & (q_min < 1.)

    
#     def _get_support(self, s, a, b, c, d, q_min):
#         res = (a, b)

#         return res
        
#     def _pdf(self, x, s, a, b, c, d, q_min):
#         res = self.pdf_interp(s, a, b, c, d, q_min)(x)

#         return res

#     def _cdf(self, x, s, a, b, c, d, q_min):
#         res = self.cdf_interp(s, a, b, c, d, q_min)(x)

#         return res

#     def _ppf(self, x, s, a, b, c, d, q_min):
#         res = self.ppf_interp(s, a, b, c, d, q_min)(x)

#         return res


# splitpowerlaw = splitpowerlaw_gen(name="primary_mass.uniform.splitpowerlaw")
