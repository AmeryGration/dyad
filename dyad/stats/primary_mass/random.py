__all__ = [
    "kroupa2001",
    "salpeter1955",
    "splitpowerlaw",
]

import numpy as np
import scipy as sp

from dyad.stats import mass
from .. import _distn_infrastructure


class kroupa2001_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    xxx
    
    %(after_notes)s

    See also
    --------

    References
    ----------

    %(example)s

    """
    def _argcheck(self, a, b):
        res = (0. < a) & (a < 0.5) & (0.5 < b)

        return res

    def _get_support(self, a, b):
        res = a, b

        return res
    
    def _pdf(self, x, a, b):
        res = 2.*mass.kroupa2001(a, b).pdf(x)*mass.kroupa2001(a, b).cdf(x)
        
        return res

    def _cdf(self, x, a, b):
        res = mass.kroupa2001(a, b).cdf(x)**2.
                
        return res

    def _ppf(self, q, a, b):
        res = mass.kroupa2001(a, b).ppf(np.sqrt(q))
        
        return res


kroupa2001 = kroupa2001_gen("primary_mass.random.kroupa2001")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    xxx
    
    %(after_notes)s

    See also
    --------

    References
    ----------

    %(example)s

    """
    def _argcheck(self, a, b):
        res = (0. < a) & (a < b)

        return res

    def _get_support(self, a, b):
        res = a, b

        return res
    
    def _pdf(self, x, a, b):
        res = 2.*mass.salpeter1955(a, b).pdf(x)*mass.salpeter1955(a, b).cdf(x)
        
        return res

    def _cdf(self, x, a, b):
        res = mass.salpeter1955(a, b).cdf(x)**2.
                
        return res

    def _ppf(self, q, a, b):
        res = mass.salpeter1955(a, b).ppf(np.sqrt(q))
        
        return res


salpeter1955 = salpeter1955_gen("primary_mass.random.salpeter1955")


class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    xxx
    
    %(after_notes)s

    See also
    --------

    References
    ----------

    %(example)s

    """
    def _argcheck(self, s, a, b, c, d):
        res = (0. < a) & (a < b) & (a < s) & (s < b) & (c < 0.) & (d < 0.)

        return res

    def _get_support(self, s, a, b, c, d):
        res = a, b

        return res
    
    def _pdf(self, x, s, a, b, c, d):
        res = (
            2.
            *mass.splitpowerlaw(s, a, b, c, d).pdf(x)
            *mass.splitpowerlaw(s, a, b, c, d).cdf(x)
        )
        
        return res

    def _cdf(self, x, s, a, b, c, d):
        res = mass.splitpowerlaw(s, a, b, c, d).cdf(x)**2.
                
        return res

    def _ppf(self, q, s, a, b, c, d):
        res = mass.splitpowerlaw(s, a, b, c, d).ppf(np.sqrt(q))
        
        return res

    
splitpowerlaw = splitpowerlaw_gen("primary_mass.random.splitpowerlaw")


# class uniform_gen(_distn_infrastructure.rv_continuous):
#     r"""The primary-star mass random variable for random pairing

#     %(before_notes)s

#     Notes
#     -----
#     xxx
    
#     %(after_notes)s

#     See also
#     --------

#     References
#     ----------

#     %(example)s

#     """
#     def _pdf(self, x, a, b):
#         res = sol_1.interpolator().__call__(np.log10(x))
        
#         return res


# uniform = uniform_gen("primary_mass.random.uniform")

# def f(x):
#     """The known function"""
#     x = 10.**x
#     res = 2.*rv_mass.pdf(x)
#     # print(res)

#     return res

# def kernel_1(x, y):
#     """The kernel"""
#     x = 10.**x
#     y = 10.**y
#     res = (
#         np.log(10.)
#         *y
#         *np.ones_like(x)
#         /(y*(1. - np.maximum(q_min, 0.08/y)))
#     )
#     # res = np.where(x <= y, res, 0.)
    
#     return res

# def kernel_2(x, y):
#     """The kernel"""
#     x = 10.**x
#     y = 10.**y
#     res = (x/y > q_min) & (x/y < 1.)
    
#     return res

# rv_mass = mass.kroupa2001(0.08, 150.)
# # rv_mass = dyad.stats.mass.salpeter1955(0.08, 150.)
# q_min = 0.1

# epsilon = 1.e-15
# a = np.log10(0.08 + epsilon)
# b = np.log10(150. - epsilon)
# c = -1.
# x = np.linspace(a, b, 2**9)
# # x = np.logspace(np.log10(a), np.log10(b))

# # sol = flr.solve_fie2(
# #     f, kernel_1, a, b, c, method="ProdTrapRule", x_eval=x, dense_output=True,
# #     ker_singular=kernel_2, rtol=1.e-2
# # )
# # y_1 = sol.sol.interpolator(x)

# # sol = flr.solve_fie2(
# #     f, kernel_1, a, b, c, x_eval=x, dense_output=True, atol=1.
# # )
# # print(sol)
# # print(sol.sol.interpolator.x_nodes.shape)

# import flourmill as flr

# n_nodes = 2**6
# sol_1 = flr.ProdTrapRule(f, kernel_1, a, b, c, ker_singular=kernel_2)
# sol_1.n_nodes = n_nodes
# sol_1.solve_system()

