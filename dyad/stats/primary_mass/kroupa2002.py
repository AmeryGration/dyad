"""
======================================================
Kroupa2002 (:mod:`dyad.stats.primary_mass.kroupa2002`)
======================================================

.. currentmodule:: dyad.stats.primary_mass.kroupa2002

This module contains probability distributions for the masses of the
primary components of a population of binary stars. It assumes that
the masses of the population of individual stars are distributed
according to the canoncical initial mass function of Kroupa [K02]_.

Probability distributions
=========================

.. autosummary::
   :toctree: generated/

   random

References
----------

.. [K02]

   Kroupa, P. 2002. \'The initial mass function and its variation
   (review)\'. *ASP conference series* 285 (January): 86.

"""

__all__ = [
    "random",
]

import os
import numpy as np
import scipy as sp

from dyad.stats import _distn_infrastructure
from dyad.stats import mass

_rv_mass = mass.kroupa2002


class random_gen(_distn_infrastructure.rv_continuous):
    r"""The primary-star mass random variable for random pairing

    %(before_notes)s

    Notes
    -----
    The probability density function for `random` is:

    .. math::
       f_{M_{1}}(m_{1}) = 2f_{M}(m_{1})F_{M}(m_{1})

    where :math:`f_{M}` and :math:`F_{M}` are the PDF and CDF for the
    mass random variable of Kroupa (2002).

    %(after_notes)s

    See also
    --------
    mass.kroupa2002 : The mass random variable of Kroupa (2002)

    References
    ----------
    Malkov, O., and H. Zinnecker. 2001. \‘Binary Stars and the
    Fundamental Initial Mass Function\’. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149–54.

    %(example)s

    """
    def _pdf(self, x):
        x = np.asarray(x)
        res = 2.*_rv_mass.pdf(x)*_rv_mass.cdf(x)
        
        return res

    def _cdf(self, x):
        res = _rv_mass.cdf(x)**2.
                
        return res

    def _ppf(self, q):
        res = _rv_mass.ppf(q**0.5)

        return res


# _a = _rv_mass.a
# _b = _rv_mass.b
# random = random_gen(a=_a, b=_b, name="primary_mass.kroupa2002.random")

# x = np.logspace(np.log10(0.1), np.log10(60.), 500)
# f = random.pdf(x)
# F = random.cdf(x)
# rvs = random.rvs(size=1_000_000)

# plt.hist(rvs, bins=x, density=True)
# plt.plot(x, f, color="red")
# plt.plot(x, F)
# plt.xscale("log")
# plt.yscale("log")
# plt.show()
