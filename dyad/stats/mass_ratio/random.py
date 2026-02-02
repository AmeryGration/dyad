import numpy as np
import scipy as sp

from dyad.stats import mass
from .. import _distn_infrastructure


__all__ = [
    "kroupa2001",
    "salpeter1955",
    "splitpowerlaw",
]


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
    # def _argcheck(self, a, b, y):
    #     res = (0. < a) & (a < 0.5) & (0.5 < b)

    #     return res

    # def _get_support(self, a, b, y):
    #     res = 0., 1.

    #     return res
    def _pdf(self, x, y, a, b):
        res = y*mass.kroupa2001(a, b).pdf(x*y)/mass.kroupa2001(a, b).cdf(y)

        return res

    def _cdf(self, x, y, a, b):
        num = (
            mass.kroupa2001(a, b).cdf(x*y) - mass.kroupa2001(a, b).cdf(a/y)
        )
        denom = mass.kroupa2001(a, b).cdf(y)
        res = num/denom

        return res

    def _ppf(self, q, y, a, b):
        num = mass.kroupa2001(a, b).ppf(
            q*mass.kroupa2001(a, b).cdf(y) + mass.kroupa2001(a, b).cdf(a/y)
        )
        denom = y
        res = num/denom

        return res


kroupa2001 = kroupa2001_gen(name="mass_ratio.random.kroupa2001")


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
    # def _argcheck(self, a, b, y):
    #     res = (0. < a) & (a < 0.5) & (0.5 < b)

    #     return res

    # def _get_support(self, a, b, y):
    #     res = 0., 1.

    #     return res
    def _pdf(self, x, y, a, b):
        res = y*mass.salpeter1955(a, b).pdf(x*y)/mass.salpeter1955(a, b).cdf(y)

        return res

    def _cdf(self, x, y, a, b):
        num = (
            mass.salpeter1955(a, b).cdf(x*y) - mass.salpeter1955(a, b).cdf(a/y)
        )
        denom = mass.salpeter1955(a, b).cdf(y)
        res = num/denom

        return res

    def _ppf(self, q, y, a, b):
        num = mass.salpeter1955(a, b).ppf(
            q*mass.salpeter1955(a, b).cdf(y) + mass.salpeter1955(a, b).cdf(a/y)
        )
        denom = y
        res = num/denom

        return res


salpeter1955 = salpeter1955_gen(name="mass_ratio.random.salpeter1955")


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
    def _pdf(self, x, y, s, a, b, c, d):
        res = (
            y*mass.splitpowerlaw(a, b).pdf(x*y)/mass.splitpowerlaw(a, b).cdf(y)
        )

        return res

    def _cdf(self, x, y, s, a, b, c, d):
        num = (
            mass.splitpowerlaw(a, b).cdf(x*y)
          - mass.splitpowerlaw(a, b).cdf(a/y)
        )
        denom = mass.splitpowerlaw(a, b).cdf(y)
        res = num/denom

        return res

    def _ppf(self, x, y, s, a, b, c, d):
        num = mass.splitpowerlaw(a, b).ppf(
            q*mass.splitpowerlaw(a, b).cdf(y)
            + mass.splitpowerlaw(a, b).cdf(a/y)
        )
        denom = y
        res = num/denom

        return res


splitpowerlaw = splitpowerlaw_gen(name="mass_ratio.random.splitpowerlaw")
