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
    r"""The random-pairing mass-ratio random variable (Kroupa mass function)

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.kroupa2001` is:

    .. math::

       f_{Q|M_{1}}(q|m_{1}) = 
       = m_{1}\dfrac{f_{M}(qm_{1})}{F_{M}(m_{1})}

    for mass ratio :math:`q \in (0, 1]` and primary_mass :math:`m_{1}
    \in [a, b]` where :math:`f_{M}` and :math:`F_{M}` are the
    probability density function and the cumulative distribution
    function for the mass random variable of Kroupa
    (`dyad.stats.mass.kroupa2001`), which iself has support on the
    interval :math:`[a, b]`.

    `random.kroupa2001` takes ``m_{1}`` as a shape parameter for
    :math:`m_{1}`, the primary mass, ``a`` as a shape parameter for
    :math:`a`, the minimum allowed mass, and ``b`` as a shape
    parameter for :math:`b`, the maximum allowed mass.

    %(after_notes)s

    References
    ----------
    Malkov, O., and H. Zinnecker. 2001. \'Binary stars and the
    fundamental initial mass function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149–54.

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
    r"""The mass-ratio random variable for random pairing

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
    r"""The mass-ratio random variable for random pairing

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


class Test(_distn_infrastructure.rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
