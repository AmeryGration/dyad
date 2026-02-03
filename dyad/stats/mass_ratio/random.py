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
    r"""A random-pairing mass-ratio random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.kroupa2001` is:

    .. math::

       f_{Q|M_{1}}(q|m_{1}) = m_{1}\dfrac{f_{M}(qm_{1})}{F_{M}(m_{1})}

    for mass ratio :math:`q \in (0, 1]` and primary_mass :math:`m_{1}
    \in [m_{\min}, m_{\max}]` where :math:`f_{M}` and :math:`F_{M}`
    are the probability density function and the cumulative
    distribution function for the mass random variable of Kroupa
    (2001). The probability density function :math:`f_{M}` is given by

    .. math::
       f_{M}(m)
       =
       A_{M}
       \begin{cases}
       m^{-1.3} &\text{ if $m \in [a, 0.5)$}\\
       0.5m^{-2.3} &\text{ if $m \in [0.5, b]$}\\
       \end{cases}

    for

    .. math::
       A_{M} := \dfrac{0.5(0.5^{-1.3} - b^{-1.3})}{1.3} +
       \dfrac{a^{-0.3} - 0.5^{-0.3}}{0.3}

    and :math:`m \in [m_{\min}, m_{\max}]` where :math:`m_{\min} <
    0.5` and :math:`m_{\max} > 0.5`. The cumulative distribution
    function is given by

    .. math::
       F_{M}(m) = \int_{m_{\min}}^{m}f_{M}(u}\mathrm{d}\,u.
  
    `random.kroupa2001` takes ``m_{1}`` as a shape parameter for
    :math:`m_{1}`, the primary mass, ``m_min`` as a shape parameter
    for :math:`m_{\min}`, the minimum allowed mass, and ``m_max``
    as a shape parameter for :math:`m_{\max}`, the maximum allowed mass.

    %(after_notes)s

    See also
    --------
    `dyad.stats.mass.kroupa2001`

    References
    ----------
    Malkov, O., and H. Zinnecker. 2001. \'Binary stars and the
    fundamental initial mass function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149–54.

    Kroupa, P. 2002. \'The initial mass function and its variation
    (review)\'. *ASP conference series* 285 (January): 86.

    %(example)s

    """
    def _argcheck(self, m_1, m_min, m_max):
        res = (0. < m_min) & (m_min < 0.5) & (0.5 < m_max)

        return res

    def _get_support(self, m_1, m_min, m_max):
        res = m_min/m_1, 1.

        return res
    
    def _pdf(self, x, m_1, m_min, m_max):
        res = (
            m_1
            *mass.kroupa2001(m_min, m_max).pdf(x*m_1)
            /mass.kroupa2001(m_min, m_max).cdf(m_1)
        )

        return res

    def _cdf(self, x, m_1, m_min, m_max):
        num = (
            mass.kroupa2001(m_min, m_max).cdf(x*m_1)
            - mass.kroupa2001(m_min, m_max).cdf(m_min/m_1)
        )
        denom = mass.kroupa2001(m_min, m_max).cdf(m_1)
        res = num/denom

        return res

    def _ppf(self, q, m_1, m_min, m_max):
        num = mass.kroupa2001(m_min, m_max).ppf(
            q*mass.kroupa2001(m_min, m_max).cdf(m_1)
            + mass.kroupa2001(m_min, m_max).cdf(m_min/m_1)
        )
        denom = m_1
        res = num/denom

        return res


kroupa2001 = kroupa2001_gen(name="mass_ratio.random.kroupa2001")


class salpeter1955_gen(_distn_infrastructure.rv_continuous):
    r"""A random-pairing mass-ratio random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.salpeter1955` is:

    .. math::

       f_{Q|M_{1}}(q|m_{1}) = m_{1}\dfrac{f_{M}(qm_{1})}{F_{M}(m_{1})}

    for mass ratio :math:`q \in (0, 1]` and primary_mass :math:`m_{1}
    \in [m_{\min}, m_{\max}]` where :math:`f_{M}` and :math:`F_{M}`
    are the probability density function and the cumulative
    distribution function for the mass random variable of Salpeter
    (1955). The probability density function :math:`f_{M}` is given by

    .. math::
       f_{M}(m) = \dfrac{c - 1}{a^{1 - c} - b^{1 - c}}\dfrac{1}{m^{c}}

    for :math:`m \in [a, b]` and :math:`c = 2.35`. The cumulative
    distribution function is given by

    .. math::
       F_{M}(m) = \int_{m_{\min}}^{m}f_{M}(u}\mathrm{d}\,u.
  
    `random.salpeter1955` takes ``m_{1}`` as a shape parameter for
    :math:`m_{1}`, the primary mass, ``m_min`` as a shape parameter
    for :math:`m_{\min}`, the minimum allowed mass, and ``m_max``
    as a shape parameter for :math:`m_{\max}`, the maximum allowed mass.

    %(after_notes)s

    See also
    --------
    `dyad.stats.mass.salpeter1955`

    References
    ----------
    Malkov, O., and H. Zinnecker. 2001. \'Binary stars and the
    fundamental initial mass function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149–54.

    Salpeter, Edwin E. 1955. \'The luminosity function and stellar
    evolution.\' *The Astrophysical Journal* 121 (January): 161.

    %(example)s

    """
    def _argcheck(self, m_1, m_min, m_max):
        res = (0. < m_min) & (0. < m_max) & (m_min < m_max)

        return res

    def _get_support(self, m_1, m_min, m_max):
        res = m_min/m_1, 1.

        return res

    def _pdf(self, x, m_1, m_min, m_max):
        res = (
            m_1
            *mass.salpeter1955(m_min, m_max).pdf(x*m_1)
            /mass.salpeter1955(m_min, m_max).cdf(m_1)
        )

        return res

    def _cdf(self, x, m_1, m_min, m_max):
        num = (
            mass.salpeter1955(m_min, m_max).cdf(x*m_1)
            - mass.salpeter1955(m_min, m_max).cdf(m_min/m_1)
        )
        denom = mass.salpeter1955(m_min, m_max).cdf(m_1)
        res = num/denom

        return res

    def _ppf(self, q, m_1, m_min, m_max):
        num = mass.salpeter1955(m_min, m_max).ppf(
            q*mass.salpeter1955(m_min, m_max).cdf(m_1)
            + mass.salpeter1955(m_min, m_max).cdf(m_min/m_1)
        )
        denom = m_1
        res = num/denom

        return res


salpeter1955 = salpeter1955_gen(name="mass_ratio.random.salpeter1955")


class splitpowerlaw_gen(_distn_infrastructure.rv_continuous):
    r"""A random-pairing mass-ratio random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `random.salpeter1955` is:

    .. math::

       f_{Q|M_{1}}(q|m_{1}) = m_{1}\dfrac{f_{M}(qm_{1})}{F_{M}(m_{1})}

    for mass ratio :math:`q \in (0, 1]` and primary_mass :math:`m_{1}
    \in [m_{\min}, m_{\max}]` where :math:`f_{M}` and :math:`F_{M}`
    are the probability density function and the cumulative
    distribution function for the mass random variable of Salpeter
    (1955). The probability density function :math:`f_{M}` is given by

    .. math::
       f_{M}(m)
       =
       A
       \begin{cases}
       m^{c}&\text{ if $m \in [a, s)$,}\\
       s^{c - d}m^{d}&\text{ if $m \in [s, b]$}
       \end{cases}

    where

    .. math::

       A = \dfrac{1}{c + 1}(s^{c + 1} - a^{c + 1}) +
       \dfrac{s^{c - d}}{d + 1}(b^{d + 1} - s^{d + 1}),

    and :math:`m \in [a, b]` for :math:`0 < a`, :math:`a < s`,
    :math:`s < b`, :math:`c < 0`, and :math:`d < 0`. The cumulative
    distribution function is given by

    .. math::
       F_{M}(m) = \int_{m_{\min}}^{m}f_{M}(u}\mathrm{d}\,u.
  
    `random.splitpowerlaw` takes ``m_{1}`` as a shape parameter for
    :math:`m_{1}`, the primary mass, ``s`` as a shape parameter for
    :math:`s`, the break mass, ``m_min`` as a shape parameter for
    :math:`m_{\min}`, the minimum allowed mass, ``m_max`` as a shape
    parameter for :math:`m_{\max}`, the maximum allowed mass, ``c`` as
    a shape parameter for :math:`c`, the inner slope, and ``d`` as a
    shape parameter for :math:`d`, the outer slope.
    
    %(after_notes)s

    See also
    --------
    `dyad.stats.mass.splitpowerlaw`

    References
    ----------
    Malkov, O., and H. Zinnecker. 2001. \'Binary stars and the
    fundamental initial mass function\'. *Monthly Notices of the Royal
    Astronomical Society* 321 (1): 149–54.

    %(example)s

    """
    def _argcheck(self, m_1, s, m_min, m_max, c, d):
        res = (
            (0. < m_min)
            & (m_min < m_max)
            & (m_min < s)
            & (s < m_max)
            & (c < 0.)
            & (d < 0.)
        )

        return res

    def _get_support(self, m_1, s, m_min, m_max, c, d):
        res = m_min/m_1, 1.

        return res

    def _pdf(self, x, m_1, s, m_min, m_max, c, d):
        res = (
            m_1
            *mass.splitpowerlaw(s, m_min, m_max, c, d).pdf(x*m_1)
            /mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(m_1)
        )

        return res

    def _cdf(self, x, m_1, s, m_min, m_max, c, d):
        num = (
            mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(x*m_1)
            - mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(m_min/m_1)
        )
        denom = mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(m_1)
        res = num/denom

        return res

    def _ppf(self, q, m_1, s, m_min, m_max, c, d):
        num = mass.splitpowerlaw(s, m_min, m_max, c, d).ppf(
            q*mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(m_1)
            + mass.splitpowerlaw(s, m_min, m_max, c, d).cdf(m_min/m_1)
        )
        denom = m_1
        res = num/denom

        return res


splitpowerlaw = splitpowerlaw_gen(name="mass_ratio.random.splitpowerlaw")


# class Test(_distn_infrastructure.rv_continuous):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
