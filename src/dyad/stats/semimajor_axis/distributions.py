"""Distributions

"""

__all__ = [
    "opik1924",
]

import numpy as np
import scipy as sp

class _opik1924_gen(sp.stats._continuous_distns.reciprocal_gen):
    r"""The semimajor-axis random variable of Öpik (1924)

    %(before_notes)s

    Notes
    -----
    The probability density function for `opik1924` is:

    .. math::

        f(x) =

    where

    .. math::

        A :=

    :math:`x > 0` [1]_.

    %(after_notes)s

    References
    ----------

    .. [1] Öpik, E. 1924. `Statistical studies of double stars: on the
    distribution of relative luminosities and distances of double stars in
    the Harvard Revised Photometry North of Declination---31°’. Publications
    of the Tartu Astrofizica Observatory 25 (January):1.
    
    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

opik1924 = _opik1924_gen(name="opik1924")
