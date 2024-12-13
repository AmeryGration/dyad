"""
.. _stats_api:

==========================================
Statistical properties (:mod:`dyad.stats`)
==========================================

.. currentmodule:: dyad.stats

This module contains probability distributions for the masses and
orbital elements of a population of binary stars.

Probability distributions
=========================

Each univariate distribution is an instance of a subclass of Scipy's
:class:`scipy.stats.rv_continuous` class in the same way that
univariate distributions in Scipy are.

.. autosummary::
   :toctree: generated

   longitude_of_ascending_node
   inclination
   argument_of_pericentre
   true_anomaly

Submodules
----------

Where there is a choice of distribution the options are made available
in a submodule. For the distributions provided by a particular
submodule see that module's documentation. Conditional random
variables are implemented using shape parameters to specify the value
of the condition.

.. autosummary::
   :toctree: generated

   mass
   mass_ratio
   semimajor_axis
   period
   log_period
   eccentricity

Warnings/errors used in :mod:`dyad.stats`
-----------------------------------------

Dyad inherits the warnings and errors used in :mod:`scipy.stats`.

See also
--------
scipy.stats.rv_continuous

"""

__all__ = [
    "longitude_of_ascending_node",
    "inclination",
    "argument_of_pericentre",
    "true_anomaly",
]

import numpy as np
import scipy as sp

from . import eccentricity
from . import mass
from . import mass_ratio
from . import period
from . import log_period
from . import semimajor_axis

class _true_anomaly_gen(sp.stats.rv_continuous):
    r"""The random variable for true anomaly

    %(before_notes)s

    Notes
    -----
    The probability density function for ``true_anomaly`` is:

    .. math::
        f_{\Theta}(theta, e) = \dfrac{1}{2\pi}
        \left(1 - e\cos(\eta(theta))\right)
        \dfrac{a(e)\sec^{2}(theta/2)}{a(e)^{2}\tan^{2}(theta/2) + 1}

    where

    .. math::
        \eta(theta) =
        2\arctan\left(a(e)\tan\left(\dfrac{theta}{2}\right)\right).

    and

    .. math::

        a(e) = \sqrt{\dfrac{1 - e}{1 + e}}.

    for :math:`theta \in [0, 2\pi)` and :math:`e \in [0, 1)`.

    The probability density function ``true_anomaly`` takes ``e`` as a
    shape parameter for :math:`e`.

    References
    ----------

    %(example)s

    """
    def _shape_info(self):
        return [sp.stats._ShapeInfo("e", False, (0., 1.), (True, False))]

    def _argcheck(self, e):
        return (0. <= e) & (e < 1.)

    def _pdf(self, x, e):
        A = 2.*np.pi
        eta = 2*np.arctan(np.sqrt(1. - e)*np.tan(x/2.)/np.sqrt(1. + e))
        # eta = 2*np.arctan2(np.sqrt(1. + e), np.sqrt(1. - e)*np.tan(x/2.))
        Y = (
            (np.sqrt(1. - e)/(np.sqrt(1. + e)*np.cos(x/2.)**2.))
            /((1. - e)*np.tan(x/2.)**2./(1. + e) + 1.)
        )
        res = (1. - e*np.cos(eta))*Y/A

        return res

    def _cdf(self, x, e):
        A = 2.*np.pi
        eta = 2.*np.arctan(np.sqrt(1. - e)*np.tan(x/2.)/np.sqrt(1. + e))
        eta = eta%(2.*np.pi)
        res = (eta - e*np.sin(eta))/A
        
        return res

    def _ppf(self, x, e):
        #####################################################################
        #####################################################################
        #####################################################################
        # This is wrong. Correct it
        #####################################################################
        #####################################################################
        #####################################################################
        def f(x, t, e):
            return x - e*np.sin(x) - t

        def fprime(x, t, e):
            return 1. - e*np.cos(x)
        
        def fsolve(x, e):
            # True anomaly must be computed numerically
            eta = sp.optimize.fsolve(f, x, (x, e))#, fprime)
            eta = np.array(eta).squeeze()
            # eta = eta%(2.*np.pi)
            res = 2.*np.arctan(
                np.sqrt((1. + e)/(1. - e))
                *np.tan(eta/2.)
            )

            return res%(2.*np.pi)

        x = 2.*np.pi*np.atleast_1d(x)
        e = np.atleast_1d(e)

        res = np.vectorize(fsolve)(x, e)

        return res

    
class _rv_uniform_gen(sp.stats.rv_continuous):
    r"""A uniform continuous random variable

    The distribution is uniform on ``[0, 2\pi)``. Using the parameters
    ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + 2.*np.pi*scale]``.

    %(before_notes)s

    See Also
    --------
    scipy.stats.uniform

    %(example)s

    """
    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.uniform(0.0, 2.*np.pi, size)

    def _pdf(self, x):
        A = 2.*np.pi
        return (x == x)/A

    def _cdf(self, x):
        A = 2.*np.pi
        return x/A

    def _ppf(self, q):
        A = 2.*np.pi
        return A*q

    # def _stats(self):
    #     return 0.5, 1.0/12, 0, -1.2

    def _entropy(self):
        return 0.0


class _longitude_of_ascending_node_gen(_rv_uniform_gen):
    r"""The random variable for the longitude of the ascending node

    The distribution is uniform on ``[0, 2\pi)``. Using the parameters
    ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + 2.*np.pi*scale]``.

    %(before_notes)s

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _inclination_gen(sp.stats.rv_continuous):
    r"""The random variabe for inclination

    %(before_notes)s

    Notes
    -----
    The probability density function for ``inclination`` is:

    .. math::

        f(x) = \dfrac{1}{2}\sin(x)

    where :math:`0 \le x \le \pi`.

    References
    ----------

    %(example)s

    """
    def _pdf(self, x):
        return 0.5*np.sin(x)

    def _cdf(self, x):
        return 0.5*(1. - np.cos(x))

    def _ppf(self, q):
        return np.arccos(1. - 2.*q)


class _argument_of_pericentre_gen(_rv_uniform_gen):
    r"""The random variabe for the argument of pericentre

    The distribution is uniform on ``[0, 2\pi)``. Using the parameters
    ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + 2.*np.pi*scale]``.

    %(before_notes)s

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


true_anomaly = _true_anomaly_gen(
    a=0., b=2.*np.pi, name="true_anomaly"
)
longitude_of_ascending_node = _longitude_of_ascending_node_gen(
    a=0., b=2.*np.pi, name="longitude_of_ascending_node"
)
inclination = _inclination_gen(
    a=0., b=np.pi, name="inclination"
)
argument_of_pericentre = _argument_of_pericentre_gen(
    a=0., b=2.*np.pi, name="argument_of_pericentre"
)
