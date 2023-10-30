__all__ = [
    "longitude_of_ascending_node",
    "inclination",
    "argument_of_pericentre",
    "true_anomaly"
]

import numpy as np
import scipy as sp


class _true_anomaly_gen(sp.stats.rv_continuous):
    r"""The true anomaly random variable

    %(before_notes)s

    Notes
    -----
    The probability density function for `true_anomaly` is:

    .. math::

        f(x, e) = \dfrac{1}{2\pi}\left(1 - e\cos(\eta(x))\right)
        \dfrac{a(e)\sec^{2}(x/2)}{a(e)^{2}\tan^{2}(x/2) + 1}

    where

    .. math::

        \eta(x) = 2\arctan\left(a(e)\tan\left(\dfrac{x}{2}\right)\right).

    and

    .. math::

        a(e) = \sqrt{\dfrac{1 - e}{1 + e}}.

    and where :math:`0 \le x < 2\pi` and :math:`0 \le e < 0` [1]_.

    `true_anomaly` takes ``e`` as a shape parameter for :math:`e`.

    %(after_notes)s

    References
    ----------
    .. [1] Reference

    %(example)s

    """
    def _shape_info(self):
        return [sp.stats._ShapeInfo("e", False, (0., 1.), (True, False))]

    def _argcheck(self, e):
        return (0. <= e) & (e < 1.)

    def _pdf(self, x, e):
        A = 2.*np.pi
        eta = 2*np.arctan(np.sqrt(1. - e)*np.tan(x/2.)/np.sqrt(1. + e))
        # eta = 2*np.arctan2(np.sqrt(1. + e), np.sqrt(1. - e)*np.tan(theta/2.))
        Y = (
            (np.sqrt(1. - e)/(np.sqrt(1. + e)*np.cos(x/2.)**2.))
            /((1. - e)*np.tan(x/2.)**2./(1. + e) + 1.)
        )

        return (1. - e*np.cos(eta))*Y/A

    def _cdf(self, x, e):
        A = 2.*np.pi
        eta = 2*np.arctan(np.sqrt(1. - e)*np.tan(x/2.)/np.sqrt(1. + e))

        return (eta - e*np.sin(eta))/A

    def _rvs(self, e, size=None, random_state=None):
        def f(eta, t):
            return eta - e*np.sin(eta) - t

        def fprime(eta, t):
            return 1. - e*np.cos(eta)

        # Compute mean anomaly
        mu = sp.stats.uniform(0., 2.*np.pi).rvs(size, random_state)
        if e == 0.:
            # True anomaly equal to mean anomaly
            return mu
        else:
            # True anomaly must be computed numerically
            eta = np.array(
                [sp.optimize.fsolve(f, mu_i, mu_i, fprime) for mu_i in mu]
            )
            res = (
                2.*np.arctan(np.sqrt((1. + e)/(1. - e))*np.tan(eta/2.))
            )
            # res = 2.*np.arctan2(
            #     np.sqrt(1. - e), np.sqrt(1. + e)*np.tan(eta/2.)
            # )
            return res.squeeze()


class _rv_uniform_gen(sp.stats.rv_continuous):
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

    def _stats(self):
        raise NotImplementedError
        # return 0.5, 1.0/12, 0, -1.2

    def _entropy(self):
        return 0.0

    @sp.stats._continuous_distns._call_super_mom
    def fit(self, data, *args, **kwds):
        ######################################################################
        # Check this method
        # Add docstring: see method ``fit`` in class ``scipy.stats.uniform``
        ######################################################################
        if len(args) > 0:
            raise TypeError("Too many arguments.")

        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)

        _remove_optimizer_parameters(kwds)

        if floc is not None and fscale is not None:
            # This check is for consistency with `rv_continuous.fit`.
            raise ValueError(
                "All parameters fixed. There is nothing to optimize."
            )

        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        if fscale is None:
            # scale is not fixed.
            if floc is None:
                # loc is not fixed, scale is not fixed.
                loc = data.min()
                scale = data.ptp()
            else:
                # loc is fixed, scale is not fixed.
                loc = floc
                scale = data.max() - loc
                if data.min() < loc:
                    raise FitDataError(
                        "uniform", lower=loc, upper=loc + scale
                    )
        else:
            # loc is not fixed, scale is fixed.
            ptp = data.ptp()
            if ptp > fscale:
                raise FitUniformFixedScaleDataError(ptp=ptp, fscale=fscale)
            # If ptp < fscale, the ML estimate is not unique; see the comments
            # above.  We choose the distribution for which the support is
            # centered over the interval [data.min(), data.max()].
            loc = data.min() - 0.5*(fscale - ptp)
            scale = fscale

        # We expect the return values to be floating point, so ensure it
        # by explicitly converting to float.
        return float(loc), float(scale)


class _longitude_of_ascending_node_gen(_rv_uniform_gen):
    r"""The longitude of the ascending node random variable

    The distribution is uniform on ``[0, 2\pi)``. Using the parameters
    ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + 2.*np.pi*scale]``.

    %(before_notes)s

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _inclination_gen(sp.stats.rv_continuous):
    r"""The inclination random variabe

    %(before_notes)s

    Notes
    -----
    The probability density function for `i` is:

    .. math::

        f(x) = \dfrac{1}{2}\sin(x)

    where :math:`0 \le x \le \pi` [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Reference

    %(example)s

    """
    def _pdf(self, x):
        return 0.5*np.sin(x)

    def _cdf(self, x):
        return 0.5*(1. - np.cos(x))

    def _ppf(self, q):
        return np.arccos(1. - 2.*q)


class _argument_of_pericentre_gen(_rv_uniform_gen):
    r"""The argument of pericentre random variable

    The distribution is uniform on ``[0, 2\pi)``. Using the parameters
    ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + 2.*np.pi*scale]``.

    %(before_notes)s

    %(example)s

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


true_anomaly = _true_anomaly_gen(a=0., b=2.*np.pi, name="true_anomaly")
longitude_of_ascending_node = _longitude_of_ascending_node_gen(
    a=0., b=2.*np.pi, name="longitude_of_ascending_node"
)
inclination = _inclination_gen(
    a=0., b=np.pi, name="inclination"
)
argument_of_pericentre = _argument_of_pericentre_gen(
    a=0., b=2.*np.pi, name="argument_of_pericentre"
)

