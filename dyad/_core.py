"""
This module contains Dyad's core classes and functions. It is private
but the objects it contains are available under the ``dyad``
namespace.

"""

__all__ = [
    "semimajor_axis_from_period",
    "period_from_semimajor_axis",
    "true_anomaly_from_mean_anomaly",
    "true_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_mean_anomaly",
    "primary_semimajor_axis_from_semimajor_axis",
    "secondary_semimajor_axis_from_semimajor_axis",
    "primary_semimajor_axis_from_secondary_semimajor_axis",
    "secondary_semimajor_axis_from_primary_semimajor_axis",
    "delaunay_elements_from_orbital_elements",
    "orbital_elements_from_delaunay_elements",
    "modified_delaunay_elements_from_orbital_elements",
    "orbital_elements_from_modified_delaunay_elements",
    "Orbit",
    "TwoBody",
]

import dyad
import numpy as np
import scipy as sp
import dyad._constants as _constants

def _check_mass(m):
    if not np.all(np.isreal(m)):
        raise TypeError("m must be scalar.")
    if np.any(m <= 0.):
        raise ValueError("m must be positive.")
    res = m

    return res

def _check_eccentricity(e):
    # The the number 0.9999999999999999 < 1.
    # But the number 0.99999999999999999 == 1.
    if not np.all(np.isreal(e)):
        raise TypeError("e must be scalar.")
    if np.any((e < 0.) | (e >= 1.)):
        raise ValueError("e must be nonnegative and less than one.")
    res = e

    return res

def _check_semimajor_axis(a):
    if not np.all(np.isreal(a)):
        raise TypeError("a must be scalar.")
    if np.any(a <= 0.):
        raise ValueError("a must be positive.")
    res = a

    return res

def _check_angle(x):
    if not np.all(np.isreal(x)):
        raise TypeError("Omega, i, and omega must be scalar.")
    res = x

    return res

def _check_period(p):
    if not np.all(np.isreal(p)):
        raise TypeError("p must be scalar.")
    if np.any(p <= 0.):
        raise ValueError("p must be positive.")
    res = p

    return res

def semimajor_axis_from_period(p, m_1, m_2):
    """Return the semimajor axis given the period

    Parameters
    ----------

    p : array-like

        Period

    m_1 : array-like

        Mass of the primary body, :math:`m_{1}`.

    m_2 : array-like

        Mass of the secondary body, :math:`m_{2}`.

    Returns
    -------

    res : ndarray

        Semimajor axis.

    Examples
    --------

    Scalar parameters.

    >>> semimajor_axis_from_period(365.25, 1., 3.00362e-6)
    0.9999884101100887

    Array-like parameters defining multiple orbits.

    >>> p, m_1, m_2 = [365.25, 365.25], [1., 1.], [3.00362e-6, 3.00362e-6]
    >>> semimajor_axis_from_period(p, m_1, m_2)
    array([0.99998841, 0.99998841])

    """
    p = np.asarray(p)
    m_1 = np.asarray(m_1)
    m_2 = np.asarray(m_2)
    p = _check_period(p)
    m_1 = _check_mass(m_1)
    m_2 = _check_mass(m_2)
    res = np.cbrt(_constants.GRAV_CONST*(m_1 + m_2)*p**2./(4.*np.pi**2.))
    res = res[()]

    return res

def period_from_semimajor_axis(a, m_1, m_2):
    """Return the period given the semimajor axis

    Parameters
    ----------

    a : array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    m_1 : array-like

        Mass of the primary body, :math:`m_{1}`.

    m_2 : array-like

        Mass of the secondary body, :math:`m_{2}`.

    Returns
    -------

    res : ndarray

        Semimajor axis.

    Examples
    --------

    Scalar parameters.

    >>> period_from_semimajor_axis(1., 1., 3.00362e-6)
    365.25634990292843

    Array-like parameters defining multiple orbits.

    >>> a, m_1, m_2 = [1., 1.], [1., 1.], [3.00362e-6, 3.00362e-6]
    >>> period_from_semimajor_axis(a, m_1, m_2)
    array([365.2563499, 365.2563499])

    """
    a = np.asarray(a)
    m_1 = np.asarray(m_1)
    m_2 = np.asarray(m_2)
    res = np.sqrt(4.*np.pi**2.*a**3./(_constants.GRAV_CONST*(m_1 + m_2)))
    res = res[()]

    return res

def mean_anomaly_from_eccentric_anomaly(eta, e):
    """Return the mean anomaly

    Parameters
    ----------

    eta : array-like

        Eccentric anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        Mean anomaly.

    Examples
    --------

    Scalar parameters.

    >>> mean_anomaly_from_eccentric_anomaly(1., 0.5)
    0.5792645075960517

    Array-like parameters defining multiple orbits.

    >>> eta, e = [1., 1.], [0.5, 0.5]
    >>> mean_anomaly_from_eccentric_anomaly(eta, e)
    array([0.57926451, 0.57926451])

    """
    eta = np.asarray(eta)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    mu = eta - e*np.sin(eta)
    mu = mu[()]

    return mu

def eccentric_anomaly_from_true_anomaly(theta, e):
    """Return the eccentric anomaly

    Parameters
    ----------

    theta : array-like

        True anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        Eccentric anomaly.

    Examples
    --------

    Scalar parameters.

    >>> eccentric_anomaly_from_true_anomaly(1., 0.5)
    0.611063702733245

    Array-like parameters defining multiple orbits.

    >>> theta, e = [1., 1.], [0.5, 0.5]
    >>> eccentric_anomaly_from_true_anomaly(theta, e)
    array([0.6110637, 0.6110637])

    """
    theta = np.asarray(theta)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    # The function np.arctan2 returns the principal angle,
    # :math:`\eta \mod 2\pi`. So we work with the principal angle
    # :math:`\theta \mod 2\pi`.
    theta_principal = theta%(2.*np.pi)
    with np.errstate(divide="ignore"):
        eta = 2.*np.arctan2(
            np.sqrt(1. - e)/np.sqrt(1. + e), 1./np.tan(theta_principal/2.)
        )
    eta = eta + 2.*np.pi*(theta//(2.*np.pi))
    eta = eta[()]

    return eta

def true_anomaly_from_eccentric_anomaly(eta, e):
    """Return the true anomaly

    Parameters
    ----------

    eta : array-like

        Eccentric anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        True anomaly.

    Examples
    --------

    Scalar parameters.

    >>> true_anomaly_from_eccentric_anomaly(1., 0.5)
    1.515548152879973

    Array-like parameters defining multiple orbits.

    >>> eta, e = [1., 1.], [0.5, 0.5]
    >>> true_anomaly_from_eccentric_anomaly(eta, e)
    array([1.51554815, 1.51554815])

    """
    eta = np.asarray(eta)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    # The function np.arctan2 returns the principal angle,
    # :math:`\theta \mod 2\pi`. So we work with the principal angle
    # :math:`\eta \mod 2\pi`.
    eta_principal = eta%(2.*np.pi)
    with np.errstate(divide="ignore"):
        theta = 2.*np.arctan2(
            np.sqrt(1. + e)/np.sqrt(1. - e), 1./np.tan(eta_principal/2.)
        )
    theta = theta + 2.*np.pi*(eta//(2.*np.pi))
    theta = theta[()]

    return theta

def mean_anomaly_from_true_anomaly(theta, e):
    """Return the mean anomaly

    Parameters
    ----------

    theta : array-like

        True anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        Mean anomaly.

    Examples
    --------

    Scalar parameters.

    >>> mean_anomaly_from_true_anomaly(1., 0.5)
    0.3241942038914112

    Array-like parameters defining multiple orbits.

    >>> theta, e = [1., 1.], [0.5, 0.5]
    >>> mean_anomaly_from_true_anomaly(theta, e)
    array([0.3241942, 0.3241942])

    """
    theta = np.asarray(theta)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    eta = eccentric_anomaly_from_true_anomaly(theta, e)
    mu = mean_anomaly_from_eccentric_anomaly(eta, e)

    return mu

def eccentric_anomaly_from_mean_anomaly(mu, e):
    """Return the eccentric anomaly

    Parameters
    ----------

    mu : array-like

        Mean anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        Eccentric anomaly.

    Examples
    --------

    Scalar parameters.

    >>> eccentric_anomaly_from_mean_anomaly(1., 0.5)
    1.4987011335178482

    Array-like parameters defining multiple orbits.

    >>> mu, e = [1., 1.], [0.5, 0.5]
    >>> eccentric_anomaly_from_mean_anomaly(mu, e)
    array([1.49870113, 1.49870113])

    """
    def f(eta, t, e):
        return mean_anomaly_from_eccentric_anomaly(eta, e) - t

    def f_gradient(eta, t, e):
        return 1. - e*np.cos(eta)

    def solve(x, e):
        # Keyword factor=1. required to avoid numerical instability for big e
        res = sp.optimize.fsolve(f, x, (x, e), f_gradient, factor=1.)
        # Fake it: ensure that the function returns np.float64:
        # 1. res.item() extracts a float;
        # 2. np.asarray(res.item()) casts this as a (1,) np.ndarray;
        # 3. 1*np.asarray(res.item()) casts this as a np.float64.
        # I suspect that the problem can only be solved by making
        # sp.optimize.fsolve a ufunc.
        res = 1.*np.asarray(res.item())

        return res

    mu = np.asarray(mu)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    mu_principal = mu%(2.*np.pi)
    eta = np.vectorize(solve)(mu_principal, e)
    eta = eta + 2.*np.pi*(mu//(2.*np.pi))

    return eta

def true_anomaly_from_mean_anomaly(mu, e):
    """Return the true anomaly

    Parameters
    ----------

    mu : array-like

        Mean anomaly.

    e : array-like

        Eccentricity.

    Returns
    -------

    res : ndarray

        True anomaly.

    Examples
    --------

    Scalar parameters.

    >>> true_anomaly_from_mean_anomaly(1., 0.5)
    2.0308062148491555

    Array-like parameters defining multiple orbits.

    >>> mu, e = [1., 1.], [0.5, 0.5]
    >>> true_anomaly_from_mean_anomaly(mu, e)
    array([2.03080621, 2.03080621])

    """
    mu = np.asarray(mu)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    eta = eccentric_anomaly_from_mean_anomaly(mu, e)
    theta = true_anomaly_from_eccentric_anomaly(eta, e)

    return theta

def primary_semimajor_axis_from_semimajor_axis(a, q):
    """Return the primary semimajor axis given the relative semimajor axis

    Parameters
    ----------

    a : array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    q : array-like

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the primary body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> primary_semimajor_axis_from_semimajor_axis(1., 0.5)
    0.3333333333333333

    Array-like parameters defining multiple orbits.

    >>> a, q = [1., 1.], [0.5, 0.5]
    >>> primary_semimajor_axis_from_semimajor_axis(a, q)
    array([0.33333333, 0.33333333])

    """
    a = np.asarray(a)
    q = np.asarray(q)
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)
    res = q*a/(1. + q)
    res = res[()]

    return res

def secondary_semimajor_axis_from_semimajor_axis(a, q):
    """Return the secondary semimajor axis given the relative semimajor axis

    Parameters
    ----------

    a : array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    q : array-like

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the secondary body, :math:`a_{2}`.

    Examples
    --------

    """
    a = np.asarray(a)
    q = np.asarray(q)
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)
    res = a/(1. + q)
    res = res[()]

    return res

def primary_semimajor_axis_from_secondary_semimajor_axis(a, q):
    """Return the primary semimajor axis given the secondary semimajor axis

    Parameters
    ----------

    a : array-like

        Semimajor axis of the secondary body, :math:`a_{2}`.

    q : array-like

        Mass ratio, :math:`q := m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the primary body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> primary_semimajor_axis_from_secondary_semimajor_axis(1., 0.5)
    0.5

    Array-like parameters defining multiple orbits.

    >>> a, q = [1., 1.], [0.5, 0.5]
    >>> primary_semimajor_axis_from_secondary_semimajor_axis(a, q)
    array([0.5, 0.5])

    """
    a = np.asarray(a)
    q = np.asarray(q)
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)
    res = a*q
    res = res[()]

    return res

def secondary_semimajor_axis_from_primary_semimajor_axis(a, q):
    """Return the secondary semimajor axis given the primary semimajor axis

    Parameters
    ----------

    a : array-like

        Semimajor axis of the primary body, :math:`a_{1}`.

    q : array-like

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the secondary body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> secondary_semimajor_axis_from_primary_semimajor_axis(1., 0.5)
    2.0

    Array-like parameters defining multiple orbits.

    >>> a, q = [1., 1.], [0.5, 0.5]
    >>> secondary_semimajor_axis_from_primary_semimajor_axis(a, q)
    array([2., 2.])

    """
    a = np.asarray(a)
    q = np.asarray(q)
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)
    res = a/q
    res = res[()]

    return res

def delaunay_elements_from_orbital_elements(a, e, Omega, i, omega, theta, m):
    r"""Return the Delaunay elements given the orbital elements

    Consider a body moving on an elliptical orbit in a gravitational
    central potential generated by a central mass of :math:`m`. The
    Delaunay elements are

    .. math::

       J_{1} &= \sqrt{\mathrm{G}ma(1 - e^{2})}\cos(i)\\
       J_{2} &= \sqrt{\mathrm{G}ma(1 - e^{2})}\\
       J_{3} &= \sqrt{\mathrm{G}ma}\\
       \Theta_{1} &= \Omega\\
       \Theta_{2} &= \omega\\
       \Theta_{3} &= \mu(\theta)

    where :math:`a/\text{AU} \in (0, \infty)` is the semimajor axis,
    :math:`e = (0, 1)` is the eccentricity, :math:`\Omega \in [0,
    2\pi)` is the longitude of the ascending node, :math:`i \in (0,
    \pi)` is the inclination, :math:`\omega \in [0, 2\pi)` is the
    argument of pericentre, and :math:`\mu(\theta) \in [0, 2\pi)` is
    the mean anomaly.

    Parameters
    ----------

    a : array-like

        Semimajor axis

    e : array-like

        Eccentricity

    Omega : array-like

        Longitude of the ascending node

    i : array-like

        Incination

    omega : array-like

        Argument of pericentre

    theta : array-like

        True anomaly

    m : array-like

        Central mass

    Returns
    -------

    res : list

        Delaunay elements ``(J_1, J_2, J_3, Theta_1, Theta_2, Theta_3)``

    Warnings
    --------

    Note that :math:`e \neq 0` and :math:`i \neq 0`.

    Examples
    --------

    Scalar parameters.

    >>> dyad.delaunay_elements_from_orbital_elements(1., 0., 0., 0., 0.,
    ...     0., 1.)
    array([0.017202098944262, 0.017202098944262, 0.017202098944262,
           0.               , 0.               , 0.               ])

    Array-like parameters defining multiple orbits.

    >>> a, e, Omega, i, omega, theta, m  = [1., 1.], [0., 0.], [0., 0.],
    ...     [0., 0.], [0., 0.], [0., 0.], [1., 1.]
    >>> dyad.delaunay_elements_from_orbital_elements(a, e, Omega, i,
    ...     omega, theta, m)
    array([[0.017202098944262, 0.017202098944262, 0.017202098944262,
            0.               , 0.               , 0.               ],
           [0.017202098944262, 0.017202098944262, 0.017202098944262,
            0.               , 0.               , 0.               ]])

    """
    a = np.asarray(a)[()]
    e = np.asarray(e)[()]
    theta = np.asarray(theta)[()]
    Omega = np.asarray(Omega)[()]
    i = np.asarray(i)[()]
    omega = np.asarray(omega)[()]
    m = np.asarray(m)[()]

    a = _check_semimajor_axis(a)
    if not np.all(np.isreal(e)):
        raise TypeError("e must be scalar.")
    if np.any((e <= 0.) | (e >= 1.)):
        raise ValueError("e must be positive and less than one.")
    Omega = _check_angle(Omega)
    i = _check_angle(i)
    if np.any(i%(2.*np.pi) == 0.):
        raise ValueError("i mod 2.*pi must be nonzero.")
    omega = _check_angle(omega)
    m = _check_mass(m)

    J_1 = np.sqrt(_constants.GRAV_CONST*m*a*(1. - e**2.))*np.cos(i)
    J_2 = np.sqrt(_constants.GRAV_CONST*m*a*(1. - e**2.))
    J_3 = np.sqrt(_constants.GRAV_CONST*m*a)
    Theta_1 = Omega
    Theta_2 = omega
    Theta_3 = dyad.mean_anomaly_from_true_anomaly(theta, e)
    res = J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

    return res

def orbital_elements_from_delaunay_elements(
        J_1, J_2, J_3, Theta_1, Theta_2, Theta_3, m):
    r"""Return the orbital elements given the Delaunay elements

    Consider a body moving on an elliptical orbit in a gravitational
    central potential generated by a central mass of :math:`m`. The
    orbital elements are

    .. math::

       a &= \dfrac{J_{3}^{2}}{\mathrm{G}m}\\
       e &= \sqrt{1 - J_{2}^{2}/J_{3}^{2}}\\
       \theta &= \theta(\Theta_{3})\\
       \Omega &= \Theta_1\\
       i &= \cos^{-1}(J_{1}/J_{2})\\
       \omega &= \Theta_{2}

    where
    :math:`J_{1}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in (-\infty, \infty)`,
    :math:`J_{2}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in (0, \infty)`,
    :math:`J_{3}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in (0, \infty)`,
    :math:`\Theta_{1} \in (-\infty, \infty)`,
    :math:`\Theta_{2} \in (-\infty, \infty)`, and
    :math:`\Theta_{3} \in (-\infty, \infty)`,
    are the Delaunay elements and where
    :math:`|J_{1}| < J_{2}` and :math:`J_{2} < J_{3}`.

    Parameters
    ----------

    J_1 : array-like

        First Delaunay action

    J_2 : array-like

        Second Delaunay action

    J_3 : array-like

        Third Delaunay action

    Theta_1 : array-like

        First Delaunay angle (longitude of the ascending node)

    Theta_2 : array-like

        Second Delaunay angle (argument of pericentre)

    Theta_3 : array-like

        Third Delaunay angle (mean anomaly)

    m : array-like

        Central mass

    Returns
    -------

    res : tuple

        Orbital elemennts ``(a, e, Omega, i, omega, theta)``

    Examples
    --------

    Scalar parameters.

    >>> dyad.orbital_elements_from_delaunay_elements(1., 0., 0., 0., 0.,
    ...     0., 1.)

    Array-like parameters defining multiple orbits.

    >>> J_1, J_2, J_3, Theta_1, Theta_2, Theta_3, m  = [1., 1.],
    ...     [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.]
    >>> dyad.orbital_elements_from_delaunay_elements(a, e, Omega, i,
    ...     omega, theta, m)

    """
    J_1 = np.asarray(J_1)[()]
    J_2 = np.asarray(J_2)[()]
    J_3 = np.asarray(J_3)[()]
    Theta_1 = np.asarray(Theta_1)[()]
    Theta_2 = np.asarray(Theta_2)[()]
    Theta_3 = np.asarray(Theta_3)[()]
    m = np.asarray(m)[()]

    if not np.all(np.isreal(J_1)):
        raise TypeError("J_1 must be scalar.")
    if not np.all(np.isreal(J_2)):
        raise TypeError("J_2 must be scalar.")
    if np.any(J_2 <= 0.):
        raise ValueError("J_2 must be positive.")
    if not np.all(np.isreal(J_3)):
        raise TypeError("J_3 must be scalar.")
    if np.any(J_3 <= 0.):
        raise ValueError("J_3 must be positive.")
    if not np.any(np.abs(J_1) < J_2):
        raise ValueError("J_1 must be less than J_2.")
    if not np.any(J_2 < J_3):
        raise ValueError("J_2 must be less than J_3.")

    Theta_1 = _check_angle(Theta_1)
    Theta_2 = _check_angle(Theta_2)
    Theta_3 = _check_angle(Theta_3)
    m = _check_mass(m)

    a = J_3**2./(_constants.GRAV_CONST*m)
    e = np.sqrt(1. - (J_2/J_3)**2.)
    theta = dyad.true_anomaly_from_mean_anomaly(Theta_3, e)
    Omega = Theta_1
    i = np.arccos(J_1/J_2)
    omega = Theta_2
    res = a, e, Omega, i, omega, theta

    return res

def modified_delaunay_elements_from_orbital_elements(
        a, e, Omega, i, omega, theta, m):
    r"""Return the modified Delaunay elements given the orbital elements

    Consider a body moving on an elliptical orbit in a gravitational
    central potential generated by a central mass of :math:`m`. The
    modified Delaunay elements are

    .. math::

       J_{\varpi} &= \sqrt{\mathrm{G}ma}(1 - \sqrt{1 - e^{2}})\\
       J_{\Omega} &= \sqrt{\mathrm{G}ma(1 - e^{2})}(1 - \cos(i))\\
       J_{\lambda} &= \sqrt{\mathrm{G}ma}\\
       \Theta_{\varpi} &= -(\Omega + \omega)\\
       \Theta_{\Omega} &= -\Omega\\
       \Theta_{\lambda} &= \Omega + \omega + \mu(\theta)

    where :math:`a/\text{AU} \in (0, \infty)` is the semimajor axis,
    :math:`e = [0, 1)` is the eccentricity, :math:`\Omega \in [0,
    2\pi)` is the longitude of the ascending node, :math:`i \in [0,
    \pi)` is the inclination, :math:`\omega \in [0, 2\pi)` is the
    argument of pericentre, and :math:`\mu(\theta) \in [0, 2\pi)` is
    the mean anomaly.

    Parameters
    ----------

    a : array-like

        Semimajor axis

    e : array-like

        Eccentricity

    Omega : array-like

        Longitude of the ascending node

    i : array-like

        Incination

    omega : array-like

        Argument of pericentre

    theta : array-like

        True anomaly

    m : array-like

        Central mass

    Returns
    -------

    res : tuple

        Modified Delaunay elements ``(J_pi, J_Omega, J_lambda, Theta_pi,
        Theta_Omega, Theta_lambda)``

    Examples
    --------

    Scalar parameters.

    >>> dyad.delaunay_elements_from_orbital_elements(1., 0., 0., 0., 0.,
    ...     0., 1.)
    array([ 0.               ,  0.               ,  0.017202098944262,
           -0.               , -0.               ,  0.               ])

    Array-like parameters defining multiple orbits.

    >>> a, e, Omega, i, omega, theta, m  = [1., 1.], [0., 0.], [0., 0.],
    ...     [0., 0.], [0., 0.], [0., 0.], [1., 1.]
    >>> dyad.delaunay_elements_from_orbital_elements(a, e, Omega, i,
    ...     omega, theta, m)
    array([[ 0.               ,  0.               ,  0.017202098944262,
            -0.               , -0.               ,  0.               ],
           [ 0.               ,  0.               ,  0.017202098944262,
            -0.               , -0.               ,  0.               ]])

    """
    a = np.asarray(a)[()]
    e = np.asarray(e)[()]
    theta = np.asarray(theta)[()]
    Omega = np.asarray(Omega)[()]
    i = np.asarray(i)[()]
    omega = np.asarray(omega)[()]
    m = np.asarray(m)[()]

    a = _check_semimajor_axis(a)
    e = _check_eccentricity(e)
    Omega = _check_angle(Omega)
    i = _check_angle(i)
    omega = _check_angle(omega)
    m = _check_mass(m)

    J_pi = np.sqrt(_constants.GRAV_CONST*m*a)*(1. - np.sqrt(1. - e**2.))
    J_Omega = np.sqrt(_constants.GRAV_CONST*m*a*(1. - e**2.))*(1. - np.cos(i))
    J_lambda = np.sqrt(_constants.GRAV_CONST*m*a)
    Theta_pi = -Omega - omega
    Theta_Omega = -Omega
    Theta_lambda = (
        Omega
        + omega
        + dyad.mean_anomaly_from_true_anomaly(theta, e)
    )
    res = J_pi, J_Omega, J_lambda, Theta_pi, Theta_Omega, Theta_lambda

    return res

def orbital_elements_from_modified_delaunay_elements(
        J_pi, J_Omega, J_lambda, Theta_pi, Theta_Omega, Theta_lambda, m):
    r"""Return the orbital elements given the modified Delaunay elements

    Consider a body moving on an elliptical orbit in a gravitational
    central potential generated by a central mass of :math:`m`. The
    orbital elements are

    .. math::

       a &= \dfrac{J_{\lambda}^{2}}{\mathrm{G}m}\\
       e &= \sqrt{1 - \left(1 - \dfrac{J_{\varpi}}{J_{\lambda}}\right)^{2}}\\
       \theta &= \theta(\Theta_{\lambda} + \Theta_{\varpi})\\
       \Omega &= -\Theta_{\Omega}\\
       i &= \cos^{-1}\left(
           1 - \dfrac{J_{\Omega}}{J_{\lambda} - J_{\varpi}}
       \right)\\
       \omega &= -\Theta_{\varpi} + \Theta_{\Omega}

    where
    :math:`J_{\varpi}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in [0, \infty)`,
    :math:`J_{\Omega}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in [0, \infty)`,
    :math:`J_{\lambda}/(\mathrm{AU}^{2}~\mathrm{d}^{-1}) \in (0, \infty)`,
    :math:`\Theta_{\varpi} \in (-\infty, \infty)`,
    :math:`\Theta_{\Omega} \in (-\infty, \infty)`, and
    :math:`\Theta_{\lambda} \in (-\infty, \infty)`
    are the modified Delaunay elements and where
    :math:`J_{\varpi} < J_{\lambda}` and 
    :math:`J_{\Omega} \le 2(J_{\lambda} - J_{\varpi})`.

    Parameters
    ----------

    J_pi : array-like

        First modified Delaunay action

    J_Omega : array-like

        Second modified Delaunay action

    J_lambda : array-like

        Third modified Delaunay action

    Theta_pi : array-like

        First modified Delaunay angle (longitude of pericentre)

    Theta_Omega : array-like

        Second modified Delaunay angle (longitude of the ascending node)

    Theta_lambda : array-like

        Third modified Delaunay angle (mean longitude)

    m : array-like

        Central mass

    Returns
    -------

    res : tuple

        Orbital elemennts ``(a, e, Omega, i, omega, theta)``

    Examples
    --------

    Scalar parameters.

    >>> dyad.orbital_elements_from_modified_elements(1., 0., 0., 0., 0.,
    ...     0., 1.)
    array([3379.38068342,    0.        ,    0.        ,    0.        ,
              0.        ,    0.        ])

    Array-like parameters defining multiple orbits.

    >>> J_1, J_2, J_3, Theta_1, Theta_2, Theta_3, m  = [1., 1.],
    ...     [0., 0.], [0., 0.],
    ...     [0., 0.], [0., 0.], [0., 0.], [1., 1.]
    >>> dyad.orbital_elements_from_modified_delaunay_elements(a, e,
    ...     Omega, i, omega, theta, m)

    """
    J_pi = np.asarray(J_pi)[()]
    J_Omega = np.asarray(J_Omega)[()]
    J_lambda = np.asarray(J_lambda)[()]
    Theta_pi = np.asarray(Theta_pi)[()]
    Theta_Omega = np.asarray(Theta_Omega)[()]
    Theta_lambda = np.asarray(Theta_lambda)[()]
    m = np.asarray(m)[()]
    
    if not np.all(np.isreal(J_pi)):
        raise TypeError("J_pi must be scalar.")
    if np.any(J_pi < 0.):
        raise ValueError("J_pi must be nonnegative.")
    if not np.all(np.isreal(J_Omega)):
        raise TypeError("J_Omega must be scalar.")
    if np.any(J_Omega < 0.):
        raise ValueError("J_Omega must be nonnegative.")
    if not np.all(np.isreal(J_lambda)):
        raise TypeError("J_lambda must be scalar.")
    if np.any(J_lambda <= 0.):
        raise ValueError("J_lambda must be positive.")
    Theta_pi = _check_angle(Theta_pi)
    Theta_Omega = _check_angle(Theta_Omega)
    Theta_lambda = _check_angle(Theta_lambda)
    m = _check_mass(m)
    
    a = J_lambda**2./(_constants.GRAV_CONST*m)
    e = np.sqrt(1. - (1. - J_pi/J_lambda)**2.)
    theta = true_anomaly_from_mean_anomaly(Theta_lambda + Theta_pi, e)
    Omega = -Theta_Omega
    i = np.arccos(1 - J_Omega/(J_lambda - J_pi))
    omega = -Theta_pi + Theta_Omega
    res = a, e, Omega, i, omega, theta

    return res


class Orbit:
    """A class representing an elliptical orbit

    Represents the bound orbit of a body in a gravitational central
    potential.

    Parameters
    ----------
    m: array-like

        The mass of the body generating the central potential

    a: array-like

        Semimajor axis.

    e: array-like

        Eccentricity.

    Omega: array-like

        Longitude of ascending node.

    i: array-like

        Inclination.

    omega: array-like

        Argument of pericentre.

    Examples
    --------

    Scalar parameters defining a single orbit in the perifocal plane.

    >>> dyad.Orbit(1., 1., 0.)
    <dyad._core.Orbit object at 0x...>
    
    Scalar parameters defining a single orbit in the observer's frame.

    >>> dyad.Orbit(1., 1., 0., 1., 1., 1.)
    <dyad._core.Orbit object at 0x...>

    Array-like parameters defining multiple orbits.

    >>> m, a, e = [1., 1.], [1., 1.], [0., 0.]
    >>> dyad.Orbit(m, a, e)
    <dyad._core.Orbit object at 0x...>
    
    """
    def __init__(self, m, a, e, Omega=0., i=0., omega=0.):
        m = np.asarray(m)[()]
        a = np.asarray(a)[()]
        e = np.asarray(e)[()]
        Omega = np.asarray(Omega)[()]
        i = np.asarray(i)[()]
        omega = np.asarray(omega)[()]
        m = _check_mass(m)
        a = _check_semimajor_axis(a)
        e = _check_eccentricity(e)
        Omega = _check_angle(Omega)
        i = _check_angle(i)
        omega = _check_angle(omega)

        self._central_mass = m
        self._semimajor_axis = a
        self._eccentricity = e
        self._longitude_of_ascending_node = Omega
        self._inclination = i
        self._argument_of_pericentre = omega

    # @property
    def orbital_elements(self, theta=None):
        """Get the orbital elements of the orbit"""
        if theta == None:
            res = (
                self.semimajor_axis,
                self.eccentricity,
                self.longitude_of_ascending_node,
                self.inclination,
                self.argument_of_pericentre,
            )
        else:
            res = (
                self.semimajor_axis,
                self.eccentricity,
                self.longitude_of_ascending_node,
                self.inclination,
                self.argument_of_pericentre,
                theta
            )
        res = np.array(list(np.broadcast(*res))).squeeze()

        return res

    # @property
    def delaunay_elements(self, theta=None):
        """Get the Delaunay elements of the orbit"""
        if theta:
            res = delaunay_elements_from_orbital_elements(
                *self.orbital_elements(theta=theta).T, m=self.central_mass
            )
        else:
            res = delaunay_elements_from_orbital_elements(
                *self.orbital_elements().T, theta=0., m=self.central_mass
            )
            res = res[:-1]
        res = np.array(list(np.broadcast(*res))).squeeze()

        return res

    # @property
    def modified_delaunay_elements(self, theta=None):
        """Get the modified Delaunay elements of the orbit"""
        if theta:
            res = modified_delaunay_elements_from_orbital_elements(
                *self.orbital_elements(theta=theta).T, m=self.central_mass
            )
        else:
            res = modified_delaunay_elements_from_orbital_elements(
                *self.orbital_elements().T, theta=0., m=self.central_mass
            )
            res = res[:-1]
        res = np.array(list(np.broadcast(*res))).squeeze()

        return res

    @property
    def central_mass(self):
        """Get the central mass"""
        return self._central_mass

    @property
    def semimajor_axis(self):
        """Get the orbit's semimajor axis"""
        return self._semimajor_axis

    @property
    def eccentricity(self):
        """Get the orbit's eccentricity"""
        return self._eccentricity

    @property
    def longitude_of_ascending_node(self):
        """Get the longitude of the ascending node of the orbit"""
        return self._longitude_of_ascending_node

    @property
    def inclination(self):
        """Get the inclination of the orbit"""
        return self._inclination

    @property
    def argument_of_pericentre(self):
        """Get the orbit's argument of pericentre"""
        return self._argument_of_pericentre

    @property
    def semiminor_axis(self):
        """Get the orbit's semiminor axis"""
        return self.semimajor_axis*np.sqrt(1. - self.eccentricity**2.)

    @property
    def semilatus_rectum(self):
        """Get the orbit's semilatus rectum"""
        return self.semimajor_axis*(1. - self.eccentricity**2.)

    @property
    def apoapsis(self):
        """Get the apoapsis of the orbit"""
        return self.semimajor_axis*(1. + self.eccentricity)

    @property
    def periapsis(self):
        """Get the periapsis of the orbit"""
        return self.semimajor_axis*(1. - self.eccentricity)

    @property
    def area(self):
        """Get the area contained by the orbit"""
        return np.pi*self.semimajor_axis*self.semiminor_axis

    @property
    def period(self):
        """Get the orbital period"""
        return (
            2.*np.pi*np.sqrt(
                self.semimajor_axis**3.
                /(_constants.GRAV_CONST*self.central_mass)
            )
        )

    @property
    def eccentricity_vector(self):
        """Get the eccentricity vector of the orbit"""
        res = (
            np.cross(self._velocity(0.), self.angular_momentum)
            /(_constants.GRAV_CONST*self.central_mass)
            - self._position(0.)/self.radius(0.)
        )

        return res

    @property
    def lrl_vector(self):
        """Get the Laplace-Runge-Lenz vector of the orbit"""
        res = (
            np.cross(self._velocity(0.), self.angular_momentum)
            - (
                _constants.GRAV_CONST*self.central_mass*self._position(0.)
                /self.radius(0.)
            )
        )

        return res

    @property
    def energy(self):
        """Get the body's specific total energy"""
        res = self.potential(0.) + self.kinetic_energy(0.)

        return res

    @property
    def angular_momentum_magnitude(self):
        """Get the magnitude of the body's specific angular momentum"""
        res = np.sqrt(
            _constants.GRAV_CONST
            *self.central_mass
            *self.semimajor_axis
            *(1. - self.eccentricity**2.)
        )

        return res

    @property
    def angular_momentum(self):
        """Get the body's specific angular momentum"""
        h_x = self.angular_momentum_magnitude*(
            np.sin(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_y = -self.angular_momentum_magnitude*(
            np.cos(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_z = self.angular_momentum_magnitude*np.cos(self.inclination)
        res = np.hstack([[h_x, h_y, h_z]]).T

        return res
    
    def potential(self, theta):
        """Return the gravitational potential at the body's position

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Potential

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.potential([0., 1.])
        array([-0.00059182, -0.00050114])
        
        """
        res = -_constants.GRAV_CONST*self.central_mass/self.radius(theta)

        return res

    def kinetic_energy(self, theta):
        """Return the body's specific kinetic energy

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Kinetic energy

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.kinetic_energy([0., 1.])
        array([3.98933787e+09, 3.17427591e+09])
        
        """
        res = 0.5*self.speed(theta)**2.

        return res

    def mean_anomaly(self, theta):
        """Return the body's mean anomaly

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Mean anomaly

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.mean_anomaly([0., 1.])
        array([0.       , 0.3241942])
        
        """
        return mean_anomaly_from_true_anomaly(theta, self.eccentricity)

    def eccentric_anomaly(self, theta):
        """Return the body's eccentric anomaly

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Eccentric anomaly

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.eccentric_anomaly([0., 1.])
        array([0.       , 0.6110637])
        
        """
        return eccentric_anomaly_from_true_anomaly(theta, self.eccentricity)

    def radius(self, theta):
        """Return the body's radius

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Radius

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.radius([0., 1.])
        array([0.5      , 0.5904809])

        """
        return self.semilatus_rectum/(1. + self.eccentricity*np.cos(theta))

    def speed(self, theta):
        """Return the body's speed

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Speed

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.speed([0., 1.])
        array([0.02979491, 0.02657749])

        """
        return np.sqrt(
            _constants.GRAV_CONST
            *self.central_mass
            *(2./self.radius(theta) - 1./self.semimajor_axis)
        )

    def _position(self, theta):
        r = self.radius(theta)
        x = r*(
            np.cos(self.longitude_of_ascending_node)
            *(
                np.cos(self.argument_of_pericentre)*np.cos(theta)
                - np.sin(self.argument_of_pericentre)*np.sin(theta)
            )
            - np.cos(self.inclination)*np.sin(self.longitude_of_ascending_node)
            *(
                np.sin(self.argument_of_pericentre)*np.cos(theta)
                + np.cos(self.argument_of_pericentre)*np.sin(theta)
            )
        )
        y = r*(
            np.sin(self.longitude_of_ascending_node)
            *(
                np.cos(self.argument_of_pericentre)*np.cos(theta)
                - np.sin(self.argument_of_pericentre)*np.sin(theta)
            )
            + np.cos(self.inclination)*np.cos(self.longitude_of_ascending_node)
            *(
                np.sin(self.argument_of_pericentre)*np.cos(theta)
                + np.cos(self.argument_of_pericentre)*np.sin(theta)
            )
        )
        z = r*np.sin(self.inclination)*(
            np.sin(self.argument_of_pericentre)*np.cos(theta)
            + np.cos(self.argument_of_pericentre)*np.sin(theta)
        )

        return np.hstack([[x, y, z]]).T

    def _velocity(self, theta):
        A = (
            2.*np.pi*self.semimajor_axis
            /(self.period*np.sqrt(1. - self.eccentricity**2.))
        )
        v_x = -A*(
            np.cos(self.longitude_of_ascending_node)
            *np.sin(theta + self.argument_of_pericentre)
            + (
                np.cos(self.inclination)
                *np.sin(self.longitude_of_ascending_node)
                *np.cos(theta + self.argument_of_pericentre)
            )
            + self.eccentricity*(
                np.cos(self.inclination)
                *np.sin(self.longitude_of_ascending_node)
                *np.cos(self.argument_of_pericentre)
                + (
                    np.cos(self.longitude_of_ascending_node)
                    *np.sin(self.argument_of_pericentre)
                )
            )
        )
        v_y = -A*(
            np.sin(self.longitude_of_ascending_node)
            *np.sin(theta + self.argument_of_pericentre)
           - (
                np.cos(self.inclination)
                *np.cos(self.longitude_of_ascending_node)
                *np.cos(theta + self.argument_of_pericentre)
            )
            - self.eccentricity*(
                np.cos(self.inclination)
                *np.cos(self.longitude_of_ascending_node)
                *np.cos(self.argument_of_pericentre)
                - (
                    np.sin(self.longitude_of_ascending_node)
                    *np.sin(self.argument_of_pericentre)
                )
            )
        )
        v_z = A*np.sin(self.inclination)*(
            np.cos(theta + self.argument_of_pericentre)
            + self.eccentricity*np.cos(self.argument_of_pericentre)
        )

        return np.hstack([[v_x, v_y, v_z]]).T

    def state(self, theta):
        """Return the orbital state vector in Cartesian coordinates

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Orbital state in form :math:`x, y, z, v_{x}, v_{y}, v_{z}`.

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.state([0., 1.])
        array([[ 0.5       ,  0.        ,  0.        , -0.        ,  0.02979491,
                 0.        ],
               [ 0.31903819,  0.49687255,  0.        , -0.01671437,  0.02066381,
                 0.        ]])

        """
        return np.hstack([self._position(theta), self._velocity(theta)])


    @property
    def eccentricity_vector(self):
        """Get the eccentricity vector of the orbit"""
        res = (
            np.cross(self._velocity(0.), self.angular_momentum)
            /(_constants.GRAV_CONST*self.central_mass)
            - self._position(0.)/self.radius(0.)
        )

        return res

    @property
    def lrl_vector(self):
        """Get the Laplace-Runge-Lenz vector of the orbit"""
        res = (
            np.cross(self._velocity(0.), self.angular_momentum)
            - (
                _constants.GRAV_CONST*self.central_mass*self._position(0.)
                /self.radius(0.)
            )
        )

        return res

    @property
    def energy(self):
        """Get the body's specific total energy"""
        res = self.potential(0.) + self.kinetic_energy(0.)

        return res

    @property
    def angular_momentum_magnitude(self):
        """Get the magnitude of the body's specific angular momentum"""
        res = np.sqrt(
            _constants.GRAV_CONST
            *self.central_mass
            *self.semimajor_axis
            *(1. - self.eccentricity**2.)
        )

        return res

    @property
    def angular_momentum(self):
        """Get the body's specific angular momentum"""
        h_x = self.angular_momentum_magnitude*(
            np.sin(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_y = -self.angular_momentum_magnitude*(
            np.cos(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_z = self.angular_momentum_magnitude*np.cos(self.inclination)
        res = np.hstack([[h_x, h_y, h_z]]).T

        return res
    
    def potential(self, theta):
        """Return the gravitational potential at the body's position

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Potential

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.potential([0., 1.])
        array([-0.00059182, -0.00050114])
        
        """
        res = -_constants.GRAV_CONST*self.central_mass/self.radius(theta)

        return res

    def kinetic_energy(self, theta):
        """Return the body's specific kinetic energy

        Parameters
        ----------

        theta : array-like

            True anomaly

        Returns
        -------

        res : ndarray

            Kinetic energy

        Examples
        --------

        >>> import dyad
        >>> orbit = dyad.Orbit(1., 1., 0.5)
        >>> orbit.kinetic_energy([0., 1.])
        array([3.98933787e+09, 3.17427591e+09])
        
        """
        res = 0.5*self.speed(theta)**2.

        return res


class TwoBody(Orbit):
    """A class representing the elliptical orbits of a two-body system

    Represents the orbit of the secondary body with respect to the
    primary body as well as the orbit of the primary and secondary
    bodies with respect to the system's centre of mass.

    Parameters
    ----------

    m_1: array-like

        Mass of the primary body, :math:`m_{1}`.

    m_2: array-like

        Mass of the secondary body, :math:`m_{2}`.

    a: array-like

        Semimajor axis of the relative orbit, :math:`a`.

    e: array-like
    
        Eccentricity of the relative orbit, :math:`e`.

    Omega: array-like

        Longitude of the ascending node of the relative orbit,
        :math:`\Omega`.

    i: array-like

        Inclination of the relative orbit, :math:`i`.

    omega: array-like

        Argument of pericentre of the relative orbit, :math:`\omega`.

    Examples
    --------

    Scalar parameters defining a single binary system in the perifocal
    plane.

    >>> dyad.TwoBody(1., 1., 1., 0.)
    <dyad._core.TwoBody object at 0x...>

    Scalar parameters defining a single orbit in the observer's frame.

    >>> dyad.TwoBody(1., 1., 1., 0., 1., 1., 1.)
    <dyad._core.TwoBody object at 0x...>
    
    Array-like parameters defining multiple orbits.

    >>> m, q, a, e = [1., 1.], [1., 1.], [1., 1.], [0., 0.]
    >>> dyad.TwoBody(m, q, a, e)
    <dyad._core.TwoBody object at 0x...>

    The relative position and velocity of the secondary body with
    respect to the primary body in the form :math:`(x, y, z, v_{x}, v_{y},
    v_{z})`.

    >>> dyad.TwoBody(1., 1., 1., 0.).state(1.)
    array([ 0.54030231,  0.84147098,  0.        , -0.02047084,  0.01314417,
            0.        ])

    The position and velocity of the primary and secondary bodies with
    respect to the centre-of-mass frame each again in the form
    :math:`x, y, z, v_{x}, v_{y}, v_{z}`.

    >>> dyad.TwoBody(1., 1., 1., 0.).primary.state(1.)
    array([ 0.27015115,  0.42073549,  0.        , -0.01023542,  0.00657209,
            0.        ])
    >>> dyad.TwoBody(1., 1., 1., 0.).secondary.state(1.)
    array([-0.27015115, -0.42073549, -0.        ,  0.01023542, -0.00657209,
           -0.        ])

    """
    def __init__(self, m_1, m_2, a, e, Omega=0., i=0., omega=0.):
        super().__init__(m_1 + m_2, a, e, Omega, i, omega)

        self.reduced_mass = m_1*m_2/(m_1 + m_2)
        self._primary = Orbit(
            m_2**3./(m_1 + m_2)**2.,
            # m_1*q**3/(1. + q)**2.,
            primary_semimajor_axis_from_semimajor_axis(a, m_2/m_1),
            e,
            Omega,
            i,
            omega
        )
        self._primary.mass = m_1
        self._secondary = Orbit(
            m_1**3./(m_1 + m_2)**2.,
            # m_1/(1. + q)**2.,
            secondary_semimajor_axis_from_semimajor_axis(a, m_2/m_1),
            e,
            Omega,
            i,
            omega + np.pi
        )
        self._secondary.mass = m_2

    
    @property
    def primary(self):
        """Get the primary body's orbit as an instance of :class:`Orbit`"""
        return self._primary

    @property
    def secondary(self):
        """Get the secondary body's orbit as an instance of :class:`Orbit`"""
        return self._secondary
