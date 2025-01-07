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
    "Orbit",
    "TwoBody",
]

import numpy as np
import scipy as sp
import dyad.constants as constants

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

        Mass of the more-massive body, :math:`m_{1}`.

    m_2 : array-like

        Mass of the less-massive body, :math:`m_{2}`.

    Returns
    -------

    res : ndarray

        Semimajor axis.

    Examples
    --------

    Scalar parameters.

    >>> semimajor_axis_from_period(365.25, 1., 3.00362e-6)
    np.float64(0.9999884101100887)
    
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
    res = np.cbrt(constants.GRAV_CONST*(m_1 + m_2)*p**2./(4.*np.pi**2.))
    res = res[()]

    return res

def period_from_semimajor_axis(a, m_1, m_2):
    """Return the period given the semimajor axis

    Parameters
    ----------

    a : array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    m_1 : array-like

        Mass of the more-massive body, :math:`m_{1}`.

    m_2 : array-like

        Mass of the less-massive body, :math:`m_{2}`.

    Returns
    -------

    res : ndarray

        Semimajor axis.

    Examples
    --------

    Scalar parameters.

    >>> period_from_semimajor_axis(1., 1., 3.00362e-6)
    np.float64(365.25634990292843)
    
    Array-like parameters defining multiple orbits.

    >>> a, m_1, m_2 = [1., 1.], [1., 1.], [3.00362e-6, 3.00362e-6]
    >>> period_from_semimajor_axis(a, m_1, m_2)
    array([365.2563499, 365.2563499])

    """
    a = np.asarray(a)
    m_1 = np.asarray(m_1)
    m_2 = np.asarray(m_2)
    res = np.sqrt(4.*np.pi**2.*a**3./(constants.GRAV_CONST*(m_1 + m_2)))
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
    np.float64(0.5792645075960517)
    
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

    Returns
    -------

    res : ndarray

        Semimajor axis.

    Examples
    --------

    Scalar parameters.

    >>> eccentric_anomaly_from_true_anomaly(1., 0.5)
    np.float64(0.611063702733245)
    
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
    np.float64(1.5155481528799728)
    
    Array-like parameters defining multiple orbits.

    >>> eta, e = [1., 1.], [0.5, 0.5]
    >>> true_anomaly_from_eccentric_anomaly(theta, e)
    array([1.51554815, 1.51554815])    

    """
    eta = np.asarray(eta)
    e = np.asarray(e)
    e = _check_eccentricity(e)
    # The function np.arctan2 returns the principal angle,
    # :math:`\theta \mod 2\pi`. So we work with the principal angle
    # :math:`\eta \mod 2\pi`.
    eta_principal = eta%(2.*np.pi)
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
    np.float64(0.3241942038914112)
    
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
    np.float64(1.4987011335178482)
    
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
    np.float64(2.030806214849156)
    
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

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> primary_semimajor_axis_from_semimajor_axis(1., 0.5)
    np.float64(0.3333333333333333)
    
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

        Semimajor axis of the less-massive body, :math:`a_{2}`.

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

        Semimajor axis of the less-massive body, :math:`a_{2}`.

    q : array-like

        Mass ratio, :math:`q := m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> primary_semimajor_axis_from_secondary_semimajor_axis(1., 0.5)
    np.float64(0.5)
    
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

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    q : array-like

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : ndarray

        Semimajor axis of the less-massive body, :math:`a_{1}`.

    Examples
    --------

    Scalar parameters.

    >>> secondary_semimajor_axis_from_primary_semimajor_axis(1., 0.5)
    np.float64(2.0)
    
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

    Example
    -------

    Scalar parameters defining a single orbit in the perifocal plane.

    >>> dyad.Orbit(1., 1., 0.)

    Scalar parameters defining a single orbit in the observer's frame.

    >>> dyad.Orbit(1., 1., 0., 1., 1., 1.)

    Array-like parameters defining multiple orbits.

    >>> m, a, e = [1., 1.], [1., 1.], [0., 0.]
    >>> orb.Orbit(m, a, e)

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

        self._mass = m
        self._semimajor_axis = a
        self._eccentricity = e
        self._longitude_of_ascending_node = Omega
        self._inclination = i
        self._argument_of_pericentre = omega

    @property
    def mass(self):
        """Get the central mass"""
        return self._mass

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

    # @property
    # def energy(self):
    #     """Get the body's specific total energy"""
    #     res = self.potential(0.) + self.kinetic_energy(0.)

    #     return res

    # @property
    # def angular_momentum_magnitude(self):
    #     """Get the magnitude of the body's specific angular momentum"""
    #     res = np.sqrt(
    #         constants.GRAV_CONST
    #         *self.mass
    #         *self.semimajor_axis
    #         *(1. - self.eccentricity**2.)
    #     )
    #     res = res*constants.KPS

    #     return res

    # @property
    # def angular_momentum(self):
    #     """Get the body's specific angular momentum"""
    #     h_x = self.angular_momentum_magnitude*(
    #         np.sin(self.longitude_of_ascending_node)*np.sin(self.inclination)
    #     )
    #     h_y = -self.angular_momentum_magnitude*(
    #         np.cos(self.longitude_of_ascending_node)*np.sin(self.inclination)
    #     )
    #     h_z = self.angular_momentum_magnitude*np.cos(self.inclination)
    #     res = np.hstack([[h_x, h_y, h_z]]).T

    #     return res

    @property
    def period(self):
        """Get the orbital period"""
        return (
            2.*np.pi*np.sqrt(
                self.semimajor_axis**3.
                /(constants.GRAV_CONST*self.mass)
            )
        )

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
        array([51.58859953, 46.01778013])

        """
        return np.sqrt(
            constants.GRAV_CONST
            *self.mass
            *(2./self.radius(theta) - 1./self.semimajor_axis)
        )*constants.KPS

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

        return np.hstack([[v_x, v_y, v_z]]).T*constants.KPS

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
        array([[  0.5       ,   0.        ,   0.        ,  -0.        ,
                 51.58859953,   0.        ],
               [  0.31903819,   0.49687255,   0.        , -28.94020643,
                 35.7784927 ,   0.        ]])

        """
        return np.hstack([self._position(theta), self._velocity(theta)])

    # def potential(self, theta):
    #     """Return the gravitational potential at the body's position

    #     Parameters
    #     ----------

    #     theta : array-like

    #         True anomaly

    #     Returns
    #     -------

    #     res : ndarray

    #         Potential

    #     Examples
    #     --------

    #     >>> import dyad
    #     >>> orbit = dyad.Orbit(1., 1., 0.5)
    #     >>> orbit.potential([0., 1.])
    #     array([-0.00059182, -0.00050114])
        
    #     """
    #     res = -constants.GRAV_CONST*self.mass/self.radius(theta)

    #     return res

    # def kinetic_energy(self, theta):
    #     """Return the body's specific kinetic energy

    #     Parameters
    #     ----------

    #     theta : array-like

    #         True anomaly

    #     Returns
    #     -------

    #     res : ndarray

    #         Kinetic energy

    #     Examples
    #     --------

    #     >>> import dyad
    #     >>> orbit = dyad.Orbit(1., 1., 0.5)
    #     >>> orbit.kinetic_energy([0., 1.])
    #     array([3.98933787e+09, 3.17427591e+09])
        
    #     """
    #     res = 0.5*self.speed(theta)**2.
    #     res = res*constants.KPS**2.

    #     return res


class TwoBody:
    """A class representing the elliptical orbits of a two-body system

    Represents the two orbits of a bound gravitational two-body
    system. The orbit of the more-massive (resp. less-massive) body is
    defined by its semimajor axis, :math:`a_{1}`
    (resp. :math:`a_{2}`), and eccentricity, :math:`e_{1}`
    (resp. :math:`e_{2}`). Its orientation is defined by the longitude
    of its ascending node, :math:`\Omega_{1}`
    (resp. :math:`\Omega_{2}`), inclination, :math:`i_{1}`
    (resp. :math:`i_{2}`), and argument of pericentre,
    :math:`\omega_{1}` (resp. :math:`\omega_{2}`). It is the case that
    :math:`a_{2} = a_{1}/q`, :math:`e_{2} = e_{1}`. :math:`\Omega_{2}
    = \Omega_{1}`, :math:`i_{2} = i_{1}`, and :math:`\omega_{2} =
    \omega_{1} + \pi`.

    Parameters
    ----------
    m: array-like

        Mass of the more-massive body, :math:`m_{1}`.

    q: array-like
    
        Ratio of the less-massive star to the more-massive star,
        :math:`q := m_{2}/m_{1}`.

    a: array-like

        Semimajor axis of the more-massive body's orbit,
        :math:`a_{1}`.

    e: array-like
    
        Eccentricity of the more-massive body's orbit, :math:`e_{1}`.

    Omega: array-like

        Longitude of the ascending node of the more-massive body's
        orbit, :math:`\Omega_{1}`.

    i: array-like

        Inclination of the more-massive body's orbit, :math:`i_{1}`.

    omega: array-like

        Argument of pericentre of the more-massive body's orbit,
        :math:`\omega_{1}`.

    Example
    -------

    Scalar parameters defining a single binary system in the perifocal plane.

    >>> dyad.TwoBody(1., 1., 1., 0.)

    Scalar parameters defining a single orbit in the observer's frame.

    >>> dyad.TwoBody(1., 1., 1., 0., 1., 1., 1.)

    Array-like parameters defining multiple orbits.

    >>> m, q, a, e = [1., 1.], [1., 1.], [1., 1.], [0., 0.]
    >>> orb.TwoBody(m, q, a, e)

    """
    def __init__(self, m, q, a, e, Omega=0., i=0., omega=0.):
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

        self._mass = m
        self._semimajor_axis = a
        self._eccentricity = e
        self._longitude_of_ascending_node = Omega
        self._inclination = i
        self._argument_of_pericentre = omega

        m_1 = m
        m_2 = q*m_1
        self.primary = Orbit(
            m_2**3./(m_1 + m_2)**2.,
            a,
            e,
            Omega,
            i,
            omega
        )
        self.secondary = Orbit(
            m_1**3./(m_1 + m_2)**2.,
            a/q,
            e,
            Omega,
            i,
            omega + np.pi
        )

    # @property
    # def energy(self):
    #     """Get the total energy of the orbit"""
    #     raise NotImplementedError

    # @property
    # def period(self):
    #     """Get the orbital period"""
    #     return (
    #         2.*np.pi
    #         *np.sqrt(
    #             self.semimajor_axis**3.
    #             /(constants.GRAV_CONST*self.mass)
    #         )
    #     )

    # @property
    # def total_mass(self):
    #     """Get the total energy of the orbit"""
    #     return (1. + q)*self._mass
    
