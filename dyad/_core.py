"""
This module contains Dyad's core classes and functions. It is private
but the objects it contains are available under the ``dyad``
namespace.

"""

__all__ = [
    "Orbit",
    "TwoBody",
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
]

import numpy as np
import scipy as sp
import dyad.constants as constants

def _check_mass(m):
    if not isinstance(m, (int, float)) or isinstance(m, bool):
        raise TypeError("m must be scalar.")

    if m <= 0.:
        raise ValueError("m must be positive.")

    return m

def _check_eccentricity(e):
    # The the number 0.9999999999999999 < 1.
    # But the number 0.99999999999999999 == 1.
    if not isinstance(e, (int, float)) or isinstance(e, bool):
        raise TypeError("e must be scalar.")

    if e < 0. or e >= 1.:
        raise ValueError("e must be nonnegative and less than one.")

    return e

def _check_semimajor_axis(a):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        raise TypeError("a must be scalar.")

    if a <= 0.:
        raise ValueError("a must be positive.")

    return a

def _check_period(p):
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise TypeError("p must be scalar.")

    if p <= 0.:
        raise ValueError("p must be positive.")

    return p

def semimajor_axis_from_period(p, m_1, m_2):
    """Return the semimajor axis given the period

    Parameters
    ----------

    p : (d, 2) array-like

        Period

    m_1 : 

        Mass of the more-massive body, :math:`m_{1}`.

    m_2 : 

        Mass of the less-massive body, :math:`m_{2}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    p = _check_period(p)
    m_1 = _check_mass(m_1)
    m_2 = _check_mass(m_2)

    if not np.isscalar(p):
        p = np.asarray(p)
    if not np.isscalar(m_1):
        m_1 = np.asarray(m_1)
    if not np.isscalar(m_2):
        m_2 = np.asarray(m_2)
    
    return np.cbrt(
        constants.GRAV_CONST*(m_1 + m_2)*p**2.
        /(4.*np.pi**2.)
    )

def period_from_semimajor_axis(a, m_1, m_2):
    """Return the period given the semimajor axis

    Parameters
    ----------

    a : (d, 2) array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    m_1 : 

        Mass of the more-massive body, :math:`m_{1}`.

    m_2 : 

        Mass of the less-massive body, :math:`m_{2}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    a = _check_semimajor_axis(a)
    m_1 = _check_mass(m_1)
    m_2 = _check_mass(m_2)

    if not np.isscalar(a):
        a = np.asarray(a)
    if not np.isscalar(m_1):
        m_1 = np.asarray(m_1)
    if not np.isscalar(m_2):
        m_2 = np.asarray(m_2)
    
    return np.sqrt(4.*np.pi**2.*a**3./constants.GRAV_CONST/(m_1 + m_2))

def true_anomaly_from_mean_anomaly(mu, e):
    """Return the true anomaly modulo :math:`2\pi`

    Parameters
    ----------

    mu : (d, 2) array-like

        Mean anomaly.

    e : 

        Eccentricity.

    Returns
    -------

    res : (n, d) ndarray

        True anomaly.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    e = _check_eccentricity(e)

    if not np.isscalar(mu):
        mu = np.asarray(mu)

    if e == 0.:
        theta = mu%(2.*np.pi)

        return theta

    eta = eccentric_anomaly_from_mean_anomaly(mu, e)
    theta = true_anomaly_from_eccentric_anomaly(eta, e)
    theta = theta%(2.*np.pi)

    return theta

def true_anomaly_from_eccentric_anomaly(eta, e):
    """Return the true anomaly modulo :math:`2\pi`

    Parameters
    ----------

    eta : (d, 2) array-like

        Eccentric anomaly.

    e : 
    
        Eccentricity.
    
    Returns
    -------

    res : (n, d) ndarray

        True anomaly.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    e = _check_eccentricity(e)

    if not np.isscalar(eta):
        eta = np.asarray(eta)

    if e == 0.:
        theta = eta%(2.*np.pi)

        return theta

    theta = 2.*np.arctan(np.sqrt(1. + e)*np.tan(eta/2.)/np.sqrt(1. - e))
    theta = (theta + 2.*np.pi)%(2.*np.pi)

    return theta

def mean_anomaly_from_eccentric_anomaly(eta, e):
    """Return the mean anomaly modulo :math:`2\pi`

    Parameters
    ----------

    eta : (d, 2) array-like

        Eccentric anomaly.

    e : 
    
        Eccentricity.
    
    Returns
    -------

    res : (n, d) ndarray

        Mean anomaly.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    e = _check_eccentricity(e)

    if not np.isscalar(eta):
        eta = np.asarray(eta)

    if e == 0.:
        mu = eta%(2.*np.pi)

        return mu

    mu = eta - e*np.sin(eta)
    mu = mu%(2.*np.pi)

    return mu

def mean_anomaly_from_true_anomaly(theta, e):
    """Return the mean anomaly modulo :math:`2\pi`

    Parameters
    ----------

    theta : (d, 2) array-like

        True anomaly.

    e : 
    
        Eccentricity.
    
    Returns
    -------

    res : (n, d) ndarray

        Mean anomaly.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    e = _check_eccentricity(e)

    if not np.isscalar(theta):
        theta = np.asarray(theta)

    if e == 0.:
        mu = theta%(2.*np.pi)

        return mu

    eta = eccentric_anomaly_from_true_anomaly(theta, e)
    mu = mean_anomaly_from_eccentric_anomaly(eta, e)

    return mu

def eccentric_anomaly_from_true_anomaly(theta, e):
    """Return the eccentric anomaly modulo :math:`2\pi`

    Parameters
    ----------

    theta : (d, 2) array-like

        True anomaly.

    e : 
    
        Eccentricity.
    
    Returns
    -------

    res : (n, d) ndarray

        Eccentric anomaly.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    e = _check_eccentricity(e)

    if not np.isscalar(theta):
        theta = np.asarray(theta)

    if e == 0.:
        eta = theta%(2.*np.pi)

        return eta

    eta = 2.*np.arctan(np.sqrt(1. - e)*np.tan(theta/2.)/np.sqrt(1. + e))
    eta = (eta + 2.*np.pi)%(2.*np.pi)

    return eta

def eccentric_anomaly_from_mean_anomaly(mu, e):
    """Return the eccentric anomaly modulo :math:`2\pi`

    Parameters
    ----------

    mu : (d, 2) array-like

        Mean anomaly.

    e : 
    
        Eccentricity.
    
    Returns
    -------

    res : (n, d) ndarray

        Eccentric anomaly.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    def f(eta, t):
        return mean_anomaly_from_eccentric_anomaly(eta, e) - t

    def f_gradient(eta, t):
        return 1. - e*np.cos(eta)

    def solve(x):
        # Keyword factor=1. required to avoid numerical instability for big e
        x = x%(2.*np.pi)

        res = sp.optimize.fsolve(f, x, x, f_gradient, factor=1.)
        # Fake it: ensure that the function returns np.float64:
        # 1. res.item() extracts a float;
        # 2. np.asarray(res.item()) casts this as a (1,) np.ndarray;
        # 3. 1*np.asarray(res.item()) casts this as a np.float64.
        # I suspect that the problem can only be solved by making
        # sp.optimize.fsolve a ufunc.
        res = 1.*np.asarray(res.item())

        return res

    e = _check_eccentricity(e)

    if not np.isscalar(mu):
        mu = np.asarray(mu)

    if e == 0.:
        eta = mu%(2.*np.pi)

        return eta

    eta = np.vectorize(solve)(mu)
    # eta = solve(mu)

    return eta

def primary_semimajor_axis_from_semimajor_axis(a, q):
    """Return the primary semimajor axis given the relative semimajor axis

    Parameters
    ----------

    a : (d, 2) array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    q : 

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)

    if not np.isscalar(a):
        a = np.asarray(a)
    if not np.isscalar(q):
        q = np.asarray(q)
    
    return q*a/(1. + q)

def secondary_semimajor_axis_from_semimajor_axis(a, q):
    """Return the secondary semimajor axis given the relative semimajor axis

    Parameters
    ----------

    a : (d, 2) array-like

        Total semimajor axis, :math:`a = a_{1} + a_{2}`.

    q : 

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis of the less-massive body, :math:`a_{2}`.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)

    if not np.isscalar(a):
        a = np.asarray(a)
    if not np.isscalar(q):
        q = np.asarray(q)
    
    return a/(1. + q)

def primary_semimajor_axis_from_secondary_semimajor_axis(a, q):
    """Return the primary semimajor axis given the secondary semimajor axis

    Parameters
    ----------

    a : (d, 2) array-like

        Semimajor axis of the less-massive body, :math:`a_{2}`.

    q : 

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)

    if not np.isscalar(a):
        a = np.asarray(a)
    if not np.isscalar(q):
        q = np.asarray(q)

    return a*q

def secondary_semimajor_axis_from_primary_semimajor_axis(a, q):
    """Return the secondary semimajor axis given the primary semimajor axis

    Parameters
    ----------

    a : (d, 2) array-like

        Semimajor axis of the more-massive body, :math:`a_{1}`.

    q : 

        Mass ratio, :math:`q = m_{2}/m_{1}`.

    Returns
    -------

    res : (n, d) ndarray

        Semimajor axis of the less-massive body, :math:`a_{1}`.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Examples
    --------

    """
    a = _check_semimajor_axis(a)
    q = _check_semimajor_axis(q)

    if not np.isscalar(a):
        a = np.asarray(a)
    if not np.isscalar(q):
        q = np.asarray(q)

    return a/q


class Orbit:
    """A class representing an elliptical orbit

    Represents the bound orbit of a body in a gravitational central
    potential.

    Parameters
    ----------
    m: (n,) array-like

        The mass of the body generating the central potential

    a: (n,) array-like

        Semimajor axis.

    e: (n,) array-like
    
        Eccentricity.

    Omega: (n,) array-like

        Longitude of ascending node.

    i: (n,) array-like

        Inclination.

    omega: (n,) array-like

        Argument of pericentre.

    Example
    -------

    Scalar parameters defining a single orbit in the perifocal plane.

    >>> dyad.Orbit(1., 1., 0., 0.)

    Scalar parameters defining a single orbit in the observer's frame.

    >>> dyad.Orbit(1., 1., 0., 0., 1., 1., 1.)

    Array-like parameters defining multiple orbits.

    >>> m, a, e = [1., 1.], [1., 1.], [0., 0.]
    >>> orb.Orbit(m, a, e)

    """
    def __init__(self, m, a, e, Omega=0., i=0., omega=0.):
        if not np.isscalar(m):
            m = np.asarray(m)
        if not np.isscalar(a):
            a = np.asarray(a)
        if not np.isscalar(e):
            e = np.asarray(e)

        m = _check_mass(m)
        a = _check_semimajor_axis(a)
        e = _check_eccentricity(e)
        # if np.any(m <= 0.):
        #     raise ValueError("m must be positive.")
        # if np.any(a <= 0.):
        #     raise ValueError("a must be positive.")
        # if not (np.any(0. <= e) and np.any(e < 1.)):
        #     raise ValueError("e must be nonnegative and less than one.")
        
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

    # @property
    # def orbital_elements(self):
    #     """Get the orbital elements as a dictionary"""
    #     return dict(
    #         semimajor_axis=self.semimajor_axis,
    #         eccentricity=self.eccentricity,
    #         true_anomaly=self.true_anomaly,
    #         longitude_of_ascending_node=self.longitude_of_ascending_node,
    #         inclination=self.inclination,
    #         argument_of_pericentre=self.argument_of_pericentre
    #     )

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
    def energy(self):
        """Get the energy of the orbit"""
        raise NotImplementedError

    @property
    def angular_momentum_magnitude(self):
        """Get the magnitude of the body's specific angular momentum"""
        return np.sqrt(
            constants.GRAV_CONST*self.mass*self.semilatus_rectum
        )*constants.AU*constants.KPS

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

        return np.hstack([[h_x, h_y, h_z]]).T*constants.AU*constants.KPS

    # @property
    # def laplace_runge_lenz_magnitude(self):
    #     return self.eccentricity

    # @property
    # def laplace_runge_lenz(self):
    #     lrl_magnitude = self.laplace_runge_lenz_magnitude
    #     lrl_x = lrl_magnitude*(
    #         np.cos(self.longitude_of_ascending_node)
    #         *np.cos(self.argument_of_pericentre)
    #         - (
    #             np.sin(self.longitude_of_ascending_node)
    #             *np.cos(self.inclination)
    #             *np.sin(self.argument_of_pericentre)
    #         )
    #     )
    #     lrl_y = lrl_magnitude*(
    #         np.sin(self.longitude_of_ascending_node)
    #         *np.cos(self.argument_of_pericentre)
    #         + (
    #             np.cos(self.longitude_of_ascending_node)
    #             *np.cos(self.inclination)
    #             *np.sin(self.argument_of_pericentre)
    #         )
    #     )
    #     lrl_z = lrl_magnitude*(
    #         np.sin(self.inclination)*np.sin(self.argument_of_pericentre)
    #     )

    #     return np.hstack([lrl_x, lrl_y, lrl_z])

    @property
    def period(self):
        """Get the orbital period"""
        return (
            2.*np.pi*np.sqrt(
                self.semimajor_axis**3.
                /(constants.GRAV_CONST*self.mass)
            )
        )

    @property
    def energy(self):
        """Get the total energy of the body"""
        raise NotImplementedError

    def mean_anomaly(self, theta):
        """Get the body's mean anomaly"""
        return mean_anomaly_from_true_anomaly(theta, self.eccentricity)

    def eccentric_anomaly(self, theta):
        """Get the body's eccentric anomaly"""
        return eccentric_anomaly_from_true_anomaly(theta, self.eccentricity)

    def state(self, theta):
        """Get the orbital state vector in Cartesian coordinates"""
        return np.hstack([self._position, self._velocity])

    def radius(self, theta):
        """Get the body's radius"""
        return self.semilatus_rectum/(1. + self.eccentricity*np.cos(theta))

    def _position(self, theta):
        """Get the body's position"""
        r = self.radius
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
    
    def potential(self, theta):
        """Get the gravitational potential at the body's position"""
        raise NotImplementedError
    
    def speed(self, theta):
        """Get the body's speed"""
        return np.sqrt(
            constants.GRAV_CONST
            *self.mass
            *(2./self.radius - 1./self.semimajor_axis)
        )*constants.KPS

    def _velocity(self, theta):
        """Get the body's velocity"""
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
    m: (n,) array-like

        Mass of the more-massive body, :math:`m_{1}`.

    q: (n,) array-like
    
        Ratio of the less-massive star to the more-massive star,
        :math:`q := m_{2}/m_{1}`.

    a: (n,) array-like

        Semimajor axis of the more-massive body's orbit,
        :math:`a_{1}`.

    e: (n,) array-like
    
        Eccentricity of the more-massive body's orbit, :math:`e_{1}`.

    Omega: (n,) array-like

        Longitude of the ascending node of the more-massive body's
        orbit, :math:`\Omega_{1}`.

    i: (n,) array-like

        Inclination of the more-massive body's orbit, :math:`i_{1}`.

    omega: (n,) array-like

        Argument of pericentre of the more-massive body's orbit,
        :math:`\omega_{1}`.

    Example
    -------

    """
    def __init__(self, m, q, a, e, Omega=0., i=0., omega=0.):
        if not np.isscalar(m):
            m = np.asarray(m)
        if not np.isscalar(1):
            q = np.asarray(q)
        if not np.isscalar(a):
            a = np.asarray(a)
        if not np.isscalar(e):
            e = np.asarray(e)

        # m = _check_mass(m)
        # q = _check_mass(q)
        # a = _check_semimajoraxis(a)
        # e = _check_eccentricity(e)

        self._mass = m
        self._mass_ratio = q
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

    
