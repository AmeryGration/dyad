"""
Kinematics

A binary kinematics module. The module contains:
(1) functions that allow pairwise transformations between true anomaly, mean anomaly, and eccentric anomaly,
(2) a class `Orbit` that holds kinematic information about an elliptical orbit, and
(3) a class `Binary` that holds kinematic information about the two orbits of a bound binary system.

Distances are in AU. Times are in days. Velocities are in km/s.

"""

__all__ = [
    "Orbit",
    "Binary",
    "semimajor_axis_from_period",
    "period_from_semimajor_axis",
    "true_anomaly_from_mean_anomaly",
    "true_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_mean_anomaly",
]

import numpy as np
import scipy as sp
import dyad.constants as constants

# from scipy import constants as const
# M_SUN = 1.98847e30
# AU = const.astronomical_unit
# DAY = const.day
# GRAV_CONST = sp.constants.gravitational_constant*M_SUN*DAY**2./AU**3.
# KPS = AU/DAY/1.e3

def _check_eccentricity(e):
    # The the number 0.9999999999999999 < 1.
    # But the number 0.99999999999999999 == 1.
    if not isinstance(e, (int, float)):
        raise TypeError("e must be scalar.")

    if e < 0. or e >= 1.:
        raise ValueError("e must be nonnegative and less than one.")

    return e

def semimajor_axis_from_period(P, m_1, m_2):
    """Return the semimajor axis given the period"""
    return np.cbrt(constants.GRAV_CONST*(m_1 + m_2)*P**2./4./np.pi**2.)

def period_from_semimajor_axis(a, m_1, m_2):
    """Return the period given the semimajor axis"""
    return np.sqrt(4.*np.pi**2.*a**3./constants.GRAV_CONST/(m_1 + m_2))

def true_anomaly_from_mean_anomaly(mu, e):
    """Return the true anomaly modulo :math:`2\pi`."""
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
    """Return the true anomaly modulo :math:`2\pi`."""
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
    """Return the mean anomaly modulo :math:`2\pi`."""
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
    """Return the mean anomaly modulo :math:`2\pi`."""
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
    """Return the eccentric anomaly modulo :math:`2\pi`."""
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


class Orbit:
    def __init__(self, central_mass, elements):
        """A class for specifying a body's orbit

        A test particle in a central potential.

        Parameters
        ----------
        mass: (n,) array-like

            The mass of the attracting body

        elements: (6, n) array-like

            The orbital elements

        Example
        -------

        Scalar parameters defining a single orbit.

        >>> dyad.Orbit(1., [1., 0., 0., 0., 0., 0.])

        Array-like parameters defining multiple orbits.

        >>> mass = [1., 1.]
        >>> elements = [[1., 1.], [0., 0.], [0., 0.], [0., 0.],
        [0., 0.], [0., 0.]]
        >>> orb.Orbit(mass, elements)

        The shape of the items in `elements` must be the same as the
        shape of `mass` but that shape may be arbitrary.

        >>> mass = np.ones(2*3).reshape(2, 3)
        >>> elements = 0.5*np.zeros(6*2*3).reshape(6, 2, 3)
        >>> orb.Orbit(mass, elements)

        """
        ################################################################
        # Check shapes of args are consistent
        ################################################################
        elements = np.asarray(elements)

        if np.isscalar(central_mass):
            if not elements.shape == (6,):
                raise ValueError(
                    "central_mass must be (n,) array_like and elements must "
                    "be (6, n) array_like."
                )
        else:
            central_mass = np.asarray(central_mass)
            if not elements.shape == (6,) + central_mass.shape:
                raise ValueError(
                    "central_mass must be (n,) array_like and elements must "
                    "be (6, n) array_like."
                )

        ################################################################
        # Check values of args
        ################################################################
        if np.any(central_mass <= 0):
            raise ValueError("central_mass must be positive.")
        if np.any(elements[0] <= 0.):
            raise ValueError(
                "first item in elements (semimajor axis) must be positive."
            )
        if not (np.any(0. <= elements[1]) and np.any(elements[1] < 1.)):
            raise ValueError(
                "second item in elements (eccentricity) must be nonnegative "
                "and less than one."
            )
        # if not (np.any(0. <= elements[2]) and np.any(elements[2] < 2.*np.pi)):
        #     raise ValueError(
        #         "third item in elements (true anomaly) must be nonnegative "
        #         "and less than 2 pi."
        #     )
        # if not (np.any(0. <= elements[3]) and np.any(elements[3] < 2.*np.pi)):
        #     raise ValueError(
        #         "fourth item in elements (longitude of ascending node) must "
        #         "be nonnegative and less than 2 pi."
        #     )
        # if not (np.any(0. <= elements[4]) and np.any(elements[4] < np.pi)):
        #     raise ValueError(
        #         "fifth item in elements (inclination) must be nonnegative "
        #         "and less than pi."
        #     )
        # if not (np.any(0. <= elements[5]) and np.any(elements[5] < 2.*np.pi)):
        #     raise ValueError(
        #         "sixth item in elements (argument of pericentre) must be "
        #         "nonnegative and less than 2 pi."
        #     )

        ################################################################
        # Attributes
        ################################################################
        self._central_mass = central_mass
        self._semimajor_axis = elements[0]
        self._eccentricity = elements[1]
        self._true_anomaly = elements[2]
        self._longitude_of_ascending_node = elements[3]
        self._inclination = elements[4]
        self._argument_of_pericentre = elements[5]

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
    def true_anomaly(self):
        """Get the body's true anomaly"""
        return self._true_anomaly

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
    def orbital_elements(self):
        """Get the orbital elements as a dictionary"""
        return dict(
            semimajor_axis=self.semimajor_axis,
            eccentricity=self.eccentricity,
            true_anomaly=self.true_anomaly,
            longitude_of_ascending_node=self.longitude_of_ascending_node,
            inclination=self.inclination,
            argument_of_pericentre=self.argument_of_pericentre
        )

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
    def mean_anomaly(self):
        """Get the body's mean anomaly"""
        return mean_anomaly_from_true_anomaly(
            self.true_anomaly, self.eccentricity
        )

    @property
    def eccentric_anomaly(self):
        """Get the body's eccentric anomaly"""
        return eccentric_anomaly_from_true_anomaly(
            self.true_anomaly, self.eccentricity
        )

    @property
    def angular_momentum_magnitude(self):
        """Get the magnitude of the body's specific angular momentum"""
        return np.sqrt(
            constants.GRAV_CONST*self.central_mass*self.semilatus_rectum
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
            2.*np.pi
            *np.sqrt(
                self.semimajor_axis**3.
                /(constants.GRAV_CONST*self.central_mass)
            )
        )

    @property
    def state(self):
        """Get the orbital state vector in Cartesian coordinates"""
        return np.hstack([self._position, self._velocity])

    @property
    def radius(self):
        """Get the body's radius"""
        return (
            self.semilatus_rectum
            /(1. + self.eccentricity*np.cos(self.true_anomaly))
        )

    @property
    def _position(self):
        """Get the body's position"""
        r = self.radius
        x = r*(
            np.cos(self.longitude_of_ascending_node)
            *(
                np.cos(self.argument_of_pericentre)*np.cos(self.true_anomaly)
                - np.sin(self.argument_of_pericentre)*np.sin(self.true_anomaly)
            )
            - np.cos(self.inclination)*np.sin(self.longitude_of_ascending_node)
            *(
                np.sin(self.argument_of_pericentre)*np.cos(self.true_anomaly)
                + np.cos(self.argument_of_pericentre)*np.sin(self.true_anomaly)
            )
        )
        y = r*(
            np.sin(self.longitude_of_ascending_node)
            *(
                np.cos(self.argument_of_pericentre)*np.cos(self.true_anomaly)
                - np.sin(self.argument_of_pericentre)*np.sin(self.true_anomaly)
            )
            + np.cos(self.inclination)*np.cos(self.longitude_of_ascending_node)
            *(
                np.sin(self.argument_of_pericentre)*np.cos(self.true_anomaly)
                + np.cos(self.argument_of_pericentre)*np.sin(self.true_anomaly)
            )
        )
        z = r*np.sin(self.inclination)*(
            np.sin(self.argument_of_pericentre)*np.cos(self.true_anomaly)
            + np.cos(self.argument_of_pericentre)*np.sin(self.true_anomaly)
        )

        return np.hstack([[x, y, z]]).T
    
    @property
    def speed(self):
        """Get the body's speed"""
        return np.sqrt(
            constants.GRAV_CONST
            *self.central_mass
            *(2./self.radius - 1./self.semimajor_axis)
        )*constants.KPS

    @property
    def _velocity(self):
        """Get the body's velocity"""
        A = (
            2.*np.pi*self.semimajor_axis
            /(self.period*np.sqrt(1. - self.eccentricity**2.))
        )
        v_x = -A*(
            np.cos(self.longitude_of_ascending_node)
            *np.sin(self.true_anomaly + self.argument_of_pericentre)
            + (
                np.cos(self.inclination)
                *np.sin(self.longitude_of_ascending_node)
                *np.cos(self.true_anomaly + self.argument_of_pericentre)
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
            *np.sin(self.true_anomaly + self.argument_of_pericentre)
           - (
                np.cos(self.inclination)
                *np.cos(self.longitude_of_ascending_node)
                *np.cos(self.true_anomaly + self.argument_of_pericentre)
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
            np.cos(self.true_anomaly + self.argument_of_pericentre)
            + self.eccentricity*np.cos(self.argument_of_pericentre)
        )

        return np.hstack([[v_x, v_y, v_z]]).T*constants.KPS


class Binary:
    def __init__(self, m, q, a, e, theta, Omega=0., i=0., omega=0.):
        """A class for specifying the two orbits of a binary system

        Quantities m, a, e, theta, Omega, i, omega are for the primary star.
        
        Parameters
        ----------
        m: (n,) array-like

            The mass of the primary component

        q: (n,) array-like

            The mass ratio, i.e. the ratio of the mass of the
            secondary component to the mass of the primary component.

        elements: (6, n) array-like

            The orbital elements of the primary component

        Example
        -------

        """
        m, q, a, e, theta, Omega, i, omega = np.asarray(
            [m, q, a, e, theta, Omega, i, omega]
        )
        m1 = m
        m2 = q*m1
        elements1 = [a, e, theta, Omega, i, omega]
        elements2 = [a/q, e, theta, Omega, i, omega + np.pi]
        # Propertize these
        self.primary = Orbit(m2**3./(m1 + m2)**2., elements1)
        self.secondary = Orbit(m1**3./(m1 + m2)**2., elements2)
