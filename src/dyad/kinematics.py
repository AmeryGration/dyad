__all__ = [
    "Orbit",
    "true_anomaly_from_mean_anomaly",
    "true_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_mean_anomaly",
]
    
import numpy as np
import scipy as sp

GRAV_CONST = sp.constants.gravitational_constant

def _check_eccentricity(e):
    # The the number 0.9999999999999999 < 1.
    # But the number 0.99999999999999999 == 1.
    if not isinstance(e, (int, float)):
        raise TypeError("e must be scalar.")

    if e < 0. or e >= 1.: 
        raise ValueError("e must be nonnegative and less than one.")

    return e

def true_anomaly_from_mean_anomaly(mu, e):
    """Return the true anomaly modulo :math:`2\pi`."""
    e = _check_eccentricity(e)

    # Coerce array `m` here. Later squeeze it to get float as required. See
    # https://stackoverflow.com/questions/29318459/python-function-that-
    # handles-scalar-or-arrays
    mu = np.asarray(mu)
    if np.isscalar(mu):
        is_scalar = True
    else:
        is_scalar = False
    
    if e == 0.:
        theta = mu%(2.*np.pi)
        if is_scalar:
            return theta.squeeze()

        return theta

    eta = eccentric_anomaly_from_mean_anomaly(mu, e)
    theta = true_anomaly_from_eccentric_anomaly(eta, e)
    theta = theta%(2.*np.pi)

    if is_scalar:
        return theta.squeeze()
    
    return theta

def true_anomaly_from_eccentric_anomaly(eta, e):
    """Return the true anomaly modulo :math:`2\pi`."""
    e = _check_eccentricity(e)

    eta = np.asarray(eta)
    if np.isscalar(eta):
        is_scalar = True
    else:
        is_scalar = False
    
    if e == 0.:
        theta = eta%(2.*np.pi)
        if is_scalar:
            return theta.squeeze()

        return theta

    theta = 2.*np.arctan(np.sqrt(1. + e)*np.tan(eta/2.)/np.sqrt(1. - e))
    theta = (theta + 2.*np.pi)%(2.*np.pi)

    if is_scalar:
        return theta.squeeze()
    
    return theta

def mean_anomaly_from_eccentric_anomaly(eta, e):
    """Return the mean anomaly modulo :math:`2\pi`."""
    e = _check_eccentricity(e)

    eta = np.asarray(eta)
    if np.isscalar(eta):
        is_scalar = True
    else:
        is_scalar = False
    
    if e == 0.:
        mu = eta%(2.*np.pi)
        if is_scalar:
            return theta.squeeze()

        return mu

    mu = eta - e*np.sin(eta)
    mu = mu%(2.*np.pi)

    if is_scalar:
        return mu.squeeze()

    return mu

def mean_anomaly_from_true_anomaly(theta, e):
    """Return the mean anomaly modulo :math:`2\pi`."""
    e = _check_eccentricity(e)

    theta = np.asarray(theta)
    if np.isscalar(theta):
        is_scalar = True
    else:
        is_scalar = False

    if e == 0.:
        mu = theta%(2.*np.pi)
        if is_scalar:
            return mu.squeeze()
        
        return mu

    eta = eccentric_anomaly_from_true_anomaly(theta, e)
    mu = mean_anomaly_from_eccentric_anomaly(eta, e)

    if is_scalar:
        return mu.squeeze()

    return mu

def eccentric_anomaly_from_true_anomaly(theta, e):
    """Return the eccentric anomaly modulo :math:`2\pi`."""
    e = _check_eccentricity(e)

    theta = np.asarray(theta)
    if np.isscalar(theta):
        is_scalar = True
    else:
        is_scalar = False

    if e == 0.:
        eta = theta%(2.*np.pi)
        if is_scalar:
            return eta.squeeze()
        
        return eta

    eta = 2.*np.arctan(np.sqrt(1. - e)*np.tan(theta/2.)/np.sqrt(1. + e))
    eta = (eta + 2.*np.pi)%(2.*np.pi)
    if is_scalar:
        return eta.squeeze()

    return eta

def eccentric_anomaly_from_mean_anomaly(mu, e):
    # Numerically for e > 0.98
    def f(eta, t):
        return mean_anomaly_from_eccentric_anomaly(eta, e) - t

    def fprime(eta, t):
        return 1. - e*np.cos(eta)

    def solve(x):
        # Keyword factor=1. required to avoid numerical instability for large e
        x = x%(2.*np.pi)

        res = sp.optimize.fsolve(f, x, x, fprime, factor=1.)
        # Fake it: ensure that the function returns np.float64:
        # 1. res.item() extracts a float;
        # 2. np.asarray(res.item()) casts this as a (1,) np.ndarray;
        # 3. 1*np.asarray(res.item()) casts this as a np.float64.
        # I suspect that the problem can only be solved by making
        # sp.optimize.fsolve a ufunc.
        res = 1.*np.asarray(res.item())

        return res
        
    e = _check_eccentricity(e)

    mu = np.asarray(mu)
    if np.isscalar(mu):
        is_scalar = True
    else:
        is_scalar = False

    if e == 0.:
        eta = mu%(2.*np.pi)
        if is_scalar:
            return eta.squeeze()

        return eta

    # eta = np.vectorize(solve)(mu)
    eta = solve(mu)
    
    if is_scalar:
        return eta.squeeze()

    return eta


class Orbit:
    def __init__(self, mass, elements):
        mass = np.asarray(mass)
        elements = np.asarray(elements)

        if not (mass.ndim == 0 or mass.ndim == 1):
            raise ValueError("mass must be a scalar or 1d array.")
        if not (len(elements) == 6):
            raise ValueError("elements must contain 6 items.")
        if not (elements.ndim == 1 or elements.ndim == 2):
            raise ValueError("elements must be a 1d or 2d array.")
        try:
            # If m is not a 0d array
            if not (mass.shape[0] == elements.shape[0]):
                raise ValueError("mass and elements must have same length.")
        except IndexError:
            # If m is a 0d array 
            if not elements.ndim == 1:
                raise ValueError("mass and elements must have same length.")
        if np.any(mass <= 0.):
            raise ValueError("mass must be positive.")
        if np.any(elements[0] <= 0.):
            raise ValueError(
                "first item in elements (semimajor axis) must be positive."
            )
        if np.any(elements[1] < 0.) or np.any(elements[1] >= 1.): 
            raise ValueError(
                "second item in elements (eccentricity) must be nonnegative "
                "and less than one."
            )

        self.mass = mass
        self.semimajor_axis = elements[0]
        self.eccentricity = elements[1]
        self.true_anomaly = elements[2]%(2.*np.pi)
        self.longitude_of_ascending_node = elements[3]%(2.*np.pi)
        self.inclination = elements[4]%np.pi
        self.argument_of_pericentre = elements[5]%(2.*np.pi)

    @property
    def orbital_elements(self):
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
        return self.semimajor_axis*np.sqrt(1. - self.eccentricity**2.)

    @property
    def semilatus_rectum(self):
        return np.sqrt(self.semimajor_axis*(1. - self.eccentricity**2.))

    @property
    def area(self):
        return np.pi*self.semimajor_axis*self.semiminor_axis

    @property
    def mean_anomaly(self):
        return mean_anomaly_from_true_anomaly(
            self.true_anomaly, self.eccentricity
        )

    @property
    def eccentric_anomaly(self):
        return eccentric_anomaly_from_mean_anomaly(
            self.mean_anomaly, self.eccentricity
        )

    @property
    def energy(self):
        return -0.5*GRAV_CONST*self.mass/self.semimajor_axis

    @property
    def angular_momentum_magnitude(self):
        return np.sqrt(
            GRAV_CONST
            *self.mass
            *self.semimajor_axis
            *(1. - self.eccentricity**2.)
        )
    
    @property
    def angular_momentum(self):
        h = self.angular_momentum_magnitude
        h_x = h*(
            np.sin(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_y = -h*(
            np.cos(self.longitude_of_ascending_node)*np.sin(self.inclination)
        )
        h_z = h*np.cos(self.inclination)

        return np.hstack([h_x, h_y, h_z])

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
        return 2.*np.pi*self.semimajor_axis**1.5/np.sqrt(GRAV_CONST*self.mass)

    @property
    def state(self):
        # tuple of floats
        return np.hstack([self._position(), self._velocity()])

    @property
    def radius(self):
        return (
            self.semilatus_rectum
            /(1. + self.eccentricity*np.cos(self.true_anomaly))
        )

    def _position(self):
        # tuple of floats
        r = self.radius()
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

        return np.hstack([x, y, z])

    @property
    def speed(self):
        return (
            2.*np.pi*self.semimajor_axis
            /(self.period*np.sqrt(1. - self.eccentricity**2.))
        )

    def _velocity(self):
        # tuple of floats
        v_magnitude = self.speed()
        v_x = -v_magnitude*(
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
        v_y = -v_magnitude*(
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
        v_z = v_magnitude*np.sin(self.inclination)*(
            np.cos(self.true_anomaly + self.argument_of_pericentre)
            + self.eccentricity*np.cos(self.argument_of_pericentre)
        )

        return np.hstack([v_x, v_y, v_z])
