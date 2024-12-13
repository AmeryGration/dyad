def delaunay_variables(m, elements):
    """Return the modified Delaunay variables"""
    m = np.atleast_1d(m)
    a, e, theta, Omega, i, omega = np.atleast_1d(*elements)

    # Check all args and kwargs have same size
    if not all(len(element) == len(m) for element in elements):
        raiseValueError("items in elements must have save size as m.")
    
    mu = mean_anomaly_from_true_anomaly(theta, e)

    J_1 = np.sqrt(GRAV_CONST*m*a)
    J_2 = np.sqrt(GRAV_CONST*m*a)*(1. - np.sqrt(1. - e**2.))
    J_3 = np.sqrt(GRAV_CONST*m*a*(1. - e**2.))*(1. - np.cos(i))
    Theta_1 = Omega + omega + mu
    Theta_2 = -Omega - mu
    Theta_3 = -Omega

    return np.array([J_1, J_2, J_3, Theta_1, Theta_2, Theta_3])


class Orbit:
    def __init__(self, m, delaunay_vars):
        self.m = m
        self.J_1 = delaunay_vars[0]
        self.J_2 = delaunay_vars[1]
        self.J_3 = delaunay_vars[2]
        self.Theta_1 = delaunay_vars[3]
        self.Theta_2 = delaunay_vars[4]
        self.Theta_3 = delaunay_vars[5]

    @property
    def semimajor_axis(self):
        return 0. # Some function of J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

    @property
    def semiminor_axis(self):
        return np.pi*self.semimajor_axis*self.semiminor_axis

    @property
    def semilatus_rectum(self):
        return np.sqrt(self.semimajor_axis*(1. - self.eccentricity**2.))

    @property
    def area(self):
        return np.pi*self.semimajor_axis*self.semiminor_axis

    @property
    def eccentricity(self):
        return 0. # Some function of J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

    @property
    def true_anomaly(self):
        return true_anomaly_from_mean_anomaly(self.mean_anomaly)

    @property
    def mean_anomaly(self):
        return 0. # Some function of J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

    @property
    def eccentric_anomaly(self):
        return eccentric_anomaly_from_mean_anomaly(self.mean_anomaly)

    @property
    def longitude_of_ascending_node(self):
        return 0.

    @property
    def inclination(self):
        return 0. # Some function of J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

    @property
    def argument_of_pericentre(self):
        return 0. # Some function of J_1, J_2, J_3, Theta_1, Theta_2, Theta_3

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
    def energy(self):
        return 0.

    @property
    def angular_momentum(self):
        # tuple
        return 0., 0., 0.

    @property
    def laplace_runge_lenz(self):
        return 0., 0., 0.

    @property
    def period(self):
        return 0.

    @property
    def radius(self):
        return self._radius()

    @property
    def speed(self):
        return self._speed()

    @property
    def state(self):
        # tuple of floats
        return self._position() + self._velocity()

    def _radius(self):
        return (
            self.semilatus_rectum
            /(1. + self.eccentricity*np.cos(self.true_anomaly))
        )

    def _position(self):
        # tuple of floats
        r = radius(self.true_anomaly, self.semimajor_axis, self.eccentricity)
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

        return x, y, z

    def _speed(self):
        return (
            2.*np.pi*self.semimajor_axis
            /(self.period*np.sqrt(1. - self.eccentricity**2.))
        )

    def _velocity(self):
        # tuple of floats
        v = self.speed(self.true_anomaly)
        v_x = -v*(
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
        v_y = -v*(
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
        v_z = v*np.sin(self.inclination)*(
            np.cos(self.true_anomaly + self.argument_of_pericentre)
            + self.eccentricity*np.cos(self.argument_of_pericentre)
        )

        return v_x, v_y, v_z

