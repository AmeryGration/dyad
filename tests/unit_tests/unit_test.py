"""
Test functions of _core module

"""

import unittest
import numpy as np
import dyad
import data_test

from parameterized import parameterized, parameterized_class


class TestFunctions(unittest.TestCase):
    @parameterized.expand(data_test.check_eccentricity_type)
    def test_check_eccentricity(self, x):
        self.assertRaises(TypeError, dyad._core._check_eccentricity, x)
        
    @parameterized.expand(data_test.check_eccentricity_value)
    def test_check_eccentricity(self, x):
        self.assertRaises(ValueError, dyad._core._check_eccentricity, x)

    @parameterized.expand(data_test.true_anomaly_from_mean_anomaly)
    def test_true_anomaly_from_mean_anomaly(self, x, target):
        result = dyad.true_anomaly_from_mean_anomaly(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.true_anomaly_from_eccentric_anomaly)
    def test_true_anomaly_from_eccentric_anomaly(self, x, target):
        result = dyad.true_anomaly_from_eccentric_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.mean_anomaly_from_eccentric_anomaly)
    def test_mean_anomaly_from_eccentric_anomaly(self, x, target):
        result = dyad.mean_anomaly_from_eccentric_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.mean_anomaly_from_true_anomaly)
    def test_mean_anomaly_from_true_anomaly(self, x, target):
        result = dyad.mean_anomaly_from_true_anomaly(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.eccentric_anomaly_from_true_anomaly)
    def test_eccentric_anomaly_from_true_anomaly(self, x, target):
        result = dyad.eccentric_anomaly_from_true_anomaly(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.eccentric_anomaly_from_mean_anomaly)
    def test_eccentric_anomaly_from_mean_anomaly(self, x, target):
        result = dyad.eccentric_anomaly_from_mean_anomaly(*x)
        self.assertAlmostEqual(result, target)


class TestOrbit(unittest.TestCase):
    def setUp(self):
        self.m = 0.5488135039273248
        self.a = 0.7151893663724195
        self.e = 0.6027633760716439
        self.Omega = 0.4236547993389047
        self.i = 0.6458941130666561
        self.omega = 0.4375872112626925
        # self.theta = 

        self.orbit = dyad.Orbit(
            self.m, self.a, self.e, self.Omega, self.i, self.omega
        )

    @parameterized.expand(data_test.initialization)
    def test_initialization(self, x):
        self.assertRaises(ValueError, dyad.Orbit, *x)

    def test_mass(self):
        result = self.orbit.mass
        target = self.m

        self.assertEqual(result, target)

    def test_semimajor_axis(self):
        result = self.orbit.semimajor_axis
        target = self.a

        self.assertEqual(result, target)
        
    def test_eccentricity(self):
        result = self.orbit.eccentricity
        target = self.e

        self.assertEqual(result, target)

    # def true_anomaly(self):
    #     result = self.orbit.true_anomaly
    #     target = self.theta

    #     self.assertEqual(result, target)

    def test_longitude_of_ascending_node(self):
        result = self.orbit.longitude_of_ascending_node
        target = self.Omega

        self.assertEqual(result, target)

    def test_inclination(self):
        result = self.orbit.inclination
        target = self.i

        self.assertEqual(result, target)

    def test_argument_of_pericentre(self):
        result = self.orbit.argument_of_pericentre
        target = self.omega

        self.assertEqual(result, target)

    # def test_orbital_elements(self):
    #     result = self.orbit.orbital_elements
    #     target = dict(
    #         semimajor_axis=self.a,
    #         eccentricity=self.e,
    #         true_anomaly=self.theta,
    #         longitude_of_ascending_node=self.Omega,
    #         inclination=self.i,
    #         argument_of_pericentre=self.omega
    #     )

    #     self.assertEqual(result, target)

    def test_semiminor_axis(self):
        result = self.orbit.semiminor_axis
        target = 0.5706638929715594

        self.assertEqual(result, target)

    def test_semilatus_rectum(self):
        result = self.orbit.semilatus_rectum
        target = 0.4553441284973978
        
        self.assertEqual(result, target)

    def test_apoapsis(self):
        result = self.orbit.apoapsis
        target = 1.146279323377599

        self.assertEqual(result, target)

    def test_periapsis(self):
        result = self.orbit.periapsis
        target = 0.2840994093672401

        self.assertEqual(result, target)

    def test_area(self):
        result = self.orbit.area
        target = 1.2821868428877317

        self.assertEqual(result, target)

    # def test_mean_anomaly(self):
    #     result = self.orbit.mean_anomaly
    #     target = 0.11191292831895089

    #     self.assertEqual(result, target)

    # def test_eccentric_anomaly(self):
    #     result = self.orbit.eccentric_anomaly
    #     target = 0.2764082754722965

    #     self.assertEqual(result, target)

    # def test_energy(self):
    #     result = self.orbit.energy
    #     target = -2.5608224489140006e-11

    #     self.assertEqual(result, target)

    # def test_angular_momentum_magnitude(self):
    #     result = self.orbit.angular_momentum_magnitude
    #     target = 0.

    #     self.assertEqual(result, target)

    # def test_angular_momentum(self):
    #     result = self.orbit.angular_momentum
    #     target = (0., 0., 0.)

    #     self.assertEqual(result, target)

    # # def test_laplace_runge_lenz_magnitude(self):
    # #     result = self.orbit.laplace_runge_lenz_magnitude
    # #     target = 0.

    # # def test_laplace_runge_lenz(self):
    # #     result = self.orbit.laplace_runge_lenz
    # #     target = 0., 0., 0.

    # #     self.assertEqual(result, target)

    # def test_period(self):
    #     result = self.orbit.period
    #     target = 0.

    #     self.assertEqual(result, target)

    # def test_state(self):
    #     result = self.orbit.state
    #     target = 0., 0., 0., 0., 0., 0.

    #     self.assertEqual(result, target)

    # def test_radius(self):
    #     result = self.orbit.radius
    #     target = 0.

    #     self.assertEqual(result, target)

    # def test_speed(self):
    #     result = self.orbit.speed
    #     target = 0.

    #     self.assertEqual(result, target)


if __name__ == "__main__":
    unittest.main()
