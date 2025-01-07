"""
Test functions of _core module

"""

import unittest
import numpy as np
import dyad
import data_test

from parameterized import parameterized, parameterized_class


class TestFunctions(unittest.TestCase):
    @parameterized.expand(data_test.check_mass_type)
    def test_check_mass_type(self, x):
        self.assertRaises(TypeError, dyad._core._check_mass, x)

    @parameterized.expand(data_test.check_mass_value)
    def test_check_mass_value(self, x):
        self.assertRaises(ValueError, dyad._core._check_mass, x)

    @parameterized.expand(data_test.check_eccentricity_type)
    def test_check_eccentricity_type(self, x):
        self.assertRaises(TypeError, dyad._core._check_eccentricity, x)

    @parameterized.expand(data_test.check_eccentricity_value)
    def test_check_eccentricity_value(self, x):
        self.assertRaises(ValueError, dyad._core._check_eccentricity, x)

    @parameterized.expand(data_test.check_semimajor_axis_type)
    def test_check_semimajor_axis_type(self, x):
        self.assertRaises(TypeError, dyad._core._check_semimajor_axis, x)

    @parameterized.expand(data_test.check_semimajor_axis_value)
    def test_check_semimajor_axis_value(self, x):
        self.assertRaises(ValueError, dyad._core._check_semimajor_axis, x)

    @parameterized.expand(data_test.check_period_type)
    def test_check_period_type(self, x):
        self.assertRaises(TypeError, dyad._core._check_period, x)

    @parameterized.expand(data_test.check_period_value)
    def test_check_period_value(self, x):
        self.assertRaises(ValueError, dyad._core._check_period, x)

    @parameterized.expand(data_test.semimajor_axis_from_period)
    def test_semimajor_axis_from_period(self, x, target):
        result = dyad.semimajor_axis_from_period(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.period_from_semimajor_axis)
    def test_period_from_semimajor_axis(self, x, target):
        result = dyad.period_from_semimajor_axis(*x)
        self.assertAlmostEqual(result, target)
    
    @parameterized.expand(data_test.mean_anomaly_from_eccentric_anomaly)
    def test_mean_anomaly_from_eccentric_anomaly(self, x, target):
        result = dyad.mean_anomaly_from_eccentric_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.eccentric_anomaly_from_true_anomaly)
    def test_eccentric_anomaly_from_true_anomaly(self, x, target):
        result = dyad.eccentric_anomaly_from_true_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.true_anomaly_from_eccentric_anomaly)
    def test_true_anomaly_from_eccentric_anomaly(self, x, target):
        result = dyad.true_anomaly_from_eccentric_anomaly(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.mean_anomaly_from_true_anomaly)
    def test_mean_anomaly_from_true_anomaly(self, x, target):
        result = dyad.mean_anomaly_from_true_anomaly(*x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.eccentric_anomaly_from_mean_anomaly)
    def test_eccentric_anomaly_from_mean_anomaly(self, x, target):
        result = dyad.eccentric_anomaly_from_mean_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.true_anomaly_from_mean_anomaly)
    def test_true_anomaly_from_mean_anomaly(self, x, target):
        result = dyad.true_anomaly_from_mean_anomaly(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(data_test.primary_semimajor_axis_from_semimajor_axis)
    def test_primary_semimajor_axis_from_semimajor_axis(self, x, target):
        result = dyad.primary_semimajor_axis_from_semimajor_axis(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(
        data_test.secondary_semimajor_axis_from_semimajor_axis
    )
    def test_secondary_semimajor_axis_from_semimajor_axis(self, x, target):
        result = dyad.secondary_semimajor_axis_from_semimajor_axis(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(
        data_test.primary_semimajor_axis_from_secondary_semimajor_axis
    )
    def test_primary_semimajor_axis_from_secondary_semimajor_axis(
        self, x, target
    ):
        result = dyad.primary_semimajor_axis_from_secondary_semimajor_axis(*x)
        self.assertAlmostEqual(result, target)
        
    @parameterized.expand(
        data_test.secondary_semimajor_axis_from_primary_semimajor_axis
    )
    def test_secondary_semimajor_axis_from_primary_semimajor_axis(
            self, x, target
    ):
        result = dyad.secondary_semimajor_axis_from_primary_semimajor_axis(*x)
        self.assertAlmostEqual(result, target)
        
    # @parameterized.expand(data_test.)
    # def test_(self, x, target):
    #     result = dyad.(*x)
    #     self.assertAlmostEqual(result, target)
        
class TestOrbit(unittest.TestCase):
    def setUp(self):
        self.m = 0.5488135039273248
        self.a = 0.7151893663724195
        self.e = 0.6027633760716439
        self.Omega = 0.4236547993389047
        self.i = 0.6458941130666561
        self.omega = 0.4375872112626925
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

    # def test_energy(self):
    #     result = self.orbit.energy
    #     target = 1373.3458311223064

    #     self.assertEqual(result, target)

    def test_angular_momentum_magnitude(self):
        result = self.orbit.angular_momentum_magnitude
        target = 14.889337552664351

        self.assertEqual(result, target)

    def test_angular_momentum(self):
        result = self.orbit.angular_momentum
        target = (3.684265656621784, -8.169766498909887, 11.890057808181918)

        self.assertIsNone(np.testing.assert_array_equal(result, target))

    # def test_laplace_runge_lenz_magnitude(self):
    #     result = self.orbit.laplace_runge_lenz_magnitude
    #     target = 0.

    # def test_laplace_runge_lenz(self):
    #     result = self.orbit.laplace_runge_lenz
    #     target = 0., 0., 0.

    #     self.assertEqual(result, target)

    def test_period(self):
        result = self.orbit.period
        target = 298.20684329677647

        self.assertEqual(result, target)

    @parameterized.expand(data_test.radius)
    def test_radius(self, x, target):
        result = self.orbit.radius(x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.speed)
    def test_speed(self, x, target):
        result = self.orbit.speed(x)
        self.assertAlmostEqual(result, target)

    @parameterized.expand(data_test.position)
    def test_position(self, x, target):
        result = self.orbit._position(x)
        print(result)
        self.assertIsNone(np.testing.assert_array_almost_equal(result, target))

    @parameterized.expand(data_test.velocity)
    def test_velocity(self, x, target):
        result = self.orbit._velocity(x)
        self.assertIsNone(np.testing.assert_array_almost_equal(result, target))

    @parameterized.expand(data_test.state)
    def test_state(self, x, target):
        result = self.orbit.state(x)
        self.assertIsNone(np.testing.assert_array_almost_equal(result, target))

    # @parameterized.expand(data_test.potential)
    # def test_potential(self, x, target):
    #     result = self.orbit.potential(x)
    #     self.assertAlmostEqual(result, target)

    # @parameterized.expand(data_test.kinetic_energy)
    # def test_kinetic_energy(self, x, target):
    #     result = self.orbit.kinetic_energy(x)
    #     self.assertAlmostEqual(result, target)


if __name__ == "__main__":
    unittest.main()
