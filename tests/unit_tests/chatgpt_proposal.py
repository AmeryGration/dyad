class TestCheckMass(unittest.TestCase):

    def test_valid_scalar(self):
        self.assertEqual(_check_mass(1.0), 1.0)

    def test_valid_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_check_mass(arr), arr)

    def test_non_real_raises_typeerror(self):
        self.assertRaises(TypeError, _check_mass, np.array([1.0, 2.0 + 3j]))

    def test_non_real_array_mixed_values_raises_typeerror(self):
        self.assertRaises(TypeError, _check_mass, np.array([1.0, 2.0, 3.0 + 1j]))

    def test_non_positive_scalar_raises_valueerror(self):
        self.assertRaises(ValueError, _check_mass, -1.0)

    def test_non_positive_array_raises_valueerror(self):
        self.assertRaises(ValueError, _check_mass, np.array([1.0, 0.0, -2.0]))

if __name__ == '__main__':
    unittest.main()
