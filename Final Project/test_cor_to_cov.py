import unittest
import cor_to_cov as ctc
import numpy as np

class TestConvert(unittest.TestCase):

    def setUp(self) -> None:
        self.correlations = np.array([[1,0.118368,0.143822,0.252213],\
                        [0.118368, 1, 0.164589, 0.099763],\
                        [0.143822,0.164589,1,0.083122],\
                        [0.252213,0.099763,0.083122,1]])
        self.stdev = np.array([0.046351, 0.03058, 0.030474, 0.035770])
        self.known_covariance = np.array([[0.0021484, 0.0001678, 0.0002031, \
                            0.0004182],\
                            [0.0001678, 0.0009351, 0.0001534, 0.0001091],\
                            [0.0002031, 0.0001534, 0.0009287, 9.06e-05],\
                            [0.0004182, 0.0001091, 9.06e-05, 0.0012795]])

    def test_valid_input(self):
        covariance = ctc.convert(self.correlations, self.stdev)
        np.testing.assert_array_equal(covariance, self.known_covariance)

    def test_cor_wrong_data_type(self):
        invalid_correlations = self.correlations.tolist()
        self.assertRaises(TypeError, ctc.convert, invalid_correlations, \
            self.stdev)

    def test_stdev_wrong_data_type(self):
        invalid_stdev = self.stdev.tolist()
        self.assertRaises(TypeError, ctc.convert, self.correlations, \
            invalid_stdev)

    def test_cor_not_square(self):
        invalid_correlations = np.delete(self.correlations, 1, 0)
        self.assertRaises(ValueError, ctc.convert, invalid_correlations, \
            self.stdev)

    def test_cor_stdev_different_shape(self):
        invalid_stdev = np.delete(self.stdev, 1, 0)
        self.assertRaises(ValueError, ctc.convert, self.correlations, \
            invalid_stdev)


if __name__ == '__main__':
    unittest.main()