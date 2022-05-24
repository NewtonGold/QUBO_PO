import unittest
import simulated_annealing
import numpy as np

class TestQubo(unittest.TestCase):

    def setUp(self) -> None:
        self.Q = np.array([[-5, 2, 4, 0],
                    [2, -3, 1, 0],
                    [4, 1, -8, 5],
                    [0, 0, 5, -6]])

    def test_zero_array(self):
        x = np.zeros(4, dtype=int)
        self.assertEqual(simulated_annealing.qubo(x, self.Q), 0)
    
    def test_optimal_array(self):
        x = np.array([1,0,0,1])
        self.assertEqual(simulated_annealing.qubo(x, self.Q), -11)

class TestNeighbor(unittest.TestCase):

    def test_flip_each_value(self):
        x = np.zeros(4, dtype=int)
        flipped = np.array([0,0,0,0])
        for i in range(len(x)):
            flipped[i] = 1
            np.testing.assert_array_equal(simulated_annealing.neighbor(x, i),\
            flipped)



if __name__ == '__main__':
    unittest.main()