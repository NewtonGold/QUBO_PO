import unittest
import qubo_formulation as qf
import numpy as np

class TestDataTypes(unittest.TestCase):

    def setUp(self) -> None:
        self.l = 0.6
        self.K = 2
        self.cov = [[0.0021484, 0.0001678, 0.0002031, 0.0004182], 
        [0.0001678, 0.0009351, 0.0001534, 0.0001091], 
        [0.0002031, 0.0001534, 0.0009287, 9.06e-05], 
        [0.0004182, 0.0001091, 9.06e-05, 0.0012795]]
        self.mew = [0.004798, 0.000659, 0.003174, 0.001377]

    def test_create_cc_model(self):
        Q = qf.cc_qubo(self.K, self.cov, self.mew, self.l)
        self.assertIsInstance(Q, np.ndarray)
    
    def test_unmatched_input_size(self):
        del self.mew[0]
        with self.assertRaises(ValueError):
            Q = qf.cc_qubo(self.K, self.cov, self.mew, self.l)

    def test_different_lambda(self):
        l1 = 0.5
        l2 = 0.6
        Q1 = qf.cc_qubo(self.K, self.cov, self.mew, l1)
        Q2 = qf.cc_qubo(self.K, self.cov, self.mew, l2)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(Q1, Q2)

class TestInitialiseVariables(unittest.TestCase):

    def setUp(self) -> None:
        self.N = 2
        self.lb = 13
        self.ub = 58
        self.m, self.total_var, self.diff = qf.calc_variables(self.N, self.ub, \
            self.lb)

    def test_return_value(self):
        w, aux = qf.initialise_variables(self.N, self.lb, self.diff, self.m, \
            self.total_var)
        print(w)
        print(aux)
        self.assertEqual(len(aux), self.N)




if __name__ == '__main__':
    unittest.main(verbosity=2)