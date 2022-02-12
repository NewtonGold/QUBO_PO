import unittest
import QUBO_Formulation

class TestDataTypes(unittest.TextCase):

    def test_create_cc_model(self):
        self.assertIsInstace(QUBO_Formulation.create_cc_model(), int)


if __name__ == '__main__':
    unittest.main()