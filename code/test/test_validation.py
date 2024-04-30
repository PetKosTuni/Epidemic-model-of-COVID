import unittest
import validation
from unittest.mock import patch
from data import *

#patch for the parser?

# Integration tests for validation.py

class TestValidation(unittest.TestCase):
    """! Class for testing the validation.py file. Tests will aim to check that the functions within the file are functional.
    """
    
    def setUp(self):
        self.data = NYTimes(level='states')
        self.pop_in = 0.01
        self.new_sus = 0
        self.pred_range = 10
        self.N = 60000000
        self.E = self.N/50
        self.a, self.decay = 0.75, 0.033
        self.train_data = [self.data.get('2020-03-22', '2020-05-28', "California")] # Get values from NYTimes California
        self.data_confirm, self.data_fatality = self.train_data[0][0], self.train_data[0][1]
        self.init = [self.N-self.E-self.data_confirm[0]-self.data_fatality[0],
            self.E, self.data_confirm[0], self.data_fatality[0]]
        self.prev_params = [0.2, .5e-2, 3e-1, 0.01]
        self.model = validation.Learner_SuEIR(N=self.N, E_0=self.E, I_0=self.data_confirm[0], R_0=self.data_fatality[0], a=self.a, decay=self.decay) # Model for training
    
    def test_validation_loss(self):

        val_data = self.data.get('2020-03-22', '2020-05-28', "California")

        params_all, loss_all = validation.rolling_train(self.model, self.init, self.train_data, self.new_sus, self.pop_in)
        result = validation.validation_loss(self.model, self.init, params_all, self.train_data, val_data, self.new_sus, self.pop_in)
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, 0)

    def generate_parameters(self):
        result = validation.generate_parameters()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 12)

        for value in result:
            self.assertIsNotNone(value)
        
        self.assertIsInstance(result['a'], float)
        self.assertIsInstance(result['decay'], float)
        self.assertIsInstance(result['pop_in'], float)

    def generate_validation_results(self):
        result = validation.generate_validation_results()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(value, str) for value in result))
    
if __name__ == '__main__':
    unittest.main()