import unittest
import validation
from unittest.mock import patch
from data import *
from datetime import datetime

# Integration tests for validation.py

class TestValidation(unittest.TestCase):
    """! Class for testing the validation.py file. Tests will aim to check that the functions within the file are functional.
    """
    def setUp(self):
        # Inject arguments to validation.py
        self.args_patch = patch('validation.args', create=True)
        self.args = self.args_patch.start()
        # CHANGE TESTING ARGUMENTS HERE!
        self.args.START_DATE = 'default'
        self.args.MID_DATE = 'default'
        self.args.RESURGE_DATE = 'default'
        self.args.END_DATE = '2020-05-28'
        self.args.VAL_END_DATE = '2020-06-06'
        self.args.level = 'state'
        self.args.state = 'California'
        self.args.nation = 'default'
        self.args.county = 'default'
        self.args.dataset = 'NYtimes'
        self.args.dataset_filepath = 'default'
        self.args.popin = 0.01
        self.args.bias = 0
    
    def tearDown(self):
        self.args_patch.stop()
    
    def test_validation_loss(self):
        N = 60000000
        E = N/50
        self.data = NYTimes('states')
        self.train_data = [self.data.get('2020-03-22', self.args.END_DATE, self.args.state)]
        self.data_confirm, self.data_fatality = self.train_data[0][0], self.train_data[0][1]
        self.init = [N-E-self.data_confirm[0]-self.data_fatality[0],
            E, self.data_confirm[0], self.data_fatality[0]]
        self.model = validation.Learner_SuEIR(N, E_0=E, I_0=self.data_confirm[0],
            R_0=self.data_fatality[0], a=0.75, decay=0.033)

        val_data = self.data.get('2020-03-22', self.args.END_DATE, self.args.state)
        params_all, loss_all = validation.rolling_train(self.model, self.init, self.train_data, 0, self.args.popin)
        result = validation.validation_loss(self.model, self.init, params_all, self.train_data, val_data, 0, self.args.popin)
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, 0)

    def test_region_list(self):
        result = validation.get_region_list()
        self.assertEqual(len(result), 7)
        for key, value in result.items():
            self.assertIsNotNone(value, f'{key} has value None')
            if key == 'write_dir':
                directory = os.path.dirname(value)
                self.assertTrue(os.path.exists(directory), f'no directory {directory}')
            elif key == 'mid_dates':
                for state, date in value.items():
                    try:
                        datetime.strptime(date, '%Y-%m-%d')
                    except ValueError:
                        self.fail(f'date {date} is invalid format. State is {state}')

    def test_generate_training_parameters(self):
        param_dict = validation.get_region_list()
        result = validation.generate_training_parameters(self.args.state, param_dict)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 12)
        for key, value in result.items():
            self.assertIsNotNone(value)
            if key == 'train_data':
                self.assertTrue(isinstance(value, list), f'train_data is not an array')
            elif key == 'full_data':
                self.assertTrue(isinstance(value, list), f'full_data is not an array')
            elif key == 'second_start_date' or key == 'start_date':
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    self.fail(f'date {value} of {key} is invalid format')
        self.assertIsInstance(result['a'], float)
        self.assertIsInstance(result['decay'], float)
        self.assertIsInstance(result['pop_in'], float)

    def test_get_county_list(self):
        county_data_1 = validation.get_county_list(1000, 10000000)
        county_data_2 = validation.get_county_list(1000, 1000000)
        self.assertGreaterEqual(len(county_data_2), len(county_data_1),
            'Stricter params should return less or equal amount of results.')

    
if __name__ == '__main__':
    unittest.main()