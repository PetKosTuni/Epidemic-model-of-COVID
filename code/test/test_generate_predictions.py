import unittest
import generate_predictions
from unittest.mock import patch
from data import *
from rolling_train_modified import rolling_prediction, rolling_train

class TestGeneratePredictions(unittest.TestCase):
    """! Class for testing the generate_predictions.py file. Tests will aim to check that the functions within the file are functional.
    """

    def setUp(self):
        self.data = JHU_global()
        self.region = 'Argentina'
        self.NE0_region = {'Argentina': [9039154.8, 11298.943500000001, 1.1937193195968993e-05, 5633490.968299583, 124580.94968087264, 17627.0, 0.08532036812096722, 0.017700621375364954]}
        self.pop_in = 0.01
        self.new_sus = 0
        self.pred_range = 100
        self.N, self.E_0 = self.NE0_region[self.region][0], self.NE0_region[self.region][1]
        self.a, self.decay = 0.75, 0.033
        self.train_data = [self.data.get("2020-04-03", "2020-08-01", "Argentina"), self.data.get("2020-08-01", "2021-07-07", "Argentina")]
        self.full_data = [self.data.get("2020-04-03", "2020-08-01", "Argentina"), self.data.get("2020-08-01", "2021-07-14", "Argentina")]
        self.data_confirm, self.data_fatality = self.train_data[0][0], self.train_data[0][1]
        self.init = [self.N-self.E_0-self.data_confirm[0]-self.data_fatality[0],
            self.E_0, self.data_confirm[0], self.data_fatality[0]]
        self.prev_params = [0.2, .5e-2, 3e-1, 0.01]
        self.I_0, self.R_0 = self.train_data[0][0], self.train_data[0][1]
        self.bias = 0.02
        self.model = generate_predictions.Learner_SuEIR(N=self.N, E_0=self.E_0, I_0=self.data_confirm[0], R_0=self.data_fatality[0], a=self.a, decay=self.decay) # Model for training
        self.params_all, self.loss_all = rolling_train(self.model, self.init, self.train_data, self.new_sus, pop_in=self.pop_in)
        self.loss_true = [self.NE0_region[self.region][-2], self.NE0_region[self.region][-1]]
        self.pred_true = rolling_prediction(self.model, self.init, self.params_all, self.full_data, self.new_sus, pred_range=self.pred_range, pop_in=self.pop_in, daily_smooth=True)
        self.confirm = self.full_data[0][0][0:-1].tolist() + self.full_data[1][0][0:-1].tolist() + self.pred_true[0].tolist()
        self.frames = []
        self.state = 0
        self.county = 0
        with patch('sys.argv'):
            self.args = generate_predictions.create_parser()
            self.args.END_DATE = '2021-07-07'
            self.args.VAL_END_DATE = '2021-07-14'
            self.args.level = 'nation'
            self.args.nation = 'Argentina'
        
    def test_read_validation_files(self):
        
        result = generate_predictions.read_validation_files(self.args)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        for value in result:
            self.assertIsNotNone(value)

    def test_generate_training_parameters(self):

        result = generate_predictions.generate_training_parameters(self.region, self.data, self.NE0_region, self.args)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 13)
        for value in result:
            self.assertIsNotNone(value)

    def test_plot_results(self):
        generate_predictions.plot_results(self.confirm, self.region, self.loss_all, self.loss_true, self.pred_true, self.args)

    def test_generate_prediction_frames(self):
        result = generate_predictions.generate_prediction_frames(self.params_all, self.model, self.init, self.full_data, self.new_sus, self.pred_range, self.pop_in, self.train_data, self.loss_true, self.pred_true, self.region, self.county, self.state, self.frames, self.args)

        self.assertIsInstance(result, list)
        self.assertNotEqual(len(result), 0)
        for value in result:
            self.assertIsNotNone(value)

    def test_generate_prediction_files(self):
        generate_predictions.generate_prediction_files(self.args)




if __name__ == '__main__':
    unittest.main()