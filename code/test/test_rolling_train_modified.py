import unittest
from rolling_train_modified import *
from data import *

# With the tests in this file we find out if the functions of rolling_train_modified are functional
# note: tests do not determine if functions are giving good results / working correctly model-wise (though they should be if nothing has been broken)
class TestRollingTrainModified(unittest.TestCase):
    """! Class for testing the rolling_train_modified.py file. The file contains no classes, but it has 
    many functions which are important to the model. Tests will aim to check that the functions are functional.
    """
    
    def setUp(self):
        """! Set up needed variables for the tests. This functions is run every time a test case of the class is run.
        We need some actual data in order to test these functions meant for training the model, so we need to read some data
        like in validation.py.
        """
        # Implementing tests using NYTimes class for now, we'll see how it goes.
        # Gotta test it in a "dumb" way, since we don't want to run validation.py 
        # to get the actual parameters in unit tests for rolling_train_modified.
        self.nytimes_data = NYTimes(level='states')
        
        
    def testLoss(self):
        """! Test the loss function, which is used in various points of rolling_train_modified and validation.
        Using bogus data, since I don't think we wanna run validation for this 
        """
        smoothing = 0.1
        actual = [1, 2, 3, 4, 5]
        actual = np.array([1, 2, 3, 4, 5])
        prediction = np.array([2, 3, 4, 5, 6])
        returnVal = loss(prediction, actual, smoothing)
        self.assertIsInstance(returnVal, float)
        self.assertIsNotNone(returnVal)
        
        
    def testTrain(self):
        """! Test the training function using somewhat bogus values to get something from the function.
        This tests functionality, not that the results are correct or useful.
        """
        # Setup necessary params
        N = 60000000
        E = N/50
        a, decay = 0.75, 0.033
        train_data = [self.nytimes_data.get('2020-03-22', '2020-05-28', "California")] # Get values from NYTimes California
        data_confirm, data_fatality = train_data[0][0], train_data[0][1]
        init = [N-E-data_confirm[0]-data_fatality[0],
            E, data_confirm[0], data_fatality[0]]
        prev_params = [0.2, .5e-2, 3e-1, 0.01]
        model = Learner_SuEIR(N=N, E_0=E, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay) # Model for training
        
        train_data = [data_confirm, data_fatality]
        params, train_loss = train(model, init, prev_params, train_data, N/2, 0) # Running the train function to get return values

        # See that return values exist and are good types
        self.assertIsNotNone(params)
        self.assertIsInstance(params, np.ndarray)
        self.assertTrue(all(isinstance(param, (int, float)) for param in params))
        self.assertIsNotNone(train_loss)
        self.assertIsInstance(train_loss, float)
        
    def testRollingTrain(self):
        """! Test the Rolling Train function in a similar manner to the train test.
        """
         # Setup necessary params
        pop_in = 0.01
        new_sus = 0
        N = 60000000
        E = N/50
        a, decay = 0.75, 0.033
        train_data = [self.nytimes_data.get('2020-03-22', '2020-05-28', "California")] # Get values from NYTimes California
        data_confirm, data_fatality = train_data[0][0], train_data[0][1]
        init = [N-E-data_confirm[0]-data_fatality[0],
            E, data_confirm[0], data_fatality[0]]
        prev_params = [0.2, .5e-2, 3e-1, 0.01]
        model = Learner_SuEIR(N=N, E_0=E, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay) # Model for training
        
        params, loss = rolling_train(model, init, train_data, new_sus, pop_in)

        # See that return values exist and are good types
        self.assertIsNotNone(params)
        self.assertIsInstance(params, list)
        self.assertTrue(all(isinstance(param, (int, float)) for param in params[0]))
        self.assertIsInstance(loss, list)
        self.assertTrue(len(loss) > 0)
        
        # change this to check each value that it is a float 
        for value in loss:
            self.assertIsNotNone(value)
            self.assertIsInstance(value, float)
            
    def testRollingPrediction(self):
        """! Test the rolling prediction function in a similar manner to the train test.
        """
         # Setup necessary params
        pop_in = 0.01
        new_sus = 0
        pred_range = 10
        N = 60000000
        E = N/50
        a, decay = 0.75, 0.033
        train_data = [self.nytimes_data.get('2020-03-22', '2020-05-28', "California")] # Get values from NYTimes California
        data_confirm, data_fatality = train_data[0][0], train_data[0][1]
        init = [N-E-data_confirm[0]-data_fatality[0],
            E, data_confirm[0], data_fatality[0]]
        prev_params = [0.2, .5e-2, 3e-1, 0.01]
        model = Learner_SuEIR(N=N, E_0=E, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay) # Model for training
        
        params, loss = rolling_train(model, init, train_data, new_sus, pop_in) # Need to get params in order to run rolling_prediction
        self.assertIsNotNone(params)
        self.assertIsInstance(params, list)
        self.assertTrue(all(isinstance(param, (int, float)) for param in params[0]))
        self.assertIsInstance(loss, list)
        self.assertTrue(len(loss) > 0)
        
        confirm, fatality, act = rolling_prediction(model, init, params, train_data, new_sus, pred_range, pop_in)      
        # Confrim, fatality and act should be numpy arrays with 10 elements
        self.assertIsNotNone(confirm)
        self.assertIsNotNone(fatality)
        self.assertIsNotNone(act)
        self.assertIsInstance(confirm, np.ndarray)
        self.assertIsInstance(fatality, np.ndarray)
        self.assertIsInstance(act, np.ndarray)
        self.assertTrue(len(confirm) == pred_range)
        self.assertTrue(len(fatality) == pred_range)
        self.assertTrue(len(act) == pred_range)
        
    def testRollingLikelihood(self):
        """! Test the rolling likelihood function in a similar manner to the train test.
        """
         # Setup necessary params
        pop_in = 0.01
        new_sus = 0
        N = 60000000
        E = N/50
        a, decay = 0.75, 0.033
        train_data = [self.nytimes_data.get('2020-03-22', '2020-05-28', "California")] # Get values from NYTimes California
        data_confirm, data_fatality = train_data[0][0], train_data[0][1]
        init = [N-E-data_confirm[0]-data_fatality[0],
            E, data_confirm[0], data_fatality[0]]
        prev_params = [0.2, .5e-2, 3e-1, 0.01]
        model = Learner_SuEIR(N=N, E_0=E, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay) # Model for training
        
        params, loss = rolling_train(model, init, train_data, new_sus, pop_in)

        # See that return values exist and are good types
        self.assertIsNotNone(params)
        self.assertIsInstance(params, list)
        self.assertTrue(all(isinstance(param, (int, float)) for param in params[0]))
        self.assertIsInstance(loss, list)
        self.assertTrue(len(loss) > 0)
        
        first, final = rolling_likelihood(model, init, params, train_data, new_sus, pop_in)
        self.assertIsNotNone(first)
        self.assertIsInstance(first, float)
        self.assertIsNotNone(final)
        self.assertIsInstance(final, float)
    
if __name__ == '__main__':
    unittest.main()