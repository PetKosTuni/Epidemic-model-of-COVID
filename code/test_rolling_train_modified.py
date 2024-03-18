import unittest


class TestRollingTrainModified(unittest.TestCase):
    """! Class for testing the rolling_train_modified.py file. The file contains no classes, but it has 
    many functions which are important to the model. Tests will aim to check that the functions are functional.
    """
    #Functions to test are rolling_train, rolling_prediction and rolling_likelihood.
    def setUp(self):
        """! Set up needed variables for the tests. This functions is run every time a test case of the class is run.
        We need some actual data in order to test these functions meant for training the model, so we need to read some data
        like in validation.py.
        """
        