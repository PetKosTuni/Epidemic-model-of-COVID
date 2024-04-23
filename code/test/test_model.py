import unittest
import numpy as np
from scipy.integrate import solve_ivp
import warnings

from model import Learner_SuEIR, Learner_SEIR, Learner_SuEIR_H

class TestLearner_SuEIR(unittest.TestCase):
    """!
    Class for unit testing Learner_SuEIR class of model.py.
    """

    def setUp(self):
        """!
        Sets up learner for test cases.
        """
        self.N = 100000
        self.E_0 = 1000
        self.I_0 = 500
        self.R_0 = 300
        self.a = 0.7
        self.decay = 0.02

        self.learner = Learner_SuEIR(N=self.N, E_0=self.E_0, I_0=self.I_0, R_0=self.R_0, a=self.a, decay=self.decay)

    def testInit(self):
        """!
        Tests if the initialization of learner was successful and correct values were used.
        """
        self.assertIsNotNone(self.learner)

        self.assertEqual(self.learner.N, self.N)
        self.assertEqual(self.learner.E_0, self.E_0)
        self.assertEqual(self.learner.I_0, self.I_0)
        self.assertEqual(self.learner.R_0, self.R_0)
        self.assertEqual(self.learner.a, self.a)
        self.assertEqual(self.learner.decay, self.decay)

        self.assertIsNotNone(self.learner.FRratio)
        self.assertIsNotNone(self.learner.pop_in)
        self.assertIsNotNone(self.learner.pop)
        self.assertIsNotNone(self.learner.bias)

        self.assertIsNotNone(self.learner.initial_N)
        self.assertIsNotNone(self.learner.initial_bias)
        self.assertIsNotNone(self.learner.initial_pop_in)

    def testCall(self):
        """!
        Tests if the learner can be called and the correct amount of results are returned.
        """
        size = 10
        params = [0.2, 0.3, 0.4, 0.5]
        init = [100, 50, 30, 20]

        self.result = self.learner(size, params, init)

        self.assertIsNotNone(self.result)
        self.assertEqual(len(self.result), 6)

    def testReset(self):
        """!
        Tests if the reset function works properly.
        """
        self.learner.N = 21000
        self.learner.pop_in = 2
        self.learner.bias = 23

        self.learner.reset()

        self.assertEqual(self.learner.N, self.N)
        self.assertEqual(self.learner.pop_in, self.learner.initial_pop_in)
        self.assertEqual(self.learner.bias, self.learner.initial_bias)

class TestLearner_SEIR(unittest.TestCase):
    """!
    Class for unit testing Learner_SEIR class of model.py.
    """

    def setUp(self):
        """!
        Sets up learner for test cases.
        """
        self.N = 100000
        self.E_0 = 1000
        self.I_0 = 500
        self.R_0 = 300
        self.a = 0.7
        self.decay = 0.02

        self.learner = Learner_SEIR(N=self.N, E_0=self.E_0, I_0=self.I_0, R_0=self.R_0, a=self.a, decay=self.decay)

    def testInit(self):
        """!
        Tests if the initialization of learner was successful and correct values were used.
        """
        self.assertIsNotNone(self.learner)

        self.assertEqual(self.learner.N, self.N)
        self.assertEqual(self.learner.E_0, self.E_0)
        self.assertEqual(self.learner.I_0, self.I_0)
        self.assertEqual(self.learner.R_0, self.R_0)
        self.assertEqual(self.learner.a, self.a)

        self.assertIsNotNone(self.learner.FRratio)

    def testCall(self):
        """!
        Tests if the learner can be called and the correct amount of results are returned.
        """
        size = 10
        params = [0.2, 0.3, 0.4]
        init = [100, 50, 30, 20]

        self.result = self.learner(size, params, init)

        self.assertIsNotNone(self.result)
        self.assertEqual(len(self.result), 6)

class TestLearner_SuEIR_H(unittest.TestCase):
    """!
    Class for unit testing Learner_SuEIR_H class of model.py.
    """

    def setUp(self):
        """!
        Sets up learner for test cases.
        """
        self.N = 100000
        self.E_0 = 1000

        self.learner = Learner_SuEIR_H(N=self.N, E_0=self.E_0)

    def testInit(self):
        """!
        Tests if the initialization of learner was successful and correct values were used.
        """
        self.assertIsNotNone(self.learner)

        self.assertEqual(self.learner.N, self.N)
        self.assertEqual(self.learner.E_0, self.E_0)

if __name__ == '__main__':
    unittest.main()