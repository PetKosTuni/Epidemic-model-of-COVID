import unittest
import pandas as pd
from datetime import datetime
from convert_JHU import get_JHU

class TestConvertJHU(unittest.TestCase):
    """!
    Class for unit testing the convert_JHU.py file.
    """

    def testGetJHUNation(self):
        """!
        Test the get_JHU function for nation level data.
        """
        # Get the JHU data for nation level
        df = get_JHU(level='nation')

        # Check if the returned object is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check if the returned DataFrame has some rows
        self.assertGreater(len(df), 0)

        # Check if the returned DataFrame has the required columns
        self.assertIn('Country_Region', df.columns)
        self.assertIn('ConfirmedCases', df.columns)
        self.assertIn('Fatalities', df.columns)
        self.assertIn('Recovered', df.columns)
        self.assertIn('Date', df.columns)

    def testGetJHUStates(self):
        """!
        Test the get_JHU function for state level data.
        """
        # Get the JHU data for state level
        df = get_JHU(level='states')

        # Check if the returned object is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check if the returned DataFrame has some rows
        self.assertGreater(len(df), 0)

        # Check if the returned DataFrame has the required columns
        self.assertIn('state', df.columns)
        self.assertIn('date', df.columns)
        self.assertIn('cases', df.columns)
        self.assertIn('deaths', df.columns)

    def testGetJHUCounties(self):
        """!
        Test the get_JHU function for county level data.
        """
        # Get the JHU data for county level
        df = get_JHU(level='counties')

        # Check if the returned object is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check if the returned DataFrame has some rows
        self.assertGreater(len(df), 0)

        # Check if the returned DataFrame has the required columns
        self.assertIn('county', df.columns)
        self.assertIn('state', df.columns)
        self.assertIn('date', df.columns)
        self.assertIn('cases', df.columns)
        self.assertIn('deaths', df.columns)
        self.assertIn('fips', df.columns)

if __name__ == '__main__':
    unittest.main()
