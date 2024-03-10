from contextlib import AbstractContextManager
from typing import Any
import unittest
from data import NYTimes, JHU_US, JHU_global, Hospital_CA, Hospital_US

# This testing file contains unit tests for the data classes of data.py.
# At time of writing, no clear test plan / framework has been decided, so I'll just write some
# basic tests to make sure things are working within the data classs.
# Need to expand this file when users will be able to use custom datasets

class TestNYTimes(unittest.TestCase):
    """! Class for unit testing NYTimes-class of data.py
    """
    def setUp(self):
        """!
        Set up the test environment before each test case.
        This method is called before each individual test case is run. It is used to set up any necessary
        objects or resources that are needed for the tests.
        In this case, it initializes two instances of the `NYTimes` class, one for states and one for counties.

        @param self  The current test case instance.
        """
        self.nytimes_states = NYTimes(level='states')
        self.nytimes_counties = NYTimes(level='counties')
        
    def test_init(self):
        """!
        Test the initialization of the NYTimes class.
        """
        # Not really sure what states mean the initialization is messed up
        # But when the NYTimes data is read from the correct csv's, there should be 
        # a dataframe and an array, and they should probably never be empty
        self.assertIsNotNone(self.nytimes_states.table) #  Tables are pandas dataframes
        self.assertIsNotNone(self.nytimes_counties.table)
        
        self.assertIsNotNone(self.nytimes_states.state_list) # Lists are numpy arrays
        self.assertIsNotNone(self.nytimes_counties.state_list) 
        
        self.assertFalse(self.nytimes_states.table.empty) # table is not empty
        self.assertFalse(self.nytimes_counties.table.empty)
        
        self.assertFalse(self.nytimes_states.state_list.size == 0) # list is not empty
        self.assertFalse(self.nytimes_counties.state_list.size == 0)
        
    def test_date_range_states(self):
        """!
        Test the date_range method of the NYTimes class for every state.
        date_range should return a tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        for state in self.nytimes_states.state_list:
            date_first, date_last = self.nytimes_states.date_range(state)
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertTrue(len(date_first) > 0)
            self.assertTrue(len(date_last) > 0)

    def test_date_range_counties(self):
        """!
        Test the date_range method of the NYTimes class for every county.
        date_range should return a tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        for state in self.nytimes_counties.state_list:
            date_first, date_last = self.nytimes_counties.date_range(state)
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertTrue(len(date_first) > 0)
            self.assertTrue(len(date_last) > 0)
        
        
    