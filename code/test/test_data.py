from contextlib import AbstractContextManager
from typing import Any
import unittest
import os
os.environ['DC_STATEHOOD'] = '1'
import us


from data import NYTimes, JHU_US, JHU_global, HospitalUs, HospitalCa

# This testing file contains unit tests for the data classes of data.py.
# At time of writing, no clear test plan / framework has been decided, so I'll just write some
# basic tests to make sure things are working within the data class.
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
        
        self.assertNotEqual(self.nytimes_states.state_list.size, 0) # list is not empty
        self.assertNotEqual(self.nytimes_counties.state_list.size, 0)
        
    def test_date_range_states(self):
        """!
        Test the date_range method of the NYTimes class for every state.
        date_range should return a tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        for state in self.nytimes_states.state_list:
            date_first, date_last = self.nytimes_states.date_range(state)
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertGreater(len(date_first), 0)
            self.assertGreater(len(date_last), 0)

    def test_date_range_counties(self):
        """!
        Test the date_range method of the NYTimes class for every county. Runtime for test is long.
        date_range should return a tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        for state in self.nytimes_counties.state_list:
            counties = self.nytimes_counties.table[self.nytimes_counties.table['state'] == state]['county'].unique()
            for county in counties:
                date_first, date_last = self.nytimes_counties.date_range(state, county)
                self.assertIsNotNone(date_first)
                self.assertIsNotNone(date_last)
                self.assertGreater(len(date_first), 0)
                self.assertGreater(len(date_last), 0)
        
    def test_state_list_not_empty(self):
        """!
        Test that the state_list attribute of the NYTimes class is not empty for either level.
        """
        self.assertGreater(len(self.nytimes_states.state_list), 0)
        self.assertGreater(len(self.nytimes_counties.state_list), 0)
        
    def test_get_whole_range_states(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range(). For states level.
        """
        for state in self.nytimes_states.state_list: # Surely each state should have a valid number of cases and deaths
            date_first, date_last = self.nytimes_states.date_range(state)
            array_cases, array_deaths = self.nytimes_states.get(date_first, date_last, state)
            self.assertIsNotNone(array_cases)
            self.assertIsNotNone(array_deaths)
            self.assertNotEqual(len(array_cases), 0)
            self.assertNotEqual(len(array_deaths), 0)
            
    def test_get_whole_range_counties(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range(). For counties level. Running this takes a long time.
        """
        for state in self.nytimes_counties.state_list:
            counties = self.nytimes_counties.table[self.nytimes_counties.table['state'] == state]['county'].unique()
            for county in counties:
                date_first, date_last = self.nytimes_counties.date_range(state, county)
                array_cases, array_deaths = self.nytimes_counties.get(date_first, date_last, state, county)
                self.assertIsNotNone(array_cases)
                self.assertIsNotNone(array_deaths)
                self.assertNotEqual(len(array_cases), 0)
                self.assertNotEqual(len(array_deaths), 0)
            
            
class TestJhuUs(unittest.TestCase): # Unable to test these now, JHU States (and prolly counties) is still broken
    """! Class for unit testing JHU_US-class of data.py
    """
    def setUp(self):
        """!
        Set up the test environment before each test case.
        This method is called before each individual test case is run. It is used to set up any necessary
        objects or resources that are needed for the tests.
        In this case, it initializes an instance of the `JHU_US` class.

        @param self  The current test case instance.
        """
        self.jhu_us_states = JHU_US(level='states')
        self.jhu_us_counties = JHU_US(level='counties')
        
    def test_init(self):
        """!
        Test the initialization of the JHU_US class.
        """
        self.assertIsNotNone(self.jhu_us_states.table) # Tables are pandas dataframes
        self.assertIsNotNone(self.jhu_us_counties.table)

        self.assertIsNotNone(self.jhu_us_states.state_list) # Lists are numpy arrays
        self.assertIsNotNone(self.jhu_us_counties.state_list)

        self.assertFalse(self.jhu_us_states.table.empty) # table is not empty
        self.assertFalse(self.jhu_us_counties.table.empty)

        self.assertGreater(self.jhu_us_states.state_list.size, 0) # list is not empty
        self.assertGreater(self.jhu_us_counties.state_list.size, 0)
        
    def test_date_range_states(self):
        """!
        Test the date_range method of the JHU_US class for every state.
        """
        for state in self.jhu_us_states.state_list:
            date_first, date_last = self.jhu_us_states.date_range(state)
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertGreater(len(date_first), 0)
            self.assertGreater(len(date_last), 0)
            
    def test_date_range_counties(self):
        """!
        Test the date_range method of the JHU_US class for every county. Runtime for test is long.
        """
        for state in self.jhu_us_counties.state_list:
            counties = self.jhu_us_counties.table[self.jhu_us_counties.table['state'] == state]['county'].unique()
            for county in counties:
                date_first, date_last = self.jhu_us_counties.date_range(state, county)
                self.assertIsNotNone(date_first)
                self.assertIsNotNone(date_last)
                self.assertGreater(len(date_first) > 0)
                self.assertGreater(len(date_last) > 0)
                
    def test_state_list_not_empty(self):
        """!
        Test that the state_list attribute of the JHU_US class is not empty for either level.
        """
        self.assertTrue(len(self.nytimes_states.state_list) > 0)
        self.assertTrue(len(self.nytimes_counties.state_list) > 0)
        
    def test_get_whole_range_states(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range(). For states level.
        """
        for state in self.jhu_us_states.state_list: # Surely each state should have a valid number of cases and deaths
            date_first, date_last = self.jhu_us_states.date_range(state)
            array_cases, array_deaths = self.jhu_us_states.get(date_first, date_last, state)
            self.assertIsNotNone(array_cases)
            self.assertIsNotNone(array_deaths)
            self.assertNotEqual(len(array_cases), 0)
            self.assertNotEqual(len(array_deaths), 0)
            
    def test_get_whole_range_counties(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range(). For counties level. Running this takes a long time.
        """
        for state in self.jhu_us_counties.state_list:
            counties = self.jhu_us_counties.table[self.jhu_us_counties.table['state'] == state]['county'].unique()
            for county in counties:
                date_first, date_last = self.jhu_us_counties.date_range(state, county)
                array_cases, array_deaths = self.jhu_us_counties.get(date_first, date_last, state, county)
                self.assertIsNotNone(array_cases)
                self.assertIsNotNone(array_deaths)
                self.assertNotEqual(len(array_cases), 0)
                self.assertNotEqual(len(array_deaths), 0)
            
            
class TestJhuGlobal(unittest.TestCase):
    """! Class for unit testing JHU_global-class of data.py
    """
    def setUp(self):
        """!
        Set up the test environment before each test case.
        This method is called before each individual test case is run. It is used to set up any necessary
        objects or resources that are needed for the tests.
        In this case, it initializes an instance of the `JHU_global` class.

        @param self  The current test case instance.
        """
        self.jhu_global = JHU_global()
        
    def test_init(self):
        """!
        Test the initialization of the JHU_global class.
        """
        self.assertIsNotNone(self.jhu_global.confirm_table) # Tables are pandas dataframes
        self.assertIsNotNone(self.jhu_global.death_table)
        self.assertIsNotNone(self.jhu_global.recover_table)

        self.assertFalse(self.jhu_global.confirm_table.empty) # table is not empty
        self.assertFalse(self.jhu_global.death_table.empty)
        self.assertFalse(self.jhu_global.recover_table.empty)
        
    def test_date_range_countries(self):
        """!
        Test the date_range method of the JHU_global class. Original implementation of the method is funny,
        it doesn't actually look up for any specific country, but just returns the first and last date of the data.
        """
        dummy_country = 'Sweden' # because Sweden is for dummies
        date_first, date_last = self.jhu_global.date_range(dummy_country)
        self.assertIsNotNone(date_first)
        self.assertIsNotNone(date_last)
        self.assertGreater(len(date_first), 0)
        self.assertGreater(len(date_last), 0)
            
        
    def test_get_whole_range_countries(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range(). Note that the dates are not country specific
        """
        country_list = self.jhu_global.confirm_table.keys()
        for country in country_list:
            date_first, date_last = self.jhu_global.date_range(country)
            array_cases, array_deaths, array_recoveries = self.jhu_global.get(date_first, date_last, country)
            self.assertIsNotNone(array_cases)
            self.assertIsNotNone(array_deaths)
            self.assertIsNotNone(array_recoveries)
            self.assertNotEqual(len(array_cases), 0)
            self.assertNotEqual(len(array_deaths), 0)
            self.assertNotEqual(len(array_recoveries), 0)
        
        
class TestHospitalCa(unittest.TestCase):
    """! Class for unit testing HospitalCa-class of data.py
    """
    def setUp(self):
        """!
        Set up the test environment before each test case.
        This method is called before each individual test case is run. It is used to set up any necessary
        objects or resources that are needed for the tests.
        In this case, it initializes an instance of the `Hospital_CA` class.

        @param self  The current test case instance.
        """
        self.hospital_ca = HospitalCa()
        self.county_list = self.hospital_ca.table["county"].unique()

    def test_init(self):
        """!
        Test the initialization of the Hospital_CA class.
        """
        self.assertIsNotNone(self.hospital_ca.table) # Table is a pandas dataframe
        self.assertFalse(self.hospital_ca.table.empty) # Table is not empty

    def test_date_range(self):
        """!
        Test the date_range method of the Hospital_CA class. For every country in California.
        """
        for county in self.county_list:
            date_first, date_last = self.hospital_ca.date_range(county)
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertGreater(len(date_first.isoformat()), 0)
            self.assertGreater(len(date_last.isoformat()), 0)

    def test_get_whole_range(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range().
        """
        for county in self.county_list:
            date_first, date_last = self.hospital_ca.date_range(county)
            # parse the datettime dates into strings of yyyy-mm-dd
            date_first = date_first.isoformat()
            date_last = date_last.isoformat()
            
            array_hospital, array_icu = self.hospital_ca.get(date_first, date_last, county)
            self.assertIsNotNone(array_hospital)
            self.assertIsNotNone(array_icu)
            self.assertNotEqual(len(array_hospital), 0)
            self.assertNotEqual(len(array_icu), 0)
       
class testHospitalUs(unittest.TestCase):
    """! Class for unit testing HospitalUs-class of data.py
    """
    def setUp(self):
        """! Set up the test environment before each test case. Create instances of the HospitalUs class for each state.
        """
        self.hospital_us_instances = []
        for state in us.states.STATES:
            instance = HospitalUs(state.name)
            if not (instance.table.empty): # only take the states that the dataset has data for
                self.hospital_us_instances.append(HospitalUs(state.name))
        
    def test_init(self):
        """!
        Test the initialization of the Hospital_US class. Each instance should have a table that is not empty.
        """
        for state in self.hospital_us_instances:
            self.assertIsNotNone(state.table) # Tables are pandas dataframes
            self.assertFalse(state.table.empty) # Table is not empty
                    
    def test_date_range(self):
        """!
        Test the date_range method of the Hospital_US class for every state.
        """
        for state in self.hospital_us_instances:
            date_first, date_last = state.date_range()
            self.assertIsNotNone(date_first)
            self.assertIsNotNone(date_last)
            self.assertGreater(len(date_first.isoformat()), 0)
            self.assertGreater(len(date_last.isoformat()), 0)          
            
    def test_get_whole_range(self):
        """!
        Test that the get() method of the class returns actual data when the range is the entire range of the data,
        which is gotten with method date_range().
        """
        for state in self.hospital_us_instances:
            date_first, date_last = state.date_range()
            # parse the datettime dates into strings of yyyy-mm-dd
            date_first = date_first.isoformat()
            date_last = date_last.isoformat()
            
            array_hospital, array_icu = state.get(date_first, date_last)
            self.assertIsNotNone(array_hospital)
            self.assertIsNotNone(array_icu)
            self.assertNotEqual(len(array_hospital), 0)
            self.assertNotEqual(len(array_icu), 0)
                 
if __name__ == '__main__':
    unittest.main()