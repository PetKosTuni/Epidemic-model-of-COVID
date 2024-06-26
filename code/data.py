import pandas as pd
import numpy as np
import datetime
import warnings
import os
# According to 'us' -package changelog, DC_STATEHOOD env variable should be set truthy before (FIRST) import
# In case this is the first import, needs to be set here.
os.environ['DC_STATEHOOD'] = '1'
import us
import prediction_data as pdata

from convert_JHU import get_JHU

class Data(object):
    """! The base class of data objects. Does nothing by itself, is meant to be 
    inherited by a class implementation of a dataset.
    """
    def date_range(self):
        """! Virtual function (kind of) for getting the timeframe the dataset spans - meant to return
        first date and last date.
        """
        warnings.warn('Data range method does not implement')
        raise NotImplementedError

    def get(self, start_date, end_date):
        """! Virtual function (kind of) for getting data of a specific timeframe
        from the dataset. Parameters may vary by class. Meant to be implemented by each dataset class.
        @param start_date  The date the wanted timeframe begins
        @param end_date  The date the wanted timeframe ends
        """
        warnings.warn('Data get method does not implement')
        raise NotImplementedError

# An alternative way to implement own datasets:
# When implementing your own datasets, uncomment the class below and copy paste as many as you need.
# Every class is its own dataset. The type of functions required depend on the format of the dataset,
# but to work nicely the date_range- and get-functions have to be implemented like the ready datasets
# below. The dataset can include either US county, US-state or country data. CSV-files would be the
# easiest way, by including them into the data folder, as done with the ready datasets (NYtimes etc).

'''
Inherit the Data Object:
class OWN_DATASET(Data):
    def __init__(self):
        pass
    def data_range(self, start_date, end_date):
        pass
    def get(self, start_date, end_date):
        pass
'''
    
class NYTimes(Data):
    """! Class for NYTimes dataset, inherits the Data base class. Data in the dataset is available 
    in two levels, states and counties. Dataset is specifically USA only.
    """
    def __init__(self, level='states'):
        """! NYTimes class initializer. Creates the used pandas dataframes.
        @param level  The level of data, either 'states' or 'counties'. States by default.
        """
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'
        url = 'data/nytimes_us-' + level + '.csv' # .csv files localized to ../code/data from the nytimes github
        self.table = pd.read_csv(url).drop('fips', axis=1)
                
        if level == 'counties':                 # Apparently the massive counties csv has null values, which causes the following assert to fail
            self.table = self.table.dropna()    # I have no idea if dropping those messes up the model, but it's either doing this or abadoning the NYTimes counties functionality...
            
        assert not self.table.isnull().values.any(), 'We do not handle nan cases in NYTimes'
        self.level = level
        self.state_list = self.table["state"].unique()

    def date_range(self, state, county=None):
        """! Get the first and last dates of available data in the dataset. If level is counties, a county should be specified.
        @param state  Name of the state for which range is being looked up
        @param county  Name of the county for which range is being looked up, IF level of data is counties. Default is 'None'.
        @return  Tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @param state  Name of the state data is being looked from
        @param county  Name of the county data is being looked from if level is counties
        @return  Tuple of arrays, arrays contain numbers of cases and deaths respectively.
        """
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        date = pd.to_datetime(tab['date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)
        return tab[mask]['cases'].to_numpy(), tab[mask]['deaths'].to_numpy()

class JHU_US(Data):
    """! Class for Johns Hopkins University US datasets. Inherits the Data base class. 
    Data is already localized in base code, but not used.
    Data is available in states and counties like NYTimes.
    """
    def __init__(self, level='states'):
        """! JHU_US class initializer. Creates the used pandas dataframes.
        @param level  The level of data, either 'states' or 'counties'. States by default.
        """
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'
        self.table = get_JHU(level)#.drop('fips', axis=1)
        assert not self.table.isnull().values.any(), 'We do not handle nan cases in NYTimes'
        self.level = level
        self.state_list = self.table["state"].unique()

    def date_range(self, state, county=None):
        """! Get the first and last dates of available data in the dataset.
        @param state  Name of the state for which range is being looked up
        @param county  Name of the county for which range is being looked up, IF level of data is counties. Default is 'None'.
        @return  Tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @param state  Name of the state data is being looked from
        @param county  Name of the county data is being looked from if level is counties
        @return  Tuple of arrays, arrays contain numbers of cases and deaths respectively.
        """
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        date = pd.to_datetime(tab['date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)
        return tab[mask]['cases'].to_numpy(dtype=float), tab[mask]['deaths'].to_numpy(dtype=float)

class JHU_global(Data):
    """! Class for Johns Hopkins University Global data. Inherits the Data base class.
    """
    def __init__(self):
        """! JHU_global class initializer. Creates the used pandas dataframes.
        """
        confirm = 'data/jhu_confirmed_global.csv'
        death = 'data/jhu_deaths_global.csv'
        recover = 'data/jhu_recovered_global.csv'
        confirm_table = pd.read_csv(confirm).drop(['Lat', 'Long'], axis=1)
        death_table = pd.read_csv(death).drop(['Lat', 'Long'], axis=1)
        recover_table = pd.read_csv(recover).drop(['Lat', 'Long'], axis=1)
        self.confirm_table = confirm_table.groupby(
            'Country/Region').sum().transpose()
        self.death_table = death_table.groupby(
            'Country/Region').sum().transpose()
        self.recover_table = recover_table.groupby(
            'Country/Region').sum().transpose()

    def date_range(self, country): # should this be changed to actually look up the country?
        """! Get the first and last dates of available data in the dataset for confirmed cases.
        @param country  Name of the country for which range is being looked up.
        @return  Tuple of strings, first date and last date, in format 'yyyy-mm-dd'
        """
        date = pd.to_datetime(self.confirm_table.iloc[1:].index).date
        start = str(date[0])
        end = str(date[-1])
        return start, end

    # Get the data of the country from dates within the given period
    def get(self, start_date, end_date, country):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @param country  The name of the country for which data is wanted.
        @return  Tuple of arrays, arrays contain numbers of confirmed cases, deaths and recoveries respectively.
        """
        countryconfirm = self.confirm_table[country].iloc[1:] # remove Province/State'
        countrydeath = self.death_table[country].iloc[1:]
        countryrecover = self.recover_table[country].iloc[1:]
        date = pd.to_datetime(self.confirm_table.index[1:], format="%m/%d/%y")
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        confirm = countryconfirm.loc[(date >= start) & (date <= end)]
        death = countrydeath.loc[(date >= start) & (date <= end)]
        recover = countryrecover.loc[(date >= start) & (date <= end)]
        return confirm.to_numpy(), death.to_numpy(), recover.to_numpy()

class HospitalCa(Data): 
    """! Class for hospital data of California Health and Human Services. Seemingly not used in the code.
    Inherits the Data base class.
    """ 
    def __init__(self):
        """! Hospital_CA class initializer. Creates pandas dataframe.
        """
        datafile = 'data/covid19hospitalbycounty_california.csv'
        self.table = pd.read_csv(datafile)[['todays_date', 'county', 'hospitalized_covid_confirmed_patients', 'icu_covid_confirmed_patients']]

    def date_range(self, region):
        """! Get the first and last dates of available data in the dataset.
        @param region  Name of the region (county) for which range is being looked up.
        @return  Tuple datetime dates
        """
        table = self.table[self.table['county'] == region] # was County Name
        dates = pd.to_datetime(table['todays_date']).dt.date.to_numpy() # was Most Recent Date
        return dates[0], dates[-1]

    def get(self, start_date, end_date, region):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @param region  Name of the region (county) for which data is wanted.
        @return  Tuple of arrays, arrays contain numbers of COVID positive patients and positive patients in ICU (intensive care).
        """
        table = self.table[self.table['county'] == region] # ^
        dates = pd.to_datetime(table['todays_date'])       # ^
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (dates >= start) & (dates <= end)
        return table[mask]['hospitalized_covid_confirmed_patients'].to_numpy(), table[mask]['icu_covid_confirmed_patients'].to_numpy() # ^

class HospitalUs(Data):
    """! Class for hospital data across the United States. Inherits the Data base class.
    """
    def __init__(self, state): 
        """! Hospital_US class initializer. Creates pandas dataframe.
        @param state  Name of the state for which data is wanted.
        """
        datafile = 'data/all-states-history.csv'
        stateabbr = us.states.lookup(state).abbr #.lower() # in the ...daily.csv the API call gives, the abbreviation is apparently lowercase
        table = pd.read_csv(datafile)[['state', 'date', 'hospitalizedCurrently', 'inIcuCurrently']]
        self.table = table[table.notnull().all(axis=1)]
        statetable = table[table['state'] == stateabbr] # select only the state we want
        statetable = statetable.drop(columns=['state']) # remove the 'state' column to maintain base-code dataframe format
        # Here we assume that once there is data, then the data is cumulative   # (WTF does this mean? Cumulative of what? -Eetu)
        self.table = statetable[table.notnull().all(axis=1)] # Make table only include rows where all values are not null. This is base code, from this I assume that in the .csv NaN does not mean 0
    
    def date_range(self):
        """! Get the first and last dates of available data in the dataset.
        @return  Tuple of datetime dates
        """
        dates = pd.to_datetime(self.table['date'], format='%Y-%m-%d').dt.date.to_numpy()
        return dates[-1], dates[0]
    
    def get(self, start_date, end_date):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @return  Tuple of arrays, arrays contain numbers of currently hospitalized covid patients and covid patients in ICU (intensive care) respectively.
        """
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        dates = pd.to_datetime(self.table['date'], format='%Y-%m-%d').dt.date.to_numpy() # The format is %Y%m%d in the ....daily.csv the base code uses with the API call, 
        mask = (dates >= start) & (dates <= end)                                         # but with the localized history csv, this format is used. Seems to work fine...
        masked = self.table[mask].sort_values(by='date')
        return masked['hospitalizedCurrently'].to_numpy(), masked['inIcuCurrently'].to_numpy()

# The DATASET_template uses a filepath supplied as an argument when launching the validation/generation files.
# If a new dataset is wanted to be implemented as an csv, the template should be able to create useful training data,
# if it includes confirmed cases, deaths (and recovery amounts if wanted).
# The names for these columns may be slightly different in different dataset.csv-files, so they can be changed in either
# the csv-files or by changing the hard coded column names in the prediction_data.py-file.

class DATASET_template(Data):
    """! Class for a custom dataset, inherits the Data base class. Data in the dataset is available 
    in three levels, nations, states and counties.
    """
    def __init__(self, filepath, columns, level):
        """! Custom dataset class initializer. Creates the used pandas dataframes.
        @param filepath The filepath of the custom dataset as given by the user.
        @columns Names of the columns in a list, so different names can be used. Hard-coded in prediction_data.py.
        @param level The level of data, either 'nation', 'states' or 'counties'.
        """
        self.level = level
        self.table = pd.read_csv(filepath)
        self.column_names = columns

        if level == "states" or level == "counties":
            self.state_list = self.table[self.column_names[2]].unique()
        
    def date_range(self, state = None, county = None, country = None):
        """! Get the first and last dates of available data in the dataset.
        @param state  Name of the state for which range is being looked up.
        @param county  Name of the county for which range is being looked up.
        @param country  Name of the country for which range is being looked up.
        @return  Tuple of strings, first date and last date, in format 'yyyy-mm-dd'.
        """
        if self.level == 'states':
            state_table = self.table[self.table['state']
                            == us.states.lookup(state).name]
            tab = state_table
        elif self.level == 'counties':
            state_table = self.table[self.table['state']
                            == us.states.lookup(state).name]
            tab = state_table[state_table['county'] == county]
        else:
            tab = self.table[self.table[self.column_names[1]] == country]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]
    
    def get(self, start_date, end_date, state = None, county = None, country = None):
        """! Get data from the dataset from a specific timeframe.
        @param start_date  The start date of the wanted timeframe, 'yyyy-mm-dd'
        @param end_date  The end date of the wanted timeframe, 'yyyy-mm-dd'
        @param state  Name of the state data is wanted from
        @param county  Name of the county data is wanted from
        @param country  Name of the country data is wanted from
        @return  Tuple of arrays, arrays contain numbers of cases and deaths (and possibly recoveries, which are not used?) respectively.
        """
        if self.level == 'states':
            state_table = self.table[self.table[self.column_names[2]]
                            == us.states.lookup(state).name]
            tab = state_table

        elif self.level == 'counties':
            state_table = self.table[self.table[self.column_names[2]]
                            == us.states.lookup(state).name]
            
            tab = state_table[state_table[self.column_names[3]] == county]
        else:
            tab = self.table[self.table[self.column_names[1]] == country]

        date = pd.to_datetime(tab[self.column_names[0]])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)

        if self.column_names[6] in tab:
            return tab[mask][self.column_names[4]].to_numpy(), tab[mask][self.column_names[5]].to_numpy(), tab[mask][self.column_names[6]].to_numpy()
        else:
            return tab[mask][self.column_names[4]].to_numpy(), tab[mask][self.column_names[5]].to_numpy()

if __name__ == '__main__':

    data = DATASET_template('data/custom2.csv', pdata.custom_dataset_columns, level = 'states')
    a,b  = data.get('2020-03-30', '2020-07-20', state = "Kentucky")
    print(len(a))
    print(len(b))

    data = DATASET_template('data/custom_dataset.csv', pdata.custom_dataset_columns, level = 'states')
    a,b  = data.get('2020-03-30', '2020-07-20', state = "Kentucky")
    print(len(a))
    print(len(b))

    data = DATASET_template('data/custom_dataset.csv', pdata.custom_dataset_columns, level = 'nation')
    a,b  = data.get('2020-07-20', '2021-07-07', country = "Brazil")
    print(a)
    print(b)

    data = NYTimes(level='states')
    a,b  = data.get('2020-04-01', '2020-04-02', state = "Kentucky")
    print(a)
    print(b)

    #data = Hospital_US('california')
    #a, b = data.get('2020-04-01', '2020-04-02')

    pass
