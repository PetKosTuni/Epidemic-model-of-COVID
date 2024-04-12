import numpy as np
import pandas as pd
from datetime import datetime
from util import first_valid_date

def get_JHU(level):
    """! Get county/state/nation level COVID-19 data from JHU global data files. Source of data is linked in commented code.
    @param level Specifies the level of data to be retrieved (county/state/nation).
    @return Pandas DataFrame containing COVID-19 information about the specific level or 0 if level value is not valid.
    """
    
    # Calculate the number of days from the specified date.
    length = (datetime.today() - datetime.strptime("2020-03-10", "%Y-%m-%d")).days

    if level == "nation":
        # Load data for global confirmed cases, deaths, and recoveries
        confirm_file = 'data/jhu_confirmed_global.csv'
        death_file = 'data/jhu_deaths_global.csv'
        recover_file = 'data/jhu_recovered_global.csv'
        df_confirm = pd.read_csv(confirm_file)
        df_death = pd.read_csv(death_file)
        df_recover = pd.read_csv(recover_file)

        # Get unique regions (countries)
        regions = df_confirm["Country/Region"].unique()

        # Iterate over regions to extract data
        data_frames = []
        for region in regions:
            # Filter data for the current region
            confirm_region = df_confirm[df_confirm["Country/Region"] == region]
            death_region = df_death[df_death["Country/Region"] == region]
            recover_region = df_recover[df_recover["Country/Region"] == region]

            # Sum data over all provinces/states within the region
            data_confirm = confirm_region.iloc[:, 1:].sum(axis=0)
            data_death = death_region.iloc[:, 1:].sum(axis=0)
            data_recover = recover_region.iloc[:, 1:].sum(axis=0)

            # Get dates
            dates = confirm_region.columns

            # Trim data to the last "length" days
            data_confirm = np.asarray(data_confirm)[-length:]
            data_death = np.asarray(data_death)[-length:]
            data_recover = np.asarray(data_recover)[-length:]
            dates = np.asarray(dates)[-length:].tolist()

            # Change date format
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]

            # Construct DataFrame for the region
            data = {
                "Country_Region": region,
                "ConfirmedCases": data_confirm,
                "Fatalities": data_death,
                "Recovered": data_recover,
                "Date": dates
            }
            df_region = pd.DataFrame(data)
            data_frames.append(df_region)

        # Combine all DataFrames
        results = pd.concat(data_frames).reset_index(drop=True)

    elif level == "states":
        # Load data for US confirmed cases and deaths
        confirm_file = 'data/jhu_confirmed_us.csv'
        death_file = 'data/jhu_deaths_us.csv'
        df_confirm = pd.read_csv(confirm_file)
        df_death = pd.read_csv(death_file)

        # Get unique states
        states = df_confirm["Province_State"].unique()

        # Iterate over states to extract data
        data_frames = []
        for state in states:
            # Filter data for the current state
            confirm_state = df_confirm[df_confirm["Province_State"] == state]
            death_state = df_death[df_death["Province_State"] == state]

            # Sum data over all counties within the state
            data_confirm = confirm_state.iloc[:, 1:].sum(axis=0)
            data_death = death_state.iloc[:, 1:].sum(axis=0)

            # Get dates
            dates = confirm_state.columns

            # Trim data to the last "length" days
            data_confirm = np.asarray(data_confirm)[-length:]
            data_death = np.asarray(data_death)[-length:]
            dates = np.asarray(dates)[-length:].tolist()

            # Find the first valid date after possible invalid data items
            first_date_index = first_valid_date(dates)

            # Change date format
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates[first_date_index:]]

            # Construct DataFrame for the state
            data = {
                "state": state,
                "date": dates,
                "cases": data_confirm[-len(dates):],
                "deaths": data_death[-len(dates):]
            }
            df_state = pd.DataFrame(data)
            data_frames.append(df_state)

        # Combine all DataFrames
        results = pd.concat(data_frames).reset_index(drop=True)

    elif level == "counties":
        # Load data for US confirmed cases and deaths
        confirm_file = 'data/jhu_confirmed_us.csv'
        death_file = 'data/jhu_deaths_us.csv'
        df_confirm = pd.read_csv(confirm_file)
        df_death = pd.read_csv(death_file)

        # Get unique states
        states = df_confirm["Province_State"].unique()

        # Iterate over states and counties to extract data
        data_frames = []
        for state in states:
            confirm_state = df_confirm[df_confirm["Province_State"] == state]
            death_state = df_death[df_death["Province_State"] == state]
            counties = confirm_state["Admin2"].unique()

            for county in counties:
                if isinstance(county, str):
                    confirm_county = confirm_state[confirm_state["Admin2"] == county]
                    death_county = death_state[death_state["Admin2"] == county]
                    fips = confirm_county.FIPS.to_numpy()[0]

                    data_confirm = confirm_county.values[:, 1:].sum(axis=0)
                    data_death = death_county.values[:, 1:].sum(axis=0)
                    dates = confirm_county.columns

                    data_confirm = np.asarray(data_confirm)[-length:]
                    data_death = np.asarray(data_death)[-length:]
                    dates = np.asarray(dates)[-length:].tolist()

                    first_date_index = first_valid_date(dates)
                    dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates[first_date_index:]]

                    if not np.isnan(fips):
                        data = {
                            "county": county,
                            "state": state,
                            "date": dates,
                            "cases": data_confirm[-len(dates):],
                            "deaths": data_death[-len(dates):],
                            "fips": "0" + str(int(fips)) if fips < 9999 else str(int(fips))
                        }
                        df_county = pd.DataFrame(data)
                        data_frames.append(df_county)

        # Combine all DataFrames
        results = pd.concat(data_frames).reset_index(drop=True)

    else:
        # If "level" is not valid, return 0
        return 0

    return results
