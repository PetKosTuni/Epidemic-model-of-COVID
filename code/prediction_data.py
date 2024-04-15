

# Starting dates for predictions for different countries.
START_nation = {
    "Brazil": "2020-03-30",
    "Canada": "2020-03-28",
    "Mexico": "2020-03-30",
    "India": "2020-03-28",
    "Turkey": "2020-03-22",
    "Russia": "2020-04-01",
    "Saudi Arabia": "2020-03-28",
    "US": "2020-03-22",
    "United Arab Emirates": "2020-04-10",
    "Qatar": "2020-04-06",
    "France": "2020-03-20",
    "Spain": "2020-03-15",
    "Indonesia":"2020-03-28",
    "Peru": "2020-04-06",
    "Chile": "2020-05-08",
    "Pakistan": "2020-04-01",
    "Germany":"2020-03-15",
    "Italy": "2020-03-10",
    "South Africa": "2020-04-10",
    "Sweden": "2020-03-25",
    "United Kingdom": "2020-03-25",
    "Colombia": "2020-04-03",
    "Argentina": "2020-04-03",
    "Bolivia": "2020-04-26",
    "Ecuador": "2020-03-28",
    "Iran": "2020-03-15"
}

# Decays and "a" values for different countries.
# First element represents the "a" value and the second element represents the decay.
FR_nation = {
    "Brazil": [0.2, 0.02],
    "Canada": [0.1, 0.015],
    "Mexico": [0.35, 0.015],
    "India": [0.20, 0.02],
    "Turkey": [1, 0.04],
    "Russia": [0.1, 0.022],
    "Saudi Arabia": [0.2, 0.035],
    "US": [0.75, 0.02],
    "United Arab Emirates": [0.07, 0.04],
    "Qatar": [0.02, 0.05],
    "France": [0.25, 0.015],
    "Spain": [0.4, 0.02],
    "Indonesia": [0.5, 0.02],
    "Peru": [0.1, 0.013],
    "Chile": [0.08, 0.025],
    "Pakistan": [0.16, 0.025],
    "Germany":[0.4, 0.1],
    "Italy":[0.35, 0.02],
    "South Africa": [0.1, 0.026],
    "Sweden": [0.5, 0.028],
    "United Kingdom": [0.5, 0.028],
    "Colombia": [0.17, 0.01],
    "Argentina": [0.1, 0.012],
    "Bolivia": [0.2, 0.015],
    "Ecuador": [0.5, 0.015],
    "Iran": [0.5, 0.02]
}

# Decays and a values for different US states.
# First element represents the "a" value and the second element represents the decay.
decay_state = {
    "Pennsylvania": [0.7, 0.024],
    "New York": [0.7, 0.042],
    "Illinois": [0.7, 0.035],
    "California": [0.5, 0.016],
    "Massachusetts": [0.7, 0.026],
    "New Jersey": [0.7, 0.03],
    "Michigan": [0.8, 0.035],
    "Virginia": [0.7, 0.034],
    "Maryland": [0.7, 0.024],
    "Washington": [0.7, 0.036],
    "North Carolina": [0.7, 0.018],
    "Wisconsin": [0.7, 0.034],
    "Texas": [0.3, 0.016],
    "New Mexico": [0.7, 0.02],
    "Louisiana": [0.4, 0.02],
    "Arkansas": [0.7, 0.02],
    "Delaware": [0.7, 0.03],
    "Georgia": [0.7, 0.015],
    "Arizona": [0.7, 0.02],
    "Connecticut": [0.7, 0.026],
    "Ohio": [0.7, 0.024],
    "Kentucky": [0.7, 0.023],
    "Kansas": [0.7, 0.02],
    "New Hampshire": [0.7, 0.014],
    "Alabama": [0.7, 0.024],
    "Indiana": [0.7, 0.03],
    "South Carolina": [0.7, 0.02],
    "Colorado": [0.7, 0.02],
    "Florida": [0.4, 0.016],
    "West Virginia": [0.7, 0.022],
    "Oklahoma": [0.7, 0.03],
    "Mississippi": [0.7, 0.026],
    "Missouri": [0.7, 0.02],
    "Utah": [0.7, 0.018],
    "Alaska": [0.7, 0.04],
    "Hawaii": [0.7, 0.04],
    "Wyoming": [0.7, 0.04],
    "Maine": [0.7, 0.025],
    "District of Columbia": [0.7, 0.024],
    "Tennessee": [0.7, 0.027],
    "Idaho": [0.7, 0.02],
    "Oregon": [0.7, 0.036],
    "Rhode Island": [0.7, 0.024],
    "Nevada": [0.5, 0.022],
    "Iowa": [0.7, 0.02],
    "Minnesota": [0.7, 0.025],
    "Nebraska": [0.7, 0.02],
    "Montana": [0.5, 0.02]
}

# Middle dates for predictions for different US states.
mid_dates_state = {
    "Alabama": "2020-06-03",
    "Arizona": "2020-05-28",
    "Arkansas": "2020-05-11",
    "California": "2020-05-30",
    "Georgia": "2020-06-05",
    "Nevada": "2020-06-01",
    "Oklahoma": "2020-05-31",
    "Oregon": "2020-05-29",
    "Texas": "2020-06-15",
    "Ohio": "2020-06-09",
    "West Virginia": "2020-06-08",
    "Florida": "2020-06-01",
    "South Carolina": "2020-05-25",
    "Utah": "2020-05-28",
    "Iowa": "2020-06-20",
    "Idaho": "2020-06-15",
    "Montana": "2020-06-15",
    "Minnesota": "2020-06-20",
    "Illinois": "2020-06-30",
    "New Jersey": "2020-06-30",
    "North Carolina": "2020-06-20",
    "Maryland":  "2020-06-25",
    "Kentucky": "2020-06-30",
    "Pennsylvania": "2020-07-01",
    "Colorado": "2020-06-20",
    "New York": "2020-06-30",
    "Alaska": "2020-06-30",
    "Washington": "2020-06-01"
}

# Resurge middle dates for predictions for different US states.
mid_dates_state_resurge = {
    "Colorado": "2020-09-10",
    "California": "2020-09-30",
    "Florida": "2020-09-20",
    "Illinois": "2020-09-10",
    "New York": "2020-09-10",
    "Texas": "2020-09-15"
}

# Middle dates for predictions for different counties in California.
mid_dates_county = {
    "San Joaquin": "2020-05-26",
    "Contra Costa": "2020-06-02",
    "Alameda": "2020-06-03",
    "Kern": "2020-05-20",
    "Tulare": "2020-05-30",
    "Sacramento": "2020-06-02",
    "Fresno": "2020-06-07",
    "San Bernardino": "2020-05-25",
    "Los Angeles": "2020-06-05",
    "Santa Clara": "2020-05-29",
    "Orange": "2020-06-12",
    "Riverside": "2020-05-26",
    "San Diego": "2020-06-02"
}

# Middle dates for predictions for different nations,
mid_dates_nation = {
    "US": "2020-06-15",
    "Mexico": "2020-07-05",
    "India": "2020-07-30",
    "South Africa": "2020-06-01",
    "Brazil": "2020-07-20",
    "Iran": "2020-08-30",
    "Bolivia": "2020-05-25",
    "Indonesia": "2020-08-01",
    "Italy": "2020-07-15",
    "Canada": "2020-08-15",
    "Russia": "2020-08-20",
    "United Kingdom": "2020-07-08",
    "Spain": "2020-07-30",
    "France": "2020-06-28",
    "Argentina": "2020-08-01"#,
    # Duplicates from original code??
    #"United Kingdom": "2020-07-20",
    #"Canada": "2020-08-30"
}

# Counties in Northern California.
north_cal = [
    "Santa Clara",
    "San Mateo",
    "Alameda",
    "Contra Costa",
    "Sacramento",
    "San Joaquin",
    "Fresno"
]

# Change these values depending on the dataset implemented either in the csv-file, or here.
custom_dataset_filepath = 'data/custom_dataset.csv'
custom_dataset_columns = ['date', 'country', 'state', 'county', 'cases', 'deaths', 'recoveries']