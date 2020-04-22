import pandas as pd
from datetime import datetime


# Filter the data based on the input paramaters
def get_subset_data(filepath, gender, start_date, end_date):
    data = pd.read_csv(filepath)
    data = data[data.Date.str.startswith('Date') == False]
    if gender:
        data = data[data.Gender == gender]

    data['Date'] = pd.to_datetime(data.Date, format="%Y-%m-%d")

    if start_date:
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        data = data[data.Date >= date_obj]

    if end_date:
        date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        data = data[data.Date <= date_obj]

    return data

# Example
# get_subset_data("Tweet_Data_With_Topic_Proportion_15.csv", "F", "2018-10-10", "2019-01-01")
