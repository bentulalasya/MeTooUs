# Import libraries
import pandas as pd
import os


# Reads file and removes re tweets and tweets that does not have gender
def read_tweet_file(file_path):
    print(file_path)
    file_data = pd.read_excel(file_path, skiprows=[0], usecols="A,B,D,M")
    file_data = file_data[file_data.Contents.str.startswith('RT @') == False]
    file_data = file_data[(file_data.Gender == "M") | (file_data.Gender == "F")]
    file_data.rename(columns={"Date (GMT)": "Date"})
    return file_data


# Given a directory read all the files and write the result to new csv file.
def read_all_files_in_dir(dir_path):
    all_files = os.listdir(dir_path)
    for file in all_files:
        cur_data = read_tweet_file(dir_path + '/' + file)
        cur_data.to_csv('Tweet_data_With_Cols.csv', mode='a', header=True, index=False, date_format='%Y-%m-%d')


# Call the main method
read_all_files_in_dir('../Data/')
