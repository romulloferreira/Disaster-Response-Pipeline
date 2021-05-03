# import packages
import sys

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """Function to load data (csv file) and merges datasets.

    Args:
        messages_filepath: String. The file path of csv file that contains disaster messages.
        categories_filepath: String. The file path of csv file that contains disaster categories for messages.

    Returns:
        df (DataFrame): Dataframe of messages and categories.
        
    """

    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id', how='outer')

    return df


def clean_data(df):

    """Function to clean df.

    Args:
        df: pandas DataFrame. Dataframe of messages and categories.
      
    Returns:
        df: pandas DataFrame. Dataframe after the cleaning process. A cleaned dataframe (messages and categories).
        
    """

    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe 
    # extract a list of new column names for categories
    # apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    row = categories.iloc[0]
    category_colnames = list(map(lambda x : x[:-2] ,row))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    # iterate through the category columns in df
    for column in categories:
        
        # keep only the last character of each string (the 1 or 0)
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from the dataframe (df)
    df = df.drop(['categories'], axis = 1)

    # concatenate the original df with the new categories dataframe
    df = pd.concat([df,categories], axis=1)

    # drop the duplicates
    df.drop_duplicates(inplace = True)
    
    # remove non-binary values
    df = df.query('related == 0 | related == 1')

    return df
    


def save_data(df, database_filename):
    
    """Function to save df in a database.

    Args:
        df: pandas DataFrame. A cleaned dataframe (messages and categories).
        database_filename: String. Database file name.

    """  
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace', chunksize = 600)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
              
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()