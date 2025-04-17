# This is for the data preprocessing; different preprocessing techniques will be used in this file.

import numpy as np
import pandas as pd
import os
from src.logger import logging


# ================================= Preprocessing Steps ==================================================

def remove_unwanted_columns(df):
    columns_to_remove = ['mid','date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
    try:
        df = df.drop(labels=columns_to_remove, axis=1)
        return df
    except Exception as e:
        print(f"Error removing columns: {e}")
        raise

def filter_consistent_teams(df):
    consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                        'Delhi Daredevils', 'Sunrisers Hyderabad']
    try:
        df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
        return df
    except Exception as e:
        print(f"Error filtering teams: {e}")
        raise

def remove_initial_overs(df, min_overs=4.0):
    try:
        df = df[df['overs'] >= min_overs]
        return df
    except Exception as e:
        print(f"Error removing initial few overs: {e}")
        raise


# ================================== Preprocessing Pipeline =============================

def preprocess_df(df):
    try:
        df = remove_unwanted_columns(df)
        df = filter_consistent_teams(df)
        df = remove_initial_overs(df)
        return df

    except Exception as e:
        print(f"Error occurred in preprocessing step: {e}")
        raise


# ======================================== Main Function =============================================

def main():
    try:
        # Fetch the train & test sets from data/raw
        logging.info("Fetching the data from the raw folder i.e. inside the data folder")
        train_data = pd.read_csv('./data_folder/raw/train.csv')
        test_data = pd.read_csv('./data_folder/raw/test.csv')
        logging.info("Both sets loaded properly")

        # Apply preprocessing to both train and test sets
        train_processed_data = preprocess_df(train_data)
        test_processed_data = preprocess_df(test_data)

        # Store the data inside data/interim
        data_path = os.path.join("./data_folder", 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)

        logging.info("Preprocessed data saved to %s", data_path)

    except Exception as e:
        logging.error("An error occurred during the data preprocessing: %s", e)
        print(f"Error: {e}")


# ===============================================================================================================
# Execution
if __name__ == "__main__":
    main()