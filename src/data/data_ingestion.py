# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
import yaml
import logging
from src.logger import logging
from datetime import datetime
# from src.connections import s3_connection

# ========================================================================================================

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


# ========================================================================================================

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ========================================================================================================

def convert_and_split_by_date(df: pd.DataFrame, year_split: int = 2015):
    """
    Converts the 'date' column to datetime format and splits the DataFrame 
    into training and testing sets based on a given 'YEAR' threshold.

    - Rows with 'date' year <= year_split go into the training set.
    - Rows with 'date' year > year_split go into the testing set.

    Returns:
    train_data : a DataFrame containing rows with dates up to and including the split year.
    test_data : a DataFrame containing rows with dates after the split year.
    """
    try:
        if 'date' not in df.columns:
            raise ValueError("The DataFrame must contain a 'date' column.")

        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        train_data = df[df['date'].dt.year <= year_split].copy()
        test_data = df[df['date'].dt.year > year_split].copy()
        return train_data, test_data

    except Exception as e:
        print(f"Error in convert_and_split_by_date: {e}")
        raise


# ========================================================================================================

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


# ========================================================================================================

def main():
    try:
        df = load_data(data_url='https://raw.githubusercontent.com/Nayan027/Datasets/refs/heads/main/ipl-data.csv')

        # Load from S3 instead
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")

        train_data, test_data = convert_and_split_by_date(df, year_split=2015)

        save_data(train_data, test_data, data_path='./data_folder')

        logging.info("Dataset successfully split into train-test sets by YEAR.")
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


# ========================================================================================================
# execution
if __name__ == '__main__':
    main()
