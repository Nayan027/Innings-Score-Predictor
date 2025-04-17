import numpy as np
import pandas as pd
import os
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import joblib


# ===========================================================================================================

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


# ===========================================================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna(0)
        logging.info('Loaded data from %s with parsed dates and filled missing values with 0.', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ===========================================================================================================

def apply_feat_engg(train_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_encode: list, columns_to_scale: list):
    """
    Apply OneHotEncoder to selected categorical columns and StandardScaler to selected numerical columns.

    Args:
    - train_df (pd.DataFrame): Training dataset
    - test_df (pd.DataFrame): Testing dataset
    - columns_to_encode (list): List of categorical columns to encode
    - columns_to_scale (list): List of numerical columns to scale

    Returns:
    - train_df (pd.DataFrame): Transformed training dataset
    - test_df (pd.DataFrame): Transformed testing dataset
    - encoder (OneHotEncoder): Fitted encoder
    - scaler (StandardScaler): Fitted scaler
    """
    try:
        logging.info("Entered Feature Engineering pipeline...")

        target_col = 'total'
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        logging.info("Splitting dataframe into train and test sets")

        # Ensure expected categorical columns exist
        missing_in_train = [col for col in columns_to_encode if col not in X_train.columns]
        missing_in_test = [col for col in columns_to_encode if col not in X_test.columns]

        if missing_in_train:
            raise ValueError(f"Missing columns in train_df: {missing_in_train}")
        if missing_in_test:
            raise ValueError(f"Missing columns in test_df: {missing_in_test}")
        
        # OneHotEncode categorical features
        logging.info("Applying One-Hot-Encoding on: %s", columns_to_encode)

        encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
        train_encoded = encoder.fit_transform(X_train[columns_to_encode])
        test_encoded = encoder.transform(X_test[columns_to_encode])

        train_encoded_df = pd.DataFrame(train_encoded, 
                                        columns=encoder.get_feature_names_out(columns_to_encode),
                                        index=X_train.index)
        test_encoded_df = pd.DataFrame(test_encoded, 
                                       columns=encoder.get_feature_names_out(columns_to_encode),
                                       index=X_test.index)

        # Drop categorical columns after encoding
        X_train = X_train.drop(columns=columns_to_encode)
        X_test = X_test.drop(columns=columns_to_encode)

        
        logging.info("Scaling numerical features: %s", columns_to_scale)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train[columns_to_scale])
        test_scaled = scaler.transform(X_test[columns_to_scale])

        train_scaled_df = pd.DataFrame(train_scaled, columns=columns_to_scale, index=X_train.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=columns_to_scale, index=X_test.index)

        # Drop Unscaled numeric columns and join Scaled ones
        X_train = X_train.drop(columns=columns_to_scale).join(train_scaled_df)
        X_test = X_test.drop(columns=columns_to_scale).join(test_scaled_df)

        # Combine all i.e. Join encoded categorical columns to scaled numeric columns
        X_train = X_train.join(train_encoded_df)
        X_test = X_test.join(test_encoded_df)
        logging.info('Combined Scaled Numeric columns and Encoded Categorical columns successfully...')

        # Reattach target
        X_train[target_col] = y_train
        X_test[target_col] = y_test

        # Save encoder and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(encoder, open('models/encoder.pkl', 'wb'))
        joblib.dump(scaler, open('models/scaler.pkl', 'wb'))
        logging.info('Encoder and scaler saved to models/ directory.')

        return X_train, X_test, encoder, scaler

    except Exception as e:
        logging.error('Error during feature engineering: %s', e)
        raise


# ===========================================================================================================

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f'Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


# ===========================================================================================================

def main():
    try:
        params = load_params('params.yaml')
        columns_to_encode = params['feature_engineering']['columns_to_encode']
        columns_to_scale = params['feature_engineering']['columns_to_scale']
        # columns_to_encode = ['bat_team','bowl_team']                                 # hard-coded for testing
        # columns_to_scale = ['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5']

        train_data = load_data('./data_folder/interim/train_preprocessed.csv')
        test_data = load_data('./data_folder/interim/test_preprocessed.csv')

        
        train_df, test_df, _, _  = apply_feat_engg(train_data, test_data, columns_to_encode, columns_to_scale)

        save_data(train_df, os.path.join("./data_folder", "processed", "train_transformed.csv"))
        save_data(test_df, os.path.join("./data_folder", "processed", "test_transformed.csv"))

        logging.info("Feature engineering process completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


# ===========================================================================================================
#Execution
if __name__ == '__main__':
    main()



    