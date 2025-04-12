import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from src.logger import logging


# ==================================================================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ==================================================================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Lasso:
    """Train the Lasso regression model."""
    try:
        regressor = Lasso(alpha=0.1, fit_intercept=True, max_iter=1000, positive=False)
        regressor.fit(X_train, y_train)
        logging.info('Lasso model training completed')
        return regressor

    except Exception as e:
        logging.error('Error during Lasso model training: %s', e)
        raise


# ==================================================================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and print accuracy."""
    try:
        y_pred = model.predict(X_test)
        R2score = r2_score(y_test, y_pred)
        logging.info('Model evaluation completed with an R2score of: %.4f', R2score)
        print(f'R2-score recorded for this algorithm is: {R2score:.4f}')

    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


# ==================================================================================================================

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)

    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise


# ==================================================================================================================

def main():
    try:
        train_data = load_data('./data/processed/train_transformed.csv')
        test_data = load_data('./data/processed/test_transformed.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        regressor = train_model(X_train, y_train)
        evaluate_model(regressor, X_test, y_test)
        save_model(regressor, 'models/model.pkl')
        logging.info("Model building step successfully executed.")

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


# ==================================================================================================================
# execution
if __name__ == '__main__':
    main()