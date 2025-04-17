import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from src.logger import logging

# -------------------------------------------------------------------------------------
# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "nayanparvez90"
# repo_name = "Innings-Score-Predictor"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------

mlflow.set_tracking_uri('https://dagshub.com/nayanparvez90/Innings-Score-Predictor.mlflow')
dagshub.init(repo_owner='nayanparvez90', repo_name='Innings-Score-Predictor', mlflow=True)


# ==================================================================================================================

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise


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

def evaluate_model(regressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = regressor.predict(X_test)

        r2score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics_dict = {
            'r2score': r2score,
            'mae': mae,
            'mse': mse,
            'mape': mape
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


# ==================================================================================================================

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def log_preprocessing_artifacts(encoder_path: str, scaler_path: str, mlflow_artifact_subdir: str = "model") -> None:
    """Log encoder and scaler pickle files as artifacts to MLflow."""
    try:
        mlflow.log_artifact(encoder_path, artifact_path=mlflow_artifact_subdir)
        logging.info(f"Encoder logged to {mlflow_artifact_subdir}")
        
        mlflow.log_artifact(scaler_path, artifact_path=mlflow_artifact_subdir)
        logging.info(f"Scaler logged to {mlflow_artifact_subdir}")
    except Exception as e:
        logging.error('Error logging encoder/scaler: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


# ==================================================================================================================

def main():
    mlflow.set_experiment("My-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            regressor = load_model('./models/model.pkl')
            test_data = load_data('./data_folder/processed/test_transformed.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(regressor, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(regressor, 'get_params'):
                params = regressor.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(regressor, "model")

            # Log encoder and scaler using helper function
            log_preprocessing_artifacts('./models/encoder.pkl', './models/scaler.pkl')

            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            logging.info("Model Evaluation step completed.")

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


# ==================================================================================================================
# execution
if __name__ == '__main__':
    main()