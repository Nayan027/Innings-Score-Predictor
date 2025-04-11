from flask import Flask, render_template, request
import mlflow
import dagshub
import pandas as pd
import warnings
import traceback
import joblib
import os
import sys
import yaml
from logger import logging  

warnings.filterwarnings("ignore")

# ------------------- Initialize tracking -------------------
mlflow.set_tracking_uri('https://dagshub.com/nayanparvez90/Innings-Score-Predictor.mlflow')
dagshub.init(repo_owner='nayanparvez90', repo_name='Innings-Score-Predictor', mlflow=True)

# ------------------- Flask app -------------------
app = Flask(__name__)

# ------------------- Load params -------------------
def load_params(path="params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

params = load_params()
columns_to_encode = params['feature_engineering']['columns_to_encode']
columns_to_scale = params['feature_engineering']['columns_to_scale']

# ------------------- Load model and transformers -------------------
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_versions:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
    return latest_versions[0].version if latest_versions else None

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# ------------------- Load encoder and scaler -------------------
local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
encoder_path = os.path.join(local_path, "encoder.pkl")
scaler_path = os.path.join(local_path, "scaler.pkl")

# Load encoder and scaler using pickle
try:
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    logging.error("Missing encoder or scaler file. Check your model artifacts.")
    raise
except Exception as e:
    logging.error(f"Unexpected error while loading transformers: {e}")
    raise

# Handle encoded column names such as [bat_team_Chennai Super Kins]
encoded_columns = encoder.get_feature_names_out(columns_to_encode)

# ------------------- Routes -------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        print("Received form data:", form_data)

        # Step 1: Convert form to DataFrame
        input_dict = {
            'bat_team': form_data.get('bat_team'),
            'bowl_team': form_data.get('bowl_team'),
            'overs': float(form_data.get('overs') or 4.1),
            'runs': int(form_data.get('runs') or 0),
            'wickets': int(form_data.get('wickets') or 0),
            'runs_last_5': int(form_data.get('runs_last_5') or 10),
            'wickets_last_5': int(form_data.get('wickets_last_5') or 0)
        }


    # Custom Validations
        if input_dict['bat_team'] == input_dict['bowl_team']:
            raise ValueError("Batting and Bowling teams cannot be the same.")

        if not (3.6 <= input_dict['overs'] <= 20):
            raise ValueError("Overs must be between 4.1 and 20.")

        if input_dict['wickets'] > 10:
            raise ValueError("Wickets cannot exceed 10.")
        
        if input_dict['wickets_last_5'] > input_dict["wickets"]:
            raise ValueError('wickets in last 5 overs cannot exceed more than total wickets fell') 
    

        input_df = pd.DataFrame([input_dict])
        logging.info(f"Input: {input_df.to_dict(orient='records')}")

        # Step 2: One-hot-encoder for categorical features
        try:
            encoded_df = pd.DataFrame(encoder.transform(input_df[columns_to_encode]),
                                      columns=encoded_columns)
        except Exception as e:
            logging.error("Encoding error: " + str(e))
            raise

        # Step 3: Scaling numerical columns
        scaled_df = pd.DataFrame(
            scaler.transform(input_df[columns_to_scale]),
            columns=columns_to_scale
        )

        # Step 4: Combining encoded + scaled data
        final_df = pd.concat([encoded_df, scaled_df], axis=1)

        # Step 5: Ensuring column order and handling missing columns
        for col in encoded_columns:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[encoded_columns.tolist() + columns_to_scale]

        logging.info(f"Final DataFrame for model prediction:\n{final_df}")
        logging.info(f"Shape: {final_df.shape}")

        # Step 6: Prediction
        try:
            prediction = int(model.predict(final_df)[0])
            logging.info(f"Prediction: {prediction}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise

        return render_template("result.html",
                               prediction=prediction,
                               lower_limit=prediction - 10,
                               upper_limit=prediction + 5)

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return render_template("result.html",
                               prediction=str(ve),
                               lower_limit="error",
                               upper_limit="error")

    except Exception as e:
        logging.exception("Prediction error occurred")
        traceback.print_exc()
        return render_template("result.html",
                               prediction="Error",
                               lower_limit="Error",
                               upper_limit="Check input or model")


# app execution
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
