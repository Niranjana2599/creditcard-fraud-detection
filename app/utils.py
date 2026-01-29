import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_system.pkl")


def load_system():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)


def predict(features, system):
    model = system["model"]
    scaler = system["scaler"]
    threshold = system["threshold"]

    # Convert input into DataFrame with correct feature names
    df = pd.DataFrame([features], columns=scaler.feature_names_in_)

    # Scale exactly same way as training
    X_scaled = scaler.transform(df)

    probability = model.predict_proba(X_scaled)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    return {
        "fraud_probability": float(probability),
        "prediction": prediction
    }
