import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_system.pkl")

def load_system():
    model_path = os.path.join("models", "fraud_system.pkl")
    return joblib.load(model_path)

def predict(features, system):
    model = system["model"]
    scaler = system["scaler"]
    threshold = system["threshold"]

    expected_features = len(scaler.feature_names_in_)

    if len(features) != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, got {len(features)}"
        )

    df = pd.DataFrame(
        [features],
        columns=scaler.feature_names_in_
    )

    X_scaled = scaler.transform(df)

    prob = model.predict_proba(X_scaled)[:, 1][0]
    pred = 1 if prob >= threshold else 0

    return {
        "fraud_probability": float(prob),
        "prediction": pred
    }

