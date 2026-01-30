import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_system.pkl")


def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale ALL features consistently
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Handle class imbalance
    fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = XGBClassifier(
        scale_pos_weight=fraud_ratio,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    system = {
        "model": model,
        "scaler": scaler,
        "threshold": 0.50,
        "metadata": {
            "model_type": "XGBoost",
            "trained_at": str(datetime.datetime.now()),
            "version": "1.0.0"
        }
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(system, MODEL_PATH)

    print("✅ Training complete. Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()
