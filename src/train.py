import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

# 1. Load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "creditcard.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train model
best_xgb = XGBClassifier(random_state=42)
best_xgb.fit(X_train_scaled, y_train)

# 5. Save system
system = {
    "model": best_xgb,
    "scaler": scaler,
    "threshold": 0.50
}

# DEBUG PRINTS
print("Number of features:", X.shape[1])
print("Scaler features:", len(scaler.feature_names_in_))

os.makedirs("models", exist_ok=True)
joblib.dump(system, "models/fraud_system.pkl")

print("✅ Training complete and model saved.")

