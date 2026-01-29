import joblib
import pandas as pd

# Load system
system = joblib.load("fraud_system.pkl")

model = system["model"]
scaler = system["scaler"]
threshold = system["threshold"]


def predict_transaction(transaction):

    # Create DataFrame with correct feature names
    transaction_df = pd.DataFrame(
        [transaction],
        columns=model.feature_names_in_
    )

    # Scale ONLY the column that scaler was trained on
    scaled_column = scaler.feature_names_in_[0]  

    transaction_df[scaled_column] = scaler.transform(
        transaction_df[[scaled_column]]
    )

    # Predict
    probability = model.predict_proba(transaction_df)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    return {
        "fraud_probability": float(probability),
        "prediction": prediction
    }


if __name__ == "__main__":
    sample_transaction = [0] * len(model.feature_names_in_)
    result = predict_transaction(sample_transaction)
    print(result)