from fastapi import FastAPI
from app.schema import PredictRequest, PredictResponse
from app.utils import load_system, predict
import logging

# --------------------------------------------------
# App Initialization
# --------------------------------------------------

app = FastAPI(title="Credit Card Fraud Detection API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
system = load_system()

# --------------------------------------------------
# Simple In-Memory Monitoring Counters
# --------------------------------------------------

prediction_count = 0
fraud_count = 0

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return system.get("metadata", {})


@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count,
        "fraud_predictions": fraud_count
    }


@app.post("/predict", response_model=PredictResponse)
def predict_fraud(request: PredictRequest):

    global prediction_count, fraud_count

    logger.info("Prediction request received")

    result = predict(request.features, system)

    prediction_count += 1

    if result["prediction"] == 1:
        fraud_count += 1

    logger.info(
        f"Model version: {system.get('metadata', {}).get('version')} | "
        f"Prediction: {result['prediction']} | "
        f"Probability: {result['fraud_probability']}"
    )

    return result
