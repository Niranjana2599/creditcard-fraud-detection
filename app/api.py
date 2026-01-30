from fastapi import FastAPI
from app.schema import PredictRequest, PredictResponse
from app.utils import load_system, predict
import logging

app = FastAPI(title="Credit Card Fraud Detection API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
system = load_system()


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}


@app.post("/predict", response_model=PredictResponse)
def predict_fraud(request: PredictRequest):

    logger.info("Prediction request received")

    result = predict(request.features, system)

    logger.info(
        f"Model version: {system.get('metadata', {}).get('version')} | "
        f"Prediction: {result['prediction']} | "
        f"Probability: {result['fraud_probability']}"
    )

    return result


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return system.get("metadata", {})
