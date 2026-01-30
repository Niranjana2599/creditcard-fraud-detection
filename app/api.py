from fastapi import FastAPI
from fastapi.responses import Response
from app.schema import PredictRequest, PredictResponse
from app.utils import load_system, predict
from prometheus_client import Counter, generate_latest
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
# Prometheus Monitoring Counters
# --------------------------------------------------

prediction_counter = Counter(
    "total_predictions",
    "Total number of predictions made"
)

fraud_counter = Counter(
    "fraud_predictions",
    "Total number of fraud predictions"
)

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
    """
    Prometheus-compatible metrics endpoint
    """
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictResponse)
def predict_fraud(request: PredictRequest):

    logger.info("Prediction request received")

    result = predict(request.features, system)

    # Increment Prometheus counters
    prediction_counter.inc()

    if result["prediction"] == 1:
        fraud_counter.inc()

    logger.info(
        f"Model version: {system.get('metadata', {}).get('version')} | "
        f"Prediction: {result['prediction']} | "
        f"Probability: {result['fraud_probability']}"
    )

    return result
