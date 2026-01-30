from fastapi import FastAPI
from app.schema import PredictRequest, PredictResponse
from app.utils import load_system, predict

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model at startup
system = load_system()


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}


@app.post("/predict", response_model=PredictResponse)
def predict_fraud(request: PredictRequest):
    return predict(request.features, system)

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/metadata")
def metadata():
    return system.get("metadata", {})