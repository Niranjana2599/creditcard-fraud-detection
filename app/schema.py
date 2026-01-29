from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    fraud_probability: float
    prediction: int
