from pydantic import BaseModel, validator
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

    @validator("features")
    def validate_feature_length(cls, v):
        if len(v) != 30:
            raise ValueError("Model expects exactly 30 features")
        return v


class PredictResponse(BaseModel):
    fraud_probability: float
    prediction: int