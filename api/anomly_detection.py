from fastapi import APIRouter, HTTPException
from sklearn.ensemble import IsolationForest
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict

'''
Anaomly Detection API Route V1: /api/v1/anomaly_detection
Takes in a request with time series data, returns anomaly detection results.
'''

class AnomalyDetectionRequest(BaseModel):
    time_series: Dict[str, List[float]] = Field(
        ...,
        example={
            "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z"],
            "values": [10.0, 20.0, 15.0, 30.0, 25.0]
        }
    )
    contamination: float = Field(0.1, example=0.1, description="The proportion of outliers in the data set.")
    random_state: int = Field(42, example=42, description="The seed used by the random number generator.")

class AnomalyDetectionResponse(BaseModel):
    anomalies: List[int] = Field(
        ...,
        example=[0, 1, 0, 1, 0]
    )

router = APIRouter()

@router.post("/anomaly_detection", response_model=AnomalyDetectionResponse, tags=["Anomaly Detection"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    try:
        # Extract the time series values
        values = np.array(request.time_series["values"]).reshape(-1, 1)

        # Initialize and fit the Isolation Forest model with custom parameters
        model = IsolationForest(contamination=request.contamination, random_state=request.random_state)
        model.fit(values)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = model.predict(values)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        anomalies = [1 if pred == -1 else 0 for pred in predictions]

        return AnomalyDetectionResponse(anomalies=anomalies)

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")