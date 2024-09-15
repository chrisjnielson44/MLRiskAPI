from pydantic import BaseModel, Field
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class TimeSeriesRequest(BaseModel):
    time_series: Dict[str, List[float]] = Field(
        ...,
        example={
            "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z"],
            "values": [10.0, 20.0, 15.0, 30.0, 25.0]
        }
    )
    forecast_steps: int = Field(5, example=5, description="Number of steps to forecast into the future.")
    p: int = Field(5, example=5, description="ARIMA model parameter p (autoregressive order).")
    d: int = Field(1, example=1, description="ARIMA model parameter d (differencing order).")
    q: int = Field(0, example=0, description="ARIMA model parameter q (moving average order).")

class TimeSeriesResponse(BaseModel):
    forecast: List[float] = Field(
        ...,
        example=[26.0, 27.0, 28.0, 29.0, 30.0]
    )

router = APIRouter()

@router.post("/time_series", response_model=TimeSeriesResponse, tags=["Time Series"])
async def forecast_time_series(request: TimeSeriesRequest):
    try:
        # Extract the time series values
        values = np.array(request.time_series["values"])

        # Fit the ARIMA model with customizable parameters
        model = ARIMA(values, order=(request.p, request.d, request.q))
        model_fit = model.fit()

        # Forecast future values
        forecast = model_fit.forecast(steps=request.forecast_steps)

        return TimeSeriesResponse(forecast=forecast.tolist())

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

