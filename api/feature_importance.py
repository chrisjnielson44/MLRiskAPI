from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

'''
Feature Importance API Route V1: /api/v1/feature_importance
Takes in a request with feature names, features, and target values, returns feature importance scores.
'''

router = APIRouter()

from pydantic import BaseModel, Field
from typing import Dict, List

class FeatureImportanceRequest(BaseModel):
    feature_names: Dict[str, str] = Field(
        ...,
        example={
            "feature1": "Age",
            "feature2": "Income",
            "feature3": "Education Level"
        }
    )
    features: Dict[str, List[float]] = Field(
        ...,
        example={
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
            "feature3": [1.0, 3.0, 5.0, 7.0, 9.0]
        }
    )
    target: List[float] = Field(..., example=[10.0, 20.0, 30.0, 40.0, 50.0])
    n_estimators: int = Field(100, example=100, description="Number of trees in the Random Forest")
    rf_weight: float = Field(0.6, example=0.6, description="Weight for Random Forest importance")
    mi_weight: float = Field(0.4, example=0.4, description="Weight for Mutual Information importance")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feature_names": {
                        "feature1": "Age",
                        "feature2": "Income",
                        "feature3": "Education Level"
                    },
                    "features": {
                        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
                        "feature3": [1.0, 3.0, 5.0, 7.0, 9.0]
                    },
                    "target": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "n_estimators": 100,
                    "rf_weight": 0.6,
                    "mi_weight": 0.4
                }
            ]
        }
    }

class FeatureImportanceResponse(BaseModel):
    feature_importance: Dict[str, float] = Field(
        ...,
        example={
            "feature1": 0.3,
            "feature2": 0.5,
            "feature3": 0.2
        }
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feature_importance": {
                        "feature1": 0.3,
                        "feature2": 0.5,
                        "feature3": 0.2
                    }
                }
            ]
        }
    }

@router.post("/feature_importance", response_model=FeatureImportanceResponse, tags=["Feature Importance"])
async def get_feature_importance(request: FeatureImportanceRequest):
    try:
        # Convert input data to numpy arrays
        X = np.array(list(request.features.values())).T
        y = np.array(request.target)

        # Check if the shapes of X and y match
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in features and target do not match.")

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate feature importance using Random Forest
        rf = RandomForestRegressor(n_estimators=request.n_estimators, random_state=42)
        rf.fit(X_scaled, y)
        rf_importance = rf.feature_importances_

        # Calculate mutual information scores
        mi_importance = mutual_info_regression(X_scaled, y)

        # Combine the two methods using the provided weights
        combined_importance = request.rf_weight * rf_importance + request.mi_weight * mi_importance

        # Normalize the importance scores
        normalized_importance = combined_importance / np.sum(combined_importance)

        # Create a dictionary of feature names and their importance scores
        feature_importance = {
            request.feature_names[key]: value
            for key, value in zip(request.features.keys(), normalized_importance)
        }

        # Sort the features by importance (descending order)
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        return FeatureImportanceResponse(feature_importance=feature_importance)

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")