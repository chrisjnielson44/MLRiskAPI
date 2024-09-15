from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Literal
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

class DimensionalityReductionRequest(BaseModel):
    features: Dict[str, List[float]] = Field(
        ...,
        example={
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
            "feature3": [1.0, 3.0, 5.0, 7.0, 9.0]
        }
    )
    n_components: int = Field(2, example=2, description="Number of principal components to keep.")
    method: Literal["PCA", "LDA", "TSNE"] = Field("PCA", example="PCA", description="Dimensionality reduction method to use.")

class DimensionalityReductionResponse(BaseModel):
    reduced_features: List[List[float]] = Field(
        ...,
        example=[[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5]]
    )

router = APIRouter()

@router.post("/dimensionality_reduction", response_model=DimensionalityReductionResponse, tags=["Dimensionality Reduction"])
async def perform_dimensionality_reduction(request: DimensionalityReductionRequest):
    try:
        # Convert input data to numpy array
        X = np.array(list(request.features.values())).T

        # Initialize and fit the appropriate model based on the method
        if request.method == "PCA":
            model = PCA(n_components=request.n_components)
        elif request.method == "LDA":
            model = LDA(n_components=request.n_components)
        elif request.method == "TSNE":
            model = TSNE(n_components=request.n_components)
        else:
            raise ValueError("Unsupported dimensionality reduction method.")

        reduced_features = model.fit_transform(X)

        return DimensionalityReductionResponse(reduced_features=reduced_features.tolist())

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")