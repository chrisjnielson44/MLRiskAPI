from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
from sklearn.cluster import KMeans
import numpy as np

class ClusteringRequest(BaseModel):
    features: Dict[str, List[float]] = Field(
        ...,
        example={
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0]
        }
    )
    n_clusters: int = Field(3, example=3, description="Number of clusters to form.")
    init: str = Field("k-means++", example="k-means++", description="Method for initialization.")
    max_iter: int = Field(300, example=300, description="Maximum number of iterations.")
    n_init: int = Field(10, example=10, description="Number of time the k-means algorithm will be run with different centroid seeds.")
    random_state: int = Field(42, example=42, description="The seed used by the random number generator.")

class ClusteringResponse(BaseModel):
    labels: List[int] = Field(
        ...,
        example=[0, 1, 1, 0, 2]
    )

router = APIRouter()

@router.post("/clustering", response_model=ClusteringResponse, tags=["Clustering"])
async def perform_clustering(request: ClusteringRequest):
    try:
        # Convert input data to numpy array
        X = np.array(list(request.features.values())).T

        # Initialize and fit the KMeans model with customizable parameters
        kmeans = KMeans(
            n_clusters=request.n_clusters,
            init=request.init,
            max_iter=request.max_iter,
            n_init=request.n_init,
            random_state=request.random_state
        )
        kmeans.fit(X)

        # Get the cluster labels
        labels = kmeans.labels_.tolist()

        return ClusteringResponse(labels=labels)

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

