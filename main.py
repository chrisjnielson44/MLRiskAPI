# main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api import feature_importance, anomly_detection, time_series, clustering, dimensionality_reduction
# time_series, anomaly_detection

app = FastAPI(
    title="MLRiskAPI",
    description="API for various machine learning analyses in the risk department",
    version="1.0.0",
)

# Include routers from different API modules
app.include_router(feature_importance.router, prefix="/api/v1", tags=["Feature Importance"])
app.include_router(time_series.router, prefix="/api/v1", tags=["Time Series"])
app.include_router(anomly_detection.router, prefix="/api/v1", tags=["Anomaly Detection"])
app.include_router(clustering.router, prefix="/api/v1", tags=["Clustering"])
app.include_router(dimensionality_reduction.router, prefix="/api/v1", tags=["Dimensionality Reduction"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="MLRiskAPI",
        version="1.0.0",
        description="This API provides various machine learning analyses for the risk metrics.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)