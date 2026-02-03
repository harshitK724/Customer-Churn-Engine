"""Run: uvicorn src.serving.app:app --reload --port 8000"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.serving.model_service import model_service
from src.serving.schemas import ModelInfoResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Customer Churn Engine",
    description="Context-aware churn prediction with retention drivers",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return model_service.model_info()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_service.predict(request.model_dump(by_alias=False))


@app.post("/predict/batch", response_model=list[PredictResponse])
async def predict_batch(requests: list[PredictRequest]):
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [model_service.predict(r.model_dump(by_alias=False)) for r in requests]
