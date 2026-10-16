import os
import time
import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import PatientData, PredictionResponse

# Configure Structured Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("heart_disease_api")

MODEL_PATH = "models/best_model.joblib"
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    logger.info("Starting up FastAPI application...")
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    else:
        logger.warning(f"Model file {MODEL_PATH} not found. Ensure model training is completed.")
    yield
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API serving Heart Disease Risk predictions with Prometheus metrics and structured logging.",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument Prometheus Metrics
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"]
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"Method={request.method} Path={request.url.path} Status={response.status_code} "
        f"Client={client_ip} Duration={duration_ms}ms"
    )
    return response

@app.get("/health", tags=["System"])
async def health_check():
    """Service health and model readiness probe."""
    is_ready = model_pipeline is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }

@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_heart_disease(patient: PatientData):
    """Predict risk of heart disease for a patient."""
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready or has not been trained yet."
            )

    try:
        # Convert Pydantic model to DataFrame for pipeline preprocessing
        input_dict = patient.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction = int(model_pipeline.predict(input_df)[0])
        probability = float(model_pipeline.predict_proba(input_df)[0][1])

        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        diagnosis = "Heart Disease Present (Elevated Risk)" if prediction == 1 else "Heart Disease Absent (Low Risk)"

        logger.info(f"Inference output: Pred={prediction}, Prob={probability:.4f}, Risk={risk_level}")

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            diagnosis=diagnosis,
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )
