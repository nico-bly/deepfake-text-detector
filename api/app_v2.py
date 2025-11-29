"""
Production-ready FastAPI backend for deepfake text detection.
Supports: Local VPS inference, Modal serverless, and client-side inference.

SIMPLIFIED VERSION - 4 essential endpoints only:
  GET  /health       - Health check (no auth)
  GET  /models       - List available models (auth required)
  POST /predict      - Single prediction (auth required)
  POST /batch-predict - Batch predictions (auth required)
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from .config import get_settings, InferenceBackend
from .inference import InferenceRouter, InferenceBackend as IBackend
from .model_mapping import (
    resolve_model, get_model_info, list_all_mappings, 
    get_available_datasets, validate_combination
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings and router
settings = get_settings()
router = InferenceRouter(settings)

# ============================================================================
# Security - API Key Authentication
# ============================================================================

API_KEY = settings.VITE_API_KEY
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify API key from X-API-Key header"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API key. Include X-API-Key header in request."
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("=" * 60)
    logger.info("Deepfake Detection API Starting")
    logger.info("=" * 60)
    logger.info(f"Default backend: {settings.DEFAULT_INFERENCE_BACKEND}")
    logger.info(f"VPS GPU: {settings.VPS_GPU_MEMORY_GB}GB | CPU: {settings.VPS_CPU_MEMORY_GB}GB")
    logger.info("=" * 60)
    
    yield
    
    logger.info("API shutting down")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Deepfake Detection API",
    description="Multi-backend inference system for AI-generated text detection",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictRequest(BaseModel):
    """Single prediction request"""
    text: str = Field(
        ..., 
        min_length=settings.MIN_TEXT_LENGTH, 
        max_length=settings.MAX_TEXT_LENGTH,
        description="Text to analyze"
    )
    model_id: str = Field(
        ..., 
        description="Model identifier (e.g., 'small-perplexity')"
    )
    dataset: str = Field(
        ..., 
        description="Dataset name (e.g., 'human-ai-binary')"
    )
    prefer_backend: Optional[IBackend] = Field(
        None, 
        description="Preferred backend (local, modal, client)"
    )


class PredictResponse(BaseModel):
    """Single prediction response"""
    prediction: int = Field(..., description="0=human, 1=AI-generated")
    probability: float = Field(..., description="Probability percentage [0-100]")
    confidence: float = Field(..., description="Confidence percentage [0-100]")
    is_fake: bool = Field(..., description="True if AI-generated")
    backend: str = Field(..., description="Backend used")
    model_id: str = Field(..., description="Model used (dataset/model_id)")
    latency_ms: float = Field(..., description="Inference latency")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(
        ..., 
        min_items=1,
        max_items=100,
        description="List of texts to analyze"
    )
    model_id: str = Field(
        ..., 
        description="Model identifier"
    )
    dataset: str = Field(
        ..., 
        description="Dataset name"
    )
    prefer_backend: Optional[IBackend] = Field(None)


class BatchPredictResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictResponse]
    total_time_ms: float
    batch_size: int


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    dataset: str
    backend_type: str
    size_mb: Optional[float]
    description: str
    metadata: Dict[str, Any]


class ModelsResponse(BaseModel):
    """Response for /models endpoint"""
    datasets: List[str]
    total_models: int
    models_by_dataset: Dict[str, List[str]]
    models_detailed: Dict[str, Dict[str, ModelInfo]]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    backends: Dict[str, Any]
    timestamp: float


# ============================================================================
# ENDPOINT 1: GET /health (No Auth)
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """
    Health check endpoint - no authentication required.
    
    Used by: Cloudflare, monitoring systems, load balancers
    
    Response: Status of all backends + timestamp
    """
    backends_status = await router.health_check()
    
    return HealthResponse(
        status="healthy",
        backends=backends_status,
        timestamp=time.time()
    )


# ============================================================================
# ENDPOINT 2: GET /models (Auth Required)
# ============================================================================

@app.get("/models", response_model=ModelsResponse, tags=["models"])
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List all available models and datasets.
    
    Required: X-API-Key header
    
    Returns: All datasets, their models, and detailed configuration
    
    Example:
        GET /models
        Headers: X-API-Key: your-key
    
    Response:
    {
        "datasets": ["human-ai-binary", "arxiv", "fakenews"],
        "total_models": 10,
        "models_by_dataset": {
            "human-ai-binary": ["small-perplexity", "large-perplexity"],
            "arxiv": ["small-perplexity", "medium-perplexity"]
        },
        "models_detailed": {
            "human-ai-binary": {
                "small-perplexity": {
                    "model_id": "small-perplexity",
                    "dataset": "human-ai-binary",
                    "backend_type": "vps",
                    "size_mb": 45.2,
                    "description": "Small model for human-ai-binary dataset",
                    "metadata": {...}
                }
            }
        }
    }
    """
    try:
        # Get all datasets
        datasets = get_available_datasets()
        
        # Build models by dataset
        models_by_dataset = {}
        models_detailed = {}
        
        for dataset in datasets:
            model_ids = list_all_mappings(dataset)
            models_by_dataset[dataset] = model_ids
            models_detailed[dataset] = {}
            
            for model_id in model_ids:
                info = get_model_info(dataset, model_id)
                models_detailed[dataset][model_id] = ModelInfo(
                    model_id=model_id,
                    dataset=dataset,
                    backend_type=info["backend_type"],
                    size_mb=info.get("size_mb"),
                    description=info.get("description", ""),
                    metadata=info.get("metadata", {})
                )
        
        # Count total models
        total_models = sum(len(models) for models in models_by_dataset.values())
        
        return ModelsResponse(
            datasets=datasets,
            total_models=total_models,
            models_by_dataset=models_by_dataset,
            models_detailed=models_detailed
        )
    
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to list models", "message": str(e)}
        )


# ============================================================================
# ENDPOINT 3: POST /predict (Auth Required)
# ============================================================================

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    request: PredictRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if text is AI-generated or human-written.
    
    Required: X-API-Key header
    
    Request body:
    {
        "text": "This is some text to analyze",
        "model_id": "small-perplexity",
        "dataset": "human-ai-binary"
    }
    
    Response:
    {
        "prediction": 1,
        "probability": 87.0,
        "confidence": 74.0,
        "is_fake": true,
        "backend": "vps",
        "model_id": "human-ai-binary/small-perplexity",
        "latency_ms": 245.3,
        "metadata": {...}
    }
    """
    start_time = time.time()
    
    try:
        # Validate dataset + model combination
        if not validate_combination(request.dataset, request.model_id):
            available_models = list_all_mappings(request.dataset)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Model combination not found",
                    "dataset": request.dataset,
                    "model_id": request.model_id,
                    "available_for_dataset": available_models,
                    "available_datasets": get_available_datasets()
                }
            )
        
        # Get model mapping info
        model_info = get_model_info(request.dataset, request.model_id)
        backend_model_id = model_info["backend_model_file"]
        backend_type = model_info["backend_type"]
        
        logger.info(
            f"Prediction: dataset={request.dataset}, model={request.model_id} "
            f"-> backend={backend_model_id} ({backend_type})"
        )
        
        # Route to appropriate backend
        result = await router.predict(
            text=request.text,
            model_id=backend_model_id,
            prefer_backend=request.prefer_backend
        )
        
        # Check for errors
        if result.error:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Inference failed",
                    "message": result.error,
                    "backend": result.backend.value
                }
            )
        
        latency = (time.time() - start_time) * 1000
        
        # Convert probability to percentage (0-100)
        probability_percent = result.probability * 100
        confidence_percent = result.confidence * 100
        
        return PredictResponse(
            prediction=result.prediction,
            probability=probability_percent,
            confidence=confidence_percent,
            is_fake=result.prediction == 1,
            backend=result.backend.value,
            model_id=f"{request.dataset}/{request.model_id}",
            latency_ms=latency,
            metadata={
                **result.metadata,
                "frontend_model_id": request.model_id,
                "dataset": request.dataset,
                "backend_model_file": backend_model_id
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )


# ============================================================================
# ENDPOINT 4: POST /batch-predict (Auth Required)
# ============================================================================

@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["inference"])
async def batch_predict(
    request: BatchPredictRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Batch prediction for multiple texts.
    
    Required: X-API-Key header
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", "Text 3"],
        "model_id": "small-perplexity",
        "dataset": "human-ai-binary"
    }
    
    Response:
    {
        "predictions": [
            { prediction response 1 },
            { prediction response 2 },
            { prediction response 3 }
        ],
        "total_time_ms": 850.5,
        "batch_size": 3
    }
    """
    start_time = time.time()
    
    try:
        # Validate combination
        if not validate_combination(request.dataset, request.model_id):
            available_models = list_all_mappings(request.dataset)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Model combination not found",
                    "available_for_dataset": available_models
                }
            )
        
        # Get model info
        model_info = get_model_info(request.dataset, request.model_id)
        backend_model_id = model_info["backend_model_file"]
        
        logger.info(f"Batch prediction: {len(request.texts)} texts with {request.model_id}")
        
        # Process each text
        predictions = []
        for idx, text in enumerate(request.texts):
            if len(text) < settings.MIN_TEXT_LENGTH or len(text) > settings.MAX_TEXT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Text {idx} has invalid length (min: {settings.MIN_TEXT_LENGTH}, max: {settings.MAX_TEXT_LENGTH})"
                )
            
            # Run inference
            result = await router.predict(
                text=text,
                model_id=backend_model_id,
                prefer_backend=request.prefer_backend
            )
            
            if result.error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference failed for text {idx}: {result.error}"
                )
            
            predictions.append(PredictResponse(
                prediction=result.prediction,
                probability=result.probability * 100,
                confidence=result.confidence * 100,
                is_fake=result.prediction == 1,
                backend=result.backend.value,
                model_id=f"{request.dataset}/{request.model_id}",
                latency_ms=result.latency_ms,
                metadata=result.metadata
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictResponse(
            predictions=predictions,
            total_time_ms=total_time,
            batch_size=len(predictions)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standard error response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "timestamp": time.time()
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepfake Detection API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api.app_v2:app",
        host=args.host,
        port=args.port,
        workers=args.workers or settings.WORKERS,
        log_level=args.log_level or settings.LOG_LEVEL,
    )