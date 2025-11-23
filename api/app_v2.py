"""
Production-ready FastAPI backend for deepfake text detection.
Supports: Local VPS inference, Modal serverless, and client-side inference.
"""
import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from .config import get_settings, InferenceBackend
from .inference import InferenceRouter, InferenceBackend as IBackend
from .model_mapping import resolve_model, get_model_info as get_mapping_info, list_all_mappings, get_available_datasets, validate_combination

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings and router
settings = get_settings()
router = InferenceRouter(settings)

# ============================================================================
# Security - API Key Authentication
# ============================================================================

# Get API key from environment
API_KEY = os.getenv("API_KEY", "your-super-secret-api-key-here")
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
# Lifespan event handlers
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Deepfake Detection API Starting")
    logger.info("=" * 50)
    logger.info(f"Settings:")
    logger.info(f"  Default backend: {settings.DEFAULT_INFERENCE_BACKEND}")
    logger.info(f"  Allow client-side: {settings.ALLOW_CLIENT_SIDE_INFERENCE}")
    logger.info(f"  Allow Modal: {settings.ALLOW_MODAL_INFERENCE}")
    logger.info(f"  Redis enabled: {settings.REDIS_ENABLED}")
    logger.info(f"  VPS GPU memory: {settings.VPS_GPU_MEMORY_GB}GB")
    logger.info(f"  VPS CPU memory: {settings.VPS_CPU_MEMORY_GB}GB")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Deepfake Detection API shutting down")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Deepfake Detection API",
    description="Multi-backend inference system for AI-generated text detection",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
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

class DetectionRequest(BaseModel):
    """Request for text detection"""
    text: str = Field(..., min_length=settings.MIN_TEXT_LENGTH, max_length=settings.MAX_TEXT_LENGTH)
    model_id: str = Field(..., description="Model identifier (e.g., 'small-perplexity')")
    dataset: str = Field(..., description="Dataset name (e.g., 'human-ai-binary')")
    prefer_backend: Optional[IBackend] = Field(None, description="Preferred inference backend")


class DetectionResponse(BaseModel):
    """Response with detection result"""
    prediction: int = Field(..., description="0=human, 1=AI-generated")
    probability: float = Field(..., description="Probability of being AI-generated [0-1]")
    confidence: float = Field(..., description="Confidence in prediction [0-1]")
    is_fake: bool = Field(..., description="True if AI-generated")
    backend: str = Field(..., description="Backend used for inference")
    model_id: str = Field(..., description="Model used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    backends: Dict[str, Any]
    available_models: List[str]
    vps_info: Dict[str, Any]


class ModelDownloadResponse(BaseModel):
    """Info for client-side model download"""
    model_id: str
    analysis_type: str
    model_size_mb: float
    download_url: str
    metadata: Dict[str, Any]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """Root endpoint"""
    return {
        "service": "Deepfake Text Detection API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "models": "/models",
            "backends": "/backends"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """Complete health check of all backends"""
    backends_status = await router.health_check()
    
    models_dir = Path("/app/saved_models")
    available_models = [
        f.stem for f in models_dir.glob("*.pkl")
        if not f.name.endswith(("_metadata.pkl", "_vectorizer.pkl"))
    ]
    
    return HealthResponse(
        status="healthy",
        backends=backends_status,
        available_models=available_models,
        vps_info={
            "cpu_memory_gb": settings.VPS_CPU_MEMORY_GB,
            "gpu_memory_gb": settings.VPS_GPU_MEMORY_GB,
            "max_inference_size": settings.MAX_VPS_INFERENCE_SIZE,
            "redis_enabled": settings.REDIS_ENABLED
        }
    )


@app.get("/models/list", tags=["models"])
async def list_dataset_models():
    """List all available dataset + model combinations.
    
    Returns a mapping of datasets to available models for each dataset.
    This is what the frontend should use to populate dropdowns.
    
    Example response:
        {
            "human-ai-binary": ["small-perplexity", "medium-perplexity", "qwen-0.5b", "qwen-8b"],
            "human-ai-anomaly": ["small-perplexity", "medium-perplexity", "qwen-0.5b"],
            "arxiv": ["small-perplexity", "medium-perplexity"],
            "fakenews": ["small-perplexity", "large-perplexity"]
        }
    """
    result = {}
    for dataset in get_available_datasets():
        result[dataset] = list_all_mappings(dataset)
    return result


@app.get("/models/info", tags=["models"])
async def get_available_models_info():
    """Get detailed info about all dataset+model combinations.
    
    Useful for frontend to show which models are available on which backend.
    """
    from .model_mapping import DATASET_MODEL_MAPPING
    
    result = {}
    for (dataset, model_id), mapping in DATASET_MODEL_MAPPING.items():
        if dataset not in result:
            result[dataset] = {}
        result[dataset][model_id] = {
            "backend_model_file": mapping.backend_model_file,
            "backend_type": mapping.backend_type.value,
            "size_mb": mapping.size_mb,
            "description": mapping.description,
            "metadata": mapping.metadata or {}
        }
    return result


@app.get("/models", tags=["models"])
async def list_models():
    """List all available models with their configurations
    
    Note: Use /models/list for dataset+model combinations or /models/info for details
    """
    return {
        "configured_models": {
            model_id: {
                "size_category": config.size_category,
                "preferred_backend": config.preferred_backend.value,
                "fallback_backends": [b.value for b in config.fallback_backends]
            }
            for model_id, config in settings.AVAILABLE_MODELS.items()
        },
        "backends": {
            "local": settings.DEFAULT_INFERENCE_BACKEND == IBackend.LOCAL,
            "modal": settings.ALLOW_MODAL_INFERENCE,
            "client_side": settings.ALLOW_CLIENT_SIDE_INFERENCE
        },
        "note": "Use /models/list for dataset+model combinations"
    }


@app.get("/backends", tags=["models"])
async def get_backends_info():
    """Get detailed backend information"""
    return {
        "backends": {
            "local": {
                "enabled": True,
                "description": "Run inference on VPS",
                "advantages": ["Fast", "No external dependencies"],
                "disadvantages": ["Limited by VPS resources"],
                "best_for": ["small", "medium models"],
                "cost": "None"
            },
            "modal": {
                "enabled": settings.ALLOW_MODAL_INFERENCE,
                "description": "Run inference on Modal serverless",
                "advantages": ["Unlimited resources", "Pay per use"],
                "disadvantages": ["Network latency", "Requires API key"],
                "best_for": ["large models"],
                "cost": "Pay per inference"
            },
            "client_side": {
                "enabled": settings.ALLOW_CLIENT_SIDE_INFERENCE,
                "description": "Run inference in browser/client",
                "advantages": ["No server load", "Privacy", "Fast"],
                "disadvantages": ["Requires client resources"],
                "best_for": ["tiny models"],
                "cost": "None"
            }
        }
    }


@app.post("/predict", response_model=DetectionResponse, tags=["inference"])
async def predict(request: DetectionRequest, api_key: str = Depends(verify_api_key)):
    """
    Detect if text is AI-generated or human-written.
    
    Requires X-API-Key header for authentication.
    
    Frontend passes (dataset + model_id), backend resolves to actual model file
    and automatically routes to the best backend based on model configuration.
    
    Example:
        POST /predict
        Headers: X-API-Key: your-api-key
        {
            "text": "This is some text to analyze",
            "model_id": "small-perplexity",
            "dataset": "human-ai-binary"
        }
    """
    try:
        # Resolve dataset + model to backend model file
        logger.info(f"Prediction request: dataset={request.dataset}, model={request.model_id}")
        
        if not validate_combination(request.dataset, request.model_id):
            available_models = list_all_mappings(request.dataset)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Model combination not found",
                    "dataset": request.dataset,
                    "model_id": request.model_id,
                    "available_models_for_dataset": available_models,
                    "available_datasets": get_available_datasets()
                }
            )
        
        # Get mapping info
        model_info = get_mapping_info(request.dataset, request.model_id)
        backend_model_id = model_info["backend_model_file"]
        backend_type = model_info["backend_type"]
        
        logger.info(f"Resolved to backend model: {backend_model_id} (backend: {backend_type})")
        
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
        
        return DetectionResponse(
            prediction=result.prediction,
            probability=result.probability,
            confidence=result.confidence,
            is_fake=result.prediction == 1,
            backend=result.backend.value,
            model_id=f"{request.dataset}/{request.model_id}",  # Return original frontend ID
            latency_ms=result.latency_ms,
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


@app.post("/batch-predict", tags=["inference"])
async def batch_predict(
    texts: List[str],
    model_id: str,
    prefer_backend: Optional[IBackend] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Batch prediction (multiple texts at once).
    
    Requires X-API-Key header for authentication.
    
    Useful for processing multiple texts efficiently.
    """
    if len(texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large. Max: {settings.MAX_BATCH_SIZE}"
        )
    
    results = []
    for text in texts:
        result = await router.predict(text=text, model_id=model_id, prefer_backend=prefer_backend)
        results.append(result.to_dict())
    
    return {"results": results, "batch_size": len(results)}


@app.get("/models/{model_id}/info", tags=["models"])
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    if model_id not in settings.AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    config = settings.AVAILABLE_MODELS[model_id]
    
    return {
        "model_id": model_id,
        "size_category": config.size_category,
        "preferred_backend": config.preferred_backend.value,
        "fallback_backends": [b.value for b in config.fallback_backends],
        "description": f"{config.size_category} model using {config.model_id}",
        "max_text_length": settings.MAX_TEXT_LENGTH,
        "supports_batch": True,
        "max_batch_size": settings.MAX_BATCH_SIZE
    }


@app.get("/models/{model_id}/download", tags=["models"])
async def download_model(model_id: str):
    """
    Download model for client-side inference.
    
    Returns the model file for client-side execution.
    Useful for tiny/small models to run in browser.
    """
    models_dir = Path("/app/saved_models")
    model_path = models_dir / f"{model_id}.pkl"
    
    if not model_path.exists():
        model_path = models_dir / f"detector_{model_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_id}"
        )
    
    # Check file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    if size_mb > 100:  # 100MB limit for downloads
        raise HTTPException(
            status_code=400,
            detail=f"Model too large ({size_mb:.1f}MB). Use server inference instead."
        )
    
    return FileResponse(
        path=model_path,
        filename=model_path.name,
        media_type="application/octet-stream"
    )


@app.get("/models/{model_id}/download-info", response_model=ModelDownloadResponse, tags=["models"])
async def get_download_info(model_id: str):
    """Get info about downloading a model for client-side use"""
    models_dir = Path("/app/saved_models")
    model_path = models_dir / f"{model_id}.pkl"
    
    if not model_path.exists():
        model_path = models_dir / f"detector_{model_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Load metadata
    metadata = {}
    metadata_path = model_path.with_name(model_path.stem + "_metadata.pkl")
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    return ModelDownloadResponse(
        model_id=model_id,
        analysis_type=metadata.get("analysis_type", "unknown"),
        model_size_mb=size_mb,
        download_url=f"/models/{model_id}/download",
        metadata=metadata
    )


@app.get("/stats", tags=["info"])
async def get_stats():
    """Get server statistics"""
    models_dir = Path("/app/saved_models")
    model_files = list(models_dir.glob("*.pkl"))
    
    return {
        "total_models": len([f for f in model_files if not f.name.endswith(("_metadata.pkl", "_vectorizer.pkl"))]),
        "storage_used_mb": sum(f.stat().st_size for f in model_files) / (1024 * 1024),
        "available_backends": [b.value for b in [IBackend.LOCAL, IBackend.MODAL, IBackend.CLIENT_SIDE]],
        "timestamp": time.time()
    }





# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepfake Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: from settings)")
    parser.add_argument("--log-level", default=None, help="Log level (default: from settings)")
    
    args = parser.parse_args()
    
    port = args.port
    workers = args.workers or settings.WORKERS
    log_level = args.log_level or settings.LOG_LEVEL
    
    uvicorn.run(
        "api.app_v2:app",
        host=args.host,
        port=port,
        workers=workers,
        log_level=log_level,
    )
