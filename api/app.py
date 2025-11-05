"""
FastAPI backend for deepfake text detection.
Designed for deployment on Coolify/VPS with GPU support.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
from pathlib import Path
import pickle
import logging

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors import EmbeddingExtractor
from models.classifiers import BinaryDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Text Detection API",
    description="ML-powered API for detecting AI-generated text",
    version="1.0.0"
)

# CORS configuration - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for models
MODELS_CACHE = {}
EXTRACTORS_CACHE = {}


# Request/Response models
class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B", description="HuggingFace model ID")
    layer: int = Field(default=22, ge=-40, le=40, description="Layer to extract embeddings from")
    pooling: str = Field(default="mean", description="Pooling strategy")
    classifier_type: str = Field(default="svm", description="Classifier type")
    dataset_name: str = Field(default="mercor_ai", description="Dataset the model was trained on")


class DetectionResponse(BaseModel):
    prediction: int = Field(..., description="0=real, 1=fake")
    probability: float = Field(..., description="Probability of being fake [0-1]")
    confidence: float = Field(..., description="Confidence score [0-1]")
    is_fake: bool = Field(..., description="True if text is predicted as fake")
    model_info: dict = Field(..., description="Model configuration used")


class HealthResponse(BaseModel):
    status: str
    service: str
    available_models: List[str]
    gpu_available: bool


class PairDetectionRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=10000)
    text2: str = Field(..., min_length=1, max_length=10000)
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B")
    layer: int = Field(default=22, ge=-40, le=40)
    pooling: str = Field(default="mean")
    classifier_type: str = Field(default="svm")
    dataset_name: str = Field(default="mercor_ai")


class PairDetectionResponse(BaseModel):
    real_text_id: int = Field(..., description="1 if text1 is real, 2 if text2 is real")
    text1_prediction: int
    text2_prediction: int
    text1_probability: float
    text2_probability: float
    confidence: float


# Helper functions
def get_detector_path(dataset_name: str, model_slug: str, layer: int, classifier_type: str) -> Optional[Path]:
    """Find the trained detector file."""
    models_dir = Path(__file__).parent.parent / "saved_models"
    
    # Try different naming patterns
    candidates = [
        models_dir / f"detector_{dataset_name}_{model_slug}_layer{layer}_{classifier_type}.pkl",
        models_dir / f"detector_{model_slug}_layer{layer}_{classifier_type}.pkl",
        models_dir / f"detector_{dataset_name}_layer{layer}_{classifier_type}.pkl",
        models_dir / f"detector_{dataset_name}_{model_slug}.pkl",
    ]
    
    for path in candidates:
        if path.exists():
            logger.info(f"Found detector at: {path}")
            return path
    
    logger.warning(f"No detector found. Tried: {[str(p) for p in candidates]}")
    return None


def load_detector(dataset_name: str, model_slug: str, layer: int, classifier_type: str):
    """Load detector with caching."""
    cache_key = f"{dataset_name}_{model_slug}_layer{layer}_{classifier_type}"
    
    if cache_key in MODELS_CACHE:
        return MODELS_CACHE[cache_key]
    
    detector_path = get_detector_path(dataset_name, model_slug, layer, classifier_type)
    
    if detector_path is None:
        raise FileNotFoundError(f"No trained model found for {cache_key}")
    
    try:
        with open(detector_path, "rb") as f:
            detector = pickle.load(f)
        MODELS_CACHE[cache_key] = detector
        logger.info(f"Loaded and cached detector: {cache_key}")
        return detector
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        raise


def get_extractor(model_name: str, device: str = "cuda"):
    """Get or create embedding extractor with caching."""
    if model_name in EXTRACTORS_CACHE:
        return EXTRACTORS_CACHE[model_name]
    
    try:
        extractor = EmbeddingExtractor(model_name, device=device)
        EXTRACTORS_CACHE[model_name] = extractor
        logger.info(f"Loaded and cached extractor: {model_name}")
        return extractor
    except Exception as e:
        logger.error(f"Failed to load extractor {model_name}: {e}")
        raise


def list_available_models() -> List[str]:
    """List all trained model files."""
    models_dir = Path(__file__).parent.parent / "saved_models"
    
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("detector_*.pkl"))
    return [f.name for f in model_files]


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Deepfake Text Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="ok",
        service="deepfake-text-detector",
        available_models=list_available_models(),
        gpu_available=torch.cuda.is_available()
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest):
    """
    Detect if a text is AI-generated or human-written.
    
    Returns prediction, probability, and confidence scores.
    """
    try:
        # Prepare model slug
        model_slug = request.model_name.replace("/", "_")
        
        # Load detector
        detector = load_detector(
            request.dataset_name,
            model_slug,
            request.layer,
            request.classifier_type
        )
        
        # Get extractor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = get_extractor(request.model_name, device)
        
        # Extract features
        logger.info(f"Extracting features from text (length: {len(request.text)})")
        
        # Get layer embeddings
        all_layer_embeds = extractor.get_all_layer_embeddings(
            [request.text.strip() or " "],
            pooling=request.pooling,
            batch_size=1
        )
        
        features = all_layer_embeds[request.layer]
        
        # Make prediction
        logger.info("Making prediction...")
        results = detector.predict(
            features,
            return_probabilities=True,
            return_distances=False
        )
        
        # Parse results
        if isinstance(results, (list, tuple)) and len(results) >= 2:
            pred = int(results[0][0])
            prob = float(results[1][0])
        else:
            pred = int(results[0])
            prob = float(results[0])
        
        confidence = abs(prob - 0.5) * 2  # 0 = uncertain, 1 = very confident
        
        return DetectionResponse(
            prediction=pred,
            probability=prob,
            confidence=confidence,
            is_fake=bool(pred == 1),
            model_info={
                "model_name": request.model_name,
                "layer": request.layer,
                "pooling": request.pooling,
                "classifier": request.classifier_type,
                "dataset": request.dataset_name,
                "device": device
            }
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Model not found",
                "message": str(e),
                "available_models": list_available_models()
            }
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Detection failed",
                "message": str(e)
            }
        )


@app.get("/models", response_model=dict)
async def get_models():
    """List all available trained models."""
    return {
        "available_models": list_available_models(),
        "cached_models": list(MODELS_CACHE.keys()),
        "cached_extractors": list(EXTRACTORS_CACHE.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
