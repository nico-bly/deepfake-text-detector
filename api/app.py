"""
FastAPI backend for deepfake text detection.
Designed for deployment on Coolify/VPS with GPU support.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
import pickle
import logging
import numpy as np

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors import EmbeddingExtractor, TFIDFExtractor, pool_embeds_from_layer
from models.text_features import PerplexityCalculator, TextIntrinsicDimensionCalculator
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


class SimpleDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_id: str = Field(..., description="Simple model identifier matching a saved model filename stem (e.g., 'embedding_A__dataset1')")


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


# ------------------------ Simplified model loading and feature extraction ------------------------
def _sanitize_texts(texts: List[str]) -> List[str]:
    return [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]


def _replace_nan_with_column_means(features):
    try:
        from scipy.sparse import issparse  # type: ignore
        if issparse(features):
            return features
    except Exception:
        pass

    arr = np.asarray(features, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    finite_mask = np.isfinite(arr)
    arr_masked = arr.copy()
    arr_masked[~finite_mask] = np.nan
    col_means = np.nanmean(arr_masked, axis=0)
    if np.isscalar(col_means):
        col_means = np.array([col_means])
    col_means = np.nan_to_num(col_means, nan=0.0)
    bad_rows, bad_cols = np.where(~finite_mask)
    if bad_rows.size > 0:
        arr[bad_rows, bad_cols] = col_means[bad_cols]
    return arr


def _load_detector_and_metadata_by_id(model_id: str) -> tuple[BinaryDetector, Dict[str, Any], Path]:
    """Load a detector and its metadata using a simple model_id that matches the filename stem.

    Expects files:
      saved_models/{model_id}.pkl
      saved_models/{model_id}_metadata.pkl (optional)
      saved_models/{model_id}_vectorizer.pkl (for TF-IDF models)
    """
    models_dir = Path(__file__).parent.parent / "saved_models"
    model_path = models_dir / f"{model_id}.pkl"
    if not model_path.exists():
        # Also support legacy prefix 'detector_'
        legacy_path = models_dir / f"detector_{model_id}.pkl"
        if legacy_path.exists():
            model_path = legacy_path
        else:
            raise FileNotFoundError(f"No saved model found for id '{model_id}' in {models_dir}")

    with open(model_path, "rb") as f:
        detector = pickle.load(f)

    metadata_path = model_path.with_name(model_path.stem + "_metadata.pkl")
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

    # Attach explicit paths for convenience
    metadata["model_path"] = str(model_path)
    metadata["model_id"] = model_id
    return detector, metadata, model_path


def _instantiate_extractor_from_metadata(metadata: Dict[str, Any], device: str):
    analysis_type = metadata.get("analysis_type", "embedding")
    model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
    layer = metadata.get("layer", 22)

    if analysis_type == "embedding":
        return EmbeddingExtractor(model_name, device=device)
    if analysis_type == "perplexity":
        return PerplexityCalculator(model_name, device=device)
    if analysis_type == "phd":
        return TextIntrinsicDimensionCalculator(model_name, device=device, layer_idx=layer)
    if analysis_type == "tfidf":
        # Load fitted vectorizer alongside the detector
        vec_path = Path(metadata.get("model_path", "")).with_name(Path(metadata.get("model_path", "")).stem + "_vectorizer.pkl")
        if not vec_path.exists():
            raise FileNotFoundError(f"Expected TF-IDF vectorizer file not found: {vec_path}")
        with open(vec_path, "rb") as f:
            fitted_vec = pickle.load(f)
        extractor = TFIDFExtractor(
            max_features=getattr(fitted_vec, 'max_features', None),
            ngram_range=getattr(fitted_vec, 'ngram_range', (1, 2)),
            lowercase=getattr(fitted_vec, 'lowercase', True),
            min_df=getattr(fitted_vec, 'min_df', 1),
            max_df=getattr(fitted_vec, 'max_df', 1.0),
            use_idf=getattr(fitted_vec, 'use_idf', True),
            norm=getattr(fitted_vec, 'norm', 'l2'),
            sublinear_tf=getattr(fitted_vec, 'sublinear_tf', False),
            stop_words=getattr(fitted_vec, 'stop_words', None),
        )
        extractor.vectorizer = fitted_vec
        extractor._is_fitted = True
        return extractor
    raise ValueError(f"Unsupported analysis_type {analysis_type}")


def _features_from_metadata(texts: List[str], extractor, metadata: Dict[str, Any], device: str) -> np.ndarray:
    processed = _sanitize_texts(texts)
    analysis_type = metadata.get("analysis_type", "embedding")
    layer = metadata.get("layer", 22)
    pooling = metadata.get("pooling", "mean")
    normalize = bool(metadata.get("normalize", False))

    if analysis_type == "embedding":
        # Efficient single-layer pooled extraction
        try:
            feats = extractor.get_pooled_layer_embeddings(
                processed, layer_idx=layer, pooling=pooling, batch_size=8, max_length=int(metadata.get("max_length", 512)), show_progress=False, normalize=normalize
            )
        except Exception:
            # Fallback to all-layer then pool
            embeds_all = extractor.get_all_layer_embeddings(processed, batch_size=8, max_length=int(metadata.get("max_length", 512)), show_progress=False)
            # Use chosen layer, or last available
            available_layers = sorted(list(embeds_all[0].keys())) if embeds_all else []
            chosen = layer if layer in available_layers else (available_layers[-1] if available_layers else 0)
            layer_embeds = [embeds[chosen] for embeds in embeds_all]
            feats = pool_embeds_from_layer(layer_embeds, pooling=pooling)
        return _replace_nan_with_column_means(feats)
    if analysis_type == "perplexity":
        arr = np.array(extractor.calculate_batch_perplexity(processed, max_length=int(metadata.get("max_length", 512)))).reshape(-1, 1)
        return _replace_nan_with_column_means(arr)
    if analysis_type == "phd":
        arr = np.array(extractor.calculate_batch(processed, max_length=int(metadata.get("max_length", 512)))).reshape(-1, 1)
        return _replace_nan_with_column_means(arr)
    if analysis_type == "tfidf":
        return extractor.transform(processed, dense=False)
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


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


@app.get("/simple-models", response_model=Dict[str, List[str]])
async def list_simple_models():
    """List simplified model identifiers based on files in saved_models.

    Returns just filename stems without extension; you can rename files to match frontend choices.
    """
    models_dir = Path(__file__).parent.parent / "saved_models"
    if not models_dir.exists():
        return {"models": []}
    stems = []
    for pkl in models_dir.glob("*.pkl"):
        name = pkl.name
        if name.endswith("_metadata.pkl") or name.endswith("_vectorizer.pkl"):
            continue
        stems.append(pkl.stem)
    # Deduplicate stems where both bare and 'detector_*' exist
    unique = sorted(set(stems))
    return {"models": unique}


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
        
        # Extract features (use robust pooled single-layer path)
        logger.info(f"Extracting features from text (length: {len(request.text)})")
        try:
            features = extractor.get_pooled_layer_embeddings(
                [request.text.strip() or " "],
                layer_idx=request.layer,
                pooling=request.pooling,
                batch_size=1,
                max_length=512,
                show_progress=False,
            )
        except Exception:
            # Fallback to all-layer + pool
            all_layer_embeds = extractor.get_all_layer_embeddings([request.text.strip() or " "], batch_size=1)
            chosen_layer = request.layer
            available_layers = sorted(list(all_layer_embeds[0].keys())) if all_layer_embeds else []
            if chosen_layer not in available_layers and available_layers:
                chosen_layer = available_layers[-1]
            layer_embeds = [embeds[chosen_layer] for embeds in all_layer_embeds]
            features = pool_embeds_from_layer(layer_embeds, pooling=request.pooling)
        
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


@app.post("/predict", response_model=DetectionResponse)
async def predict_simple(request: SimpleDetectionRequest):
    """Simplified prediction endpoint.

    Frontend passes a single model_id that matches a saved model filename stem (without extension).
    Backend uses the saved metadata to instantiate the correct extractor and fixed classifier/layer.
    """
    import torch
    try:
        detector, metadata, model_path = _load_detector_and_metadata_by_id(request.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = _instantiate_extractor_from_metadata(metadata, device=device)

        features = _features_from_metadata([request.text], extractor, metadata, device=device)
        results = detector.predict(features, return_probabilities=True, return_distances=False)

        if isinstance(results, (list, tuple)) and len(results) >= 2:
            pred = int(results[0][0])
            prob = float(results[1][0])
        else:
            # Fallback if only probabilities returned
            if isinstance(results, np.ndarray):
                prob = float(results.squeeze())
                pred = int(prob >= 0.5)
            else:
                pred = int(results)
                prob = float(pred)

        confidence = abs(prob - 0.5) * 2
        return DetectionResponse(
            prediction=pred,
            probability=prob,
            confidence=confidence,
            is_fake=bool(pred == 1),
            model_info={
                "model_id": request.model_id,
                "analysis_type": metadata.get("analysis_type"),
                "hf_model": metadata.get("model_name"),
                "layer": metadata.get("layer"),
                "pooling": metadata.get("pooling"),
                "classifier": metadata.get("classifier_type"),
                "dataset": metadata.get("dataset_used"),
                "device": device,
                "model_path": str(model_path)
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "Model not found", "message": str(e)})
    except Exception as e:
        logger.error(f"Simplified prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Prediction failed", "message": str(e)})


## Pair prediction endpoint removed (no longer supported)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
