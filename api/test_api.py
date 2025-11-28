"""Minimal test API with proper feature extraction"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.model_mapping_simple import load_model, list_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Detector API")

class PredictRequest(BaseModel):
    text: str
    model_id: str = "human_ai_microsoft_deberta"

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    confidence: float
    is_fake: bool
    model_id: str
    metadata: dict = {}

def extract_features(text: str, metadata: dict) -> np.ndarray:
    """Extract features from text using metadata"""
    # Get analysis type from metadata
    analysis_type = metadata.get("analysis_type", "embedding")
    
    if analysis_type == "embedding":
        # Use embedding extractor
        from models.extractors import EmbeddingExtractor
        model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
        layer = metadata.get("layer", 23)
        pooling = metadata.get("pooling", "mean")
        
        extractor = EmbeddingExtractor(model_name, device="cpu")  # Use CPU for testing
        features = extractor.get_pooled_layer_embeddings(
            [text],
            layer_idx=layer,
            pooling=pooling,
            batch_size=1,
            show_progress=False
        )
        return features
    
    elif analysis_type == "perplexity":
        # Use perplexity calculator
        from models.text_features import PerplexityCalculator
        model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
        
        calculator = PerplexityCalculator(model_name, device="cpu")
        perp = calculator.calculate_batch_perplexity([text])
        features = np.array(perp).reshape(-1, 1)
        return features
    
    elif analysis_type == "phd":
        # Use intrinsic dimension calculator
        from models.text_features import TextIntrinsicDimensionCalculator
        model_name = metadata.get("model_name", "Qwen/Qwen2.5-0.5B")
        layer = metadata.get("layer", 23)
        
        calculator = TextIntrinsicDimensionCalculator(model_name, device="cpu", layer_idx=layer)
        phd = calculator.calculate_batch([text])
        features = np.array(phd).reshape(-1, 1)
        return features
    
    elif analysis_type == "tfidf":
        # Use TF-IDF vectorizer from metadata
        model_path = metadata.get("model_path")
        if not model_path:
            raise ValueError("TF-IDF requires model_path in metadata")
        
        vec_path = Path(model_path).with_name(Path(model_path).stem + "_vectorizer.pkl")
        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vec_path}")
        
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        
        features = vectorizer.transform([text])
        return features.toarray() if hasattr(features, 'toarray') else features
    
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

@app.get("/models")
async def get_models():
    return {"models": list_models()}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Load model and metadata
        detector, metadata = load_model(request.model_id)
        logger.info(f"Model loaded: {request.model_id}")
        logger.info(f"Metadata: {metadata}")
        
        # Extract features from text
        logger.info(f"Extracting features from text (length: {len(request.text)})")
        features = extract_features(request.text, metadata)
        logger.info(f"Features shape: {features.shape}")
        
        # Make prediction
        results = detector.predict(features, return_probabilities=True)
        
        if isinstance(results, tuple) and len(results) >= 2:
            pred = int(results[0][0])
            prob = float(results[1][0])
        else:
            pred = int(results[0])
            prob = float(results[0])
        
        confidence = abs(prob - 0.5) * 2
        
        return PredictResponse(
            prediction=pred,
            probability=prob,
            confidence=confidence,
            is_fake=bool(pred == 1),
            model_id=request.model_id,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
