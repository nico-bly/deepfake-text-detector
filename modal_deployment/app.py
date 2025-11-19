"""
Modal.com deployment for deepfake text detection.
This runs your ML models serverlessly with auto-scaling.

Deployment:
    pip install modal
    modal token new
    modal deploy modal_deployment/app.py

Usage:
    Get webhook URL from Modal dashboard, add to Render gateway env vars.
"""
import modal
from pathlib import Path
import sys

# Create Modal app
app = modal.App("deepfake-text-detector")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    )
    # Copy your model code into the container
    .copy_local_dir("models", "/root/models")
    .copy_local_dir("utils", "/root/utils")
)

# Persistent storage for trained models
models_volume = modal.Volume.from_name(
    "deepfake-models",
    create_if_missing=True
)

@app.function(
    image=image,
    gpu=None,  # Set to "T4" for GPU, None for CPU
    memory=4096,  # 4GB RAM
    timeout=300,  # 5 minutes max
    volumes={"/models": models_volume},
    keep_warm=1,  # Keep 1 instance warm to avoid cold starts (optional, costs ~$5/month)
)
def detect_deepfake_text(
    text: str,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    layer: int = 22,
    pooling: str = "mean",
    classifier_type: str = "svm",
    dataset_name: str = "mercor_ai"
):
    """
    Analyze text for deepfake detection.
    
    Args:
        text: Text to analyze
        model_name: HuggingFace model ID
        layer: Layer to extract embeddings from
        pooling: Pooling strategy (mean, max, last, attn_mean)
        classifier_type: Classifier type (svm, lr, neural)
        dataset_name: Dataset the model was trained on
    
    Returns:
        {
            "prediction": 0 or 1 (0=real, 1=fake),
            "probability": float in [0,1],
            "confidence": float in [0,1],
            "is_fake": bool
        }
    """
    import sys
    sys.path.insert(0, "/root")
    
    from models.extractors import EmbeddingExtractor
    from models.classifiers import BinaryDetector
    import pickle
    import numpy as np
    
    # Validate input
    if not text or not isinstance(text, str):
        return {"error": "Invalid text input"}
    
    if len(text) > 10000:
        return {"error": "Text too long (max 10000 chars)"}
    
    # Load trained detector
    model_slug = model_name.replace("/", "_")
    detector_candidates = [
        f"/models/detector_{dataset_name}_{model_slug}_layer{layer}_{classifier_type}.pkl",
        f"/models/detector_{model_slug}_layer{layer}_{classifier_type}.pkl",
        f"/models/detector_{dataset_name}_layer{layer}_{classifier_type}.pkl",
    ]
    
    detector = None
    for path in detector_candidates:
        try:
            with open(path, "rb") as f:
                detector = pickle.load(f)
            print(f"✓ Loaded detector from {path}")
            break
        except FileNotFoundError:
            continue
    
    if detector is None:
        return {
            "error": "No trained model found",
            "tried_paths": [p for p in detector_candidates]
        }
    
    # Extract features
    extractor = EmbeddingExtractor(model_name, device="cpu")  # Change to "cuda" if gpu=True
    
    try:
        features = extractor.get_pooled_layer_embeddings(
            [text.strip() or " "],  # Handle empty strings
            layer_idx=layer,
            pooling=pooling,
            batch_size=1,
            max_length=512,
            show_progress=False
        )
    except Exception as e:
        return {"error": f"Feature extraction failed: {str(e)}"}
    
    # Make prediction
    try:
        results = detector.predict(
            features,
            return_probabilities=True,
            return_distances=False
        )
        
        if isinstance(results, (list, tuple)) and len(results) >= 2:
            pred, prob = results[0][0], float(results[1][0])
        else:
            pred, prob = int(results[0]), float(results[0])
        
        return {
            "prediction": int(pred),
            "probability": float(prob),
            "confidence": abs(prob - 0.5) * 2,  # 0 = uncertain, 1 = very confident
            "is_fake": bool(pred == 1),
            "model_info": {
                "model_name": model_name,
                "layer": layer,
                "pooling": pooling,
                "classifier": classifier_type,
                "dataset": dataset_name
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.function(
    image=image,
    memory=1024,  # Lightweight, just routing
)
@modal.web_endpoint(method="POST")
def webhook(data: dict):
    """
    HTTP endpoint for your gateway to call.
    
    POST to: https://[your-modal-username]--deepfake-text-detector-webhook.modal.run
    
    Request body:
    {
        "text": "Text to analyze",
        "model_name": "Qwen/Qwen2.5-0.5B",  // optional
        "layer": 22,  // optional
        "classifier_type": "svm"  // optional
    }
    """
    try:
        result = detect_deepfake_text.remote(
            text=data.get("text", ""),
            model_name=data.get("model_name", "Qwen/Qwen2.5-0.5B"),
            layer=data.get("layer", 22),
            pooling=data.get("pooling", "mean"),
            classifier_type=data.get("classifier_type", "svm"),
            dataset_name=data.get("dataset_name", "mercor_ai")
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "message": "Internal server error"
        }


@app.function(
    image=image,
    memory=512,
)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "deepfake-text-detector",
        "available_models": list_available_models.remote()
    }


@app.function(
    image=image,
    volumes={"/models": models_volume},
    memory=512
)
def list_available_models():
    """List all trained models available."""
    import os
    from pathlib import Path
    
    models_dir = Path("/models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("detector_*.pkl"))
    return [f.name for f in model_files]


# Helper function to upload models (run locally)
@app.local_entrypoint()
def upload_models():
    """
    Upload your trained models to Modal.
    
    Run locally: modal run modal_deployment/app.py::upload_models
    """
    import os
    from pathlib import Path
    
    local_models_dir = Path("saved_models")
    
    if not local_models_dir.exists():
        print(f"❌ {local_models_dir} not found")
        return
    
    model_files = list(local_models_dir.glob("*.pkl"))
    
    if not model_files:
        print("❌ No .pkl files found in saved_models/")
        return
    
    print(f"📦 Found {len(model_files)} model files")
    
    # Upload to Modal volume
    models_volume.reload()
    
    for model_file in model_files:
        print(f"⬆️  Uploading {model_file.name}...")
        with open(model_file, "rb") as f:
            models_volume.write_file(model_file.name, f.read())
    
    print("✅ All models uploaded successfully!")
    print("\n🔗 Next steps:")
    print("1. Deploy the app: modal deploy modal_deployment/app.py")
    print("2. Get your webhook URL from Modal dashboard")
    print("3. Add to Render env: MODAL_WEBHOOK_URL=https://...")
