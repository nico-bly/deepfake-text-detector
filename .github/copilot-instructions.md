# Copilot Instructions - Deepfake Text Detection Backend

## Project Overview

This is the **backend API** for a deepfake text detection platform deployed on a VPS via **Coolify**.

**Architecture:**
- **Frontend**: Managed by Coolify on the VPS (separate service)
- **This Repo**: Backend API for **text detection only**
- **Multiple Inference Backends**: 
  1. **Local VPS** - Run inference directly on the VPS (GPU/CPU)
  2. **Modal API** - Offload to Modal serverless platform (powerful GPUs)
  3. **Client-Side** - Send model to client for local processing (privacy-focused, WIP)

The backend uses **intelligent routing** to select the best inference backend based on model size, availability, and resource constraints.

---

## System Architecture

### Request Flow

```
Frontend (Web/Mobile)
       ↓
   HTTP POST /predict
   {text, model_id, [prefer_backend]}
       ↓
   FastAPI Server (api/app_v2.py)
   • Validates input
   • Rate limiting (optional)
   ↓
   InferenceRouter (inference.py)
   • Loads model config
   • Checks backend availability
   • Applies fallback logic
   ↓
   ┌─────────┬──────────┬──────────────┐
   ↓         ↓          ↓              ↓
 LOCAL     MODAL    CLIENT-SIDE    [Error]
 (VPS)    (Cloud)    (Browser)
   ↓         ↓          ↓
 Result Result Result
   ↓         ↓          ↓
   └─────────┴──────────┴──────────────┘
            ↓
    Standard Response
    {prediction, probability, confidence, 
     is_fake, backend, latency_ms, ...}
            ↓
        Frontend
```

### Backend Selection Logic

**Model Configuration** → **Backend Assignment**:
- **TINY** (< 10 MB): `CLIENT_SIDE` → fallback `LOCAL`
- **SMALL** (50-200 MB): `LOCAL` → fallback `MODAL`
- **MEDIUM** (500MB-1GB): `LOCAL` → fallback `MODAL`
- **LARGE** (> 1GB): `MODAL` → fallback `LOCAL`

Each model in `config.py` specifies:
```python
ModelConfig(
    model_id="model-name",
    size_category="small",  # tiny, small, medium, large
    preferred_backend=InferenceBackend.LOCAL,
    fallback_backends=[InferenceBackend.MODAL]
)
```

### System Components

**Key Files:**
- `api/app_v2.py` - FastAPI server with lifespan events (replaces deprecated `on_event`)
- `api/config.py` - Settings, model registry, backend configuration
- `api/inference.py` - `InferenceRouter` (routing logic) + backend engines (LOCAL, MODAL, CLIENT_SIDE)
- `models/extractors.py` - Embedding extraction (`EmbeddingExtractor`, layer-wise pooling)
- `models/text_features.py` - Feature calculators (`PerplexityCalculator`, `TextIntrinsicDimensionCalculator`)
- `models/classifiers.py` - Binary classifiers (SVM, LR, XGB, Neural)
- `utils/data_loader.py` - DataLoader utilities for training and inference
- `Dockerfile` & `Dockerfile.prod` - Development and production images
- `docker-compose.yml` & `docker-compose.prod.yml` - Local and production orchestration

---

## Core Workflows

### 1. Single Text Prediction

```python
# API receives request
POST /predict {
    "text": "Some text to analyze...",
    "model_id": "small-perplexity",
    "prefer_backend": null  # optional
}

# Inside app_v2.py:
1. Validate text length (MIN_TEXT_LENGTH to MAX_TEXT_LENGTH)
2. Call router.predict(text, model_id, prefer_backend)
3. Router:
   a. Load model config from settings
   b. Check preferred_backend availability
   c. If unavailable, try fallback_backends
   d. Select best available backend
   e. Run inference
4. Return standard DetectionResponse
```

### 2. Batch Prediction

```python
POST /batch-predict {
    "texts": ["text1", "text2", ...],
    "model_id": "small-perplexity",
    "prefer_backend": null
}

# Processing:
- Validate batch size (MAX_BATCH_SIZE)
- Process sequentially or batch on GPU (if LOCAL backend)
- Return list of DetectionResponse objects
```

### 3. Model Training & Saving

```python
# Load training data
train_loader, test_loader = create_unified_dataloaders(
    train_path="data/train",
    test_path="data/test",
    batch_size=8
)

# Extract features (embeddings or perplexity)
extractor = EmbeddingExtractor(model_id, device="cuda")
embeddings = extractor.get_all_layer_embeddings(texts)
pooled = pool_embeds_from_layer(embeddings, layer=22)

# Train classifier
detector = BinaryDetector(classifier_type="svm")
detector.fit(pooled, labels)

# Save for serving
detector.save("saved_models/model_name.pkl")
```

### 4. Local VPS Inference

```python
# InferenceRouter selects LOCAL backend
# LocalEngine loads saved model and runs inference

1. Load model from saved_models/{model_id}.pkl
2. Extract features using EmbeddingExtractor or PerplexityCalculator
3. Run BinaryDetector.predict()
4. Return result with backend="local", latency_ms, etc.
```

### 5. Modal Serverless Inference

```python
# InferenceRouter selects MODAL backend
# ModalEngine sends request to Modal API

1. Check MODAL_TOKEN_ID, MODAL_TOKEN_SECRET configured
2. Call Modal inference function
3. Handle rate limits, timeouts, retries
4. Return result with backend="modal", latency_ms, etc.
```

### 6. Client-Side Inference (WIP)

```python
# InferenceRouter selects CLIENT_SIDE backend
# ClientEngine prepares model for download

1. Find model file (e.g., saved_models/tiny-model.pkl)
2. Check file size < 100MB
3. Return download URL or FileResponse
4. Client downloads and runs locally in browser/app
```

---

## Configuration & Environment

### Settings (api/config.py)

```python
# Server
API_HOST = "0.0.0.0"
API_PORT = 8000
WORKERS = 4

# Backends
DEFAULT_INFERENCE_BACKEND = "local"  # "local", "modal", "client"
ALLOW_CLIENT_SIDE_INFERENCE = True
ALLOW_MODAL_INFERENCE = False

# VPS Hardware
VPS_GPU_MEMORY_GB = 0   # Set if GPU available
VPS_CPU_MEMORY_GB = 8
MAX_VPS_INFERENCE_SIZE = "medium"  # Max model size to run locally

# Redis (optional queuing)
REDIS_ENABLED = False
REDIS_URL = "redis://redis:6379/0"

# Model defaults
DEFAULT_MODEL_ID = "sentence-transformers_all-MiniLM-L6-v2"
SAVED_MODELS_DIR = "saved_models"

# Modal config (if using Modal backend)
MODAL_ENABLED = False
MODAL_TOKEN_ID = None
MODAL_TOKEN_SECRET = None
```

### .env File

```bash
# Server
WORKERS=4
LOG_LEVEL=info

# Backends
DEFAULT_INFERENCE_BACKEND=local
ALLOW_CLIENT_SIDE_INFERENCE=true
ALLOW_MODAL_INFERENCE=false

# VPS Hardware
VPS_GPU_MEMORY_GB=0
VPS_CPU_MEMORY_GB=8

# Modal (if using)
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

---

## Running & Deployment

### Local Development

```bash
# Install
pip install -r requirements.txt

# Run with default port 8000
python -m api.app_v2

# Run with custom port
python -m api.app_v2 --port 8001

# Access docs
http://localhost:8000/docs
```

### Docker (Development)

```bash
# Build
docker build -f Dockerfile -t deepfake-detector:dev .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/logs:/app/logs \
  deepfake-detector:dev
```

### Docker Compose (Development)

```bash
# Start
docker-compose up --build

# Stop
docker-compose down
```

### Docker Compose (Production on VPS)

```bash
# Start with Coolify or manual deployment
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f api

# Stop
docker-compose -f docker-compose.prod.yml down
```

### Coolify Deployment

1. Connect Git repository
2. Select `Dockerfile.prod`
3. Set environment variables from `.env`
4. Configure volumes:
   - `/app/saved_models` ← mount trained models
   - `/app/logs` ← application logs
5. Set resource limits (CPU, RAM, GPU if available)
6. Deploy! Auto-rebuilds on push

---

## API Endpoints

### Info & Health

```
GET /
  → {"service": "...", "version": "2.0.0", "endpoints": {...}}

GET /health
  → {"status": "healthy", "backends": {...}, "available_models": [...]}

GET /models
  → {"configured_models": {...}, "backends": {...}}

GET /backends
  → Detailed backend info (LOCAL, MODAL, CLIENT_SIDE)

GET /stats
  → Server statistics (models count, storage, backends)
```

### Inference

```
POST /predict
  Input:  {text, model_id, [prefer_backend]}
  Output: {prediction, probability, confidence, is_fake, backend, latency_ms, metadata}

POST /batch-predict
  Input:  {texts: [...], model_id, [prefer_backend]}
  Output: {results: [...], batch_size}
```

### Models

```
GET /models/{model_id}/info
  → Model configuration and capabilities

GET /models/{model_id}/download
  → Download model file (< 100MB only)

GET /models/{model_id}/download-info
  → Metadata about downloading the model
```

---

## Data Formats

### Standard Inference Response

```json
{
  "prediction": 0,           // 0=human, 1=AI-generated
  "probability": 0.72,       // P(AI) in [0, 1]
  "confidence": 0.44,        // Confidence in prediction [0, 1]
  "is_fake": false,          // Boolean (prediction == 1)
  "backend": "local",        // "local", "modal", "client"
  "model_id": "small-perplexity",
  "latency_ms": 245.32,      // Inference time
  "metadata": {
    "features_extracted": ["perplexity", "entropy"],
    "text_length": 500,
    "tokens": 85
  }
}
```

### Health Response

```json
{
  "status": "healthy",
  "backends": {
    "local": {
      "status": "healthy",
      "available_models": 3,
      "gpu_available": false,
      "gpu_memory_mb": 0
    },
    "modal": {
      "status": "unavailable",
      "reason": "Modal not configured"
    },
    "client_side": {
      "status": "healthy",
      "available_models": 2,
      "mode": "client-side"
    }
  },
  "available_models": ["model1", "model2", "model3"],
  "vps_info": {
    "cpu_memory_gb": 8,
    "gpu_memory_gb": 0,
    "max_inference_size": "medium",
    "redis_enabled": false
  }
}
```

---

## Key Conventions & Gotchas

### Prediction Convention
- **0 = Human-written (Real)**
- **1 = AI-generated (Fake)**
- Always follow this convention when training and loading models

### Text Sanitization
- Empty strings crash HuggingFace tokenizers
- Always sanitize: `text = text.strip() or " "` before encoding

### Memory Management
- Call `clear_gpu_memory()` after inference in batch jobs
- Use `batch_size=8` by default to avoid OOM
- Models are cached in memory after first load

### Fallback Behavior
- If preferred backend unavailable → try fallbacks in order
- If all backends fail → return HTTP 500 with detailed error
- Log all backend switches for debugging

### Docker & Coolify
- Use `Dockerfile.prod` for production (production-optimized)
- Use `Dockerfile` for development
- Mount `saved_models/` as read-only in production
- Mount `logs/` for application logs
- Health check runs on `/health` endpoint every 30s

### Lifespan Events
- Use **lifespan context manager** (not deprecated `@app.on_event`)
- Startup: Initialize logger, display settings
- Shutdown: Clean up resources

---

## Common Tasks

### Add a New Model

1. Train and save to `saved_models/{model_name}.pkl`
2. Add config to `api/config.py`:
   ```python
   self.AVAILABLE_MODELS["new-model"] = ModelConfig(
       model_id="new-model",
       size_category="small",
       preferred_backend=InferenceBackend.LOCAL,
       fallback_backends=[InferenceBackend.MODAL]
   )
   ```
3. Restart API
4. Test: `curl http://localhost:8000/predict -X POST -d '{"text":"...", "model_id":"new-model"}'`

### Switch Default Backend

Edit `.env` or environment:
```bash
DEFAULT_INFERENCE_BACKEND=modal  # Switch to Modal
ALLOW_MODAL_INFERENCE=true
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```
Then restart API.

### Monitor Inference Performance

```bash
# Check logs
docker-compose logs -f api

# Test latency
time curl http://localhost:8000/predict -X POST ...

# Get stats
curl http://localhost:8000/stats
```

### Enable GPU Inference

1. Ensure VPS has GPU with CUDA support
2. Update `.env`:
   ```bash
   VPS_GPU_MEMORY_GB=4  # or your GPU VRAM
   ```
3. Models will auto-detect and use GPU when available

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Address already in use" | Use `--port 8001` or kill existing process |
| DeprecationWarning on_event | Already fixed in app_v2.py (uses lifespan) |
| Model not found | Check `saved_models/` directory exists and has `.pkl` files |
| "All backends unavailable" | Check env config, Modal token, GPU memory |
| Out of Memory | Reduce batch size, use smaller model, use Modal backend |
| Slow inference on first request | Normal (model loads into GPU). Subsequent requests faster. |
| High latency on Modal | Network + cold start. Consider keeping connections warm. |

---

## File Structure

```
deepfake-text-detector/
├── api/
│   ├── app_v2.py           # Main FastAPI app (use this!)
│   ├── config.py           # Settings & model registry
│   ├── inference.py        # Router + backend engines
│   └── test_api.py         # API tests
├── models/
│   ├── extractors.py       # EmbeddingExtractor
│   ├── text_features.py    # PerplexityCalculator, etc.
│   └── classifiers.py      # BinaryDetector
├── utils/
│   ├── data_loader.py      # DataLoader utils
│   └── utils.py            # General utilities
├── saved_models/           # Trained models (.pkl files)
├── logs/                   # Application logs
├── Dockerfile              # Development image
├── Dockerfile.prod         # Production image
├── docker-compose.yml      # Development orchestration
└── docker-compose.prod.yml # Production orchestration
```

---

## Summary

- **Single purpose**: Text detection backend API
- **Three inference backends**: LOCAL (VPS), MODAL (cloud), CLIENT_SIDE (browser)
- **Smart routing**: Automatically selects best backend per model
- **Production-ready**: Docker, health checks, logging, error handling
- **Flexible**: Easy to add models, swap backends, scale with Coolify
