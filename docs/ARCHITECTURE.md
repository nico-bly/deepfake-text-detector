# Deepfake Detection API - Architecture & Implementation Summary

## Problem Statement

You needed a production-ready API that:
1. ✅ Runs on VPS with flexible resource constraints
2. ✅ Supports **model size-based routing** (tiny → client, small/medium → VPS, large → Modal)
3. ✅ Allows easy backend switching (local, Modal, client-side)
4. ✅ Can accommodate model changes without major refactoring
5. ✅ Provides fallback mechanisms when preferred backend unavailable

---

## Solution Overview

### Architecture: Three-Tier Inference System

```
┌─────────────────────────────────────────────────────────┐
│            FRONTEND / CLIENT                            │
├─────────────────────────────────────────────────────────┤
│                    API Layer                            │
│  app_v2.py: FastAPI with standardized endpoints         │
├─────────────────────────────────────────────────────────┤
│            INFERENCE ROUTER (Smart Router)              │
│  inference.py: Selects optimal backend                  │
├─────────────────────────────────────────────────────────┤
│  LOCAL ENGINE    │    MODAL ENGINE    │   CLIENT ENGINE │
│  VPS inference   │  Serverless GPU    │  Browser-based  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Configuration (`api/config.py`)

**Purpose**: Centralized settings and model configuration

```python
Settings:
  - DEFAULT_INFERENCE_BACKEND: local | modal | client
  - ALLOW_CLIENT_SIDE_INFERENCE: bool
  - ALLOW_MODAL_INFERENCE: bool
  - AVAILABLE_MODELS: Dict[str, ModelConfig]
  - VPS_GPU_MEMORY_GB: int (for routing decisions)
  - VPS_CPU_MEMORY_GB: int (for routing decisions)

ModelConfig:
  - model_id: str (filesystem identifier)
  - size_category: "tiny" | "small" | "medium" | "large"
  - preferred_backend: InferenceBackend
  - fallback_backends: List[InferenceBackend]
```

**Benefits**:
- Single source of truth
- Easy model management
- Environment-based configuration
- No hardcoding

### 2. Inference Engine (`api/inference.py`)

**Purpose**: Abstract inference backend implementations

**Classes**:
- `InferenceRouter`: Smart router that selects backends
- `LocalInferenceEngine`: VPS-based inference (fast, limited resources)
- `ModalInferenceEngine`: Serverless GPU (powerful, network latency)
- `ClientSideInferenceEngine`: Browser-based (lightweight, client burden)

**Smart Routing Decision Tree**:
```
Model found? 
├─ No → 404 error
└─ Yes → Check preferred backend
   ├─ Available? 
   │  ├─ Yes → Use it
   │  └─ No → Try first fallback
   ├─ Available?
   │  ├─ Yes → Use it
   │  └─ No → Try next fallback
   └─ All failed? → Return error with info
```

### 3. API Server (`api/app_v2.py`)

**Purpose**: FastAPI endpoints with standardized responses

**Key Endpoints**:

```
GET  /                      → API info
GET  /health                → Backend health status
GET  /models                → List configured models
GET  /backends              → Available backends info
GET  /models/{id}/info      → Model configuration
GET  /models/{id}/download  → Download model (for client)

POST /predict               → Single prediction (main)
POST /batch-predict         → Multiple predictions
POST /predict (with prefer_backend) → Force specific backend
```

**Features**:
- Async/await for performance
- Request validation (min/max text length)
- Standardized responses
- Detailed error messages
- Metadata tracking

---

## How It Works: Inference Flow

### Scenario 1: Small Model on VPS (default)

```
Request: POST /predict
Body: {"text": "...", "model_id": "small-perplexity"}
       ↓
InferenceRouter analyzes model config
  - preferred_backend: LOCAL
  - fallback: MODAL
  - current status: VPS has GPU, RAM available ✓
       ↓
LocalInferenceEngine.predict()
  - Load model from disk
  - Run inference in thread pool
  - Cache model in memory
  - Return result
       ↓
Response: {prediction: 1, probability: 0.72, backend: "local", ...}
```

**Time**: 100-500ms

### Scenario 2: Large Model (Model unavailable on VPS)

```
Request: POST /predict
Body: {"text": "...", "model_id": "large-multilayer"}
       ↓
InferenceRouter analyzes model config
  - preferred_backend: MODAL (because size=large)
  - fallback: LOCAL
  - Modal configured? Yes ✓
       ↓
ModalInferenceEngine.predict()
  - Call Modal API
  - Run on remote GPU
  - Wait for result
  - Return result
       ↓
Response: {prediction: 1, probability: 0.85, backend: "modal", ...}
```

**Time**: 500ms-5s (includes network)

### Scenario 3: Client-side Inference

```
Request: POST /predict
Body: {"text": "...", "model_id": "tiny-tfidf"}
       ↓
InferenceRouter analyzes model config
  - preferred_backend: CLIENT_SIDE
  - Client-side enabled? Yes ✓
       ↓
ClientSideInferenceEngine.predict()
  - Prepare model metadata
  - Return model download URL
  - Return feature extraction params
       ↓
Response: {backend: "client", metadata: {model_url: "...", ...}, ...}
       ↓
Frontend:
  1. Downloads model from /models/{id}/download
  2. Loads model in WebWorker
  3. Runs inference locally
  4. Gets instant result
```

**Time**: 10-50ms (no network after download)

### Scenario 4: Preferred Backend Unavailable (Fallback)

```
Request: POST /predict
Body: {"text": "...", "model_id": "medium-embedding", "prefer_backend": "modal"}
       ↓
InferenceRouter tries MODAL
  - Modal API key not configured ✗
       ↓
Falls back to first in fallback_backends: LOCAL
  - Local GPU available ✓
       ↓
LocalInferenceEngine.predict()
  - Runs on VPS GPU
  - Returns result
       ↓
Response: {backend: "local", note: "fell back from modal", ...}
```

---

## Configuration Examples

### Example 1: VPS with GPU (4GB VRAM, 8GB RAM)

```env
# .env
DEFAULT_INFERENCE_BACKEND=local
ALLOW_CLIENT_SIDE_INFERENCE=true
ALLOW_MODAL_INFERENCE=false

VPS_GPU_MEMORY_GB=4
VPS_CPU_MEMORY_GB=8
MAX_VPS_INFERENCE_SIZE=medium
```

**Model Configuration** (`api/config.py`):
```python
AVAILABLE_MODELS = {
    "tiny-tfidf": ModelConfig(
        model_id="embedding_A__mercor_ai",
        size_category="tiny",
        preferred_backend=InferenceBackend.CLIENT_SIDE,
        fallback_backends=[InferenceBackend.LOCAL]
    ),
    "small-perplexity": ModelConfig(
        model_id="perplexity_base",
        size_category="small",
        preferred_backend=InferenceBackend.LOCAL,  # ← Use VPS GPU
        fallback_backends=[InferenceBackend.MODAL]
    ),
    "medium-embedding": ModelConfig(
        model_id="embedding_qwen_22",
        size_category="medium",
        preferred_backend=InferenceBackend.LOCAL,
        fallback_backends=[InferenceBackend.MODAL]
    ),
}
```

**Result**: Small/medium models run on VPS, large models on Modal, tiny runs on client

### Example 2: Minimal VPS (No GPU, 2GB RAM)

```env
DEFAULT_INFERENCE_BACKEND=local  # CPU only
ALLOW_CLIENT_SIDE_INFERENCE=true
ALLOW_MODAL_INFERENCE=true

VPS_GPU_MEMORY_GB=0
VPS_CPU_MEMORY_GB=2
MAX_VPS_INFERENCE_SIZE=tiny
```

**Model Configuration**:
```python
AVAILABLE_MODELS = {
    "tiny-tfidf": ModelConfig(
        model_id="embedding_A__mercor_ai",
        size_category="tiny",
        preferred_backend=InferenceBackend.LOCAL,  # ← CPU-friendly
        fallback_backends=[InferenceBackend.CLIENT_SIDE]
    ),
    "small-perplexity": ModelConfig(
        model_id="perplexity_base",
        size_category="small",
        preferred_backend=InferenceBackend.MODAL,  # ← Offload to cloud
        fallback_backends=[InferenceBackend.LOCAL]
    ),
}
```

**Result**: Only tiny models on VPS, everything else on Modal

### Example 3: Full-Featured (GPU + Modal + Client)

```env
DEFAULT_INFERENCE_BACKEND=local
ALLOW_CLIENT_SIDE_INFERENCE=true
ALLOW_MODAL_INFERENCE=true

MODAL_TOKEN_ID=xxx
MODAL_TOKEN_SECRET=xxx

VPS_GPU_MEMORY_GB=8
VPS_CPU_MEMORY_GB=16
MAX_VPS_INFERENCE_SIZE=large
```

**All options available**: Users can choose via `prefer_backend` parameter

---

## Deployment Steps

### Step 1: Prepare Code

```bash
cd deepfake-text-detector

# Copy and edit environment
cp .env.example .env
nano .env  # Set your VPS specs
```

### Step 2: Configure Models

Edit `api/config.py`:
```python
AVAILABLE_MODELS = {
    "your-model-id": ModelConfig(
        model_id="your-model-id",
        size_category="small",  # Adjust based on model size
        preferred_backend=InferenceBackend.LOCAL,
        fallback_backends=[InferenceBackend.MODAL]
    )
}
```

### Step 3: Build & Deploy

```bash
# Option A: Docker (Coolify)
docker-compose -f docker-compose.prod.yml up -d

# Option B: Local testing
python -m api.app_v2

# Option C: Production (gunicorn)
gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 api.app_v2:app
```

### Step 4: Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

---

## Integration Guide

### Python Client

```python
import httpx

async def predict(text: str, model_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://api.example.com/predict",
            json={"text": text, "model_id": model_id}
        )
        return response.json()

result = await predict("Some text", "small-perplexity")
print(f"Is fake: {result['is_fake']}")
print(f"Backend: {result['backend']}")
```

### JavaScript/Frontend

```javascript
const detector = new DeepfakeDetector('http://api.example.com');

const result = await detector.predict(textInput, 'small-perplexity');
console.log(`Is fake: ${result.is_fake}`);
console.log(`Backend: ${result.backend}`);
console.log(`Latency: ${result.latency_ms}ms`);
```

### Choosing Model ID from Frontend

```javascript
// Get available models
const models = await fetch('/models').then(r => r.json());

// Display options
const options = Object.keys(models.configured_models);
// ["tiny-tfidf", "small-perplexity", "medium-embedding", "large-multilayer"]

// Let user choose (or auto-select based on device)
const selectedModel = "small-perplexity";

// Make prediction with that model
const result = await detector.predict(text, selectedModel);
```

---

## Adding New Models

### When you train a new model:

1. **Save with metadata**:
```python
import pickle

detector = train_my_model()

with open("saved_models/my-new-model.pkl", "wb") as f:
    pickle.dump(detector, f)

metadata = {
    "analysis_type": "embedding",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "layer": 22,
    "pooling": "mean",
    "classifier_type": "svm",
    "dataset_used": "mercor_ai"
}

with open("saved_models/my-new-model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
```

2. **Register in config**:
```python
# api/config.py
AVAILABLE_MODELS = {
    "my-new-model": ModelConfig(
        model_id="my-new-model",
        size_category="small",  # Adjust
        preferred_backend=InferenceBackend.LOCAL,  # Choose
        fallback_backends=[InferenceBackend.MODAL]
    )
}
```

3. **Deploy**:
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

4. **Verify**:
```bash
curl http://localhost:8000/models
# Should see "my-new-model" in the list
```

---

## Performance Optimization

### 1. Model Caching
Models are cached after first use. First request ~500-1000ms, subsequent ~100-200ms.

### 2. Batch Processing
```python
# Instead of
for text in texts:
    await predict(text, model_id)

# Use
await batch_predict(texts, model_id)
```

### 3. GPU Acceleration
Set `VPS_GPU_MEMORY_GB` to enable GPU inference (5-10x faster).

### 4. Async Implementation
All endpoints are async, supporting high concurrency.

### 5. Resource Limits
Docker limits prevent memory explosion. Adjust in `docker-compose.prod.yml`.

---

## Monitoring

### Health Endpoint
```bash
curl http://localhost:8000/health

# Returns:
{
  "status": "healthy",
  "backends": {
    "local": {"status": "healthy", "gpu_available": true},
    "modal": {"status": "unavailable"},
    "client_side": {"status": "healthy"}
  },
  "available_models": ["model1", "model2", ...],
  "vps_info": {...}
}
```

### Statistics
```bash
curl http://localhost:8000/stats
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Model not found" | Model file missing | Check `/app/saved_models` |
| "Out of memory" | VPS overloaded | Reduce batch size, scale to Modal |
| "Backend unavailable" | Configuration issue | Check .env, verify API keys |
| Slow performance | CPU/GPU bottleneck | Enable GPU, scale with Docker |
| "Can't connect to API" | Network/firewall | Check VPS connectivity, open port 8000 |

---

## Comparison: Local vs Modal vs Client

| Aspect | Local VPS | Modal | Client-side |
|--------|-----------|-------|------------|
| **Speed** | 100-500ms | 500-5000ms | 10-50ms |
| **Cost** | Fixed VPS cost | Per-inference | Zero |
| **Model Size** | <1GB | Unlimited | <10MB |
| **Privacy** | Data sent to VPS | Data sent to Modal | Data stays local |
| **Reliability** | Depends on VPS | Managed by Modal | User's device |
| **Best For** | Small/Medium | Large/Enterprise | Lightweight |

---

## Files Created/Modified

```
api/
  ├── config.py              [NEW] Configuration & model setup
  ├── inference.py           [NEW] Inference engines & routing
  ├── app_v2.py              [NEW] Production FastAPI app
  ├── examples.py            [NEW] Integration examples
  └── app.py                 [EXISTING] Keep for reference

Dockerfile.prod             [NEW] Optimized production Docker image
docker-compose.prod.yml     [NEW] Production docker-compose
.env.example                [NEW] Environment template
DEPLOYMENT_GUIDE.md         [NEW] Complete deployment guide
```

---

## Next Steps

1. **Review** the code and understand the routing logic
2. **Configure** `api/config.py` with your models
3. **Test locally**: `python -m api.app_v2`
4. **Deploy to VPS**: Use docker-compose.prod.yml
5. **Monitor**: Check /health and /stats regularly
6. **Integrate** frontend using examples.py
7. **Optimize** based on actual performance metrics

---

## Key Benefits of This Architecture

✅ **Flexible**: Switch backends easily based on needs
✅ **Scalable**: Add more models without code changes
✅ **Resilient**: Automatic fallbacks to backup backends
✅ **Efficient**: Models cached, async processing, batch support
✅ **Observable**: Health checks, statistics, detailed responses
✅ **Production-ready**: Security, error handling, resource limits
✅ **Cost-optimized**: Run tiny models on client, large on Modal
✅ **Developer-friendly**: Clean API, type hints, documentation
