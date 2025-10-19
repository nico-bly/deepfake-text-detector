# 🏗️ Production Architecture for 512MB RAM Constraint

## Problem
Render free tier = 512MB RAM, but ML models need 2-8GB RAM.

## Solution
Separate lightweight API gateway from heavy ML inference.

---

## 🎯 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   React + Vite (Vercel/Netlify FREE)                │   │
│  │   - User interface                                   │   │
│  │   - File upload                                      │   │
│  │   - Results display                                  │   │
│  └────────────────────┬────────────────────────────────┘   │
└─────────────────────┬─┴────────────────────────────────────┘
                      │
                      │ HTTPS/REST
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                    API GATEWAY LAYER                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Lightweight Gateway (Render Free - 512MB)          │   │
│  │   FastAPI or Express.js                              │   │
│  │   - Authentication/Authorization                     │   │
│  │   - Request routing                                  │   │
│  │   - Rate limiting                                    │   │
│  │   - Response caching (Redis/Upstash)                │   │
│  │   - NO ML MODELS HERE                               │   │
│  └───────────┬──────────────────────────────────────────┘   │
└──────────────┴──────────────────────────────────────────────┘
               │
               │ Routes to appropriate service
               │
┌──────────────┴──────────────────────────────────────────────┐
│                    ML INFERENCE LAYER                        │
│                                                              │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │  Hugging Face      │  │  Modal.com         │            │
│  │  Inference API     │  │  Serverless GPU    │            │
│  │                    │  │                    │            │
│  │  • Pre-trained     │  │  • Custom models   │            │
│  │    models          │  │  • Your detectors  │            │
│  │  • FREE tier       │  │  • Auto-scaling    │            │
│  │  • 1k req/day      │  │  • Pay per use     │            │
│  └────────────────────┘  └────────────────────┘            │
│                                                              │
│  Alternative: Replicate, RunPod, AWS Lambda + EFS           │
└──────────────────────────────────────────────────────────────┘
               │
               │ Store/retrieve data
               │
┌──────────────┴──────────────────────────────────────────────┐
│                    DATA LAYER                                │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │  PostgreSQL     │  │  S3/R2/Backblaze│  │  Upstash   │  │
│  │  (Supabase)     │  │  (File Storage) │  │  (Redis)   │  │
│  │                 │  │                 │  │            │  │
│  │  • User data    │  │  • Trained      │  │  • Cache   │  │
│  │  • Predictions  │  │    models       │  │  • Session │  │
│  │  • Logs         │  │  • Datasets     │  │            │  │
│  └─────────────────┘  └─────────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────┘
               │
               │ Scheduled tasks
               │
┌──────────────┴──────────────────────────────────────────────┐
│                    AUTOMATION LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GitHub Actions (FREE)                               │   │
│  │  - Model training jobs (monthly)                     │   │
│  │  - Data pipeline                                     │   │
│  │  - Backup & cleanup                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Options

### **Option 1: Hugging Face + Render Gateway (Simplest)**

**Pros:**
- ✅ Minimal code changes
- ✅ FREE tier: 1,000 requests/day
- ✅ No infrastructure management
- ✅ Pre-trained models work out-of-box

**Cons:**
- ❌ Can't use your custom trained detectors easily
- ❌ Limited model selection
- ❌ Slower for custom embeddings

**Best for:** MVP, testing, demos

---

### **Option 2: Modal.com + Render Gateway (Recommended)**

**Pros:**
- ✅ Use your exact training code
- ✅ Load custom trained models
- ✅ Auto-scaling (0 to infinity)
- ✅ Only pay when running (~$0.10/1000 requests)
- ✅ Keep your current model architecture

**Cons:**
- ❌ Small learning curve
- ❌ Cold starts (5-10s first request, then fast)

**Best for:** Production with custom models

**Cost estimate:**
- 10,000 requests/month ≈ **$1-2/month**
- 100,000 requests/month ≈ **$10-20/month**

---

### **Option 3: AWS Lambda + EFS (Advanced)**

**Pros:**
- ✅ Generous free tier (1M requests/month)
- ✅ Load models from EFS
- ✅ Established platform

**Cons:**
- ❌ Complex setup
- ❌ 10GB Lambda limit
- ❌ Cold starts

**Best for:** High-scale production (>1M requests/month)

---

## 📝 Migration Path for Your Current Code

### Step 1: Keep Gateway Lightweight (Render Free Tier)

**File: `services/gateway_lite/main.py`**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Deepfake Gateway")

MODAL_WEBHOOK_URL = os.getenv("MODAL_WEBHOOK_URL")

class AnalyzeRequest(BaseModel):
    text: str
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer: int = 22
    classifier_type: str = "svm"

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Route to ML service (Modal/HF)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            MODAL_WEBHOOK_URL,
            json=req.dict()
        )
        return response.json()

@app.get("/health")
def health():
    return {"status": "ok", "ram_usage": "~50MB"}
```

**Memory usage: ~50-100MB** ✅

---

### Step 2: Deploy ML Models to Modal

**File: `modal_deployment/detector.py`**
```python
import modal
from pathlib import Path

# Create Modal app
stub = modal.Stub("deepfake-text-detector")

# Define the image with your dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "scikit-learn",
        "sentence-transformers",
        "numpy",
        "pandas"
    )
)

# Mount your trained models
models_volume = modal.NetworkFileSystem.from_name(
    "deepfake-models", 
    create_if_missing=True
)

@stub.function(
    image=image,
    gpu="T4",  # Optional: remove for CPU-only
    memory=4096,  # 4GB RAM
    timeout=300,
    network_file_systems={"/models": models_volume}
)
def detect_deepfake(
    text: str,
    model_name: str,
    layer: int,
    classifier_type: str
):
    """
    Your existing detection logic - no changes needed!
    """
    import sys
    sys.path.append("/models")
    
    from extractors import EmbeddingExtractor
    from classifiers import BinaryDetector
    import pickle
    
    # Load your trained detector
    detector_path = f"/models/detector_{model_name.replace('/', '_')}_layer{layer}_{classifier_type}.pkl"
    with open(detector_path, "rb") as f:
        detector = pickle.load(f)
    
    # Extract features
    extractor = EmbeddingExtractor(model_name, device="cuda")
    features = extractor.get_pooled_layer_embeddings(
        [text],
        layer_idx=layer,
        pooling="mean"
    )
    
    # Predict
    pred, prob = detector.predict(features, return_probabilities=True)
    
    return {
        "prediction": int(pred[0]),
        "probability": float(prob[0]),
        "is_fake": bool(pred[0] == 1)
    }

# Web endpoint
@stub.webhook(method="POST")
def webhook(data: dict):
    result = detect_deepfake.remote(
        text=data["text"],
        model_name=data["model_name"],
        layer=data["layer"],
        classifier_type=data.get("classifier_type", "svm")
    )
    return result
```

**Deploy:**
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy
modal deploy modal_deployment/detector.py

# Get your webhook URL (copy this to gateway env vars)
# https://yourusername--deepfake-text-detector-webhook.modal.run
```

---

### Step 3: Upload Your Trained Models to Modal

```bash
# One-time setup: upload your models
modal volume put deepfake-models saved_models/
```

Or via Python:
```python
import modal

# Get the volume
volume = modal.NetworkFileSystem.lookup("deepfake-models")

# Upload trained models
with volume.batch_upload() as upload:
    upload.put_directory("saved_models/", "/")
```

---

## 💰 Cost Comparison

| Solution | Free Tier | Cost (10k req/mo) | Cost (100k req/mo) |
|----------|-----------|-------------------|-------------------|
| **HF Inference** | 1k/day | FREE | $50-100 |
| **Modal.com** | - | $1-2 | $10-20 |
| **Replicate** | - | $5 | $50 |
| **RunPod** | - | $4 | $40 |
| **AWS Lambda** | 1M free | FREE | FREE-$10 |
| **Render 2GB** | - | $7/mo | $7/mo (but limited) |

---

## 🚀 Recommended Setup for You

### Phase 1: MVP (Start Here)
```
Frontend (Vercel) → Gateway (Render Free) → HF Inference API
                          ↓
                   PostgreSQL (Supabase)
```
**Cost:** FREE for <1k requests/day

### Phase 2: Production (Custom Models)
```
Frontend (Vercel) → Gateway (Render Free) → Modal.com
                          ↓
                   PostgreSQL + S3
```
**Cost:** ~$1-5/month for 10k-50k requests

### Phase 3: Scale (>100k requests/month)
```
Frontend (Vercel) → Gateway (Render Starter) → Modal.com + Cache
                          ↓
                   PostgreSQL + Redis + S3
```
**Cost:** ~$20-30/month

---

## 📦 What to Store Where

### Render Gateway (512MB)
- ✅ API routing logic
- ✅ Authentication
- ✅ Input validation
- ✅ Response caching (in-memory dict for last 100 results)
- ❌ NO ML models
- ❌ NO large datasets

### Modal.com / HF
- ✅ All ML models
- ✅ Embedding extractors
- ✅ Trained classifiers
- ✅ Feature computation

### S3 / Cloudflare R2
- ✅ Trained model files (.pkl, .pt)
- ✅ Training datasets
- ✅ User uploads (if needed)

### PostgreSQL (Supabase)
- ✅ User accounts
- ✅ Prediction history
- ✅ API usage logs
- ✅ Model metadata

---

## 🔄 Migration Checklist

- [ ] Create Modal account (free to start)
- [ ] Deploy detector to Modal using provided code
- [ ] Upload trained models to Modal volume
- [ ] Update gateway to call Modal webhook
- [ ] Add Modal webhook URL to Render env vars
- [ ] Test end-to-end flow
- [ ] Remove ML models from Render service
- [ ] Deploy lightweight gateway
- [ ] Monitor costs and performance

---

## 🆘 Need Help?

I can help you:
1. Set up Modal deployment with your existing code
2. Migrate specific detectors
3. Optimize for cost/performance
4. Set up caching to reduce API calls

Just let me know which option you want to pursue!
