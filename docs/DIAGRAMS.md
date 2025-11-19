# System Architecture Diagrams

## 1. Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                 │
│  (Web, Mobile, Desktop)                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ HTTP POST /predict
                           │ {text, model_id, [prefer_backend]}
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER                             │
│                  (api/app_v2.py)                                │
│                                                                 │
│  • Validates input (min/max length)                            │
│  • Authentication/Rate limiting (optional)                     │
│  • Calls InferenceRouter                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              INFERENCE ROUTER                                   │
│         (inference.py - Smart Routing Logic)                    │
│                                                                 │
│  Decision Tree:                                                │
│  1. Load model config                                          │
│  2. Check preferred backend availability                       │
│  3. If unavailable → try fallbacks                            │
│  4. Select best available backend                              │
│  5. Delegate to selected engine                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │  LOCAL   │    │  MODAL   │    │  CLIENT-SIDE │
    │ ENGINE   │    │ ENGINE   │    │   ENGINE     │
    └────┬─────┘    └────┬─────┘    └────┬─────────┘
         │                │              │
         │                │              │
         ▼                ▼              ▼
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │   VPS    │    │  Modal   │    │  Browser/    │
    │ Inference│    │ Serverless│    │  Client App  │
    │  (GPU/   │    │   GPU    │    │              │
    │  CPU)    │    │          │    │              │
    └────┬─────┘    └────┬─────┘    └────┬─────────┘
         │                │              │
         └────────────────┼──────────────┘
                          │
                          │ Standard Response Format
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            STANDARD INFERENCE RESULT                            │
│                                                                 │
│  {                                                              │
│    "prediction": 0,              # 0=human, 1=AI              │
│    "probability": 0.72,          # Probability [0-1]          │
│    "confidence": 0.44,           # Confidence [0-1]           │
│    "is_fake": true,              # Boolean                    │
│    "backend": "local",           # Which backend ran          │
│    "model_id": "model-name",     # Which model used          │
│    "latency_ms": 245.32,         # How long it took          │
│    "metadata": {...}             # Extra info                │
│  }                                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND                                   │
│              (Display result to user)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Backend Selection Logic

```
SCENARIO 1: Small Model (size_category="small")
─────────────────────────────────────────────────

Model Config:
  preferred_backend: LOCAL
  fallback_backends: [MODAL]

Request: {"text": "...", "model_id": "small-perplexity"}

Router checks:
  ✓ Is LOCAL available? YES (VPS has GPU, RAM)
  
Decision: Use LOCAL
  
Result: 
  ├─ Inference time: 200-500ms
  ├─ Cost: Free (included in VPS)
  └─ Backend: "local"


SCENARIO 2: Large Model (size_category="large")
──────────────────────────────────────────────────

Model Config:
  preferred_backend: MODAL
  fallback_backends: [LOCAL]

Request: {"text": "...", "model_id": "large-multilayer"}

Router checks:
  ✓ Is MODAL available? YES (API key configured)
  
Decision: Use MODAL
  
Result:
  ├─ Inference time: 2-30s
  ├─ Cost: $0.01-0.05 per request
  └─ Backend: "modal"


SCENARIO 3: Preferred Backend Unavailable (FALLBACK)
──────────────────────────────────────────────────────

Model Config:
  preferred_backend: MODAL
  fallback_backends: [LOCAL, CLIENT_SIDE]

Request: {"text": "...", "model_id": "model"}

Router checks:
  ✗ Is MODAL available? NO (API key not configured)
  ✓ Is first fallback (LOCAL) available? YES
  
Decision: Fallback to LOCAL
  
Result:
  ├─ Inference time: 200-500ms
  ├─ Cost: Free
  ├─ Backend: "local"
  └─ Note: Fell back from modal


SCENARIO 4: All Backends Unavailable
──────────────────────────────────────

Model Config:
  preferred_backend: LOCAL
  fallback_backends: []

Request: {"text": "...", "model_id": "model"}

Router checks:
  ✗ Is LOCAL available? NO (Out of GPU memory)
  ✗ No fallbacks configured
  
Decision: Return error

Result:
  ├─ Status: Error
  ├─ Message: "All inference backends failed"
  └─ Suggestion: Configure fallback or increase resources
```

---

## 3. System Components Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                          │
│                     (api/app_v2.py)                             │
│                                                                  │
│  Routes:                                                         │
│  ├── GET  /                    → API info                       │
│  ├── GET  /health              → Backend health                 │
│  ├── GET  /models              → List models                    │
│  ├── GET  /backends            → Backend options                │
│  ├── POST /predict             → Single inference               │
│  ├── POST /batch-predict       → Batch inference                │
│  └── GET  /models/{id}/download → Download for client           │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    Depends on
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              CONFIGURATION MANAGEMENT                            │
│                   (api/config.py)                               │
│                                                                  │
│  Settings:                                                       │
│  ├── DEFAULT_INFERENCE_BACKEND                                 │
│  ├── ALLOW_CLIENT_SIDE_INFERENCE                               │
│  ├── ALLOW_MODAL_INFERENCE                                     │
│  ├── VPS_GPU_MEMORY_GB                                         │
│  ├── VPS_CPU_MEMORY_GB                                         │
│  └── AVAILABLE_MODELS (Dict)                                   │
│      └── Each model specifies:                                 │
│          ├── model_id                                          │
│          ├── size_category                                     │
│          ├── preferred_backend                                 │
│          └── fallback_backends                                 │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    Reads from
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                ENVIRONMENT (.env file)                          │
│                                                                  │
│  DEFAULT_INFERENCE_BACKEND=local                                │
│  ALLOW_CLIENT_SIDE_INFERENCE=true                               │
│  VPS_GPU_MEMORY_GB=4                                            │
│  ALLOW_MODAL_INFERENCE=false                                    │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│               INFERENCE ROUTING ENGINE                           │
│                  (inference.py)                                 │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ InferenceRouter                                           │  │
│  │ • Loads model config                                      │  │
│  │ • Evaluates backend availability                          │  │
│  │ • Applies fallback logic                                  │  │
│  │ • Delegates to selected engine                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│            │              │                │                    │
│   ┌────────▼─────┐ ┌──────▼─────┐ ┌───────▼──────┐              │
│   │   LocalEngine│ │ModalEngine │ │ClientEngine  │              │
│   │              │ │            │ │              │              │
│   │• Loads .pkl  │ │• Calls API │ │• Prepares    │              │
│   │• Extracts    │ │• Sends to  │ │  model for   │              │
│   │  features    │ │  remote    │ │  download    │              │
│   │• Runs        │ │  GPU       │ │• Returns     │              │
│   │  inference   │ │• Caches    │ │  download    │              │
│   │  (GPU/CPU)   │ │  result    │ │  URL         │              │
│   └──────────────┘ └────────────┘ └──────────────┘              │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│                 INFERENCE EXECUTION LAYER                        │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │
│  │ VPS Server   │ │ Modal Cloud  │ │ Client Browser/App    │   │
│  │              │ │              │ │                        │   │
│  │ • GPUs       │ │ • A40/A100   │ │ • WebWorker or        │   │
│  │ • CPUs       │ │ • Fast       │ │   NodeJS worker        │   │
│  │ • RAM        │ │ • Scalable   │ │ • Limited resources    │   │
│  │ • Low latency│ │ • Pay/req    │ │ • Zero server load     │   │
│  └──────────────┘ └──────────────┘ └────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Model Configuration to Backend Routing

```
TINY MODELS (< 10 MB)
────────────────────
  Size: Very small
  Example: TF-IDF with <100 features
  
  Config:
    size_category: "tiny"
    preferred_backend: CLIENT_SIDE
    fallback_backends: [LOCAL]
  
  Routing:
    ┌─────────────────────────┐
    │ Try CLIENT_SIDE first   │  → 10-50ms (browser)
    └────────────┬────────────┘
                 │ (if unavailable)
    ┌────────────▼────────────┐
    │ Fallback to LOCAL       │  → 50-200ms (VPS)
    └─────────────────────────┘
  
  Use when: Mobile/lightweight apps, privacy-sensitive

SMALL MODELS (50-200 MB)
──────────────────────────
  Size: Manageable
  Example: Perplexity calculator, lightweight embeddings
  
  Config:
    size_category: "small"
    preferred_backend: LOCAL
    fallback_backends: [MODAL]
  
  Routing:
    ┌─────────────────────────┐
    │ Try LOCAL first         │  → 100-500ms (VPS + GPU)
    └────────────┬────────────┘
                 │ (if unavailable)
    ┌────────────▼────────────┐
    │ Fallback to MODAL       │  → 500-5000ms (remote GPU)
    └─────────────────────────┘
  
  Use when: Production server with GPU

MEDIUM MODELS (500MB - 1GB)
────────────────────────────
  Size: Large
  Example: Full embeddings + pooling
  
  Config:
    size_category: "medium"
    preferred_backend: LOCAL
    fallback_backends: [MODAL]
  
  Routing:
    ┌─────────────────────────┐
    │ Try LOCAL first         │  → 500-2000ms (needs GPU)
    └────────────┬────────────┘
                 │ (if unavailable)
    ┌────────────▼────────────┐
    │ Fallback to MODAL       │  → 2-10s (remote GPU)
    └─────────────────────────┘
  
  Use when: High-accuracy needed, GPU available

LARGE MODELS (> 1GB)
────────────────────
  Size: Very large
  Example: Ensemble, multi-layer extraction
  
  Config:
    size_category: "large"
    preferred_backend: MODAL
    fallback_backends: [LOCAL]
  
  Routing:
    ┌─────────────────────────┐
    │ Try MODAL first         │  → 2-30s (powerful GPU)
    └────────────┬────────────┘
                 │ (if unavailable)
    ┌────────────▼────────────┘
    │ Fallback to LOCAL       │  → 10-60s (VPS struggles)
    └─────────────────────────┘
  
  Use when: State-of-the-art accuracy needed
```

---

## 5. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT                             │
│                                                             │
│  Command: python -m api.app_v2                            │
│  Port: 8000                                                 │
│  Workers: 1                                                 │
│  GPU: Auto-detect                                           │
│  Perfect for: Local testing, debugging                      │
└─────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│                  STAGING/PRODUCTION                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          NGINX/Caddy (Optional)                     │  │
│  │  • HTTPS termination                               │  │
│  │  • Rate limiting                                   │  │
│  │  • Load balancing                                  │  │
│  └────────────────┬────────────────────────────────────┘  │
│                   │                                       │
│     ┌─────────────┴──────────────┐                       │
│     │                            │                       │
│  ┌──▼──────────┐      ┌──────────▼──┐                   │
│  │   API Pod 1 │      │  API Pod 2   │  (Docker replicas)
│  │  :8000      │      │  :8000       │                   │
│  │  workers=4  │      │  workers=4   │                   │
│  └──┬──────────┘      └──────────┬───┘                   │
│     │                            │                       │
│     └─────────────┬──────────────┘                       │
│                   │                                       │
│  ┌────────────────▼──────────────────┐                  │
│  │      Docker Host (Your VPS)        │                  │
│  │                                    │                  │
│  │  Volumes:                          │                  │
│  │  ├─ ./saved_models → /app/models  │                  │
│  │  ├─ ./logs → /app/logs            │                  │
│  │  └─ redis-data (optional)          │                  │
│  │                                    │                  │
│  │  Network: deepfake-net             │                  │
│  │  ├─ api container                  │                  │
│  │  └─ redis container (optional)     │                  │
│  └────────────────────────────────────┘                  │
│                                                          │
│  Deployment Tool: docker-compose.prod.yml              │
│  Restart: unless-stopped                                │
│  Health checks: /health endpoint every 30s              │
└─────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│           COOLIFY / VPS PROVIDER INTEGRATION                │
│                                                             │
│  1. Connect Git repo                                        │
│  2. Select Dockerfile.prod                                  │
│  3. Set environment variables from .env                     │
│  4. Configure volumes:                                      │
│     - /app/saved_models (mount local models)               │
│     - /app/logs (for logs)                                 │
│  5. Set resource limits                                     │
│  6. Deploy!                                                 │
│                                                             │
│  Auto features:                                             │
│  • Builds on every push                                     │
│  • Auto-restart on crash                                    │
│  • Health checks                                            │
│  • Logs available                                           │
│  • Easy scaling                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Data Flow During Inference

```
SINGLE PREDICTION FLOW:
─────────────────────

Frontend                API Server              Backend Engine
   │                        │                        │
   │─ POST /predict ──────→ │                        │
   │                        │ Load model config      │
   │                        │─────────────────────→  │ ✓ Check config
   │                        │                        │
   │                        │ Select backend         │
   │                        │─────────────────────→  │ ✓ LOCAL available
   │                        │                        │
   │                        │ Run inference          │
   │                        │─────────────────────→  │ • Load model
   │                        │                        │ • Extract features
   │                        │                        │ • Predict
   │                        │                        │
   │                        │ ←───────────────────── │ Return result
   │                        │                        │
   │                        │ Format response        │
   │ ←─ JSON response ───── │                        │
   │                        │                        │
   ✓ Display result         │                        │


BATCH PREDICTION FLOW:
──────────────────────

Frontend                API Server              Backend Engine
   │                        │                        │
   │─ POST /batch ────────→ │                        │
   │ [text1, text2, ...]    │                        │
   │                        │ Process sequentially   │
   │                        │ (or batch if GPU)      │
   │                        │─────────────────────→  │ • Load once
   │                        │                        │ • Batch features
   │                        │                        │ • Batch predict
   │                        │                        │
   │                        │ ←───────────────────── │ Results
   │                        │                        │
   │ ←─ [result1, result2]─ │                        │
   │                        │                        │


FALLBACK FLOW:
──────────────

Request                 Router                   Engines
   │                        │                        │
   │─ predict ────────────→ │                        │
   │ model=X               │ Check preferred        │
   │ prefer_backend=MODAL  │─────────────────────→  │ MODAL
   │                        │                        │ ✗ Not configured
   │                        │ ←───────────────────── │
   │                        │                        │
   │                        │ Try fallback #1        │
   │                        │─────────────────────→  │ LOCAL
   │                        │                        │ ✓ Available
   │                        │ Run inference          │
   │                        │─────────────────────→  │ • Load
   │                        │                        │ • Predict
   │                        │ ←───────────────────── │ Result
   │                        │                        │
   │ ←─ Result (from LOCAL)─ │                        │
   │   (fallback from MODAL) │                        │
   │                        │                        │
```

---

## 7. Resource Utilization

```
VPS WITH GPU (4GB VRAM, 8GB RAM)
────────────────────────────────

Timeline of requests:
┌──────────────────────────────────────────────────────────┐
│ Time(s) │ Activity           │ GPU VRAM │ CPU RAM        │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 0.0     │ API starts         │ 0MB      │ 500MB          │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 0.5     │ Model loads        │ 3500MB   │ 600MB          │
│         │ (first request)    │          │                │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 0.6     │ Inference runs     │ 3800MB   │ 700MB          │
│         │ (GPU busy)         │ (utilization: 95%) │       │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 0.8     │ Result returned    │ 3500MB   │ 600MB          │
│         │ (model cached)     │ (cached) │                │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 1.0     │ Another inference  │ 3500MB   │ 700MB          │
│         │ (no reload, fast)  │ (utilization: 98%) │       │
├─────────┼────────────────────┼──────────┼────────────────┤
│ 1.2     │ Result returned    │ 3500MB   │ 600MB          │
│         │ Latency: 100ms     │ (cached) │                │
└──────────────────────────────────────────────────────────┘

Key points:
✓ First request slower (model load) ~500ms
✓ Subsequent requests fast (cached) ~100-200ms
✓ Memory stays stable (model persistent)
✓ GPU always utilized > 90% when processing
```

---

## Summary

This architecture provides:

1. **Smart Routing**: Automatic backend selection
2. **Flexibility**: Easy to add models and backends
3. **Resilience**: Fallback mechanisms
4. **Performance**: Caching, async, batching
5. **Scalability**: Docker, multiple workers
6. **Observability**: Health checks, stats, monitoring
7. **Cost Efficiency**: Use appropriate backend per model size

The three-tier system (Local/Modal/Client) covers all use cases from minimal (tiny models) to maximal (large enterprise models).
