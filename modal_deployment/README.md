# Modal Deployment Guide

Deploy your ML models to Modal.com for serverless inference.

## 🚀 Quick Start

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal token new
```

### 3. Upload Your Trained Models
```bash
modal run modal_deployment/app.py::upload_models
```

### 4. Deploy
```bash
modal deploy modal_deployment/app.py
```

### 5. Get Webhook URL
After deployment, Modal will show:
```
✓ Deployed web endpoint webhook
  https://[your-username]--deepfake-text-detector-webhook.modal.run
```

Copy this URL!

### 6. Configure Render Gateway
Add to Render environment variables:
```
MODAL_WEBHOOK_URL=https://[your-username]--deepfake-text-detector-webhook.modal.run
```

---

## 📡 API Usage

### Endpoint
```
POST https://[your-username]--deepfake-text-detector-webhook.modal.run
```

### Request Body
```json
{
  "text": "This is the text to analyze",
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layer": 22,
  "pooling": "mean",
  "classifier_type": "svm",
  "dataset_name": "mercor_ai"
}
```

### Response
```json
{
  "prediction": 1,
  "probability": 0.87,
  "confidence": 0.74,
  "is_fake": true,
  "model_info": {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "layer": 22,
    "pooling": "mean",
    "classifier": "svm",
    "dataset": "mercor_ai"
  }
}
```

---

## 🔧 Configuration Options

### Use GPU (Faster, More Expensive)
In `app.py`, change:
```python
@app.function(
    gpu="T4",  # or "A10G" for more power
    ...
)
```

**Cost:** ~$0.60/hour GPU time (only charged when running)

### Keep Instance Warm (Faster Response)
```python
@app.function(
    keep_warm=1,  # Keep 1 instance always ready
    ...
)
```

**Cost:** ~$5/month to keep 1 instance warm
**Benefit:** Eliminates cold starts (~5-10s → ~500ms)

### Increase Memory
```python
@app.function(
    memory=8192,  # 8GB
    ...
)
```

---

## 📊 Cost Estimation

### Without Keep Warm (Default)
- First request: 5-10s (cold start)
- Subsequent requests: ~500ms
- Cost: $0.00003/second = **~$0.10 per 1,000 requests**

### With Keep Warm
- All requests: ~500ms
- Cost: $5/month + $0.10 per 1,000 requests

### With GPU
- Cost: $0.00025/second = **~$0.25 per 1,000 requests**
- Much faster for large models

---

## 🧪 Testing

### Test Health Endpoint
```bash
curl https://[your-username]--deepfake-text-detector-health.modal.run
```

### Test Detection
```bash
curl -X POST https://[your-username]--deepfake-text-detector-webhook.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test message",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "layer": 22
  }'
```

---

## 🔄 Updating Models

### Upload New Models
```bash
modal run modal_deployment/app.py::upload_models
```

### Redeploy
```bash
modal deploy modal_deployment/app.py
```

No downtime - Modal handles rolling updates!

---

## 📈 Monitoring

### View Logs
```bash
modal app logs deepfake-text-detector
```

### Monitor Usage
Go to: https://modal.com/apps

You'll see:
- Request count
- Execution time
- Memory usage
- Costs

---

## 🐛 Troubleshooting

### Error: "No trained model found"
```bash
# Upload models first
modal run modal_deployment/app.py::upload_models
```

### Error: "Import error"
Check that `models/` and `utils/` directories exist and have:
- `models/__init__.py`
- `models/extractors.py`
- `models/classifiers.py`

### Slow Cold Starts
Enable keep_warm:
```python
keep_warm=1
```

### Out of Memory
Increase memory:
```python
memory=8192  # 8GB
```

---

## 💰 Cost Optimization

### Strategy 1: Cache in Gateway
Cache repeated requests in your Render gateway:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def analyze_cached(text_hash: str):
    # Call Modal
    ...
```

### Strategy 2: Batch Requests
Send multiple texts at once to Modal.

### Strategy 3: Use Smaller Models
Deploy multiple model sizes:
- Small (Qwen 0.5B) for quick checks
- Large (Qwen 8B) for important decisions

---

## 🚀 Next Steps

1. ✅ Deploy to Modal
2. ✅ Update Render gateway to call Modal
3. ✅ Test end-to-end
4. ✅ Monitor costs for a few days
5. ✅ Optimize (keep_warm, caching, etc.)

Need help? Check Modal docs: https://modal.com/docs
