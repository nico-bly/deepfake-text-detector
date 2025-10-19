# 🚀 Coolify Deployment Guide

## Quick Start

### 1️⃣ Prepare Your Models

Before deploying, make sure you have trained models in `saved_models/`:

```bash
# Check if you have trained models
ls -lh saved_models/

# If not, train some models first
python scripts/train_and_save_detector.py
```

### 2️⃣ Test Locally with Docker

```bash
# Build the image
docker build -t deepfake-backend .

# Run locally
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models:ro \
  deepfake-backend

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence to analyze."}'
```

### 3️⃣ Deploy to Coolify

#### Option A: Using Git Repository

1. **Push your code to Git** (GitHub/GitLab/etc.)

2. **In Coolify Dashboard:**
   - Click **"+ New Resource"**
   - Select **"Docker Compose"** or **"Dockerfile"**
   - Connect your Git repository
   - Set branch: `main`
   - Set build context: `deepfake-text-detector`
   - Set Dockerfile path: `Dockerfile`

3. **Configure Environment:**
   - No environment variables needed for basic setup
   - Optional: Add `LOG_LEVEL=debug` for verbose logging

4. **Configure Ports:**
   - Internal Port: `8000`
   - Public Port: `8000` (or your preferred port)

5. **Storage (Important!):**
   - Add a volume mount:
     - Source: `/path/to/your/saved_models` (on your VPS)
     - Destination: `/app/saved_models`
     - Read-only: ✅

6. **GPU Support (Optional):**
   - If your VPS has GPU, add in Docker settings:
     ```yaml
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
     ```

7. **Click "Deploy"**

#### Option B: Using Docker Compose

1. **Upload files to your VPS:**
   ```bash
   rsync -avz --exclude 'data/' --exclude 'archive/' \
     deepfake-text-detector/ user@your-vps:/home/user/deepfake-backend/
   ```

2. **In Coolify:**
   - Click **"+ New Resource"**
   - Select **"Docker Compose"**
   - Paste your `docker-compose.yml` content
   - Set working directory: `/home/user/deepfake-backend`
   - Click **"Deploy"**

### 4️⃣ Update Frontend Configuration

Once deployed, get your backend URL from Coolify (e.g., `https://api.yourdomain.com`)

Update your frontend to point to this URL:

```typescript
// In your frontend .env or config
VITE_API_URL=https://api.yourdomain.com
```

Or update directly in your API client:

```typescript
const API_BASE_URL = 'https://api.yourdomain.com';

const detectText = async (text: string) => {
  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return response.json();
};
```

---

## 📋 API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "deepfake-text-detector",
  "available_models": ["detector_mercor_ai_Qwen_Qwen2.5-0.5B_layer22_svm.pkl"],
  "gpu_available": true
}
```

### `POST /detect`
Detect if text is AI-generated.

**Request:**
```json
{
  "text": "Your text here",
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layer": 22,
  "pooling": "mean",
  "classifier_type": "svm",
  "dataset_name": "mercor_ai"
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "confidence": 0.54,
  "is_fake": false,
  "model_info": {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "layer": 22,
    "pooling": "mean",
    "classifier": "svm",
    "dataset": "mercor_ai",
    "device": "cuda"
  }
}
```

### `POST /detect-pair`
Detect which text in a pair is real.

**Request:**
```json
{
  "text1": "First text",
  "text2": "Second text",
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layer": 22
}
```

**Response:**
```json
{
  "real_text_id": 1,
  "text1_prediction": 0,
  "text2_prediction": 1,
  "text1_probability": 0.2,
  "text2_probability": 0.8,
  "confidence": 0.6
}
```

### `GET /models`
List available models.

**Response:**
```json
{
  "available_models": ["detector_mercor_ai_Qwen_Qwen2.5-0.5B_layer22_svm.pkl"],
  "cached_models": ["mercor_ai_Qwen_Qwen2.5-0.5B_layer22_svm"],
  "cached_extractors": ["Qwen/Qwen2.5-0.5B"]
}
```

---

## 🔧 Troubleshooting

### Models not found
```bash
# Check if models directory exists in container
docker exec deepfake-text-backend ls -la /app/saved_models/

# If empty, verify volume mount in Coolify
# Or copy models manually:
docker cp ./saved_models/. deepfake-text-backend:/app/saved_models/
```

### Out of memory
```bash
# Reduce model cache or use smaller models
# In Coolify, increase memory limit:
# Settings → Resources → Memory Limit → 8GB
```

### GPU not detected
```bash
# Verify NVIDIA Docker runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Enable GPU in docker-compose.yml (see example above)
```

### Slow first request
This is normal - first request loads the model into memory (cold start).
Subsequent requests are faster. Consider:
- Using smaller models
- Pre-warming cache with a startup script
- Keeping containers running (avoid frequent restarts)

---

## 🎯 Production Checklist

- [ ] Trained models exist in `saved_models/`
- [ ] Docker image builds successfully
- [ ] Health check passes locally
- [ ] Volume mount configured in Coolify
- [ ] CORS origins restricted to your frontend domain
- [ ] HTTPS enabled (Coolify handles this)
- [ ] Monitoring/logging configured
- [ ] Backup strategy for trained models
- [ ] Consider using smaller/quantized models for faster inference
- [ ] Set up automatic restarts on failure

---

## 📊 Performance Tips

1. **Model Selection**: Smaller models = faster inference
   - `Qwen/Qwen2.5-0.5B` is fast
   - `Llama-3.1-8B` is more accurate but slower

2. **Caching**: Models are cached after first load
   - First request: ~10-30s
   - Subsequent requests: ~1-3s

3. **Batching**: Process multiple texts together (TODO: implement batch endpoint)

4. **GPU**: Significantly faster than CPU
   - Ensure GPU is available and configured

5. **Workers**: Use multiple workers for high traffic
   ```bash
   CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

---

## 🔐 Security Notes

- Limit text input size (currently 10,000 chars max)
- Use CORS restrictions in production
- Consider rate limiting
- Monitor resource usage
- Keep dependencies updated

---

## 📝 Environment Variables (Optional)

You can customize behavior with these env vars:

```bash
LOG_LEVEL=info          # Logging level (debug, info, warning, error)
MAX_WORKERS=1           # Number of workers (1 for GPU, multiple for CPU)
MODEL_CACHE_SIZE=5      # Number of models to keep in memory
DEVICE=cuda             # Force device (cuda or cpu)
```

---

## 🆘 Need Help?

Check logs in Coolify:
1. Go to your service
2. Click "Logs" tab
3. Look for errors during startup or requests

Common issues:
- **Model not found**: Check volume mount and model files
- **CUDA out of memory**: Use smaller models or reduce batch size
- **Connection refused**: Check port configuration in Coolify
- **CORS errors**: Update CORS origins in `api/app.py`
