# 🚀 Backend API Setup - Quick Start

This is your REST API for the deepfake text detection backend. No API gateway needed since you're deploying directly on Coolify!

## 📋 Architecture

```
Frontend (Coolify) → Backend API (Coolify) → ML Models (GPU)
     ↓                       ↓
  React/Vite            FastAPI + PyTorch
```

## ⚡ Quick Setup (Local Testing)

### 1. Install Dependencies

```bash
cd deepfake-text-detector
pip install -r requirements.txt
```

### 2. Ensure You Have Trained Models

```bash
# Check if models exist
ls -lh saved_models/

# If empty, train some models first
python scripts/train_and_save_detector.py
```

### 3. Start the API Server

```bash
# From deepfake-text-detector directory
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python api/app.py
```

### 4. Test the API

Open another terminal:

```bash
# Quick test
./api/test_api.sh

# Or detailed test
python api/test_api.py
```

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Detect text
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence to analyze for AI generation."}'
```

### 5. View API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🐳 Docker Deployment (Local)

### Build and Run

```bash
# Build the image
docker build -t deepfake-backend .

# Run with GPU support (if available)
docker run -p 8000:8000 \
  --gpus all \
  -v $(pwd)/saved_models:/app/saved_models:ro \
  deepfake-backend

# Or without GPU
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models:ro \
  deepfake-backend
```

### Using Docker Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## ☁️ Coolify Deployment

See **[COOLIFY_DEPLOYMENT.md](./COOLIFY_DEPLOYMENT.md)** for complete step-by-step guide.

### Quick Summary:

1. **Push to Git** or upload files to VPS
2. **In Coolify Dashboard:**
   - New Resource → Dockerfile or Docker Compose
   - Point to your repo/directory
   - Add volume mount: `./saved_models:/app/saved_models`
   - Set port: `8000`
   - Enable GPU if available
   - Deploy!
3. **Update Frontend** with your new API URL
4. **Done!** 🎉

---

## 📚 API Endpoints

### `GET /health`
Check if API is running and view available models.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "deepfake-text-detector",
  "available_models": ["detector_...pkl"],
  "gpu_available": true
}
```

### `POST /detect`
Detect if a single text is AI-generated.

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "layer": 22,
    "pooling": "mean",
    "classifier_type": "svm",
    "dataset_name": "mercor_ai"
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "confidence": 0.54,
  "is_fake": false,
  "model_info": {...}
}
```

### `POST /detect-pair`
Detect which text in a pair is real.

```bash
curl -X POST http://localhost:8000/detect-pair \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "First text",
    "text2": "Second text"
  }'
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
List all available and cached models.

```bash
curl http://localhost:8000/models
```

---

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# Set in Coolify or .env file
LOG_LEVEL=info          # debug, info, warning, error
MAX_WORKERS=1           # Number of workers (keep 1 for GPU)
DEVICE=cuda             # Force device (cuda or cpu)
```

### Model Selection

Edit in your request body:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B",     // HuggingFace model
  "layer": 22,                            // Layer to extract from
  "pooling": "mean",                      // mean, max, last
  "classifier_type": "svm",               // svm, lr, neural
  "dataset_name": "mercor_ai"             // Dataset used for training
}
```

Or keep defaults by just sending:

```json
{
  "text": "Your text here"
}
```

---

## 🎯 Frontend Integration

### Update Your Frontend

In your React/Vite frontend:

```typescript
// src/config/api.ts
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// src/services/api.ts
import { API_BASE_URL } from '../config/api';

export const detectText = async (text: string) => {
  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error('Detection failed');
  }

  return response.json();
};

export const detectPair = async (text1: string, text2: string) => {
  const response = await fetch(`${API_BASE_URL}/detect-pair`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text1, text2 }),
  });

  if (!response.ok) {
    throw new Error('Pair detection failed');
  }

  return response.json();
};
```

### Environment Variable

Create `.env` in your frontend:

```bash
# Local development
VITE_API_URL=http://localhost:8000

# Production (update after deploying backend to Coolify)
VITE_API_URL=https://api.yourdomain.com
```

---

## 🧪 Testing

### Run All Tests

```bash
# Shell script (quick)
./api/test_api.sh

# Python script (detailed)
python api/test_api.py

# Custom API URL
API_URL=https://api.yourdomain.com ./api/test_api.sh
```

### Manual Testing

```bash
# Test with different texts
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'

curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "As an AI language model, I can help you with various tasks."}'
```

---

## 🐛 Troubleshooting

### API won't start

```bash
# Check if port is in use
lsof -i :8000

# Kill process if needed
kill -9 $(lsof -t -i :8000)

# Check Python environment
which python
python --version
pip list | grep fastapi
```

### No models found

```bash
# Check models directory
ls -la saved_models/

# Train a model if empty
python scripts/train_and_save_detector.py

# Or download pre-trained models
# (if you have them stored elsewhere)
```

### Import errors

```bash
# Make sure you're in the right directory
cd deepfake-text-detector

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# Check NVIDIA driver
nvidia-smi

# API will fall back to CPU automatically
```

### CORS errors from frontend

Edit `api/app.py` and update CORS origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📁 Project Structure

```
deepfake-text-detector/
├── api/                      # ← NEW: API module
│   ├── __init__.py
│   ├── app.py               # ← Main FastAPI application
│   ├── test_api.py          # ← Python test script
│   └── test_api.sh          # ← Bash test script
├── models/                   # ML models code
│   ├── classifiers.py
│   ├── extractors.py
│   └── ...
├── utils/                    # Utilities
├── saved_models/            # ← Trained model files (.pkl)
│   └── detector_*.pkl
├── Dockerfile               # ← NEW: Docker build config
├── docker-compose.yml       # ← NEW: Docker compose config
├── .dockerignore            # ← NEW: Docker ignore
├── requirements.txt         # Python dependencies
├── README.md                # ← This file
└── COOLIFY_DEPLOYMENT.md    # ← Deployment guide
```

---

## 🚀 Deployment Checklist

Before deploying to Coolify:

- [ ] API runs successfully locally (`uvicorn api.app:app`)
- [ ] Tests pass (`./api/test_api.sh`)
- [ ] At least one trained model exists in `saved_models/`
- [ ] Docker image builds (`docker build -t deepfake-backend .`)
- [ ] Docker container runs (`docker run -p 8000:8000 deepfake-backend`)
- [ ] Health check works in Docker
- [ ] Code pushed to Git (if using Git deployment)
- [ ] CORS origins configured for production
- [ ] Frontend `.env` ready with API URL

After deploying:

- [ ] Backend accessible via public URL
- [ ] Health endpoint returns 200
- [ ] At least one model shows in `/health` response
- [ ] Frontend can reach backend (test CORS)
- [ ] Detection works end-to-end
- [ ] Logs show no errors

---

## 💡 Tips

1. **Start Small**: Test locally first, then Docker, then Coolify
2. **Model Size**: Smaller models = faster inference (`Qwen2.5-0.5B` is good)
3. **Caching**: First request is slow (loads model), subsequent requests are fast
4. **GPU**: Highly recommended for acceptable performance
5. **Monitoring**: Check Coolify logs regularly after deployment
6. **Backups**: Keep your trained models backed up!

---

## 📖 Next Steps

1. ✅ **Local Testing**: Run API locally and test endpoints
2. ✅ **Docker Testing**: Build and run Docker container
3. ✅ **Deploy to Coolify**: Follow [COOLIFY_DEPLOYMENT.md](./COOLIFY_DEPLOYMENT.md)
4. ✅ **Update Frontend**: Point to new API URL
5. ✅ **Test End-to-End**: Verify frontend can detect texts
6. 🎉 **You're Done!**

---

## 📞 Support

- **API Docs**: http://localhost:8000/docs (when running)
- **Health Check**: http://localhost:8000/health
- **Logs**: `docker-compose logs -f` or Coolify logs tab

Need help? Check:
1. Logs in Coolify
2. Health endpoint response
3. Available models list
4. This README's troubleshooting section

---

Made with ❤️ for Coolify deployment
