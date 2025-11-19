# Implementation Checklist & Verification Guide

## Pre-Deployment Checklist

### Code Review
- [ ] Read `ARCHITECTURE.md` to understand the design
- [ ] Review `api/config.py` - understand model registration
- [ ] Review `api/inference.py` - understand routing logic
- [ ] Review `api/app_v2.py` - understand endpoints
- [ ] Review `api/examples.py` - understand integrations

### Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Edit `.env` with your VPS specifications:
  - [ ] Set `VPS_GPU_MEMORY_GB` (0 if no GPU)
  - [ ] Set `VPS_CPU_MEMORY_GB` (your available RAM)
  - [ ] Set `MAX_VPS_INFERENCE_SIZE` (tiny, small, or medium)
  - [ ] Set `DEFAULT_INFERENCE_BACKEND` (local by default)
  - [ ] Enable/disable Modal if needed
  - [ ] Set CORS origins for your domain(s)

### Models Setup
- [ ] Verify models exist in `saved_models/`:
  - [ ] Model files: `*.pkl`
  - [ ] Metadata files: `*_metadata.pkl`
  - [ ] Check file sizes match expectations
- [ ] Edit `api/config.py`:
  - [ ] Add entry for each model in `AVAILABLE_MODELS`
  - [ ] Specify correct `size_category`
  - [ ] Set `preferred_backend` based on model size
  - [ ] Set `fallback_backends` for resilience
- [ ] Verify model IDs match saved filenames

### Directory Structure
- [ ] `/app/saved_models/` exists with your models
- [ ] `api/` folder exists with all files:
  - [ ] `__init__.py`
  - [ ] `config.py`
  - [ ] `inference.py`
  - [ ] `app_v2.py`
  - [ ] `examples.py`
- [ ] `models/` folder exists (your model extraction code)
- [ ] `utils/` folder exists (any utilities)

---

## Local Testing Checklist

### Install Dependencies
```bash
pip install -r requirements.txt
```
- [ ] Installation completes without errors
- [ ] Verify key packages:
  ```bash
  python -c "import torch, transformers, fastapi, pydantic"
  ```

### Test Configuration Loading
```bash
python -c "from api.config import get_settings; s = get_settings(); print(s.DEFAULT_INFERENCE_BACKEND)"
```
- [ ] Prints "local" or configured value
- [ ] No import errors

### Test Model Loading
```python
from api.config import get_model_config
config = get_model_config("your-model-id")
print(config.size_category, config.preferred_backend)
```
- [ ] Prints correct model info
- [ ] No FileNotFoundError

### Start API Locally
```bash
python -m api.app_v2
```
- [ ] Server starts without errors
- [ ] "Uvicorn running on http://0.0.0.0:8000"
- [ ] Can access http://localhost:8000

### Test Endpoints (in another terminal)
```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","backends":{...},"available_models":[...]}
```
- [ ] Returns 200 OK
- [ ] Shows available models
- [ ] Shows backend health

### Test Models Endpoint
```bash
curl http://localhost:8000/models
```
- [ ] Returns configured models
- [ ] Each model has size_category, preferred_backend, fallback_backends

### Test Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","model_id":"your-model-id"}'
```
- [ ] Returns 200 OK
- [ ] Result has: prediction, probability, confidence, is_fake, backend
- [ ] backend field matches expected inference engine
- [ ] latency_ms shows reasonable time

### Test Error Handling
```bash
# Wrong model ID
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model_id":"nonexistent"}'
```
- [ ] Returns 404 with helpful error message
- [ ] Lists available models in error

### Test with Different Backends (if configured)
```bash
# Force specific backend
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model_id":"model","prefer_backend":"local"}'
```
- [ ] Works with prefer_backend parameter
- [ ] Falls back if preferred unavailable

### Test Batch Prediction
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["text1","text2"],"model_id":"your-model"}'
```
- [ ] Returns results for all texts
- [ ] Faster than individual predictions

### Test Model Download (client-side)
```bash
curl http://localhost:8000/models/your-model/download-info
curl http://localhost:8000/models/your-model/download
```
- [ ] Returns model info and download URL
- [ ] Model file downloads successfully

### Performance Testing
```python
import time
import httpx
import asyncio

async def benchmark():
    async with httpx.AsyncClient() as client:
        # Warm-up (model loads)
        await client.post("http://localhost:8000/predict",
            json={"text":"warmup","model_id":"your-model"})
        
        # Measure
        times = []
        for i in range(5):
            start = time.time()
            await client.post("http://localhost:8000/predict",
                json={"text":"test","model_id":"your-model"})
            times.append(time.time() - start)
        
        print(f"Average: {sum(times)/len(times)*1000:.1f}ms")
        print(f"Min: {min(times)*1000:.1f}ms")
        print(f"Max: {max(times)*1000:.1f}ms")

asyncio.run(benchmark())
```
- [ ] First request: 200-1000ms (model loading)
- [ ] Subsequent requests: 50-500ms depending on model
- [ ] No memory leaks (check RAM usage)

---

## Docker Testing Checklist

### Build Docker Image
```bash
docker build -f Dockerfile.prod -t deepfake-api:latest .
```
- [ ] Build completes without errors
- [ ] Image created: `docker images | grep deepfake-api`

### Run Docker Container
```bash
docker run -d \
  --name deepfake-api-test \
  -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models:ro \
  -e VPS_GPU_MEMORY_GB=0 \
  -e VPS_CPU_MEMORY_GB=4 \
  deepfake-api:latest
```
- [ ] Container starts: `docker ps | grep deepfake-api`
- [ ] Logs show startup: `docker logs deepfake-api-test`
- [ ] No errors in logs

### Test Docker Container
```bash
# Wait 30s for startup
sleep 30

# Test
curl http://localhost:8000/health
curl http://localhost:8000/models
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model_id":"your-model"}'
```
- [ ] All endpoints work in Docker
- [ ] Same behavior as local
- [ ] No permission errors

### Test Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```
- [ ] Builds and starts all services
- [ ] API container running
- [ ] Redis running (if enabled)
- [ ] Health checks passing

### Verify Volumes
```bash
docker exec deepfake-api ls -la /app/saved_models
```
- [ ] Models visible inside container
- [ ] Files readable

### Clean Up
```bash
docker-compose -f docker-compose.prod.yml down
docker rm deepfake-api-test
docker rmi deepfake-api:latest
```
- [ ] All resources cleaned up

---

## Production Deployment Checklist

### Pre-Deployment
- [ ] All local tests passed
- [ ] Docker tests passed
- [ ] .env file prepared with production values
- [ ] Models uploaded to VPS
- [ ] Backups configured
- [ ] Domain/SSL configured (if using HTTPS)

### VPS Setup
- [ ] SSH access working
- [ ] Docker installed: `docker --version`
- [ ] Docker Compose installed: `docker-compose --version`
- [ ] Port 8000 open (or reverse proxy port)
- [ ] Disk space available for models
- [ ] Memory available for models

### Deploy to VPS
```bash
# Clone repo or upload code
git clone <repo> deepfake-api
cd deepfake-api

# Copy .env
cp .env.example .env
nano .env  # Edit with production values

# Start
docker-compose -f docker-compose.prod.yml up -d

# Check
docker ps
docker logs deepfake-api
curl http://localhost:8000/health
```
- [ ] Container running
- [ ] No errors in logs
- [ ] Health check passes

### Verify Production
```bash
curl -X POST http://your-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model_id":"your-model"}'
```
- [ ] API responds from domain
- [ ] Correct predictions returned
- [ ] No errors

### Configure Reverse Proxy (Optional)
- [ ] Nginx/Apache configured to forward to :8000
- [ ] HTTPS working
- [ ] Rate limiting configured
- [ ] CORS headers properly set

### Monitoring Setup
- [ ] Health check endpoint monitored
- [ ] Logs centralized (syslog, etc.)
- [ ] Alerts configured for failures
- [ ] Performance metrics tracked

### Backup Strategy
- [ ] Model files backed up
- [ ] Logs backed up
- [ ] Configuration backed up
- [ ] Recovery procedure documented

---

## Verification Tests

### Functionality Test
```python
import httpx

async def test_all_models():
    async with httpx.AsyncClient() as client:
        # Get models
        r = await client.get("http://localhost:8000/models")
        models = r.json()
        
        # Test each model
        for model_id in models.get("configured_models", {}).keys():
            response = await client.post(
                "http://localhost:8000/predict",
                json={"text": "test text", "model_id": model_id}
            )
            assert response.status_code == 200, f"Failed for {model_id}"
            result = response.json()
            assert "prediction" in result
            assert "backend" in result
            print(f"✓ {model_id} ({result['backend']})")
```
- [ ] All models return predictions
- [ ] Backend selection works
- [ ] Response format correct

### Fallback Test
```python
# Configure first backend as unavailable
# Verify fallback kicks in
# Check response backend field shows fallback

# Or test with prefer_backend parameter
await client.post("http://localhost:8000/predict",
    json={
        "text": "test",
        "model_id": "model",
        "prefer_backend": "modal"  # If not configured
    }
)
# Should fallback and succeed
```
- [ ] Fallbacks work when preferred unavailable
- [ ] Error handling graceful
- [ ] Response explains what happened

### Performance Test
```python
import time
import httpx
import asyncio

async def measure_performance():
    async with httpx.AsyncClient() as client:
        # Warmup
        await client.post("http://localhost:8000/predict",
            json={"text":"x"*1000, "model_id":"model"})
        
        # 10 sequential predictions
        times = []
        for _ in range(10):
            start = time.time()
            r = await client.post("http://localhost:8000/predict",
                json={"text":"x"*1000, "model_id":"model"})
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            result = r.json()
            actual_latency = result["latency_ms"]
            print(f"Total: {elapsed:.0f}ms, API: {actual_latency:.0f}ms")
        
        print(f"\nAverage: {sum(times)/len(times):.0f}ms")
        print(f"95th percentile: {sorted(times)[9]:.0f}ms")
```
- [ ] Latency acceptable for use case
- [ ] No degradation over time
- [ ] No memory leaks

### Concurrency Test
```python
import asyncio

async def test_concurrency():
    async with httpx.AsyncClient() as client:
        # 5 concurrent requests
        tasks = [
            client.post("http://localhost:8000/predict",
                json={"text": f"text {i}", "model_id": "model"})
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in results)
        print(f"✓ 5 concurrent requests successful")
```
- [ ] Handles multiple simultaneous requests
- [ ] No crashes under load
- [ ] All requests succeed

### Backend Health Test
```python
async def test_backend_health():
    async with httpx.AsyncClient() as client:
        health = await client.get("http://localhost:8000/health")
        data = health.json()
        
        print(f"Status: {data['status']}")
        for backend, info in data['backends'].items():
            print(f"  {backend}: {info.get('status', 'unknown')}")
        
        assert data['status'] == 'healthy'
```
- [ ] All backends reported
- [ ] Status shows availability
- [ ] Can make informed decisions

---

## Post-Deployment Verification

### Day 1
- [ ] Monitor logs for errors
- [ ] Check disk usage
- [ ] Verify predictions quality
- [ ] Test with real traffic samples
- [ ] Check backend selection is correct

### Week 1
- [ ] Monitor average response times
- [ ] Check for memory leaks
- [ ] Verify fallback mechanisms work
- [ ] Review error logs
- [ ] Check cost if using Modal

### Month 1
- [ ] Collect performance metrics
- [ ] Identify optimization opportunities
- [ ] Plan model updates
- [ ] Document any issues
- [ ] Update documentation if needed

---

## Troubleshooting Decision Tree

### "Model not found" error
```
├─ Check saved_models/ directory exists
│  └─ If missing: Create it with models
├─ Check .pkl file exists
│  └─ If missing: Copy model file
├─ Check metadata exists
│  └─ If missing: Not required but recommended
└─ Restart API
   └─ If still failing: Check model_id in config.py
```

### "Out of memory"
```
├─ Check /stats endpoint
│  ├─ If storage high: Clean old models
│  └─ If RAM high: Reduce workers
├─ Try smaller model
├─ Enable GPU if available
└─ Consider Modal for large models
```

### "Backend unavailable"
```
├─ Check .env configuration
│  ├─ LOCAL: Should always work
│  ├─ MODAL: Check API key set
│  └─ CLIENT: Check ALLOW_CLIENT_SIDE_INFERENCE=true
├─ Check /health endpoint
│  └─ Shows which backends available
└─ Check logs for errors
```

### Slow inference
```
├─ Check /health for GPU status
│  └─ If no GPU: Enable or use Modal
├─ Check model size
│  └─ If large: Consider Modal
├─ Check latency_ms in response
│  └─ If high: Model is computation-heavy
└─ Try batch processing for multiple texts
```

---

## Success Criteria

Your deployment is successful when:

- [ ] API starts without errors
- [ ] `/health` endpoint returns "healthy"
- [ ] All models appear in `/models` list
- [ ] Can predict with all configured models
- [ ] Predictions are correct for test inputs
- [ ] Response time acceptable (< 5s for small models)
- [ ] Docker container runs stably
- [ ] No memory leaks after 1 hour
- [ ] Fallback mechanisms work
- [ ] Error messages helpful and clear
- [ ] Can monitor via `/health` and `/stats`
- [ ] Frontend can integrate successfully

When all checked ✓, you're ready for production! 🚀
