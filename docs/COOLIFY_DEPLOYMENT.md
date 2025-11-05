# Coolify Deployment

Deploy the deepfake detector API to Coolify for always-on inference.

## Prerequisites

- Coolify instance running (self-hosted or SaaS)
- GitHub repo with this code
- Trained model saved as `saved_models/model.pkl`
- Docker Hub account (for private images, optional)

## Setup Steps

### 1. Push Code to GitHub

Ensure your repo is pushed with:
- `Dockerfile` ✓
- `requirements.txt` ✓
- `api/app.py` ✓
- Trained models in `saved_models/` (or mounted separately)

```bash
git push origin main
```

### 2. Connect Repository to Coolify

1. Log in to Coolify dashboard
2. **New Project** → Select your GitHub repo
3. Choose **Docker** as deployment type
4. Point to `Dockerfile` in root directory

### 3. Set Environment Variables

In Coolify, configure:

| Variable | Value | Notes |
|----------|-------|-------|
| `DEVICE` | `cpu` or `cuda:0` | Use `cpu` unless GPU available |
| `PORT` | `8000` | FastAPI default |
| `MODEL_PATH` | `saved_models/best_model.pkl` | Must exist in container |
| `BATCH_SIZE` | `8` | For batch predictions |

### 4. Mount Volumes (if needed)

If models are stored externally or need persistence:

```yaml
volumes:
  - /path/to/saved_models:/app/saved_models
  - /path/to/data:/app/data
```

### 5. Deploy

Click **Deploy** in Coolify. It will:
- Build Docker image
- Start container
- Assign domain (e.g., `detector.example.com`)
- Manage SSL certificate
- Handle auto-restart on failure

### 6. Verify

```bash
curl https://detector.example.com/health
# Expected response: {"status": "ok"}

curl https://detector.example.com/docs
# Opens interactive API docs
```

## Usage

### Health Check

```bash
curl https://detector.example.com/health
```

### Predict

```bash
curl -X POST https://detector.example.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence."}'
```

Response:
```json
{
  "text": "This is a test sentence.",
  "probability": 0.23,
  "label": "real",
  "confidence": 0.77
}
```

### Batch Predict

```bash
curl -X POST https://detector.example.com/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", ...]}'
```

## Monitoring

In Coolify dashboard:
- **Logs** – Check for errors
- **Metrics** – CPU, memory, request count
- **Health** – Auto-restart on crash

## Troubleshooting

### Model not found

- Ensure `saved_models/model.pkl` exists in repo
- Or mount external volume with models
- Check logs: `docker logs <container>`

### Out of memory

- Reduce `BATCH_SIZE` env var
- Use `DEVICE=cpu` instead of GPU
- Ensure Coolify has enough memory allocated

### Slow startup

- Large models take time to load
- Coolify health check timeout may be too short; increase if possible
- Use smaller models (e.g., 4B instead of 70B)

## Advanced: Scaling

For high-traffic deployments:
1. **Increase replicas** in Coolify (horizontal scaling)
2. **Load balancer** (Coolify includes one) distributes requests
3. **GPU shared** across replicas (if using GPU)

---

See [README.md](./README.md) for local development and [ARCHITECTURE.md](./ARCHITECTURE.md) for system design.