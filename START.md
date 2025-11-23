# Quick Start

## Setup

1. **Place your model file in `saved_models_prod/`:**
```bash
# Your file should be named like:
# human_ai_microsoft_deberta-v3-large_embedding_layer23_mean_std_l2norm_lr.pkl
# human_ai_microsoft_deberta-v3-large_embedding_layer23_mean_std_l2norm_lr_metadata.pkl
```

2. **Update model mapping:**
```python
# In api/model_mapping_simple.py
MODELS = {
    "human_ai_microsoft_deberta": "human_ai_microsoft_deberta-v3-large_embedding_layer23_mean_std_l2norm_lr",
}
```

## Test Locally

```bash
# Run with Docker
./test_docker.sh

# Or manually:
docker-compose -f docker-compose.test.yml up
```

## API Endpoints

```bash
# List all dataset+model combinations
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8008/models/list

# Health check
curl http://localhost:8008/health

# Make prediction (requires API key)
curl -X POST http://localhost:8008/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "text": "Your text to analyze",
    "dataset": "human-ai-binary",
    "model_id": "qwen-0.5b"
  }'

# Using environment variable for API key
export API_KEY="your-api-key-here"
curl -X POST http://localhost:8008/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text":"Your text","dataset":"human-ai-binary","model_id":"qwen-0.5b"}'
```

### Available Datasets and Models
- **human-ai-binary**: `qwen-0.5b`, `qwen-8b`, `small-embedding`
- **human-ai-anomaly**: `qwen-0.5b`, `small-perplexity`, `medium-perplexity`  
- **arxiv**: `small-perplexity`, `medium-perplexity`
- **fakenews**: `small-perplexity`, `large-perplexity`

## Response Format

```json
{
  "prediction": 1,
  "probability": 0.85,
  "confidence": 0.7,
  "is_fake": true,
  "model_id": "human_ai_microsoft_deberta"
}
```

## Logs

```bash
docker-compose -f docker-compose.test.yml logs -f
```

## Stop

```bash
docker-compose -f docker-compose.test.yml down
```
