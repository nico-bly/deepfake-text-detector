# 🎯 Architecture Decision: Single Service vs Modal

## Your Use Case Analysis

### Training Phase (Your GPU, Offline)
```python
# You do this locally with your powerful GPU
model = load_model("Qwen/Qwen2.5-8B")  # 16GB model
embeddings = model.encode(training_texts)  # Generate embeddings
classifier = SVM()
classifier.fit(embeddings, labels)
classifier.save("detector.pkl")  # Save tiny classifier (~1-10MB)
```

### Inference Phase (Production, Online)
```python
# User request comes in
text = "Check if this is fake..."

# Option A: Small model on Render
embedding = small_model.encode(text)  # ONE embedding, fast!
prediction = classifier.predict(embedding)  # Instant!

# Option B: Large model on Modal
embedding = large_model.encode(text)  # More accurate?
prediction = classifier.predict(embedding)  # Instant!
```

## 🔍 Key Question: Do you need the SAME 8B model for inference?

### Scenario 1: Can use smaller model for inference
**Answer:** No, you can use a smaller model that produces "good enough" embeddings

**Best Solution:** ✅ **Single Render Service (FREE)**

### Scenario 2: MUST use same 8B model for inference
**Answer:** Yes, the embeddings must match training exactly

**Best Solution:** ✅ **Modal.com ($0.10 per 1k requests)**

---

## 📊 Comparison Table

| Aspect | Small Model (Render) | Large Model (Modal) |
|--------|---------------------|---------------------|
| **Embedding Model** | distilroberta (250MB) | Qwen2.5-8B (16GB) |
| **RAM Needed** | 400MB | 20GB+ |
| **Cost** | **FREE** | ~$0.10/1k requests |
| **Cold Start** | None (always on) | 5-10s first request |
| **Response Time** | ~200-500ms | ~1-2s |
| **Accuracy** | Good (if retrained) | Best (exact match) |
| **Deployment** | Simple (1 service) | Complex (2 services) |

---

## 💡 Recommended Strategy

### **Option 1: Retrain with Small Model (Recommended)**

**Process:**
1. Use small model (distilroberta) to generate embeddings
2. Train your SVM/LogReg classifier on these embeddings
3. Deploy everything to Render free tier
4. Done! 🎉

**Pros:**
- ✅ Completely FREE
- ✅ Fast responses (~200ms)
- ✅ Simple architecture
- ✅ No cold starts
- ✅ Easy to maintain

**Cons:**
- ❌ Need to retrain classifiers
- ❌ Slightly lower accuracy (maybe)

**When to use:**
- You have training data available
- You can retrain classifiers (~30 min)
- You want FREE hosting
- You want simple deployment

---

### **Option 2: Keep Large Model, Use Modal**

**Process:**
1. Keep your existing Qwen-8B trained classifiers
2. Deploy to Modal.com
3. Deploy gateway to Render
4. Pay per request

**Pros:**
- ✅ Use existing trained models
- ✅ No retraining needed
- ✅ Best possible accuracy
- ✅ Can handle any model size

**Cons:**
- ❌ Costs ~$0.10 per 1,000 requests
- ❌ More complex setup
- ❌ Cold starts on first request
- ❌ Slower responses (~1-2s)

**When to use:**
- Can't retrain (no training data)
- Need exact embedding consistency
- Budget for $1-10/month
- Accuracy is critical

---

## 🎯 My Recommendation for You

### Start with Option 1 (Small Model on Render)

**Reasoning:**
1. **You have training data** (mercor, ESA, DAIGT)
2. **You can retrain** (you already have the pipeline)
3. **Small models are surprisingly good** (80-90% of large model accuracy)
4. **It's FREE** (huge win for MVP)
5. **Simpler to maintain**

**Implementation Steps:**

```bash
# 1. Retrain with small model (on your GPU)
python scripts/train_with_small_model.py \
  --model_name sentence-transformers/all-distilroberta-v1 \
  --train_path data/mercor-ai/train.csv \
  --output_path saved_models/detector_mercor_distilroberta_svm.pkl

# 2. Test locally
python test_small_model.py

# 3. Deploy to Render (single service!)
# Start command: uvicorn services.single_service.main:app --host 0.0.0.0 --port $PORT

# 4. Done! FREE deployment ✅
```

---

## 📝 Training Script for Small Model

Here's a script to retrain with a small model:

```python
# scripts/train_with_small_model.py
import argparse
from pathlib import Path
import pandas as pd
from models.extractors import EmbeddingExtractor
from models.classifiers import BinaryDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--classifier_type", default="svm")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.train_path)
    texts = df['answer'].tolist()  # or 'text' column
    labels = df['is_cheating'].tolist()
    
    print(f"Training with {len(texts)} samples...")
    
    # Extract embeddings with SMALL model
    extractor = EmbeddingExtractor(args.model_name, device="cuda")
    embeddings = extractor.get_pooled_layer_embeddings(
        texts,
        layer_idx=-1,  # Last layer
        pooling="mean",
        batch_size=32,
        show_progress=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Train classifier
    detector = BinaryDetector(
        n_components=0.95,
        contamination=0.1,
        random_state=42
    )
    
    detector.fit(
        embeddings=embeddings,
        labels=labels,
        validation_split=0.2,
        classifier_type=args.classifier_type,
        pca=True
    )
    
    # Save
    import pickle
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(detector, f)
    
    print(f"✅ Saved to {output_path}")
    
    # Test
    test_texts = ["This is a test", "Another test message"]
    test_embeddings = extractor.get_pooled_layer_embeddings(
        test_texts, layer_idx=-1, pooling="mean", batch_size=2
    )
    predictions, probs = detector.predict(test_embeddings, return_probabilities=True)
    print(f"Test predictions: {predictions}, probs: {probs}")

if __name__ == "__main__":
    main()
```

---

## 🔄 When to Switch to Modal

Switch to Modal if:
1. ❌ Small model accuracy is insufficient (<80%)
2. ❌ Users complain about quality
3. ✅ You have budget (~$10/month)
4. ✅ You get > 10k requests/month

**But honestly?** Try the small model first. You might be surprised how good it is!

---

## 💰 Cost Scenarios

### Scenario A: Small Model (Render)
```
Traffic          | Cost
10 req/day       | FREE
100 req/day      | FREE
1,000 req/day    | FREE
10,000 req/day   | FREE (but may hit rate limits)
```

### Scenario B: Large Model (Modal)
```
Traffic          | Cost
10 req/day       | $0.03/month
100 req/day      | $0.30/month
1,000 req/day    | $3.00/month
10,000 req/day   | $30/month
```

---

## 🎓 Model Size Recommendations

### For Text Classification (Your Use Case)

| Model | Size | RAM | Accuracy | Speed | Recommendation |
|-------|------|-----|----------|-------|----------------|
| `all-MiniLM-L6-v2` | 80MB | 200MB | ⭐⭐⭐ | ⚡⚡⚡⚡ | ✅ Best for free tier |
| `all-distilroberta-v1` | 250MB | 400MB | ⭐⭐⭐⭐ | ⚡⚡⚡ | ✅ Good balance |
| `Qwen2.5-0.5B` | 1GB | 2GB | ⭐⭐⭐⭐ | ⚡⚡ | ❌ Too big for free tier |
| `Qwen2.5-8B` | 16GB | 20GB | ⭐⭐⭐⭐⭐ | ⚡ | ❌ Needs Modal |

---

## ✅ Action Plan

### Week 1: MVP with Small Model (FREE)
1. Retrain classifiers with `all-distilroberta-v1`
2. Deploy to Render free tier
3. Test with real users
4. Measure accuracy

### Week 2: Evaluate
- If accuracy > 85%: ✅ Keep it! You're done.
- If accuracy < 85%: ⚠️ Consider upgrading

### Week 3: Optimize (if needed)
- Try `all-mpnet-base-v2` (420MB - might still fit)
- Fine-tune small model on your data
- Or upgrade to Modal for large model

---

## 🚀 Quick Start Command

```bash
# Option 1: Single service with small model (Recommended)
uvicorn services.single_service.main:app --host 0.0.0.0 --port $PORT

# Environment Variables for Render:
EMBEDDING_MODEL=sentence-transformers/all-distilroberta-v1
DEFAULT_CLASSIFIERS=detector_mercor_svm,detector_esa_svm
DEVICE=cpu
```

**Memory usage:** ~300-400MB ✅ Fits in 512MB!

---

## 📞 Decision Matrix

Answer these questions:

1. **Can you retrain classifiers?**
   - Yes → Use small model (Render)
   - No → Use Modal

2. **Is FREE hosting important?**
   - Yes → Use small model (Render)
   - No, I have budget → Can use Modal

3. **Do you need EXACT embeddings from training?**
   - Yes → Use Modal
   - No, good enough is fine → Use small model (Render)

4. **How many requests per day?**
   - < 1,000 → Small model (Render)
   - > 10,000 → Consider Modal or Render paid tier

---

## 🎉 TL;DR

**For your use case (single predictions with pre-trained classifiers):**

1. ✅ **Retrain with small model** (`all-distilroberta-v1`)
2. ✅ **Deploy to Render free tier** (single service)
3. ✅ **Enjoy FREE hosting** with ~300ms responses
4. ✅ **Upgrade to Modal later** only if needed

**Estimated time to deploy:** 2 hours  
**Cost:** FREE  
**Performance:** ~200-500ms per request  
**Complexity:** Low (single service)

Start simple, upgrade if needed! 🚀
