# Architecture & Design Decisions

## System Overview

The detector follows a modular pipeline:

```
Raw Text
  ↓ [Preprocessing: lowercase, normalize whitespace]
  ↓ [Feature Extraction: embedding/TF-IDF/etc]
  ↓ [Dimensionality reduction (optional): PCA, scaling]
  ↓ [Classification: LR/SVM/Outlier Detection]
  ↓ [Threshold application: optimize or fixed]
  ↓ Output: P(fake) ∈ [0, 1], Decision ∈ {0, 1}
```

## Feature Extractors

### Embedding-based (Default)

**Process:**
1. Tokenize text with HuggingFace model
2. Forward pass through transformer → hidden states per layer
3. **Extract single layer** (configurable, e.g., layer 18)
4. **Pool tokens** (mean/last/mean_std) → single vector per text
5. (Optional) **L2 normalize** → unit vectors
6. Feed to classifier

**Models tested:**
- `microsoft/deberta-v3-large` – Strong baseline, 24 layers
- `Qwen/Qwen3-Embedding-4B` – Lightweight embeddings
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` – Multilingual

**Key parameters:**
- `layer`: Which transformer layer to extract (typically mid-to-late, e.g., 12–23 of 24)
- `pooling`: How to aggregate tokens
  - `mean` – Average all tokens
  - `last` – Take last token (usually `[EOS]`)
  - `mean_std` – Concatenate mean + standard deviation
  - `statistical` – Compute full covariance matrix (high-dim, use with caution)
- `normalize`: L2 normalization before classifier

**Memory efficiency:**
- **Default (all layers)**: Load hidden states for all layers → ~35–40 GB for 4k texts × large model
- **`--memory_efficient`**: Extract one layer at a time, discard others → ~20 GB
- Strongly recommended for large-scale sweeps.

### TF-IDF

**Process:**
1. Tokenize text (whitespace, remove stopwords)
2. Build vocabulary & IDF weights
3. Vectorize each text as sparse/dense TF-IDF vector

**Pros:**
- Fast, interpretable, no GPU needed
- Works well for short-range stylistic differences

**Cons:**
- Poor generalization across domains (0.64–0.80 ROC-AUC)
- Sensitive to vocabulary choice
- High-dimensional (20k+ features)

### Other (Perplexity, PHD)

- **Perplexity**: Use language model to score text; lower perplexity ≈ human-like
- **PHD (Intrinsic Dimensionality)**: Measure local dimensionality; AI text often has lower PHD

Less reliable than embeddings; useful as ensemble features.

## Classifiers

### Supervised

**Logistic Regression** (`lr`)
- Linear decision boundary in feature space
- Fast, interpretable, often strong baseline
- Requires normalized features; good with embedding pooling

**Support Vector Machine** (`svm`)
- Non-linear boundaries (RBF kernel)
- Works well with high-dimensional embeddings
- Slower training than LR

### Outlier Detection

Assume real text is "in-distribution"; AI text is outlier.

**Isolation Forest** (`iforest`)
- Fast, parallelizable
- Works on high-dimensional data
- Often competitive on new domains

**Elliptic Envelope** (`elliptic`)
- Estimates covariance of the main cluster
- Sensitive to initialization; may fail if data is not well-clustered
- Useful when real data is homogeneous

**One-Class SVM** (`ocsvm`)
- Learns hypersphere around real data
- Slower but robust
- Best for well-separated domains

## Why Outlier Detection?

When evaluating on a new dataset (e.g., Mercor AI after training on Human vs AI):
- Domain shift is common → supervised classifiers overfit to train domain quirks
- Outlier detection may generalize better if "real" text has consistent properties across domains
- In practice: **outlier detectors ≈ supervised** in performance; sometimes +0.05 ROC-AUC on cross-dataset

## Threshold Optimization

### Default (0.5)
- Assumes equal class weight
- Works if P(fake) is well-calibrated; often isn't

### Optimized (validation-set tuning)
1. Reserve 20% of validation set
2. Sweep thresholds τ ∈ [0, 1] (step 0.01)
3. For each τ: compute F1 (or target metric)
4. Pick τ with best F1
5. Report metrics at that τ

**Why this matters:**
- Model with high ROC-AUC but 0% F1 at τ=0.5? Optimize τ → F1 jumps to 0.75+
- Class imbalance on eval set? Optimized τ adapts automatically
- Different metrics (F1 vs precision) require different τ

### Deployment strategy

**If you know the target distribution:**
- Optimize on a labeled validation set from that distribution
- Use fixed τ for production

**If unknown (e.g., Kaggle test set):**
- Optimize on best-guess validation set (e.g., Mercor AI train)
- Use that τ for final submission
- Accept that τ may not be perfect if test distribution differs significantly

## Class Imbalance & Stratified Sampling

### Problem
Without stratification, random sampling on imbalanced data → train on 70% real, 30% AI → classifier biased toward "real" → predicts mostly 0 → F1 = 0 even if AUC is high.

### Solution
`--stratified_sample` → sample exactly N/2 from each class
- Gives classifier a fair chance to learn both classes
- Metrics become interpretable (no "always predict 0" trap)
- Works orthogonally to threshold optimization:
  - **Balanced training** + **threshold optimization at eval time** = best of both worlds

## Normalization (L2)

**Effect:**
- Normalize embeddings to unit length
- Reduces variance in vector magnitudes; focuses on directions
- Can help with distance-based classifiers (SVM, kNN)
- Often neutral or slightly positive for LR

**When to use:**
- Default: ON (recommended)
- Some embeddings (e.g., BAAI bge) already return normalized vectors; double-normalization is harmless
- TF-IDF: Normalization can help or hurt; sweep both

## Ensemble Strategy

After a parameter sweep:
1. **Rank configs by F1** on cross-dataset eval
2. **Select top-15 per base model**, diversify by:
   - Layer (e.g., 12, 18, 22)
   - Pooling (mean, last, mean_std)
   - Classifier (svm, lr, ocsvm)
3. **Average probabilities** from selected configs
4. **Optimize threshold** on ensemble output
5. **Submit** with optimized threshold

Example ensemble (3 configs):
```python
probs = [
    model1.predict_proba(text),  # deberta layer 18, mean, svm
    model2.predict_proba(text),  # deberta layer 22, mean_std, lr
    model3.predict_proba(text),  # qwen layer 4, last, ocsvm
]
ensemble_prob = np.mean(probs)
prediction = ensemble_prob >= threshold  # threshold = 0.38
```

---

## See Also

- [README.md](./README.md) – Quick start & full guide
- [COOLIFY_DEPLOYMENT.md](./COOLIFY_DEPLOYMENT.md) – Deploy the API