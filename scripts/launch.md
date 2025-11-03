# Launch Guide for Deepfake Text Detection Scripts

This document provides comprehensive examples for running all the scripts in this deepfake text detection project.

## 📁 Project Structure

The main scripts are located in the `scripts/` directory:

- **`main_submission_esa.py`**: ESA challenge submission pipeline
- **`main_submission_mercor.py`**: Mercor AI challenge submission pipeline  
- **`parameter_sweep_mercor.py`**: Hyperparameter optimization with k-fold CV
- **`train_and_save_detector.py`**: Train models and save for later use
- **`load_and_evaluate.py`**: Load saved models and evaluate on datasets
- **`cross_dataset_evaluation.py`**: Cross-dataset generalization evaluation

## 🚀 Core Analysis Types

All scripts support three main analysis types:

1. **`embedding`**: Layer-wise embeddings from transformer models (requires `--layer` and `--pooling`)
2. **`perplexity`**: Text perplexity scores
3. **`phd`**: Persistent homological dimensions (requires `--layer`)

## 📊 Classifier Types

- **`svm`**: Support Vector Machine (fast, good baseline)
- **`lr`**: Logistic Regression (fastest, interpretable)
- **`xgb`**: XGBoost (ensemble method)
- **`neural`**: Neural network (requires CUDA, most powerful)

---

## 🔄 Pooling strategies (embedding)

When using `--analysis_type embedding`, set `--pooling` to control how token embeddings are aggregated per layer:

- mean: Average over tokens. Stable baseline.
- max: Element-wise maximum over tokens. Highlights salient features.
- last: Use the last token’s vector. Works well for encoder-style sentence embeddings or causal models with EOS.
- attn_mean: Attention-weighted mean; gracefully falls back to uniform mean if attentions aren’t available (e.g., flash attention).
- mean_std: Concatenate mean and standard deviation over tokens. Doubles dimensionality (2 × hidden_size) and often improves robustness by capturing dispersion.
- statistical (aliases: covariance, cov): Covariance pooling. Flattens the upper triangle of the token-embedding covariance matrix for the chosen layer, capturing style/coherence patterns (useful for authorship-like signals).
  - Dimensionality warning: hidden_size × (hidden_size + 1) / 2. To keep things practical on large models, we cap via env var `COV_MAX_HIDDEN` (default 1024). If hidden_size > cap, it falls back to the diagonal (per-dimension variances).
  - Recommended layers: mid-to-deep layers to balance syntax/semantics (e.g., for 32-layer backbones, try 20–26 first).

Notes:
- All pooling works per-layer. Select the layer with `--layer`.
- For embeddings, the downstream detector applies StandardScaler + PCA(0.95) by default, so higher-dimensional poolings are reduced automatically.

---

## 1. ESA Challenge Submission (`main_submission_esa.py`)

Generate submissions for the ESA challenge using paired text data.

### Basic Usage

```bash
python scripts/main_submission_esa.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --analysis_type embedding \
  --layer 22 \
  --pooling mean \
  --classifier_type neural \
  --device cuda:0
```

### Examples by Analysis Type

#### Embedding Analysis
```bash
# Small model, middle layer
python scripts/main_submission_esa.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --layer 22 \
  --pooling mean \
  --classifier_type svm \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --device cuda:0

# Mean+Std pooling (richer first-order stats)
python scripts/main_submission_esa.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --layer 22 \
  --pooling mean_std \
  --classifier_type svm \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --device cuda:0

# Covariance pooling (second-order stats; may be heavy on large hidden sizes)
# Tip: set COV_MAX_HIDDEN=1024 (default) to cap feature size; otherwise the diagonal variance is used when hidden_size > cap.
python scripts/main_submission_esa.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --layer 22 \
  --pooling statistical \
  --classifier_type lr \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --device cuda:0

# Large model, final layers
python scripts/main_submission_esa.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --analysis_type embedding \
  --layer 30 \
  --pooling max \
  --classifier_type neural \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --batch_size 4 \
  --device cuda:0
```

#### Perplexity Analysis
```bash
python scripts/main_submission_esa.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type perplexity \
  --classifier_type lr \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --device cuda:0
```

#### PHD (Persistent Homological Dimension) Analysis
```bash
python scripts/main_submission_esa.py \
  --model_name "FacebookAI/roberta-base" \
  --analysis_type phd \
  --layer 6 \
  --classifier_type lr \
  --train_path data/data_esa/train \
  --train_labels_path data/data_esa/train.csv \
  --test_path data/data_esa/test \
  --device cuda:0
```

---

## 2. Mercor AI Challenge Submission (`main_submission_mercor.py`)

Generate submissions for Mercor AI cheating detection challenge.

### Basic Usage

```bash
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 21 \
  --pooling mean \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_mercor.csv \
  --device cuda:0
```

### Examples by Model Size

#### Small Models (Fast Training)
```bash
# DistilRoBERTa
python scripts/main_submission_mercor.py \
  --model_name "sentence-transformers/all-distilroberta-v1" \
  --analysis_type embedding \
  --layer 5 \
  --pooling mean \
  --classifier_type neural \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_distilroberta.csv \
  --device cuda:0 \
  --batch_size 16

# Multilingual model
python scripts/main_submission_mercor.py \
  --model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  --analysis_type embedding \
  --layer 4 \
  --pooling max \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_multilingual.csv \
  --device cuda:0 \
  --batch_size 12
```

#### Large Models (Better Performance)
```bash
# Qwen 8B model
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 30 \
  --pooling attn_mean \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_attnmean.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Qwen 8B with mean+std pooling
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 26 \
  --pooling mean_std \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_meanstd.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Qwen 8B with covariance pooling (consider mid/deep layers)
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 22 \
  --pooling statistical \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_cov.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Llama 8B model
python scripts/main_submission_mercor.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --analysis_type embedding \
  --layer 26 \
  --pooling attn_mean \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_llama8b.csv \
  --device cuda:1 \
  --batch_size 4
```

#### Different Analysis Types
```bash
# Perplexity-based detection
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type perplexity \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_perplexity.csv \
  --device cuda:1 \
  --batch_size 8

# PHD-based detection
python scripts/main_submission_mercor.py \
  --model_name "FacebookAI/roberta-base" \
  --analysis_type phd \
  --layer 8 \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_phd.csv \
  --device cuda:0 \
  --batch_size 8
```

---

## 3. Parameter Sweep (`parameter_sweep_mercor.py`)

Systematic hyperparameter optimization using k-fold cross-validation.

### Quick Testing Sweep
```bash
python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "sentence-transformers/all-distilroberta-v1" \
  --layers 3 5 \
  --pooling_types "mean" "mean_std" "statistical" \
  --use_pca_options true \
  --normalize_options true \
  --classifier_types svm \
  --cv_folds 3 \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/quick_sweep.json
```

### Comprehensive Sweep (Multiple Models)
```bash
nohup python > embed06B_binary.out scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-0.6B" \
  --layers 1 2 5 10 15 20 25 26 27 \
  --pooling_types "mean" "max" "last" \
  --use_pca_options true false \
  --normalize_options true false \
  --classifier_types lr svm xgb neural \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/comprehensive_sweep_all06b_binary.json \
  --memory_efficient &

nohup python > embed06B_binary.out scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-0.6B" \
  --layers 1 2 5 10 15 20 25 26 27 \
  --pooling_types "mean" "max" "last"  \
  --use_pca_options false \
  --normalize_options false \
  --classifier_types ocsvm iforest \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/sweep_binary.json &
```

### Background Execution (Long Sweeps)
```bash
nohup python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-8B" "meta-llama/Llama-3.1-8B" \
  --layers 1 5 10 15 20 25 30 -1 -2 -3 \
  --pooling_types "mean" "max" \
  --use_pca_options true false \
  --normalize_options true false \
  --classifier_types lr svm xgb neural \
  --cv_folds 5 \
  --device cuda:1 \
  --batch_size 4 \
  --output_path results/long_sweep.json > sweep.log 2>&1 &
```

### Focused Layer Search
```bash
# Find best layer for specific model
python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-8B" \
  --layers 1 5 10 15 20 25 30 32 34 35 -1 -2 -3 \
  --pooling_types "mean" \
  --use_pca_options true \
  --normalize_options false \
  --classifier_types lr \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/layer_search_qwen.json
```

---

## 4. Train and Save Models (`train_and_save_detector.py`)

Train models on one dataset and save for later cross-dataset evaluation.

### Train on Human vs AI Dataset
sentence-transformers/all-distilroberta-v1
Qwen/Qwen2.5-0.5B
Qwen/Qwen3-8B

```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 30 \
  --pooling mean \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --sample_frac 0.01 \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --log_memory
```

```bash
# Mean+Std pooling
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 20 \
  --pooling mean_std \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient
```

```bash
# Covariance pooling (second-order); recommended mid/deep layers
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 22 \
  --pooling statistical \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient
```

```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type iforest \
  --layer 22 \
  --pooling mean \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --sample_frac 0.01 \
  --log_memory


python scripts/train_and_save_detector.py \
  --analysis_type tfidf \
  --classifier_type svm \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --tfidf_max_features 50000 \
  --tfidf_ngram_min 1 \
  --tfidf_ngram_max 2 \
  --tfidf_min_df 2 \
  --tfidf_max_df 0.9 \
  --tfidf_stop_words english \
  --svd_components 500 \
  --validation_split 0.2 \
  --sample_frac 0.1
```

### Train on Mercor AI Dataset
```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 22 \
  --pooling mean \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --batch_size 8 \
  --device cuda:0
```

### Train on DAIGT v2 Dataset

You can train directly on the DAIGT v2 dataset stored under `data/daigt_v2/`. The loader accepts either a directory (it will auto-pick a `train_v2_*.csv`) or a direct path to a CSV. Expected columns are `text` and `label` (0=human, 1=AI).

```bash
# Directory input (auto-detect CSV inside)
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 20 \
  --pooling mean \
  --dataset_name daigtv2 \
  --train_data_path data/daigt_v2 \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --sample_frac 0.05

# Explicit CSV input
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 22 \
  --pooling mean \
  --dataset_name daigtv2 \
  --train_data_path data/daigt_v2/train_v2_drcat_02.csv \
  --batch_size 8 \
  --device cuda:0
```

Notes:
- DAIGT v2 loader uses fixed columns (`text`, `label`). You do not need to set `--text_column` or `--label_column` for this dataset.
- If you pass a generic CSV with `--dataset_name generic`, you can still specify `--text_column`/`--label_column` manually.


### Training with Sampling (Large Datasets)
```bash
# Train on 50% of data for faster experiments
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --classifier_type neural \
  --layer 30 \
  --pooling mean \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --sample_frac 0.5 \
  --batch_size 4 \
  --device cuda:1
```

### Different Analysis Types
```bash
# Perplexity-based model
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type perplexity \
  --classifier_type lr \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --device cuda:0

# PHD-based model
python scripts/train_and_save_detector.py \
  --model_name "FacebookAI/roberta-base" \
  --analysis_type phd \
  --layer 6 \
  --classifier_type lr \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --device cuda:0
```

---

## 5. Load and Evaluate Models (`load_and_evaluate.py`)

Evaluate saved models on different datasets.

### Basic Evaluation
```bash
python scripts/load_and_evaluate.py \
  --model_path saved_models/human_ai_Qwen_Qwen2.5-0.5B_embedding_layer22_mean_svm_metadata.pkl \
  --dataset_name mercor_ai \
  --data_path data/mercor-ai/train.csv \
  --device cuda:0 \
  --save_predictions \
  --output_dir evaluation_results
```

### Evaluate Multiple Models (Batch)
```bash
# Evaluate all saved models on mercor dataset
for model in saved_models/*.pkl; do
    echo "Evaluating $model"
    python scripts/load_and_evaluate.py \
        --model_path "$model" \
        --dataset_name mercor_ai \
        --data_path data/mercor-ai/train.csv \
        --device cuda:0 \
        --save_predictions \
        --output_dir evaluation_results
done
```

### Generic Dataset Evaluation
```bash
python scripts/load_and_evaluate.py \
  --model_path saved_models/custom_model.pkl \
  --dataset_name generic \
  --data_path data/custom_dataset.csv \
  --text_column "text_content" \
  --label_column "is_fake" \
  --device cuda:0
```

---

## 6. Cross-Dataset Evaluation (`cross_dataset_evaluation.py`)

Test model generalization across different datasets.

### Single Model, Multiple Datasets
```bash
python scripts/cross_dataset_evaluation.py \
  --model_path saved_models/daigtv2_Qwen_Qwen2.5-0.5B_embedding_layer20_mean_lr.pkl \
  --datasets mercor_ai:data/mercor-ai/train.csv \
  --device cuda:0 \
  --save_summary \
  --output_dir evaluation_results
```

### Multiple Models, Cross-Dataset
```bash
# Compare different models across datasets
for model in saved_models/human_ai_*.pkl; do
    echo "Cross-evaluating $model"
    python scripts/cross_dataset_evaluation.py \
        --model_path "$model" \
        --datasets mercor_ai:data/mercor-ai/train.csv \
        --device cuda:0 \
        --save_summary \
        --output_dir evaluation_results/cross_dataset
done
```

---

## 🔧 Common Parameters

### Model Selection
```bash
# Small/Fast models
--model_name "microsoft/deberta-v3-large"
--model_name "sentence-transformers/all-distilroberta-v1"
--model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
--model_name "Qwen/Qwen2.5-0.5B"

# Large/Powerful models  
--model_name "Qwen/Qwen3-8B"
--model_name "meta-llama/Llama-3.1-8B"
--model_name "Qwen/Qwen3-Embedding-8B"

# Specialized models
--model_name "FacebookAI/roberta-base"  # Good for PHD analysis
```

### Layer Selection Guidelines
```bash
# Small models (0.5B-1B parameters): layers 1-24
--layer 1     # Early features
--layer 12    # Middle layer
--layer 22    # Late layer
--layer -1    # Final layer

# Large models (7B-8B parameters): layers 1-32
--layer 1     # Early features  
--layer 15    # Middle layer
--layer 30    # Late layer
--layer -1    # Final layer
```

### Hardware Optimization
```bash
# GPU memory limited
--batch_size 4 --device cuda:0

# Multiple GPUs
--device cuda:1  # Use second GPU

# CPU only
--device cpu --batch_size 1

# Large memory available
--batch_size 16 --device cuda:0
```

### Output Management
```bash
# Background execution with logging
nohup python script.py [args] > output.log 2>&1 &

# Check background job
tail -f output.log

# Monitor GPU usage
nvidia-smi -l 1
```

---

## 📈 Results Analysis

### Parameter Sweep Results
```bash
# View best configurations
python -c "
import json
import pandas as pd
with open('results/sweep.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(df.nlargest(10, 'roc_auc_mean')[['model_name', 'layer', 'pooling', 'classifier_type', 'roc_auc_mean']])
"
```

### Submission Files
All submission scripts generate CSV files with the format:
- `submission_*.csv` for competition submissions
- Contains pair predictions in the required format

### Evaluation Results
- Saved in `evaluation_results/` directory
- Includes metrics (accuracy, F1, ROC-AUC) and predictions
- Cross-dataset summaries in CSV format

---

## 💡 Tips and Best Practices

### 1. Start Small
```bash
# Test with small model first
--model_name "sentence-transformers/all-distilroberta-v1" --batch_size 16
```

### 2. Parameter Sweeps
```bash
# Use fewer CV folds for initial exploration
--cv_folds 3

# Use more folds for final evaluation
--cv_folds 5
```

### 3. GPU Memory Management
```bash
# For large models, reduce batch size
--batch_size 4

# Clear memory between runs
python -c "import torch; torch.cuda.empty_cache()"
```

### 4. Reproducibility
```bash
# Always set random state
--random_state 42
```

### 5. Monitoring Long Jobs
```bash
# Use tmux/screen for long jobs
tmux new-session -d 'python script.py [args]'

# Monitor with tail
tail -f nohup.out
```