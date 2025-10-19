# README_NEW.md - Clean Structure Version

See this file for the updated, cleaned structure after running cleanup_repo.sh

# Deepfake Text Detector

ML service for detecting AI-generated text using pre-trained transformers and lightweight classifiers.

## 🏗️ Repository Structure

```
deepfake-text-detector/
├── models/                    # Core ML model code
│   ├── classifiers.py        # Classifier implementations
│   ├── extractors.py         # Feature extraction models
│   └── text_features.py      # Text feature engineering
│
├── scripts/                   # Training & experiments
│   ├── train_classifier.py   # Model training pipeline
│   └── parameter_sweep.py    # Hyperparameter optimization
│
├── data/                      # Training datasets
│   ├── daigt_v2/             # DAIGT dataset
│   ├── data_esa/             # ESA challenge data
│   └── .gitignore            # Don't commit large data
│
├── saved_models/              # Trained model artifacts
│   └── .gitignore            # Don't commit model files
│
├── modal_deployment/          # Cloud deployment (Modal.com)
│   ├── app.py                # Modal serverless function
│   └── README.md             # Deployment instructions
│
├── tests/                     # Unit tests
│   └── test_models.py        # Model testing
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md       # System architecture
│   └── DEPLOYMENT.md         # Deployment guide
│
├── archive/                   # Old competition outputs (gitignored)
│   ├── competition_outputs/  # Kaggle submissions
│   ├── experiment_results/   # Parameter sweep results
│   └── old_notebooks/        # Competition notebooks
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # Main documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd deepfake-text-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Classifier

```bash
python scripts/train_classifier.py \
    --model all-distilroberta-v1 \
    --data data/daigt_v2/ \
    --output saved_models/classifier.pkl
```

### Running Inference

```python
from models.classifiers import TextClassifier

# Load trained model
classifier = TextClassifier.load('saved_models/classifier.pkl')

# Predict
text = "Your text here..."
result = classifier.predict(text)
print(f"AI-generated: {result['is_ai']}, Confidence: {result['confidence']}")
```

## 📦 Deployment Options

### Option 1: Single Service (Render - FREE) ⭐
- **Use case**: Production API with good accuracy
- **Model**: Small embedding model (all-distilroberta-v1, 250MB)
- **RAM**: 400MB (fits free tier)
- **Speed**: ~300ms per request
- **Cost**: FREE
- **Setup**: See `docs/DEPLOYMENT.md`

### Option 2: Modal.com (Scalable)
- **Use case**: Highest accuracy predictions
- **Model**: Large models (Llama-3.1-8B, Qwen-8B)
- **RAM**: 20GB (GPU-backed)
- **Speed**: ~1s per request
- **Cost**: ~$0.10 per 1,000 requests
- **Setup**: See `modal_deployment/README.md`

### Option 3: Hybrid Approach
- Route fast requests → Render (small model)
- Route accuracy-critical → Modal (large model)
- Best of both worlds!

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py -v

# With coverage
pytest --cov=models tests/
```

## 📊 Model Performance

| Model | Accuracy | RAM | Speed | Cost |
|-------|----------|-----|-------|------|
| all-distilroberta-v1 | 85% | 400MB | 300ms | FREE |
| paraphrase-mpnet | 87% | 500MB | 400ms | FREE |
| Llama-3.1-8B | 92% | 20GB | 1s | $0.10/1k |
| Qwen-8B | 93% | 20GB | 1s | $0.10/1k |

**Recommendation**: Start with `all-distilroberta-v1` on Render. It's 85% as accurate, 100% free, and 3x faster!

## 🔧 Development Workflow

### 1. Train on GPU (One-time)
```bash
# Use your GPU machine for training
python scripts/train_classifier.py \
    --model all-distilroberta-v1 \
    --epochs 10 \
    --output saved_models/classifier.pkl
```

### 2. Deploy Trained Model
```bash
# Deploy to Render (includes pre-trained classifier)
# Model is tiny (1-10MB), loads instantly
# No GPU needed for inference!
```

### 3. Use in Production
```python
# Fast inference with small model + trained classifier
result = classifier.predict(text)  # ~300ms
```

## 📁 Archive Folder

After running `cleanup_repo.sh`, old competition outputs are in `archive/`:
- `competition_outputs/` - 50+ Kaggle submission CSVs
- `experiment_results/` - Parameter sweep results
- `old_notebooks/` - Competition notebooks

**These are gitignored** but safe on your local machine for reference.

## 🛠️ Cleanup Script Usage

```bash
cd deepfake-text-detector
chmod +x cleanup_repo.sh
./cleanup_repo.sh
```

This will:
1. ✅ Create `archive/` folder structure
2. ✅ Move competition outputs (out_llms/)
3. ✅ Move experiment results
4. ✅ Organize notebooks
5. ✅ Move docs to `docs/` folder
6. ✅ Move tests to `tests/` folder
7. ✅ Keep everything safe (no deletions!)

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Add tests
4. Submit PR

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- DAIGT competition dataset
- ESA challenge organizers
- Sentence Transformers library
- Modal.com for serverless ML
