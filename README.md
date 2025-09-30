# ESA Challenge: Fake Text Detection

Embedding-based classification to detect AI-generated fake texts from authentic ones. Uses sentence transformers and small language models for binary classification on text pairs. Addresses data poisoning and hallucination detection challenges.

## 🎯 Overview

This package provides a comprehensive solution for detecting AI-generated (fake) texts using **mean embeddings from multiple transformer model layers**. The approach uses ensemble methods to achieve robust classification performance.

### Key Features

- **Multi-Model Ensemble**: Combines predictions from multiple transformer models
- **Layer-Specific Analysis**: Extracts embeddings from different layers to capture various linguistic patterns
- **Binary Classification**: Uses SVM classifiers to distinguish real vs. fake texts
- **Confidence Scoring**: Provides uncertainty estimates for each prediction
- **Jupyter Integration**: Ready-to-use notebooks for experimentation
- **Modular Design**: Easy to extend with new models and classifiers

## 🚀 Quick Start

### Installation

```bash
# Development installation (recommended)
pip install -e .

# With optional dependencies
pip install -e .[notebooks,gpu,dev]
```

### Basic Usage

```python
from scripts.models import EmbeddingExtractor
from scripts.classifiers import OutlierDetections
from scripts.utils import read_texts_from_dir

# Load embedding extractor
extractor = EmbeddingExtractor("sentence-transformers/all-distilroberta-v1")

# Create binary classifier
detector = OutlierDetections(detector_type="svm_binary")

# Load and process data
df_test = read_texts_from_dir("data/test/")
```

## 📁 Project Structure

```
esa-challenge-fake-texts/
├── scripts/              # Core modules
│   ├── models.py         # EmbeddingExtractor class
│   ├── classifiers.py    # OutlierDetections, TrajectoryClassifier
│   ├── utils.py          # Data loading utilities
│   └── main.py           # Main execution script
├── notebooks/            # Jupyter notebooks
│   └── kaggle_competition_submission.ipynb
├── data/                 # Dataset files
└── requirements.txt      # Dependencies
```

## 🧠 How It Works

The solution uses **mean embeddings from different transformer layers** to detect fake texts:

1. **Embedding Extraction**: Extract mean-pooled embeddings from multiple layers
2. **Pattern Learning**: Train SVM classifiers on real vs. fake text patterns  
3. **Ensemble Prediction**: Combine multiple models using majority voting
4. **Confidence Analysis**: Generate uncertainty estimates

### Supported Models

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/all-distilroberta-v1`
- `meta-llama/Llama-3.1-8B`
- `Qwen/Qwen3-8B`

## 📊 Results

The ensemble approach achieves robust performance by combining:
- **Multiple transformer architectures** for diverse perspectives
- **Different layer depths** for multi-level linguistic analysis
- **Confidence scoring** for uncertainty quantification

## 🛠️ Development

### Setup Development Environment

```bash
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black scripts/
flake8 scripts/
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- scikit-learn 1.0+
- See `requirements.txt` for full dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
