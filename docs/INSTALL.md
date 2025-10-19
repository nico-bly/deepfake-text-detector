# Installation and Usage Guide

## Quick Start Installation

### Option 1: Development Installation (Recommended)
```bash
# Clone the repository
cd /home/infres/billy-22/projets/esa_challenge_kaggle/esa-challenge-fake-texts

# Install in development mode (editable)
pip install -e .

# Or install with optional dependencies
pip install -e .[dev,notebooks,gpu]
```

### Option 2: Standard Installation
```bash
# Install from source
pip install .

# Or install with specific optional dependencies
pip install .[notebooks]
```

### Option 3: Install from Requirements File
```bash
# Install just the dependencies
pip install -r requirements.txt
```

## Package Structure

Your package will be organized as:
```
esa-fake-text-detector/
├── setup.py                 # Main setup configuration
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Dependencies
├── MANIFEST.in             # Files to include in package
├── README.md               # Documentation
├── __init__.py             # Package initialization
├── scripts/                # Main modules
│   ├── __init__.py
│   ├── models.py           # EmbeddingExtractor class
│   ├── classifiers.py      # OutlierDetections, TrajectoryClassifier
│   ├── utils.py            # Data loading utilities
│   └── main.py             # Main execution script
├── notebooks/              # Jupyter notebooks
│   └── *.ipynb
└── data/                   # Data files (CSV only)
    └── *.csv
```

## Usage Examples

### As a Python Package
```python
# Import the package
from scripts.models import EmbeddingExtractor
from scripts.classifiers import OutlierDetections
from scripts.utils import read_texts_from_dir

# Or use the convenience imports
from esa_fake_text_detector import EmbeddingExtractor, OutlierDetections

# Use the classes
extractor = EmbeddingExtractor("sentence-transformers/all-distilroberta-v1")
detector = OutlierDetections(detector_type="svm_binary")
```

### Command Line Usage
```bash
# Run the main detector
esa-fake-detector

# Process data
esa-process-data
```

### In Jupyter Notebooks
```python
# After installation, you can import anywhere
import sys
sys.path.append('/path/to/your/package')

from scripts import EmbeddingExtractor, OutlierDetections
```

## Building and Distributing

### Create Source Distribution
```bash
python setup.py sdist
```

### Create Wheel Distribution  
```bash
python setup.py bdist_wheel
```

### Build with Modern Tools
```bash
pip install build
python -m build
```

### Install from Built Package
```bash
pip install dist/esa_fake_text_detector-1.0.0-py3-none-any.whl
```

## Development Setup

### Install Development Dependencies
```bash
pip install -e .[dev]
```

### Code Formatting
```bash
black scripts/
```

### Linting
```bash
flake8 scripts/
```

### Testing (if you add tests)
```bash
pytest
```

## Troubleshooting

### Import Issues
If you get import errors, ensure you've installed the package:
```bash
pip install -e .
```

### Missing Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

### CUDA Issues
For GPU support:
```bash
pip install .[gpu]
```

### Jupyter Notebook Issues
For notebook support:
```bash
pip install .[notebooks]
jupyter lab
```