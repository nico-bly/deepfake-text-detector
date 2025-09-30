"""
Models module for ESA Challenge Fake Text Detection

Contains the main implementation modules:
- extractors: Embedding extraction from transformer models
- classifiers: Binary classification and outlier detection  
"""

from .extractors import EmbeddingExtractor
from .classifiers import OutlierDetections, TrajectoryClassifier, BinaryDetector

__all__ = [
    "EmbeddingExtractor",
    "OutlierDetections", 
    "TrajectoryClassifier",
    "BinaryDetector",
]