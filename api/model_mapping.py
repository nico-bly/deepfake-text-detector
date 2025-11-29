"""
Manual mapping system for dataset + model combinations to backend model files.

Allows frontend to request (dataset + model_id) and backend resolves to the actual
model file location and backend type (VPS, Modal, or client-side).

Usage:
    from .model_mapping import resolve_model, get_model_info, list_all_mappings
    
    backend_model_id = resolve_model("human-ai-binary", "small-perplexity")
    # Returns: "embedding_A__dataset1" (or actual model filename)
    
    info = get_model_info("human-ai-binary", "small-perplexity")
    # Returns: {"backend_model": "...", "backend_type": "vps", "size_mb": 45.2, ...}
"""

from enum import Enum
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Where the model should run"""
    VPS = "vps"              # Run on server with GPU
    CLIENT_SIDE = "client"   # Download and run in browser
    MODAL = "modal"          # Run on Modal serverless


@dataclass
class ModelMapping:
    """Represents a single dataset+model combination"""
    frontend_model_id: str      # What frontend sends (e.g., "small-perplexity")
    backend_model_file: str     # Actual model filename in saved_models_prod (without .pkl)
    backend_type: BackendType   # Where it should run
    size_mb: Optional[float] = None  # Model file size
    description: str = ""
    metadata: Dict = None  # Custom metadata (analysis type, layer, etc)


# ============================================================================
# MANUAL MAPPING CONFIGURATION
# ============================================================================
# Define your dataset + model combinations here
# Format: (dataset_name, frontend_model_id) -> ModelMapping

DATASET_MODEL_MAPPING: Dict[Tuple[str, str], ModelMapping] = {
    # ========== human-ai-binary dataset ==========
    ("human-ai-binary", "qwen-0.5b"): ModelMapping(
        frontend_model_id="qwen-0.5b",
        backend_model_file="human_ai_Qwen_Qwen2.5-0.5B_embedding_layer16_last_l2norm_lr",
        backend_type=BackendType.VPS,
        size_mb=50.0,
        description="Qwen 0.5B embedding model",
        metadata={
            "analysis_type": "embedding",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 16,
        }
    ),
    ("human-ai-binary", "miniLM"): ModelMapping(
        frontend_model_id="miniLM",
        backend_model_file="human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_last_ocsvm",
        backend_type=BackendType.VPS,
        size_mb=27.0,
        description="Mini LM",
        metadata={
            "analysis_type": "embedding",
            "model_name": "MiniLM-L6-v2",
            "layer": 2,
        }
    )
    
    
    }

"""


DATASET_MODEL_MAPPING: Dict[Tuple[str, str], ModelMapping] = {
    # ========== human-ai-binary dataset ==========
    ("human-ai-binary", "qwen-0.5b"): ModelMapping(
        frontend_model_id="qwen-0.5b",
        backend_model_file="human_ai_Qwen_Qwen2.5-0.5B_embedding_layer16_last_l2norm_lr",
        backend_type=BackendType.VPS,
        size_mb=50.0,
        description="Qwen 0.5B embedding model",
        metadata={
            "analysis_type": "embedding",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 16,
        }
    ),
    
    ("human-ai-binary", "deberta-large"): ModelMapping(
        frontend_model_id="deberta-large",
        backend_model_file="human_ai_microsoft_deberta-v3-large_embedding_layer23_mean_std_l2norm_lr",
        backend_type=BackendType.VPS,
        size_mb=250.0,
        description="DeBERTa v3 Large embedding model",
        metadata={
            "analysis_type": "embedding",
            "model_name": "microsoft/deberta-v3-large",
            "layer": 23,
        }
    ),
    
    ("human-ai-binary", "minilm-ocsvm"): ModelMapping(
        frontend_model_id="minilm-ocsvm",
        backend_model_file="human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_last_ocsvm",
        backend_type=BackendType.VPS,
        size_mb=30.0,
        description="MiniLM with One-Class SVM (anomaly detection)",
        metadata={
            "analysis_type": "embedding",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "layer": 2,
        }
    ),
    
    ("human-ai-binary", "minilm-ocsvm-std"): ModelMapping(
        frontend_model_id="minilm-ocsvm-std",
        backend_model_file="human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_mean_std_l2norm_ocsvm",
        backend_type=BackendType.VPS,
        size_mb=30.0,
        description="MiniLM with standardized features and One-Class SVM",
        metadata={
            "analysis_type": "embedding",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "layer": 2,
        }
    ),
    
    # ========== human-ai-anomaly dataset ==========
    ("human-ai-anomaly", "small-perplexity"): ModelMapping(
        frontend_model_id="small-perplexity",
        backend_model_file="embedding_A__human_ai_anomaly_small",
        backend_type=BackendType.VPS,
        size_mb=45.2,
        description="Small perplexity model for human-ai-anomaly dataset",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 16,
        }
    ),
    
    ("human-ai-anomaly", "medium-perplexity"): ModelMapping(
        frontend_model_id="medium-perplexity",
        backend_model_file="embedding_B__human_ai_anomaly_medium",
        backend_type=BackendType.VPS,
        size_mb=120.5,
        description="Medium perplexity model for human-ai-anomaly dataset",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-1.5B",
            "layer": 20,
        }
    ),
    
    ("human-ai-anomaly", "qwen-0.5b"): ModelMapping(
        frontend_model_id="qwen-0.5b",
        backend_model_file="detector_qwen05b__human_ai_anomaly",
        backend_type=BackendType.CLIENT_SIDE,
        size_mb=50.0,
        description="Qwen 0.5B model for human-ai-anomaly dataset",
        metadata={
            "analysis_type": "embedding",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 18,
        }
    ),
    
    # ========== arxiv dataset ==========
    ("arxiv", "small-perplexity"): ModelMapping(
        frontend_model_id="small-perplexity",
        backend_model_file="embedding_A__arxiv_small",
        backend_type=BackendType.VPS,
        size_mb=45.2,
        description="Small perplexity model for arxiv dataset",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 16,
        }
    ),
    
    ("arxiv", "medium-perplexity"): ModelMapping(
        frontend_model_id="medium-perplexity",
        backend_model_file="embedding_B__arxiv_medium",
        backend_type=BackendType.VPS,
        size_mb=120.5,
        description="Medium perplexity model for arxiv dataset",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-1.5B",
            "layer": 20,
        }
    ),
    
    # ========== fakenews dataset ==========
    ("fakenews", "small-perplexity"): ModelMapping(
        frontend_model_id="small-perplexity",
        backend_model_file="embedding_A__fakenews_small",
        backend_type=BackendType.VPS,
        size_mb=45.2,
        description="Small perplexity model for fakenews dataset",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "layer": 16,
        }
    ),
    
    ("fakenews", "large-perplexity"): ModelMapping(
        frontend_model_id="large-perplexity",
        backend_model_file="embedding_C__fakenews_large",
        backend_type=BackendType.MODAL,
        size_mb=450.0,
        description="Large perplexity model for fakenews - runs on Modal",
        metadata={
            "analysis_type": "perplexity",
            "model_name": "Qwen/Qwen2.5-7B",
            "layer": 24,
        }
    ),
}
"""


# ============================================================================
# FUNCTIONS
# ============================================================================

def resolve_model(dataset: str, frontend_model_id: str) -> str:
    """
    Resolve frontend (dataset, model_id) to backend model filename.
    
    Args:
        dataset: Dataset name (e.g., "human-ai-binary")
        frontend_model_id: Frontend model ID (e.g., "small-perplexity")
    
    Returns:
        Backend model filename (without .pkl extension)
    
    Raises:
        KeyError: If combination not found
    
    Example:
        >>> resolve_model("human-ai-binary", "small-perplexity")
        "embedding_A__human_ai_binary_small"
    """
    key = (dataset, frontend_model_id)
    if key not in DATASET_MODEL_MAPPING:
        available = list_all_mappings(dataset)
        raise KeyError(
            f"Model not found: dataset='{dataset}', model_id='{frontend_model_id}'. "
            f"Available models for '{dataset}': {available}"
        )
    return DATASET_MODEL_MAPPING[key].backend_model_file


def get_model_info(dataset: str, frontend_model_id: str) -> Dict:
    """
    Get full information about a dataset+model combination.
    
    Returns:
        Dict with backend_model, backend_type, size_mb, description, metadata
    
    Example:
        >>> info = get_model_info("human-ai-binary", "small-perplexity")
        >>> info["backend_type"]
        "vps"
    """
    key = (dataset, frontend_model_id)
    if key not in DATASET_MODEL_MAPPING:
        available = list_all_mappings(dataset)
        raise KeyError(
            f"Model not found: dataset='{dataset}', model_id='{frontend_model_id}'. "
            f"Available models for '{dataset}': {available}"
        )
    
    mapping = DATASET_MODEL_MAPPING[key]
    return {
        "frontend_model_id": mapping.frontend_model_id,
        "backend_model_file": mapping.backend_model_file,
        "backend_type": mapping.backend_type.value,
        "size_mb": mapping.size_mb,
        "description": mapping.description,
        "metadata": mapping.metadata or {}
    }


def list_all_mappings(dataset: Optional[str] = None) -> List[str]:
    """
    List all available model combinations.
    
    Args:
        dataset: Optional - filter by dataset. If None, returns all datasets.
    
    Returns:
        List of model IDs (or list of tuples if dataset=None)
    
    Example:
        >>> list_all_mappings("human-ai-binary")
        ["small-perplexity", "medium-perplexity", "large-perplexity", "qwen-0.5b", "qwen-8b"]
        
        >>> list_all_mappings()
        [("human-ai-binary", "small-perplexity"), ("human-ai-binary", "medium-perplexity"), ...]
    """
    if dataset is None:
        return sorted(DATASET_MODEL_MAPPING.keys())
    else:
        models = [model_id for (ds, model_id) in DATASET_MODEL_MAPPING.keys() if ds == dataset]
        return sorted(set(models))


def get_available_datasets() -> List[str]:
    """
    Get list of all available datasets.
    
    Returns:
        Sorted list of unique dataset names
    
    Example:
        >>> get_available_datasets()
        ["arxiv", "fakenews", "human-ai-anomaly", "human-ai-binary"]
    """
    datasets = {dataset for (dataset, _) in DATASET_MODEL_MAPPING.keys()}
    return sorted(datasets)


def validate_combination(dataset: str, frontend_model_id: str) -> bool:
    """
    Check if a dataset+model combination is valid.
    
    Returns:
        True if combination exists, False otherwise
    """
    return (dataset, frontend_model_id) in DATASET_MODEL_MAPPING


def get_backend_type(dataset: str, frontend_model_id: str) -> BackendType:
    """
    Get the backend type for a specific model.
    
    Returns:
        BackendType enum (VPS, CLIENT_SIDE, or MODAL)
    """
    mapping = DATASET_MODEL_MAPPING[(dataset, frontend_model_id)]
    return mapping.backend_type


def get_models_by_backend(backend_type: BackendType) -> Dict[str, List[str]]:
    """
    Group models by backend type.
    
    Returns:
        Dict mapping dataset -> list of models for that backend
    
    Example:
        >>> get_models_by_backend(BackendType.VPS)
        {
            "human-ai-binary": ["small-perplexity", "medium-perplexity", "qwen-8b"],
            "arxiv": ["small-perplexity", "medium-perplexity"],
            ...
        }
    """
    result = {}
    for (dataset, model_id), mapping in DATASET_MODEL_MAPPING.items():
        if mapping.backend_type == backend_type:
            if dataset not in result:
                result[dataset] = []
            result[dataset].append(model_id)
    
    # Sort model lists
    for dataset in result:
        result[dataset] = sorted(result[dataset])
    
    return result


# ============================================================================
# PRINT HELPER
# ============================================================================

def print_mapping_summary():
    """Print a summary of all mappings for debugging."""
    print("\n" + "=" * 80)
    print("MODEL MAPPING SUMMARY")
    print("=" * 80)
    
    for dataset in get_available_datasets():
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        for model_id in list_all_mappings(dataset):
            info = get_model_info(dataset, model_id)
            print(f"  {model_id:20} -> {info['backend_model_file']:40} ({info['backend_type']})")
    
    print("\n" + "=" * 80)
    print("BACKEND DISTRIBUTION")
    print("=" * 80)
    
    for backend in BackendType:
        models = get_models_by_backend(backend)
        print(f"\n{backend.value.upper()}:")
        for dataset, model_ids in models.items():
            print(f"  {dataset}: {', '.join(model_ids)}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Test the mapping
    print_mapping_summary()
