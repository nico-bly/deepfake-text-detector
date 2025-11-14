"""
Configuration management with support for multiple inference backends.
Supports: VPS-local, Modal, Client-side
"""
from pydantic_settings import BaseSettings
from typing import List, Literal
from enum import Enum
import os

class InferenceBackend(str, Enum):
    """Supported inference backends"""
    LOCAL = "local"          # Run on VPS
    MODAL = "modal"          # Run on Modal serverless
    CLIENT_SIDE = "client"   # Run on client (send metadata only)


class ModelConfig(BaseSettings):
    """Model size and backend selection"""
    model_id: str  # e.g., "embedding_A__mercor_ai" 
    size_category: Literal["tiny", "small", "medium", "large"]  # Size classification
    preferred_backend: InferenceBackend = InferenceBackend.LOCAL
    fallback_backends: List[InferenceBackend] = [InferenceBackend.LOCAL]
    
    model_config = {"from_attributes": True}


from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    # ===== Server Configuration =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"
    API_ROOT_PATH: str = ""
    
    # ===== CORS Configuration =====
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
    ]
    
    # ===== Server Settings =====
    PYTHONUNBUFFERED: bool = True
    LOG_LEVEL: str = "info"
    WORKERS: int = 4
    THREADS_PER_WORKER: int = 1
    
    # ===== Inference Backend Configuration =====
    DEFAULT_INFERENCE_BACKEND: str = "local"  # local, modal, client
    ALLOW_CLIENT_SIDE_INFERENCE: bool = True
    ALLOW_MODAL_INFERENCE: bool = False
    
    # ===== VPS Hardware Configuration =====
    VPS_GPU_MEMORY_GB: int = 0
    VPS_CPU_MEMORY_GB: int = 8
    MAX_VPS_INFERENCE_SIZE: str = "medium"  # tiny, small, medium
    
    # ===== Redis Configuration =====
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://redis:6379/0"
    MAX_QUEUE_SIZE: int = 100
    
    # ===== Model Configuration =====
    DEFAULT_MODEL_ID: str = "sentence-transformers_all-MiniLM-L6-v2"
    SAVED_MODELS_DIR: str = "saved_models"
    
    class Config:
        env_file = ".env"
        case_sensitive = False  # Allow lowercase env vars
        extra = "ignore"  # Ignore extra env vars (don't raise error)

@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    settings = get_settings()
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_id}' not configured")
    return settings.AVAILABLE_MODELS[model_id]
