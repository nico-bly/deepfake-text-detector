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


class Settings(BaseSettings):
    """Main application settings"""
    
    # Server
    WORKERS: int = int(os.getenv("WORKERS", "2"))
    THREADS_PER_WORKER: int = int(os.getenv("THREADS_PER_WORKER", "2"))
    MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    
    # Inference backend strategy
    DEFAULT_INFERENCE_BACKEND: InferenceBackend = InferenceBackend.LOCAL
    ALLOW_CLIENT_SIDE_INFERENCE: bool = True
    ALLOW_MODAL_INFERENCE: bool = False  # Disable by default (requires API key)
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "https://your-frontend-domain.com"
    ]
    
    # Redis (for task queuing)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    
    # Modal configuration (if using Modal backend)
    MODAL_TOKEN_ID: str = os.getenv("MODAL_TOKEN_ID", "")
    MODAL_TOKEN_SECRET: str = os.getenv("MODAL_TOKEN_SECRET", "")
    MODAL_WORKSPACE: str = os.getenv("MODAL_WORKSPACE", "deepfake-detection")
    
    # Model configuration
    AVAILABLE_MODELS: dict = {
        # Tiny models (good for client-side)
        "tiny-tfidf": ModelConfig(
            model_id="embedding_A__mercor_ai",
            size_category="tiny",
            preferred_backend=InferenceBackend.CLIENT_SIDE,
            fallback_backends=[InferenceBackend.LOCAL]
        ),
        
        # Small models (good for VPS with modest resources)
        "small-perplexity": ModelConfig(
            model_id="perplexity_base",
            size_category="small",
            preferred_backend=InferenceBackend.LOCAL,
            fallback_backends=[InferenceBackend.MODAL]
        ),
        
        # Medium models (requires GPU or Modal)
        "medium-embedding": ModelConfig(
            model_id="embedding_qwen_22",
            size_category="medium",
            preferred_backend=InferenceBackend.LOCAL,
            fallback_backends=[InferenceBackend.MODAL]
        ),
        
        # Large models (best on Modal)
        "large-multilayer": ModelConfig(
            model_id="multilayer_ensemble",
            size_category="large",
            preferred_backend=InferenceBackend.MODAL,
            fallback_backends=[InferenceBackend.LOCAL]
        ),
    }
    
    # Hardware constraints on VPS
    MAX_VPS_INFERENCE_SIZE: Literal["tiny", "small", "medium"] = "medium"
    VPS_GPU_MEMORY_GB: int = int(os.getenv("VPS_GPU_MEMORY_GB", "0"))  # 0 = no GPU
    VPS_CPU_MEMORY_GB: int = int(os.getenv("VPS_CPU_MEMORY_GB", "4"))
    
    # Request constraints
    MAX_TEXT_LENGTH: int = 10000
    MIN_TEXT_LENGTH: int = 1
    MAX_BATCH_SIZE: int = 8
    REQUEST_TIMEOUT_SECONDS: int = 30
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }


def get_settings() -> Settings:
    """Get settings instance (singleton)"""
    return Settings()


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    settings = get_settings()
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_id}' not configured")
    return settings.AVAILABLE_MODELS[model_id]
