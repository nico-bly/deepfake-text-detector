from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Literal, Optional, Dict, Any
from enum import Enum
from functools import lru_cache
import json

class InferenceBackend(str, Enum):
    LOCAL = "local"
    MODAL = "modal"
    CLIENT_SIDE = "client"


class ModelConfig:
    def __init__(
        self,
        model_id: str,
        size_category: Literal["tiny", "small", "medium", "large"],
        preferred_backend: InferenceBackend = InferenceBackend.LOCAL,
        fallback_backends: Optional[List[InferenceBackend]] = None,
    ):
        self.model_id = model_id
        self.size_category = size_category
        self.preferred_backend = preferred_backend
        self.fallback_backends = fallback_backends or [InferenceBackend.LOCAL]


class Settings(BaseSettings):
    # ===== CORS Configuration =====
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins (can be set as comma-separated string)"
    )
    
    # ===== Server Configuration =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"
    API_ROOT_PATH: str = ""
    
    # ===== Server Settings =====
    PYTHONUNBUFFERED: bool = True
    LOG_LEVEL: str = "info"
    WORKERS: int = 4
    THREADS_PER_WORKER: int = 1
    
    # ===== Inference Backend Configuration =====
    DEFAULT_INFERENCE_BACKEND: str = "local"
    ALLOW_CLIENT_SIDE_INFERENCE: bool = True
    ALLOW_MODAL_INFERENCE: bool = False
    
    # ===== VPS Hardware Configuration =====
    VPS_GPU_MEMORY_GB: int = 0
    VPS_CPU_MEMORY_GB: int = 8
    MAX_VPS_INFERENCE_SIZE: str = "medium"
    
    # ===== Redis Configuration =====
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://redis:6379/0"
    MAX_QUEUE_SIZE: int = 100
    
    # ===== Model Configuration =====
    DEFAULT_MODEL_ID: str = "sentence-transformers_all-MiniLM-L6-v2"
    SAVED_MODELS_DIR: str = "saved_models"
    
    # ===== Modal Configuration =====
    MODAL_ENABLED: bool = False
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None
    MODAL_WORKSPACE: Optional[str] = None
    
    # ===== Text Validation =====
    MIN_TEXT_LENGTH: int = 1
    MAX_TEXT_LENGTH: int = 10000
    
    # ===== Request/Response Configuration =====
    MAX_BATCH_SIZE: int = 100
    REQUEST_TIMEOUT_SECONDS: int = 300
    ENABLE_BATCH_ENDPOINT: bool = True
    
    # ===== Available Models Registry =====
    AVAILABLE_MODELS: Dict[str, ModelConfig] = {}
    
    # ===== Security - API Key =====
    API_KEY: str = "change-me-in-production"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        json_schema_extra={"ALLOWED_ORIGINS": "comma-separated string"}
    )
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse ALLOWED_ORIGINS from comma-separated string to list"""
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            # Remove any surrounding quotes
            v = v.strip().strip('"').strip("'")
            # Split by comma and clean up
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins
        
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.AVAILABLE_MODELS:
            self._init_default_models()
    
    def _init_default_models(self):
        """Initialize default model configurations"""
        self.AVAILABLE_MODELS = {
            "sentence-transformers_all-MiniLM-L6-v2": ModelConfig(
                model_id="sentence-transformers_all-MiniLM-L6-v2",
                size_category="small",
                preferred_backend=InferenceBackend.LOCAL,
                fallback_backends=[InferenceBackend.LOCAL, InferenceBackend.CLIENT_SIDE],
            ),
        }


@lru_cache()
def get_settings() -> Settings:
    """Get singleton settings instance"""
    return Settings()


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    settings = get_settings()
    if model_id not in settings.AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_id}' not configured")
    return settings.AVAILABLE_MODELS[model_id]