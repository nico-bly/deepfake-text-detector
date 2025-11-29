from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource
from pydantic import Field
from typing import List, Literal, Optional, Dict, Any
from enum import Enum
from functools import lru_cache

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


class CommaSeparatedListSource(EnvSettingsSource):
    """Custom env source that handles comma-separated lists before JSON parsing"""
    
    def prepare_field_value(self, field_name, field, value, value_is_complex):
        """Override to handle comma-separated strings for List fields"""
        # For ALLOWED_ORIGINS, parse as comma-separated string instead of JSON
        if field_name == "ALLOWED_ORIGINS" and isinstance(value, str):
            # Check if it looks like JSON (starts with [ or {)
            if not (value.strip().startswith('[') or value.strip().startswith('{')):
                # Parse as comma-separated
                return [origin.strip() for origin in value.split(',') if origin.strip()]
        
        # For other fields, use default behavior
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    # ===== CORS Configuration =====
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins (can be set as comma-separated string)"
    )
    
    # ===== Server Configuration =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    #API_BASE_URL: str = "http://localhost:8000"
    VITE_API_BASE_URL: str = "http://localhost:8000"
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
    SAVED_MODELS_DIR: str = "saved_models_prod"
    
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
    VITE_API_KEY: str = "change-me-in-production"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        json_schema_extra=None,
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.AVAILABLE_MODELS:
            self._init_default_models()
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Use custom source for environment variables"""
        _ = env_settings  # Use custom source instead of default env source
        return (
            init_settings,
            CommaSeparatedListSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
    
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