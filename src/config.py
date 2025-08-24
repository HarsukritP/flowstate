"""
FlowState Configuration Management
Production and development environment settings
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./flowstate.db", env="DATABASE_URL")
    
    # Redis Configuration (for caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Spotify API Configuration
    spotify_client_id: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_SECRET")
    spotify_redirect_uri: str = Field(default="http://localhost:8000/callback", env="SPOTIFY_REDIRECT_URI")
    
    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # ML Model Configuration
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    audio_sample_rate: int = Field(default=22050, env="AUDIO_SAMPLE_RATE")
    max_audio_duration: int = Field(default=30, env="MAX_AUDIO_DURATION")  # seconds
    
    # Performance Configuration
    queue_generation_timeout: int = Field(default=10, env="QUEUE_GENERATION_TIMEOUT")  # seconds
    max_queue_length: int = Field(default=50, env="MAX_QUEUE_LENGTH")
    default_queue_length: int = Field(default=10, env="DEFAULT_QUEUE_LENGTH")
    
    # CORS Configuration
    allowed_origins: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security
    secret_key: str = Field(default="flowstate-dev-key-change-in-production", env="SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting"""
        if self.database_url.startswith("postgres://"):
            # Fix for SQLAlchemy 2.0 compatibility
            return self.database_url.replace("postgres://", "postgresql://", 1)
        return self.database_url

# Global settings instance
settings = Settings()

# Environment-specific configurations
def get_cors_origins() -> list:
    """Get CORS origins based on environment"""
    if settings.is_production:
        return [
            "https://open.spotify.com",
            "https://accounts.spotify.com",
            "https://api.spotify.com",
            # Add your deployed frontend URL here
        ]
    else:
        return ["*"]  # Allow all origins in development

def get_logging_config() -> dict:
    """Get logging configuration"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": settings.log_file or "flowstate.log",
                "formatter": "detailed",
                "level": "DEBUG",
            },
        },
        "loggers": {
            "flowstate": {
                "handlers": ["console", "file"] if settings.log_file else ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
