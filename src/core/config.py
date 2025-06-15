"""
Configuration management for TradeKnowledge
"""

from pathlib import Path
from typing import Optional, List
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def _safe_int(value: str, default: int) -> int:
    """Safely convert string to int with fallback"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

class AppConfig(BaseModel):
    name: str = "TradeKnowledge"
    version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

class ChromaConfig(BaseModel):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "trading_books"

class QdrantConfig(BaseModel):
    """Qdrant configuration for vector storage"""
    host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = Field(default_factory=lambda: _safe_int(os.getenv("QDRANT_PORT", "6333"), 6333))
    collection_name: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "tradeknowledge"))
    use_grpc: bool = Field(default_factory=lambda: os.getenv("QDRANT_USE_GRPC", "false").lower() == "true")
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    https: bool = Field(default_factory=lambda: os.getenv("QDRANT_HTTPS", "false").lower() == "true")
    prefer_grpc: bool = Field(default_factory=lambda: os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true")
    
    @property
    def url(self) -> str:
        """Get Qdrant connection URL"""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"

class SQLiteConfig(BaseModel):
    path: str = "./data/knowledge.db"
    fts_version: str = "fts5"

class DatabaseConfig(BaseModel):
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

class EmbeddingConfig(BaseModel):
    """Embedding configuration for local setup"""
    model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "nomic-embed-text"))
    dimension: int = Field(default_factory=lambda: _safe_int(os.getenv("EMBEDDING_DIMENSION", "384"), 384))
    batch_size: int = Field(default_factory=lambda: _safe_int(os.getenv("EMBEDDING_BATCH_SIZE", "32"), 32))
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    timeout: int = Field(default_factory=lambda: _safe_int(os.getenv("OLLAMA_TIMEOUT", "30"), 30))
    cache_embeddings: bool = True
    max_concurrent_requests: int = Field(default_factory=lambda: _safe_int(os.getenv("EMBEDDING_CONCURRENCY", "5"), 5))
    
    @field_validator('dimension')
    @classmethod
    def dimension_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('dimension must be positive')
        return v
    
    @field_validator('batch_size')
    @classmethod
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v

class SearchConfig(BaseModel):
    default_results: int = 10
    max_results: int = 50
    min_score: float = 0.7
    hybrid_weight: float = 0.7

class RedisConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: _safe_int(os.getenv("REDIS_PORT", "6379"), 6379))
    db: int = 0
    ttl: int = 3600

class MemoryCacheConfig(BaseModel):
    max_size: int = 1000
    ttl: int = 600

class CacheConfig(BaseModel):
    redis: RedisConfig = Field(default_factory=RedisConfig)
    memory: MemoryCacheConfig = Field(default_factory=MemoryCacheConfig)

class PerformanceConfig(BaseModel):
    use_cpp_extensions: bool = True
    thread_pool_size: int = 8
    batch_processing: bool = True

class AuthConfig(BaseModel):
    """Authentication configuration"""
    secret_key: str = Field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production"))
    algorithm: str = "HS256"
    token_expiry_hours: int = Field(default_factory=lambda: _safe_int(os.getenv("JWT_EXPIRY_HOURS", "24"), 24))
    enable_registration: bool = Field(default_factory=lambda: os.getenv("ENABLE_REGISTRATION", "false").lower() == "true")
    min_password_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15

class ApiConfig(BaseModel):
    """API server configuration"""
    auth: AuthConfig = Field(default_factory=AuthConfig)
    cors_origins: List[str] = Field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    rate_limit_per_minute: int = Field(default_factory=lambda: _safe_int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"), 60))
    max_file_size_mb: int = Field(default_factory=lambda: _safe_int(os.getenv("MAX_FILE_SIZE_MB", "100"), 100))
    enable_docs: bool = Field(default_factory=lambda: os.getenv("ENABLE_API_DOCS", "true").lower() == "true")
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    request_timeout: int = Field(default_factory=lambda: _safe_int(os.getenv("REQUEST_TIMEOUT", "300"), 300))
    max_concurrent_uploads: int = Field(default_factory=lambda: _safe_int(os.getenv("MAX_CONCURRENT_UPLOADS", "3"), 3))

class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path("config/config.yaml")
    
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
            return Config(**config_dict)
        else:
            # Return default config if file doesn't exist
            return Config()
    except (yaml.YAMLError, IOError, OSError) as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default configuration")
        return Config()

# Singleton instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get configuration singleton"""
    global _config
    if _config is None:
        _config = load_config()
    return _config