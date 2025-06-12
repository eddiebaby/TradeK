"""
Configuration management for TradeKnowledge
"""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

class SQLiteConfig(BaseModel):
    path: str = "./data/knowledge.db"
    fts_version: str = "fts5"

class DatabaseConfig(BaseModel):
    chroma: ChromaConfig
    sqlite: SQLiteConfig

class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-ada-002"
    batch_size: int = 100
    cache_embeddings: bool = True

class SearchConfig(BaseModel):
    default_results: int = 10
    max_results: int = 50
    min_score: float = 0.7
    hybrid_weight: float = 0.7

class RedisConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = 0
    ttl: int = 3600

class MemoryCacheConfig(BaseModel):
    max_size: int = 1000
    ttl: int = 600

class CacheConfig(BaseModel):
    redis: RedisConfig
    memory: MemoryCacheConfig

class PerformanceConfig(BaseModel):
    use_cpp_extensions: bool = True
    thread_pool_size: int = 8
    batch_processing: bool = True

class Config(BaseModel):
    app: AppConfig
    server: ServerConfig
    database: DatabaseConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    search: SearchConfig
    cache: CacheConfig
    performance: PerformanceConfig

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path("config/config.yaml")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)

# Singleton instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get configuration singleton"""
    global _config
    if _config is None:
        _config = load_config()
    return _config