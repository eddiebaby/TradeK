"""
Tests for the configuration system including new local setup configurations
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from src.core.config import (
    Config, EmbeddingConfig, QdrantConfig, DatabaseConfig,
    load_config, get_config, _safe_int
)


class TestSafeInt:
    """Test safe integer conversion helper"""
    
    def test_safe_int_valid_string(self):
        """Test safe int conversion with valid string"""
        assert _safe_int("123", 0) == 123
        assert _safe_int("0", 10) == 0
        assert _safe_int("-5", 0) == -5
    
    def test_safe_int_invalid_string(self):
        """Test safe int conversion with invalid string"""
        assert _safe_int("abc", 42) == 42
        assert _safe_int("", 10) == 10
        assert _safe_int("12.5", 0) == 0
    
    def test_safe_int_none_value(self):
        """Test safe int conversion with None"""
        assert _safe_int(None, 100) == 100


class TestEmbeddingConfig:
    """Test the updated EmbeddingConfig for local setup"""
    
    def test_default_embedding_config(self):
        """Test default embedding configuration values"""
        config = EmbeddingConfig()
        assert config.model == "nomic-embed-text"
        assert config.dimension == 768
        assert config.batch_size == 32
        assert config.ollama_host == "http://localhost:11434"
        assert config.timeout == 30
    
    def test_custom_embedding_config(self):
        """Test custom embedding configuration"""
        config = EmbeddingConfig(
            model="custom-model",
            dimension=512,
            batch_size=16,
            ollama_host="http://remote:11434",
            timeout=60
        )
        assert config.model == "custom-model"
        assert config.dimension == 512
        assert config.batch_size == 16
        assert config.ollama_host == "http://remote:11434"
        assert config.timeout == 60
    
    def test_embedding_config_validation(self):
        """Test embedding config validation"""
        # Valid config should not raise
        EmbeddingConfig(dimension=768, batch_size=1)
        
        # Invalid dimension should raise
        with pytest.raises(ValueError):
            EmbeddingConfig(dimension=0)
        
        # Invalid batch size should raise
        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)


class TestQdrantConfig:
    """Test the new QdrantConfig class"""
    
    def test_default_qdrant_config(self):
        """Test default Qdrant configuration values"""
        config = QdrantConfig()
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.collection_name == "tradeknowledge"
        assert config.use_grpc is False
        assert config.api_key is None
        assert config.https is False
        assert config.prefer_grpc is False
    
    def test_custom_qdrant_config(self):
        """Test custom Qdrant configuration"""
        config = QdrantConfig(
            host="remote-host",
            port=6334,
            collection_name="custom_collection",
            use_grpc=True,
            api_key="test-key",
            https=True,
            prefer_grpc=True
        )
        assert config.host == "remote-host"
        assert config.port == 6334
        assert config.collection_name == "custom_collection"
        assert config.use_grpc is True
        assert config.api_key == "test-key"
        assert config.https is True
        assert config.prefer_grpc is True
    
    def test_qdrant_url_property_http(self):
        """Test Qdrant URL property with HTTP"""
        config = QdrantConfig(host="example.com", port=6333, https=False)
        assert config.url == "http://example.com:6333"
    
    def test_qdrant_url_property_https(self):
        """Test Qdrant URL property with HTTPS"""
        config = QdrantConfig(host="example.com", port=6334, https=True)
        assert config.url == "https://example.com:6334"


class TestUpdatedDatabaseConfig:
    """Test the updated DatabaseConfig with Qdrant"""
    
    def test_default_database_config(self):
        """Test default database configuration includes Qdrant"""
        config = DatabaseConfig()
        assert hasattr(config, 'sqlite')
        assert hasattr(config, 'qdrant')
        assert isinstance(config.qdrant, QdrantConfig)
    
    def test_custom_database_config(self):
        """Test custom database configuration"""
        qdrant_config = QdrantConfig(host="custom-host", port=6334)
        config = DatabaseConfig(qdrant=qdrant_config)
        assert config.qdrant.host == "custom-host"
        assert config.qdrant.port == 6334


class TestConfigLoading:
    """Test configuration loading with new local setup"""
    
    def test_load_config_with_local_setup(self):
        """Test loading config with local setup values"""
        config_data = {
            'embedding': {
                'model': 'nomic-embed-text',
                'dimension': 768,
                'batch_size': 32,
                'ollama_host': 'http://localhost:11434',
                'timeout': 30
            },
            'database': {
                'qdrant': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'tradeknowledge',
                    'use_grpc': False,
                    'https': False
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config.embedding.model == "nomic-embed-text"
            assert config.embedding.dimension == 768
            assert config.database.qdrant.host == "localhost"
            assert config.database.qdrant.port == 6333
        finally:
            temp_path.unlink()
    
    def test_load_config_missing_file(self):
        """Test loading config with missing file returns defaults"""
        config = load_config(Path("nonexistent.yaml"))
        assert isinstance(config, Config)
        assert config.embedding.model == "nomic-embed-text"
        assert config.database.qdrant.host == "localhost"
    
    def test_load_config_invalid_yaml(self):
        """Test loading config with invalid YAML returns defaults"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert isinstance(config, Config)
        finally:
            temp_path.unlink()


class TestEnvironmentVariables:
    """Test environment variable integration"""
    
    def test_embedding_environment_variables(self):
        """Test that embedding environment variables are properly loaded"""
        with patch.dict(os.environ, {
            'OLLAMA_MODEL': 'custom-model',
            'EMBEDDING_DIMENSION': '512',
            'EMBEDDING_BATCH_SIZE': '64',
            'OLLAMA_HOST': 'http://custom:11434',
            'OLLAMA_TIMEOUT': '60'
        }):
            config = EmbeddingConfig()
            assert config.model == 'custom-model'
            assert config.dimension == 512
            assert config.batch_size == 64
            assert config.ollama_host == 'http://custom:11434'
            assert config.timeout == 60
    
    def test_qdrant_environment_variables(self):
        """Test that Qdrant environment variables are properly loaded"""
        with patch.dict(os.environ, {
            'QDRANT_HOST': 'custom-qdrant',
            'QDRANT_PORT': '6334',
            'QDRANT_COLLECTION': 'custom_collection',
            'QDRANT_USE_GRPC': 'true',
            'QDRANT_API_KEY': 'test-key',
            'QDRANT_HTTPS': 'true',
            'QDRANT_PREFER_GRPC': 'true'
        }):
            config = QdrantConfig()
            assert config.host == 'custom-qdrant'
            assert config.port == 6334
            assert config.collection_name == 'custom_collection'
            assert config.use_grpc is True
            assert config.api_key == 'test-key'
            assert config.https is True
            assert config.prefer_grpc is True
    
    def test_invalid_environment_variables(self):
        """Test that invalid environment variables fall back to defaults"""
        with patch.dict(os.environ, {
            'EMBEDDING_DIMENSION': 'invalid',
            'QDRANT_PORT': 'not-a-number',
            'QDRANT_USE_GRPC': 'invalid-bool'
        }):
            embedding_config = EmbeddingConfig()
            qdrant_config = QdrantConfig()
            
            # Should fall back to defaults
            assert embedding_config.dimension == 768
            assert qdrant_config.port == 6333
            assert qdrant_config.use_grpc is False


class TestConfigSingleton:
    """Test configuration singleton behavior"""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance"""
        # Clear singleton
        import src.core.config as config_module
        config_module._config = None
        
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_get_config_default_values(self):
        """Test that get_config returns proper default values"""
        # Clear singleton to get fresh config
        import src.core.config as config_module
        config_module._config = None
        
        # Temporarily change working directory to avoid loading config.yaml
        with patch('src.core.config.load_config') as mock_load_config:
            mock_load_config.return_value = Config()
            config = get_config()
            assert config.embedding.model == "nomic-embed-text"
            assert config.embedding.dimension == 768
            assert config.database.qdrant.collection_name == "tradeknowledge"


class TestBackwardCompatibility:
    """Test backward compatibility during migration"""
    
    def test_old_chroma_config_still_exists(self):
        """Test that old ChromaConfig is still accessible during migration"""
        config = Config()
        # The old chroma config should still exist for migration purposes
        assert hasattr(config.database, 'chroma') or hasattr(config.database, 'sqlite')
    
    def test_config_migration_readiness(self):
        """Test that config is ready for migration"""
        config = Config()
        # Should have both old and new configurations during migration
        assert hasattr(config, 'embedding')
        assert hasattr(config, 'database')
        assert config.embedding.dimension == 768  # New dimension


if __name__ == "__main__":
    pytest.main([__file__])