"""
Test configuration settings for TradeKnowledge tests.

This module provides configuration for different test environments.
"""

import os
from pathlib import Path


class TestConfig:
    """Base test configuration"""
    
    # Test database settings
    TEST_DB_PATH = ":memory:"  # In-memory SQLite for tests
    TEST_QDRANT_HOST = "localhost"
    TEST_QDRANT_PORT = 6333
    TEST_COLLECTION_NAME = "test_collection"
    
    # Test API settings
    TEST_SECRET_KEY = "test-secret-key-not-for-production"
    TEST_JWT_ALGORITHM = "HS256"
    TEST_TOKEN_EXPIRY_HOURS = 1
    
    # Test file settings
    TEST_UPLOAD_DIR = "test_uploads"
    TEST_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for tests
    ALLOWED_EXTENSIONS = [".pdf", ".txt", ".epub"]
    
    # Test embedding settings
    TEST_EMBEDDING_MODEL = "test-model"
    TEST_EMBEDDING_DIMENSION = 384
    TEST_BATCH_SIZE = 5
    
    # Security test settings
    MAX_LOGIN_ATTEMPTS = 3
    LOCKOUT_DURATION_MINUTES = 5
    PASSWORD_MIN_LENGTH = 8
    
    # Performance test settings
    PERFORMANCE_TEST_TIMEOUT = 30  # seconds
    MAX_RESPONSE_TIME_MS = 1000
    MAX_MEMORY_USAGE_MB = 512
    
    # Test data paths
    TEST_DATA_DIR = Path(__file__).parent.parent / "fixtures" / "test_data"
    SAMPLE_PDF_PATH = TEST_DATA_DIR / "sample.pdf"
    MALICIOUS_FILE_PATH = TEST_DATA_DIR / "malicious.exe"


class UnitTestConfig(TestConfig):
    """Configuration for unit tests"""
    
    # Use mocks for external services
    USE_MOCK_QDRANT = True
    USE_MOCK_OLLAMA = True
    USE_MOCK_FILESYSTEM = True
    
    # Faster test execution
    EMBEDDING_TIMEOUT = 5
    SEARCH_TIMEOUT = 5


class IntegrationTestConfig(TestConfig):
    """Configuration for integration tests"""
    
    # Use real services but isolated instances
    USE_MOCK_QDRANT = False
    USE_MOCK_OLLAMA = False
    USE_MOCK_FILESYSTEM = False
    
    # Real service endpoints for integration
    QDRANT_HOST = os.getenv("TEST_QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("TEST_QDRANT_PORT", "6333"))
    OLLAMA_HOST = os.getenv("TEST_OLLAMA_HOST", "http://localhost:11434")
    
    # Longer timeouts for real services
    EMBEDDING_TIMEOUT = 30
    SEARCH_TIMEOUT = 30


class SecurityTestConfig(TestConfig):
    """Configuration for security tests"""
    
    # Security-specific settings
    ENABLE_RATE_LIMITING = True
    ENABLE_SECURITY_HEADERS = True
    ENABLE_INPUT_VALIDATION = True
    
    # Test payloads
    SQL_INJECTION_TESTS = True
    XSS_TESTS = True
    PATH_TRAVERSAL_TESTS = True
    COMMAND_INJECTION_TESTS = True
    
    # Security test timeouts
    BRUTE_FORCE_TEST_ATTEMPTS = 10
    TIMING_ATTACK_SAMPLES = 100
    
    # Logging for security tests
    LOG_SECURITY_EVENTS = True
    LOG_LEVEL = "DEBUG"


class PerformanceTestConfig(TestConfig):
    """Configuration for performance tests"""
    
    # Performance benchmarks
    MAX_SEARCH_TIME_MS = 500
    MAX_EMBEDDING_TIME_MS = 2000
    MAX_INGESTION_TIME_MS = 10000
    
    # Load testing
    CONCURRENT_USERS = 10
    TEST_DURATION_SECONDS = 60
    REQUESTS_PER_SECOND = 50
    
    # Resource limits
    MAX_CPU_USAGE_PERCENT = 80
    MAX_MEMORY_USAGE_PERCENT = 70
    MAX_DISK_USAGE_MB = 1000


class E2ETestConfig(TestConfig):
    """Configuration for end-to-end tests"""
    
    # Full application testing
    START_REAL_SERVER = True
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8001  # Different from production
    
    # Browser testing (if using Selenium)
    BROWSER = "chrome"
    HEADLESS = True
    IMPLICIT_WAIT = 10
    
    # Test scenarios
    TEST_USER_WORKFLOWS = True
    TEST_API_ENDPOINTS = True
    TEST_FILE_UPLOADS = True


# Test markers for pytest
TEST_MARKERS = {
    "unit": "Unit tests - fast, isolated tests",
    "integration": "Integration tests - test component interactions",
    "security": "Security tests - test for vulnerabilities",
    "performance": "Performance tests - test speed and resource usage",
    "e2e": "End-to-end tests - full application workflows",
    "slow": "Slow tests - may take longer to complete",
    "requires_qdrant": "Tests that require Qdrant vector database",
    "requires_ollama": "Tests that require Ollama embedding service",
    "requires_gpu": "Tests that require GPU resources",
}


def get_test_config(test_type: str = "unit"):
    """Get configuration for specific test type"""
    configs = {
        "unit": UnitTestConfig,
        "integration": IntegrationTestConfig,
        "security": SecurityTestConfig,
        "performance": PerformanceTestConfig,
        "e2e": E2ETestConfig,
    }
    
    return configs.get(test_type, UnitTestConfig)


# Pytest configuration helpers
def pytest_configure_markers():
    """Configure pytest markers"""
    marker_configs = []
    for marker, description in TEST_MARKERS.items():
        marker_configs.append(f"{marker}: {description}")
    return marker_configs


# Environment detection
def is_ci_environment():
    """Check if running in CI environment"""
    ci_indicators = [
        "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS",
        "TRAVIS", "JENKINS_URL", "GITLAB_CI"
    ]
    return any(os.getenv(indicator) for indicator in ci_indicators)


def should_skip_slow_tests():
    """Check if slow tests should be skipped"""
    return os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true"


def should_skip_security_tests():
    """Check if security tests should be skipped"""
    return os.getenv("SKIP_SECURITY_TESTS", "false").lower() == "true"


def get_parallel_workers():
    """Get number of parallel test workers"""
    return int(os.getenv("TEST_WORKERS", "4"))


# Test data management
def ensure_test_directories():
    """Ensure test directories exist"""
    directories = [
        TestConfig.TEST_DATA_DIR,
        Path(TestConfig.TEST_UPLOAD_DIR),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def cleanup_test_data():
    """Clean up test data after tests"""
    import shutil
    
    cleanup_paths = [
        Path(TestConfig.TEST_UPLOAD_DIR),
    ]
    
    for path in cleanup_paths:
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)


# Test database utilities
def get_test_database_url():
    """Get test database URL"""
    return f"sqlite:///{TestConfig.TEST_DB_PATH}"


def get_test_qdrant_config():
    """Get test Qdrant configuration"""
    return {
        "host": TestConfig.TEST_QDRANT_HOST,
        "port": TestConfig.TEST_QDRANT_PORT,
        "collection_name": TestConfig.TEST_COLLECTION_NAME,
    }