"""
Test utilities and helper functions for TradeKnowledge tests.

This module provides common functionality needed across test suites.
"""

import tempfile
import os
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional
import jwt
from datetime import datetime, timedelta


class MockUser:
    """Mock user for testing"""
    def __init__(self, user_id: str = "test_user", username: str = "testuser", 
                 email: str = "test@example.com", is_active: bool = True):
        self.id = user_id
        self.username = username
        self.email = email
        self.is_active = is_active


class MockSearchEngine:
    """Mock search engine for testing"""
    def __init__(self):
        self.search_results = []
        self.suggestions = []
    
    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock search method"""
        return {
            "query": query,
            "results": self.search_results,
            "total_results": len(self.search_results),
            "search_time_ms": 10,
            "suggestions": self.suggestions
        }
    
    async def get_suggestions(self, query: str, **kwargs) -> List[str]:
        """Mock suggestions method"""
        return self.suggestions
    
    def set_results(self, results: List[Dict[str, Any]]):
        """Set mock results"""
        self.search_results = results
    
    def set_suggestions(self, suggestions: List[str]):
        """Set mock suggestions"""
        self.suggestions = suggestions


class MockVectorStorage:
    """Mock vector storage for testing"""
    def __init__(self):
        self.embeddings = {}
        self.search_results = []
    
    async def save_embeddings(self, chunks: List[Any], embeddings: List[Any]):
        """Mock save embeddings"""
        for chunk, embedding in zip(chunks, embeddings):
            self.embeddings[chunk.id] = embedding
    
    async def search_semantic(self, query_embedding: List[float], **kwargs) -> List[Dict[str, Any]]:
        """Mock semantic search"""
        return self.search_results
    
    async def delete_embeddings(self, chunk_ids: List[str]):
        """Mock delete embeddings"""
        for chunk_id in chunk_ids:
            self.embeddings.pop(chunk_id, None)
    
    def set_search_results(self, results: List[Dict[str, Any]]):
        """Set mock search results"""
        self.search_results = results


class TemporaryFileManager:
    """Manager for temporary files in tests"""
    def __init__(self):
        self.temp_files = []
    
    def create_temp_file(self, content: bytes = b"test content", suffix: str = ".pdf") -> str:
        """Create a temporary file with specified content"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_malicious_file(self, suffix: str = ".pdf") -> str:
        """Create a file with malicious content"""
        # PE header for executable
        malicious_content = b'\\x4d\\x5a'
        return self.create_temp_file(malicious_content, suffix)
    
    def create_large_file(self, size_mb: int = 101, suffix: str = ".pdf") -> str:
        """Create a large file for size testing"""
        large_content = b'0' * (size_mb * 1024 * 1024)
        return self.create_temp_file(large_content, suffix)
    
    def cleanup(self):
        """Clean up all temporary files"""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        self.temp_files.clear()


class JWTTestHelper:
    """Helper for JWT token testing"""
    @staticmethod
    def create_token(payload: Dict[str, Any], secret: str = "test_secret", 
                    algorithm: str = "HS256") -> str:
        """Create a JWT token for testing"""
        return jwt.encode(payload, secret, algorithm=algorithm)
    
    @staticmethod
    def create_expired_token(user_id: str, secret: str = "test_secret") -> str:
        """Create an expired JWT token"""
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        }
        return jwt.encode(payload, secret, algorithm="HS256")
    
    @staticmethod
    def create_malformed_token() -> str:
        """Create a malformed JWT token"""
        return "not.a.valid.jwt.token"
    
    @staticmethod
    def decode_token_unsafe(token: str) -> Dict[str, Any]:
        """Decode token without verification for testing"""
        return jwt.decode(token, options={"verify_signature": False})


class AsyncTestHelper:
    """Helper for async testing"""
    @staticmethod
    def run_async(coro):
        """Run async function in test"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    def create_mock_async_function(return_value=None, side_effect=None):
        """Create a mock async function"""
        async def mock_func(*args, **kwargs):
            if side_effect:
                raise side_effect
            return return_value
        return MagicMock(side_effect=mock_func)


class SecurityTestHelper:
    """Helper for security testing"""
    @staticmethod
    def assert_no_sensitive_data_in_logs(log_output: str):
        """Assert no sensitive data appears in logs"""
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "key",
            "authorization",
            "credential",
        ]
        
        log_lower = log_output.lower()
        for pattern in sensitive_patterns:
            assert pattern not in log_lower, f"Sensitive data '{pattern}' found in logs"
    
    @staticmethod
    def assert_input_sanitized(original: str, sanitized: str):
        """Assert input has been properly sanitized"""
        dangerous_patterns = [
            "<script",
            "javascript:",
            "onerror",
            "onload",
            "DROP TABLE",
            "UNION SELECT",
            "../",
            "\\..\\",
        ]
        
        sanitized_lower = sanitized.lower()
        for pattern in dangerous_patterns:
            assert pattern.lower() not in sanitized_lower, \
                f"Dangerous pattern '{pattern}' not sanitized"
    
    @staticmethod
    def create_timing_test_decorator(max_time_variance_ms: float = 100):
        """Decorator to test timing attack resistance"""
        def decorator(test_func):
            def wrapper(*args, **kwargs):
                import time
                times = []
                
                # Run test multiple times to measure timing
                for _ in range(10):
                    start = time.time()
                    result = test_func(*args, **kwargs)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms
                
                # Check timing variance
                min_time = min(times)
                max_time = max(times)
                variance = max_time - min_time
                
                assert variance <= max_time_variance_ms, \
                    f"Timing variance {variance}ms exceeds limit {max_time_variance_ms}ms"
                
                return result
            return wrapper
        return decorator


class MockRequestHelper:
    """Helper for mocking HTTP requests"""
    @staticmethod
    def create_mock_request(method: str = "GET", url: str = "/test", 
                          headers: Dict[str, str] = None, 
                          client_host: str = "127.0.0.1") -> MagicMock:
        """Create a mock FastAPI request"""
        mock_request = MagicMock()
        mock_request.method = method
        mock_request.url = url
        mock_request.headers = headers or {}
        mock_request.client.host = client_host
        return mock_request
    
    @staticmethod
    def create_malicious_request(attack_type: str = "xss") -> MagicMock:
        """Create a request with malicious content"""
        if attack_type == "xss":
            return MockRequestHelper.create_mock_request(
                url="/search?q=<script>alert('xss')</script>",
                headers={"User-Agent": "<script>alert('xss')</script>"}
            )
        elif attack_type == "sql":
            return MockRequestHelper.create_mock_request(
                url="/search?q=' OR 1=1 --",
            )
        elif attack_type == "path_traversal":
            return MockRequestHelper.create_mock_request(
                url="/file/../../etc/passwd"
            )
        else:
            return MockRequestHelper.create_mock_request()


# Test data generators
def generate_test_search_results(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock search results"""
    results = []
    for i in range(count):
        results.append({
            "id": f"chunk_{i}",
            "content": f"Test content {i}",
            "score": 0.9 - (i * 0.1),
            "book_title": f"Test Book {i}",
            "book_author": f"Author {i}",
            "chapter": f"Chapter {i}",
            "page": i + 1
        })
    return results


def generate_test_chunks(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock chunks for testing"""
    chunks = []
    for i in range(count):
        chunks.append({
            "id": f"chunk_{i}",
            "content": f"This is test chunk content number {i}",
            "book_id": f"book_{i % 3}",  # 3 different books
            "book_title": f"Test Book {i % 3}",
            "chunk_index": i,
            "metadata": {
                "chapter": f"Chapter {i + 1}",
                "page": i + 1,
                "section": f"Section {i + 1}"
            }
        })
    return chunks


def generate_test_embeddings(count: int = 5, dimension: int = 384) -> List[List[float]]:
    """Generate mock embeddings"""
    import random
    embeddings = []
    for _ in range(count):
        embedding = [random.random() for _ in range(dimension)]
        embeddings.append(embedding)
    return embeddings