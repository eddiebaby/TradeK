"""
Critical Security Edge Cases Tests

This module contains tests for critical security vulnerabilities
and edge cases that were identified in the TDD analysis.
"""

import pytest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
import json
import time

from src.core.input_validator import InputValidator
from src.api.auth import AuthManager
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.core.qdrant_storage import QdrantStorage


class TestJWTTokenEdgeCases:
    """Test JWT token security edge cases"""
    
    def test_jwt_with_malformed_structure_raises_invalid_token_error(self):
        """Test JWT with malformed structure raises InvalidTokenError"""
        # ARRANGE
        auth_manager = AuthManager()
        malformed_tokens = [
            "not.a.valid.jwt",
            "onlyonepart",
            "two.parts",
            "too.many.parts.here.invalid",
            "header.payload.",  # missing signature
            ".payload.signature",  # missing header
            "header..signature",  # missing payload
        ]
        
        # ACT & ASSERT
        for token in malformed_tokens:
            with pytest.raises(ValueError, match="Invalid token"):
                auth_manager.verify_token(token)
    
    def test_jwt_with_missing_required_claims_raises_validation_error(self):
        """Test JWT with missing required claims raises ValidationError"""
        # ARRANGE
        auth_manager = AuthManager()
        tokens_with_missing_claims = [
            {"exp": int(time.time()) + 3600},  # missing user_id
            {"user_id": "test", "iat": int(time.time())},  # missing exp
            {"user_id": "test", "exp": int(time.time()) + 3600},  # missing iat
            {},  # completely empty payload
        ]
        
        # ACT & ASSERT
        for payload in tokens_with_missing_claims:
            with pytest.raises(ValueError, match="Missing required claim"):
                token = auth_manager._create_test_token(payload)  # Helper method
                auth_manager.verify_token(token)
    
    def test_jwt_with_future_issued_time_raises_validation_error(self):
        """Test JWT with future issued time raises ValidationError"""
        # ARRANGE
        auth_manager = AuthManager()
        future_time = int(time.time()) + 3600  # 1 hour in future
        payload = {
            "user_id": "test",
            "iat": future_time,
            "exp": future_time + 3600
        }
        
        # ACT
        token = auth_manager._create_test_token(payload)
        
        # ASSERT
        with pytest.raises(ValueError, match="Token issued in future"):
            auth_manager.verify_token(token)
    
    def test_jwt_with_clock_skew_within_tolerance_accepts_token(self):
        """Test JWT with clock skew within tolerance accepts token"""
        # ARRANGE
        auth_manager = AuthManager(clock_skew_tolerance=300)  # 5 minutes
        slightly_future_time = int(time.time()) + 60  # 1 minute future
        payload = {
            "user_id": "test",
            "iat": slightly_future_time,
            "exp": slightly_future_time + 3600
        }
        
        # ACT
        token = auth_manager._create_test_token(payload)
        result = auth_manager.verify_token(token)
        
        # ASSERT
        assert result["user_id"] == "test"


class TestUnicodeEncodingAttacks:
    """Test Unicode and encoding attack vectors"""
    
    def test_input_validator_with_overlong_utf8_encoding_raises_validation_error(self):
        """Test input validator with overlong UTF-8 encoding raises ValidationError"""
        # ARRANGE
        validator = InputValidator()
        overlong_sequences = [
            b'\xc0\xaf',  # Overlong encoding of '/'
            b'\xe0\x80\xaf',  # Overlong encoding of '/'
            b'\xf0\x80\x80\xaf',  # Overlong encoding of '/'
            b'\xc1\x9c',  # Overlong encoding of '\'
            b'\xc0\x80',  # Overlong encoding of null byte
        ]
        
        # ACT & ASSERT
        for sequence in overlong_sequences:
            with pytest.raises(ValueError, match="Invalid UTF-8"):
                validator.validate_text_input(sequence.decode('utf-8', errors='ignore'))
    
    def test_input_validator_with_mixed_encoding_attacks_raises_validation_error(self):
        """Test input validator with mixed encoding attacks raises ValidationError"""
        # ARRANGE
        validator = InputValidator()
        mixed_encoding_attacks = [
            "normal_text\x00hidden_payload",  # Null byte injection
            "test\uFEFFBOM_injection",  # BOM injection
            "test\u202Eright_to_left_override",  # RTL override
            "test\u034F\u034F\u034Fcombining_chars",  # Combining character abuse
            "test\uFFF9\uFFFAinvisible\uFFFB",  # Invisible characters
        ]
        
        # ACT & ASSERT
        for attack in mixed_encoding_attacks:
            with pytest.raises(ValueError, match="Invalid characters"):
                validator.validate_text_input(attack)
    
    def test_search_query_with_unicode_normalization_attacks_sanitized(self):
        """Test search query with Unicode normalization attacks is sanitized"""
        # ARRANGE
        validator = InputValidator()
        normalization_attacks = [
            "caf\u00E9",  # NFC form
            "cafe\u0301",  # NFD form - separate combining character
            "\u212Aelvin",  # Kelvin sign looks like K
            "\uFF21\uFF42\uFF43",  # Full-width characters ABC
        ]
        
        # ACT & ASSERT
        for attack in normalization_attacks:
            sanitized = validator.normalize_search_query(attack)
            # Should be normalized to consistent form
            assert sanitized != attack or len(sanitized) != len(attack)


class TestQdrantVectorDatabaseEdgeCases:
    """Test Qdrant vector database security and corruption scenarios"""
    
    @pytest.mark.asyncio
    async def test_qdrant_with_invalid_vector_dimensions_raises_validation_error(self):
        """Test Qdrant with invalid vector dimensions raises ValidationError"""
        # ARRANGE
        storage = QdrantStorage()
        invalid_vectors = [
            [],  # Empty vector
            [1.0] * 100,  # Wrong dimension (should be 384)
            [float('inf')] * 384,  # Infinity values
            [float('nan')] * 384,  # NaN values
            ['not', 'a', 'number'] * 128,  # Non-numeric values
        ]
        
        # ACT & ASSERT
        for vector in invalid_vectors:
            with pytest.raises(ValueError, match="Invalid vector"):
                await storage.upsert_embeddings([{
                    "id": "test",
                    "vector": vector,
                    "metadata": {"text": "test"}
                }])
    
    @pytest.mark.asyncio
    async def test_qdrant_with_corrupted_metadata_raises_validation_error(self):
        """Test Qdrant with corrupted metadata raises ValidationError"""
        # ARRANGE
        storage = QdrantStorage()
        corrupted_metadata = [
            {"circular": None},  # Will be made circular
            {"too_deep": {"level": {"deep": {"very": {"deep": "value"}}}}},
            {"huge_string": "x" * (1024 * 1024)},  # 1MB string
            {"invalid_json": json.dumps({"key": "value"}) + "corrupted"},
        ]
        
        # Make first metadata circular
        corrupted_metadata[0]["circular"] = corrupted_metadata[0]
        
        # ACT & ASSERT
        for metadata in corrupted_metadata:
            with pytest.raises(ValueError, match="Invalid metadata"):
                await storage.upsert_embeddings([{
                    "id": "test",
                    "vector": [1.0] * 384,
                    "metadata": metadata
                }])
    
    @pytest.mark.asyncio
    async def test_qdrant_concurrent_collection_modifications_handled_safely(self):
        """Test Qdrant concurrent collection modifications are handled safely"""
        # ARRANGE
        storage = QdrantStorage()
        concurrent_operations = []
        
        # ACT
        # Simulate concurrent collection operations
        import asyncio
        
        async def create_collection():
            await storage.create_collection("test_collection_1")
            
        async def delete_collection():
            await storage.delete_collection("test_collection_1")
            
        async def modify_collection():
            await storage.update_collection_config("test_collection_1", {})
        
        # Run concurrent operations
        tasks = [create_collection(), delete_collection(), modify_collection()]
        
        # ASSERT
        # Should handle race conditions gracefully without corruption
        with pytest.raises(ValueError, match="Collection operation conflict"):
            await asyncio.gather(*tasks, return_exceptions=True)


class TestFileUploadSecurityEdgeCases:
    """Test file upload security edge cases"""
    
    def test_file_processor_with_polyglot_file_raises_security_error(self):
        """Test file processor with polyglot file raises SecurityError"""
        # ARRANGE
        processor = EnhancedBookProcessor()
        
        # Create polyglot file (valid PDF + executable)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            # Write PDF header
            f.write(b'%PDF-1.4\n')
            # Write executable payload (simplified)
            f.write(b'\x7fELF')  # ELF header
            f.write(b'malicious_payload' * 100)
            polyglot_path = Path(f.name)
        
        try:
            # ACT & ASSERT
            with pytest.raises(SecurityError, match="Polyglot file detected"):
                processor.process_file(polyglot_path)
        finally:
            polyglot_path.unlink()
    
    def test_file_processor_with_zip_bomb_raises_security_error(self):
        """Test file processor with ZIP bomb raises SecurityError"""
        # ARRANGE
        processor = EnhancedBookProcessor()
        
        # Create ZIP bomb
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Create highly compressed file
                large_content = b'0' * (1024 * 1024 * 10)  # 10MB of zeros
                zf.writestr('large_file.txt', large_content)
            zip_bomb_path = Path(f.name)
        
        try:
            # ACT & ASSERT
            with pytest.raises(SecurityError, match="ZIP bomb detected"):
                processor.process_file(zip_bomb_path)
        finally:
            zip_bomb_path.unlink()
    
    def test_file_processor_with_embedded_executable_raises_security_error(self):
        """Test file processor with embedded executable raises SecurityError"""
        # ARRANGE
        processor = EnhancedBookProcessor()
        
        # Create file with embedded executable
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            # Write valid PDF content
            pdf_content = b'''%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
'''
            f.write(pdf_content)
            # Embed Windows executable signature
            f.write(b'MZ')  # DOS header
            f.write(b'\x90' * 58)  # DOS stub
            f.write(b'PE\x00\x00')  # PE header
            embedded_exe_path = Path(f.name)
        
        try:
            # ACT & ASSERT
            with pytest.raises(SecurityError, match="Embedded executable"):
                processor.process_file(embedded_exe_path)
        finally:
            embedded_exe_path.unlink()


class TestSizeBasedDoSAttacks:
    """Test size-based denial of service attacks"""
    
    def test_json_parser_with_deeply_nested_object_raises_resource_error(self):
        """Test JSON parser with deeply nested object raises ResourceError"""
        # ARRANGE
        validator = InputValidator()
        
        # Create deeply nested JSON (1000 levels)
        nested_json = "{" * 1000 + '"key":"value"' + "}" * 1000
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="JSON too deeply nested"):
            validator.validate_json_input(nested_json)
    
    def test_search_query_with_extremely_large_payload_raises_size_error(self):
        """Test search query with extremely large payload raises SizeError"""
        # ARRANGE
        validator = InputValidator()
        large_query = "search term " * 100000  # ~1MB query
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Query too large"):
            validator.validate_search_query(large_query)
    
    def test_metadata_with_circular_references_raises_validation_error(self):
        """Test metadata with circular references raises ValidationError"""
        # ARRANGE
        validator = InputValidator()
        
        # Create circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Circular reference"):
            validator.validate_metadata(circular_dict)


class TestConcurrentAccessEdgeCases:
    """Test concurrent access and race condition scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_sessions_handled_safely(self):
        """Test concurrent user sessions are handled safely"""
        # ARRANGE
        auth_manager = AuthManager()
        user_id = "test_user"
        
        # ACT
        # Simulate concurrent login attempts
        import asyncio
        
        async def login_attempt():
            return await auth_manager.create_session(user_id)
        
        # Run multiple concurrent logins
        tasks = [login_attempt() for _ in range(10)]
        sessions = await asyncio.gather(*tasks)
        
        # ASSERT
        # All sessions should be valid and unique
        session_ids = [s["session_id"] for s in sessions]
        assert len(set(session_ids)) == len(session_ids)  # All unique
    
    @pytest.mark.asyncio
    async def test_session_cleanup_during_concurrent_access_maintains_consistency(self):
        """Test session cleanup during concurrent access maintains consistency"""
        # ARRANGE
        auth_manager = AuthManager()
        user_id = "test_user"
        
        # Create initial session
        session = await auth_manager.create_session(user_id)
        session_id = session["session_id"]
        
        # ACT
        import asyncio
        
        async def access_session():
            return await auth_manager.validate_session(session_id)
        
        async def cleanup_session():
            await auth_manager.cleanup_expired_sessions()
        
        # Run concurrent access and cleanup
        access_tasks = [access_session() for _ in range(5)]
        cleanup_task = cleanup_session()
        
        results = await asyncio.gather(*access_tasks, cleanup_task, return_exceptions=True)
        
        # ASSERT
        # Should handle concurrent access gracefully
        access_results = results[:-1]
        assert all(isinstance(r, (bool, dict)) for r in access_results)  # No exceptions