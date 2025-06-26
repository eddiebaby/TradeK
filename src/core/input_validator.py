"""
Comprehensive Input Validation and Sanitization

This module provides security-focused input validation for all user inputs
including search queries, file paths, and API parameters.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails"""

    pass


class InputValidator:
    """
    Comprehensive input validator with security-focused sanitization.

    Features:
    - SQL injection prevention
    - Path traversal protection
    - Unicode normalization
    - File size and type validation
    - Search query sanitization
    """

    # Dangerous patterns that should be blocked
    SQL_INJECTION_PATTERNS = [
        r"union\s+select",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"update\s+.*\s+set",
        r"exec\s*\(",
        r"execute\s*\(",
        r"script\s*>",
        r"javascript:",
        r"<\s*script",
        r"--\s*$",
        r"/\*.*\*/",
        r";\s*--",
        r"#\s*$",
        r";\s*#",
        r"'\s*or\s+'",
        r'"\s*or\s+"',
        r"'\s*=\s*'",
        r'"\s*=\s*"',
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.",
        r"~/",
        r"\\\\",
        r"//",
        r"%2e%2e",
        r"%2f",
        r"%5c",
    ]

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = [
        ".exe",
        ".bat",
        ".cmd",
        ".com",
        ".scr",
        ".pif",
        ".vbs",
        ".js",
        ".jar",
        ".sh",
        ".ps1",
        ".php",
        ".asp",
        ".aspx",
        ".jsp",
        ".pl",
        ".py",
        ".rb",
    ]

    # Allowed file extensions for uploads
    ALLOWED_BOOK_EXTENSIONS = [".pdf", ".epub", ".txt", ".md", ".docx", ".doc"]

    def __init__(
        self, max_string_length: int = 10000, max_file_size: int = 500 * 1024 * 1024
    ):
        """Initialize validator with limits"""
        self.max_string_length = max_string_length
        self.max_file_size = max_file_size

        # Compile regex patterns for efficiency
        self.sql_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
        self.path_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PATH_TRAVERSAL_PATTERNS
        ]

    def sanitize_string(self, text: str, max_length: int | None = None) -> str:
        """
        Sanitize a string input for safe processing.

        Args:
            text: Input string to sanitize
            max_length: Optional length limit (uses default if None)

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input is invalid or dangerous
        """
        if not isinstance(text, str):
            raise ValidationError(f"Expected string, got {type(text)}")

        # Length check
        limit = max_length or self.max_string_length
        if len(text) > limit:
            raise ValidationError(f"String too long: {len(text)} > {limit}")

        # Unicode normalization to prevent encoding attacks
        normalized = unicodedata.normalize("NFKC", text)

        # Remove null bytes and control characters (except common whitespace)
        sanitized = "".join(
            char for char in normalized if ord(char) >= 32 or char in "\\t\\n\\r"
        )

        # Check for SQL injection patterns
        sanitized_lower = sanitized.lower()
        for pattern in self.sql_patterns:
            if pattern.search(sanitized_lower):
                logger.warning(f"SQL injection pattern detected: {pattern.pattern}")
                raise ValidationError("Potentially dangerous SQL patterns detected")

        # Check for path traversal patterns
        for pattern in self.path_patterns:
            if pattern.search(sanitized):
                logger.warning(f"Path traversal pattern detected: {pattern.pattern}")
                raise ValidationError("Path traversal patterns detected")

        return sanitized.strip()

    def validate_search_query(self, query: str) -> str:
        """
        Validate and sanitize search query.

        Args:
            query: Search query string

        Returns:
            Sanitized query string

        Raises:
            ValidationError: If query is invalid
        """
        if not query:
            raise ValidationError("Search query cannot be empty")

        # Basic sanitization
        sanitized = self.sanitize_string(query, max_length=500)

        # Additional search-specific validation
        # Remove excessive whitespace
        sanitized = re.sub(r"\\s+", " ", sanitized)

        # Check for minimum length
        if len(sanitized.strip()) < 2:
            raise ValidationError("Search query too short (minimum 2 characters)")

        # Check for excessive punctuation (possible attack)
        punctuation_ratio = sum(
            1 for c in sanitized if not c.isalnum() and not c.isspace()
        ) / len(sanitized)
        if punctuation_ratio > 0.5:
            raise ValidationError("Query contains too many special characters")

        return sanitized

    def validate_file_path(
        self, file_path: str, base_directory: str | None = None
    ) -> Path:
        """
        Validate file path for security and accessibility.

        Args:
            file_path: File path to validate
            base_directory: Optional base directory to restrict access

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid or dangerous
        """
        if not file_path:
            raise ValidationError("File path cannot be empty")

        # URL decode the path
        decoded_path = unquote(file_path)

        # Sanitize the path string
        sanitized_path = self.sanitize_string(decoded_path, max_length=1000)

        try:
            # Create Path object and resolve it
            path = Path(sanitized_path).resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid file path: {e}")

        # Check if path exists
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")

        # Check if it's actually a file
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")

        # Base directory restriction
        if base_directory:
            try:
                base_path = Path(base_directory).resolve()
                path.relative_to(base_path)  # Will raise ValueError if outside base
            except ValueError:
                raise ValidationError(f"File path outside allowed directory: {path}")

        # Check file extension
        extension = path.suffix.lower()
        if extension in self.DANGEROUS_EXTENSIONS:
            raise ValidationError(f"Dangerous file extension: {extension}")

        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValidationError(
                    f"File too large: {file_size} > {self.max_file_size}"
                )
        except OSError as e:
            raise ValidationError(f"Cannot access file: {e}")

        return path

    def validate_book_file(
        self, file_path: str, base_directory: str | None = None
    ) -> Path:
        """
        Validate book file specifically.

        Args:
            file_path: Path to book file
            base_directory: Base directory for restrictions

        Returns:
            Validated Path object

        Raises:
            ValidationError: If file is not a valid book
        """
        path = self.validate_file_path(file_path, base_directory)

        # Check file extension for books
        extension = path.suffix.lower()
        if extension not in self.ALLOWED_BOOK_EXTENSIONS:
            raise ValidationError(
                f"Invalid book file type: {extension}. Allowed: {self.ALLOWED_BOOK_EXTENSIONS}"
            )

        return path

    def validate_api_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate API parameters dictionary.

        Args:
            params: Dictionary of API parameters

        Returns:
            Validated and sanitized parameters

        Raises:
            ValidationError: If parameters are invalid
        """
        if not isinstance(params, dict):
            raise ValidationError("Parameters must be a dictionary")

        validated = {}

        for key, value in params.items():
            # Validate key
            if not isinstance(key, str):
                raise ValidationError(f"Parameter key must be string: {type(key)}")

            sanitized_key = self.sanitize_string(key, max_length=100)

            # Validate value based on type
            if isinstance(value, str):
                sanitized_value = self.sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                sanitized_value = value
            elif isinstance(value, list):
                sanitized_value = [
                    self.sanitize_string(item) if isinstance(item, str) else item
                    for item in value[:100]  # Limit list size
                ]
            elif isinstance(value, dict):
                sanitized_value = self.validate_api_parameters(
                    value
                )  # Recursive validation
            elif value is None:
                sanitized_value = None
            else:
                raise ValidationError(f"Unsupported parameter type: {type(value)}")

            validated[sanitized_key] = sanitized_value

        return validated

    def validate_pagination(
        self, offset: int = 0, limit: int = 10, max_limit: int = 100
    ) -> tuple[int, int]:
        """
        Validate pagination parameters.

        Args:
            offset: Record offset
            limit: Number of records
            max_limit: Maximum allowed limit

        Returns:
            Validated (offset, limit) tuple

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate offset
        if not isinstance(offset, int) or offset < 0:
            raise ValidationError("Offset must be a non-negative integer")

        if offset > 1000000:  # Prevent excessive offsets
            raise ValidationError("Offset too large (max 1,000,000)")

        # Validate limit
        if not isinstance(limit, int) or limit < 1:
            raise ValidationError("Limit must be a positive integer")

        if limit > max_limit:
            raise ValidationError(f"Limit too large (max {max_limit})")

        return offset, limit

    def validate_sort_parameters(
        self,
        sort_by: str,
        sort_order: str = "asc",
        allowed_fields: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Validate sorting parameters.

        Args:
            sort_by: Field to sort by
            sort_order: Sort direction (asc/desc)
            allowed_fields: List of allowed sort fields

        Returns:
            Validated (sort_by, sort_order) tuple

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate sort_by
        sanitized_sort_by = self.sanitize_string(sort_by, max_length=50)

        if allowed_fields and sanitized_sort_by not in allowed_fields:
            raise ValidationError(
                f"Invalid sort field: {sanitized_sort_by}. Allowed: {allowed_fields}"
            )

        # Validate sort_order
        sanitized_sort_order = self.sanitize_string(sort_order, max_length=10).lower()

        if sanitized_sort_order not in ["asc", "desc", "ascending", "descending"]:
            raise ValidationError(f"Invalid sort order: {sanitized_sort_order}")

        # Normalize sort order
        if sanitized_sort_order in ["desc", "descending"]:
            sanitized_sort_order = "desc"
        else:
            sanitized_sort_order = "asc"

        return sanitized_sort_by, sanitized_sort_order

    def validate_filter_params(
        self, filters: dict[str, Any], allowed_filters: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Validate filter parameters.

        Args:
            filters: Dictionary of filter parameters
            allowed_filters: List of allowed filter keys

        Returns:
            Validated filter dictionary

        Raises:
            ValidationError: If filters are invalid
        """
        validated_filters = {}

        for key, value in filters.items():
            sanitized_key = self.sanitize_string(key, max_length=50)

            if allowed_filters and sanitized_key not in allowed_filters:
                logger.warning(f"Ignoring unknown filter: {sanitized_key}")
                continue

            # Validate filter values
            if isinstance(value, str):
                validated_filters[sanitized_key] = self.sanitize_string(
                    value, max_length=200
                )
            elif isinstance(value, (int, float, bool)):
                validated_filters[sanitized_key] = value
            elif isinstance(value, list):
                validated_filters[sanitized_key] = [
                    self.sanitize_string(item) if isinstance(item, str) else item
                    for item in value[:50]  # Limit filter list size
                ]
            else:
                logger.warning(f"Ignoring invalid filter value type: {type(value)}")

        return validated_filters


# Global validator instance
_validator = None


def get_input_validator() -> InputValidator:
    """Get the global input validator instance"""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


# Convenience functions for common validations


def sanitize_search_query(query: str) -> str:
    """Sanitize a search query"""
    return get_input_validator().validate_search_query(query)


def validate_file_path(file_path: str, base_directory: str | None = None) -> Path:
    """Validate a file path"""
    return get_input_validator().validate_file_path(file_path, base_directory)


def validate_book_file(file_path: str, base_directory: str | None = None) -> Path:
    """Validate a book file"""
    return get_input_validator().validate_book_file(file_path, base_directory)


def sanitize_api_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sanitize API parameters"""
    return get_input_validator().validate_api_parameters(params)


def validate_pagination_params(offset: int = 0, limit: int = 10) -> tuple[int, int]:
    """Validate pagination parameters"""
    return get_input_validator().validate_pagination(offset, limit)


def sanitize_string(input_string: str, max_length: int = 1000) -> str:
    """Sanitize a general string input"""
    return get_input_validator().sanitize_string(input_string, max_length)


def validate_search_query(query: str) -> str:
    """Validate and sanitize a search query"""
    return get_input_validator().validate_search_query(query)
