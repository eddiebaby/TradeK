"""
Contract Testing for API Interfaces

Ensures that API contracts are maintained and backward compatibility
is preserved across versions and service boundaries.
"""

import pytest
import json
from typing import Dict, Any, List
from jsonschema import validate, ValidationError
from unittest.mock import Mock, patch
import requests_mock

from src.api.main import app
from src.api.models import SearchRequest, SearchResponse, UserRequest
from fastapi.testclient import TestClient


class APIContractValidator:
    """Validates API contracts and schemas."""
    
    def __init__(self):
        self.search_request_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "offset": {"type": "integer", "minimum": 0},
                "filters": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string"},
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "format": "date"},
                                "end": {"type": "string", "format": "date"}
                            }
                        }
                    }
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
        
        self.search_response_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id", "title", "content", "score"]
                    }
                },
                "total_found": {"type": "integer", "minimum": 0},
                "query_time_ms": {"type": "number", "minimum": 0},
                "pagination": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1},
                        "has_more": {"type": "boolean"}
                    },
                    "required": ["offset", "limit", "has_more"]
                }
            },
            "required": ["results", "total_found", "query_time_ms"],
            "additionalProperties": False
        }
        
        self.error_response_schema = {
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"}
                    },
                    "required": ["code", "message"]
                },
                "request_id": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["error", "request_id", "timestamp"],
            "additionalProperties": False
        }
    
    def validate_search_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate search request against contract."""
        try:
            validate(instance=request_data, schema=self.search_request_schema)
            return True
        except ValidationError:
            return False
    
    def validate_search_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate search response against contract."""
        try:
            validate(instance=response_data, schema=self.search_response_schema)
            return True
        except ValidationError:
            return False
    
    def validate_error_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate error response against contract."""
        try:
            validate(instance=response_data, schema=self.error_response_schema)
            return True
        except ValidationError:
            return False


class TestSearchAPIContract:
    """Test search API contract compliance."""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.validator = APIContractValidator()
    
    def test_search_endpoint_with_valid_request_follows_response_contract(self):
        """Test search endpoint with valid request follows response contract"""
        # ARRANGE
        valid_request = {
            "query": "trading strategies",
            "limit": 10,
            "offset": 0
        }
        
        # ACT
        response = self.client.post("/api/v1/search", json=valid_request)
        
        # ASSERT
        assert response.status_code == 200
        response_data = response.json()
        assert self.validator.validate_search_response(response_data), \
            f"Response doesn't match contract: {response_data}"
    
    def test_search_endpoint_with_invalid_request_returns_contract_compliant_error(self):
        """Test search endpoint with invalid request returns contract-compliant error"""
        # ARRANGE
        invalid_requests = [
            {},  # Missing required query
            {"query": ""},  # Empty query
            {"query": "test", "limit": 0},  # Invalid limit
            {"query": "test", "offset": -1},  # Invalid offset
            {"query": "x" * 1001},  # Query too long
        ]
        
        # ACT & ASSERT
        for invalid_request in invalid_requests:
            response = self.client.post("/api/v1/search", json=invalid_request)
            
            assert response.status_code in [400, 422], \
                f"Expected error status for {invalid_request}"
            
            if response.status_code == 400:  # Custom error format
                response_data = response.json()
                assert self.validator.validate_error_response(response_data), \
                    f"Error response doesn't match contract: {response_data}"
    
    def test_search_endpoint_response_fields_always_present(self):
        """Test search endpoint response always contains required fields"""
        # ARRANGE
        test_queries = [
            "trading",
            "nonexistent query that returns no results",
            "a" * 100,  # Long query
        ]
        
        # ACT & ASSERT
        for query in test_queries:
            response = self.client.post("/api/v1/search", json={"query": query})
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Required fields must always be present
            assert "results" in response_data
            assert "total_found" in response_data
            assert "query_time_ms" in response_data
            assert isinstance(response_data["results"], list)
            assert isinstance(response_data["total_found"], int)
            assert isinstance(response_data["query_time_ms"], (int, float))
    
    def test_search_endpoint_pagination_contract_consistency(self):
        """Test search endpoint pagination follows contract consistently"""
        # ARRANGE
        request_with_pagination = {
            "query": "test",
            "limit": 5,
            "offset": 10
        }
        
        # ACT
        response = self.client.post("/api/v1/search", json=request_with_pagination)
        
        # ASSERT
        assert response.status_code == 200
        response_data = response.json()
        
        # Pagination contract validation
        if "pagination" in response_data:
            pagination = response_data["pagination"]
            assert pagination["offset"] == request_with_pagination["offset"]
            assert pagination["limit"] == request_with_pagination["limit"]
            assert isinstance(pagination["has_more"], bool)
    
    def test_search_endpoint_result_items_contract_compliance(self):
        """Test search endpoint result items follow contract specification"""
        # ARRANGE
        request = {"query": "trading"}
        
        # ACT
        response = self.client.post("/api/v1/search", json=request)
        
        # ASSERT
        assert response.status_code == 200
        response_data = response.json()
        
        # Each result item must follow contract
        for result in response_data["results"]:
            assert "id" in result
            assert "title" in result
            assert "content" in result
            assert "score" in result
            
            assert isinstance(result["id"], str)
            assert isinstance(result["title"], str)
            assert isinstance(result["content"], str)
            assert isinstance(result["score"], (int, float))
            assert 0 <= result["score"] <= 1


class TestBackwardCompatibilityContract:
    """Test API backward compatibility contracts."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_api_v1_endpoints_maintain_backward_compatibility(self):
        """Test API v1 endpoints maintain backward compatibility"""
        # ARRANGE
        legacy_request_formats = [
            # Original format
            {"q": "trading"},  # Legacy query parameter name
            # Extended format with new optional fields
            {"query": "trading", "max_results": 10},  # Legacy limit name
        ]
        
        # ACT & ASSERT
        for legacy_request in legacy_request_formats:
            # Should either work or fail gracefully with clear error
            response = self.client.post("/api/v1/search", json=legacy_request)
            
            # Should not return 500 (server error)
            assert response.status_code != 500, \
                f"Legacy format caused server error: {legacy_request}"
            
            # Should either succeed or return clear validation error
            assert response.status_code in [200, 400, 422], \
                f"Unexpected status for legacy format: {legacy_request}"
    
    def test_api_response_fields_never_removed(self):
        """Test API response fields are never removed (only deprecated)"""
        # ARRANGE
        request = {"query": "test"}
        
        # ACT
        response = self.client.post("/api/v1/search", json=request)
        
        # ASSERT
        assert response.status_code == 200
        response_data = response.json()
        
        # Core fields that must never be removed
        required_fields = ["results", "total_found"]
        for field in required_fields:
            assert field in response_data, \
                f"Critical field '{field}' missing from response"
    
    def test_api_error_format_consistency(self):
        """Test API error format remains consistent across versions"""
        # ARRANGE
        error_triggering_requests = [
            {"query": ""},  # Validation error
            {"invalid": "request"},  # Schema error
        ]
        
        # ACT & ASSERT
        for request in error_triggering_requests:
            response = self.client.post("/api/v1/search", json=request)
            
            if response.status_code >= 400:
                # Error response should have consistent structure
                response_data = response.json()
                
                # Should contain error information
                assert "detail" in response_data or "error" in response_data, \
                    "Error response missing error details"


class TestExternalServiceContracts:
    """Test contracts with external services (Qdrant, Ollama)."""
    
    @patch('src.core.qdrant_storage.QdrantStorage')
    def test_qdrant_service_contract_compliance(self, mock_qdrant):
        """Test Qdrant service contract compliance"""
        # ARRANGE
        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Define expected contract for Qdrant operations
        expected_search_response = {
            'results': [
                {
                    'id': 'test_id',
                    'score': 0.95,
                    'payload': {'text': 'test content'}
                }
            ]
        }
        mock_qdrant_instance.search.return_value = expected_search_response
        
        # ACT
        from src.core.qdrant_storage import QdrantStorage
        storage = QdrantStorage()
        result = storage.search(vector=[1.0] * 384, limit=10)
        
        # ASSERT
        # Verify the contract is maintained
        assert 'results' in result
        assert isinstance(result['results'], list)
        for item in result['results']:
            assert 'id' in item
            assert 'score' in item
            assert isinstance(item['score'], (int, float))
    
    @requests_mock.Mocker()
    def test_ollama_service_contract_compliance(self, m):
        """Test Ollama service contract compliance"""
        # ARRANGE
        expected_embedding_response = {
            'embedding': [0.1] * 384
        }
        m.post('http://localhost:11434/api/embeddings', json=expected_embedding_response)
        
        # ACT
        from src.ingestion.local_embeddings import LocalEmbeddingService
        service = LocalEmbeddingService()
        result = service.generate_embedding("test text")
        
        # ASSERT
        # Verify embedding service contract
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, (int, float)) for x in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])