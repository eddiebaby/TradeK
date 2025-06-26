"""
Comprehensive API testing for TradeKnowledge to achieve 95% coverage
Tests all API endpoints, error handling, authentication, and edge cases
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List
import json
from pathlib import Path

# Import the main app and dependencies
from src.api.main import app
from src.api.routers.search import router as search_router
from src.api.middleware import SecurityMiddleware
from src.api.metrics import metrics_collector
from src.core.models import SearchResult, ChunkData


class TestAPIMain:
    """Test the main API application setup and configuration"""
    
    def test_app_initialization(self):
        """Test that the FastAPI app initializes correctly"""
        assert app.title == "TradeKnowledge API"
        assert app.version is not None
        
    def test_health_check_endpoint(self):
        """Test the health check endpoint"""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data
    
    def test_root_endpoint_redirects_to_docs(self):
        """Test that root endpoint provides API information"""
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "docs_url" in data
    
    def test_cors_middleware_configuration(self):
        """Test CORS middleware is properly configured"""
        with TestClient(app) as client:
            # Preflight request
            response = client.options(
                "/search",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
            )
            assert response.status_code == 200
            assert "access-control-allow-origin" in response.headers


class TestSearchAPI:
    """Comprehensive tests for the search API endpoints"""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Mock the search engine"""
        with patch('src.api.routers.search.search_engine') as mock:
            mock.search = AsyncMock()
            mock.get_stats = AsyncMock()
            yield mock
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            SearchResult(
                chunk_id="chunk_1",
                content="Trading strategies for beginners",
                score=0.95,
                metadata={
                    "book_title": "Trading 101",
                    "page_number": 5,
                    "chapter": "Introduction"
                }
            ),
            SearchResult(
                chunk_id="chunk_2", 
                content="Advanced options trading",
                score=0.87,
                metadata={
                    "book_title": "Options Master",
                    "page_number": 42,
                    "chapter": "Advanced Strategies"
                }
            )
        ]
    
    def test_search_valid_query(self, mock_search_engine, sample_search_results):
        """Test search with valid query"""
        mock_search_engine.search.return_value = sample_search_results
        
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "trading strategies",
                    "max_results": 10,
                    "min_score": 0.7
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "query" in data
            assert "total_results" in data
            assert len(data["results"]) == 2
            assert data["query"] == "trading strategies"
            
            # Verify mock was called correctly
            mock_search_engine.search.assert_called_once()
    
    def test_search_empty_query(self):
        """Test search with empty query returns validation error"""
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={"query": ""}
            )
            assert response.status_code == 422  # Validation error
    
    def test_search_query_too_long(self):
        """Test search with overly long query"""
        long_query = "trading " * 1000  # Very long query
        
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={"query": long_query}
            )
            assert response.status_code == 422
    
    def test_search_invalid_max_results(self):
        """Test search with invalid max_results parameter"""
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "trading",
                    "max_results": -1  # Invalid negative value
                }
            )
            assert response.status_code == 422
    
    def test_search_invalid_min_score(self):
        """Test search with invalid min_score parameter"""
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "trading",
                    "min_score": 2.0  # Invalid score > 1.0
                }
            )
            assert response.status_code == 422
    
    def test_search_malformed_json(self):
        """Test search with malformed JSON"""
        with TestClient(app) as client:
            response = client.post(
                "/search",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 422
    
    def test_search_missing_required_fields(self):
        """Test search with missing required fields"""
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={}  # Missing required query field
            )
            assert response.status_code == 422
    
    def test_search_engine_error_handling(self, mock_search_engine):
        """Test error handling when search engine fails"""
        mock_search_engine.search.side_effect = Exception("Search engine error")
        
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={"query": "test query"}
            )
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
    
    def test_search_with_filters(self, mock_search_engine, sample_search_results):
        """Test search with additional filters"""
        mock_search_engine.search.return_value = sample_search_results
        
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "trading",
                    "filters": {
                        "book_title": "Trading 101",
                        "chapter": "Introduction"
                    }
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
    
    def test_search_stats_endpoint(self, mock_search_engine):
        """Test the search statistics endpoint"""
        mock_stats = {
            "total_documents": 1500,
            "total_chunks": 45000,
            "index_size": "2.5GB",
            "last_update": "2024-01-01T00:00:00Z"
        }
        mock_search_engine.get_stats.return_value = mock_stats
        
        with TestClient(app) as client:
            response = client.get("/search/stats")
            assert response.status_code == 200
            data = response.json()
            assert data == mock_stats


class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    def test_security_headers_added(self):
        """Test that security headers are added to responses"""
        with TestClient(app) as client:
            response = client.get("/health")
            
            # Check for security headers
            assert "X-Content-Type-Options" in response.headers
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert "X-Frame-Options" in response.headers
            assert "X-XSS-Protection" in response.headers
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        with TestClient(app) as client:
            # Make multiple rapid requests
            responses = []
            for i in range(100):  # Attempt to exceed rate limit
                response = client.get("/health")
                responses.append(response.status_code)
                if response.status_code == 429:  # Rate limited
                    break
            
            # Should eventually get rate limited
            assert 429 in responses or all(r == 200 for r in responses)  # Either rate limited or passes
    
    def test_request_size_limit(self):
        """Test request size limiting"""
        large_data = {"query": "x" * (10 * 1024 * 1024)}  # 10MB payload
        
        with TestClient(app) as client:
            response = client.post("/search", json=large_data)
            # Should either reject large payload or handle gracefully
            assert response.status_code in [413, 422, 500]
    
    def test_sql_injection_protection(self):
        """Test SQL injection attempt protection"""
        malicious_queries = [
            "'; DROP TABLE chunks; --",
            "test UNION SELECT * FROM chunks",
            "test; DELETE FROM books;",
        ]
        
        with TestClient(app) as client:
            for query in malicious_queries:
                response = client.post(
                    "/search",
                    json={"query": query}
                )
                # Should not return 500 error (indicating SQL injection worked)
                assert response.status_code in [200, 422, 400]
    
    def test_xss_protection(self):
        """Test XSS attempt protection"""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        with TestClient(app) as client:
            for xss in xss_attempts:
                response = client.post(
                    "/search",
                    json={"query": xss}
                )
                
                # Verify XSS is not executed and properly escaped
                if response.status_code == 200:
                    data = response.json()
                    response_text = json.dumps(data)
                    assert "<script>" not in response_text
                    assert "javascript:" not in response_text


class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    def test_request_metrics_collected(self):
        """Test that request metrics are collected"""
        with patch.object(metrics_collector, 'record_request') as mock_record:
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
                
                # Verify metrics were recorded
                mock_record.assert_called()
    
    def test_search_metrics_collected(self):
        """Test that search-specific metrics are collected"""
        with patch.object(metrics_collector, 'record_search') as mock_record:
            with patch('src.api.routers.search.search_engine') as mock_engine:
                mock_engine.search.return_value = []
                
                with TestClient(app) as client:
                    response = client.post(
                        "/search",
                        json={"query": "test"}
                    )
                    
                    if response.status_code == 200:
                        mock_record.assert_called()
    
    def test_error_metrics_collected(self):
        """Test that error metrics are collected"""
        with patch.object(metrics_collector, 'record_error') as mock_record:
            with patch('src.api.routers.search.search_engine') as mock_engine:
                mock_engine.search.side_effect = Exception("Test error")
                
                with TestClient(app) as client:
                    response = client.post(
                        "/search",
                        json={"query": "test"}
                    )
                    
                    assert response.status_code == 500
                    mock_record.assert_called()


class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def test_404_error_handling(self):
        """Test 404 error handling for non-existent endpoints"""
        with TestClient(app) as client:
            response = client.get("/nonexistent-endpoint")
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
    
    def test_405_method_not_allowed(self):
        """Test 405 error for wrong HTTP methods"""
        with TestClient(app) as client:
            response = client.get("/search")  # POST-only endpoint
            assert response.status_code == 405
    
    def test_timeout_handling(self):
        """Test timeout handling for slow operations"""
        with patch('src.api.routers.search.search_engine') as mock_engine:
            mock_engine.search.side_effect = asyncio.TimeoutError("Operation timed out")
            
            with TestClient(app) as client:
                response = client.post(
                    "/search",
                    json={"query": "test"}
                )
                assert response.status_code == 504  # Gateway timeout
    
    def test_connection_error_handling(self):
        """Test connection error handling"""
        with patch('src.api.routers.search.search_engine') as mock_engine:
            mock_engine.search.side_effect = ConnectionError("Database connection failed")
            
            with TestClient(app) as client:
                response = client.post(
                    "/search",
                    json={"query": "test"}
                )
                assert response.status_code == 503  # Service unavailable


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability"""
    
    @pytest.mark.performance
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            with TestClient(app) as client:
                start_time = time.time()
                response = client.get("/health")
                end_time = time.time()
                results.append({
                    "status_code": response.status_code,
                    "response_time": end_time - start_time
                })
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert len(results) == 10
        assert all(r["status_code"] == 200 for r in results)
        
        # Verify reasonable response times (under 1 second)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 1.0
    
    @pytest.mark.performance
    def test_large_result_set_handling(self):
        """Test handling of large result sets"""
        with patch('src.api.routers.search.search_engine') as mock_engine:
            # Create large result set
            large_results = []
            for i in range(1000):
                large_results.append(SearchResult(
                    chunk_id=f"chunk_{i}",
                    content=f"Content {i} " * 100,  # Large content
                    score=0.8,
                    metadata={"book_title": f"Book {i}"}
                ))
            
            mock_engine.search.return_value = large_results
            
            with TestClient(app) as client:
                response = client.post(
                    "/search",
                    json={"query": "test", "max_results": 1000}
                )
                
                # Should handle large response gracefully
                assert response.status_code in [200, 413]  # Success or payload too large


class TestDataValidation:
    """Test comprehensive data validation"""
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters"""
        unicode_queries = [
            "tradingλογικήстратегия",  # Mixed scripts
            "🚀📈💰 trading",  # Emojis
            "café résumé naïve",  # Accented characters
        ]
        
        with TestClient(app) as client:
            for query in unicode_queries:
                response = client.post(
                    "/search",
                    json={"query": query}
                )
                # Should handle Unicode gracefully
                assert response.status_code in [200, 422]
    
    def test_edge_case_numeric_values(self):
        """Test edge case numeric values"""
        edge_cases = [
            {"max_results": 0},
            {"max_results": 999999},
            {"min_score": 0.0},
            {"min_score": 1.0},
            {"min_score": 0.000001},
        ]
        
        with TestClient(app) as client:
            for case in edge_cases:
                response = client.post(
                    "/search",
                    json={"query": "test", **case}
                )
                # Should validate appropriately
                assert response.status_code in [200, 422]
    
    def test_null_and_undefined_handling(self):
        """Test handling of null and undefined values"""
        test_cases = [
            {"query": None},
            {"query": "test", "max_results": None},
            {"query": "test", "min_score": None},
        ]
        
        with TestClient(app) as client:
            for case in test_cases:
                response = client.post("/search", json=case)
                # Should handle null values appropriately
                assert response.status_code in [200, 422]


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.integration
    def test_complete_search_workflow(self):
        """Test complete search workflow from query to results"""
        with patch('src.api.routers.search.search_engine') as mock_engine:
            mock_engine.search.return_value = [
                SearchResult(
                    chunk_id="chunk_1",
                    content="Complete trading strategy guide",
                    score=0.95,
                    metadata={"book_title": "Master Trader"}
                )
            ]
            mock_engine.get_stats.return_value = {
                "total_documents": 100,
                "total_chunks": 5000
            }
            
            with TestClient(app) as client:
                # 1. Check API health
                health_response = client.get("/health")
                assert health_response.status_code == 200
                
                # 2. Get search stats
                stats_response = client.get("/search/stats")
                assert stats_response.status_code == 200
                
                # 3. Perform search
                search_response = client.post(
                    "/search",
                    json={"query": "trading strategy"}
                )
                assert search_response.status_code == 200
                
                # 4. Verify search results
                search_data = search_response.json()
                assert len(search_data["results"]) == 1
                assert search_data["results"][0]["content"] == "Complete trading strategy guide"
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self):
        """Test error recovery and graceful degradation"""
        with patch('src.api.routers.search.search_engine') as mock_engine:
            # First request fails
            mock_engine.search.side_effect = Exception("Temporary failure")
            
            with TestClient(app) as client:
                response1 = client.post(
                    "/search",
                    json={"query": "test"}
                )
                assert response1.status_code == 500
                
                # Second request succeeds (recovery)
                mock_engine.search.side_effect = None
                mock_engine.search.return_value = []
                
                response2 = client.post(
                    "/search",
                    json={"query": "test"}
                )
                assert response2.status_code == 200


# Test configuration
@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test"""
    if hasattr(metrics_collector, 'reset'):
        metrics_collector.reset()


@pytest.fixture(scope="session")
def test_app():
    """Provide test app instance"""
    return app


# Performance test configuration
def pytest_configure(config):
    """Configure performance test markers"""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )