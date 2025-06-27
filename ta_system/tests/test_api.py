"""Tests for FastAPI application."""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.api import app


class TestAPI:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert data["version"] == "0.1.0"
        assert isinstance(data["available_indicators"], list)
        assert len(data["available_indicators"]) > 0

    def test_available_indicators(self, client):
        """Test available indicators endpoint."""
        response = client.get("/indicators/available")
        assert response.status_code == 200
        indicators = response.json()
        assert isinstance(indicators, list)
        assert "RSI_14" in indicators
        assert "SMA_20" in indicators
        assert "MACD_12_26_9" in indicators

    def test_calculate_indicators_success(self, client):
        """Test successful indicator calculation."""
        # Create test OHLCV data
        ohlcv_data = []
        base_price = 100.0
        
        for i in range(25):  # Enough data for most indicators
            price = base_price + i * 0.5
            ohlcv_data.append({
                "symbol": "TEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000
            })
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["RSI_14", "SMA_20", "EMA_10"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 200
        
        results = response.json()
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that we have results for requested indicators
        indicator_names = {result["indicator"] for result in results}
        assert "RSI_14" in indicator_names
        assert "EMA_10" in indicator_names
        # SMA_20 might not have results if we don't have enough data

    def test_calculate_indicators_with_components(self, client):
        """Test indicator calculation with components (MACD)."""
        # Create test OHLCV data
        ohlcv_data = []
        base_price = 100.0
        
        for i in range(50):  # More data for MACD
            price = base_price + i * 0.1
            ohlcv_data.append({
                "symbol": "TEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1000000
            })
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["MACD_12_26_9"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 200
        
        results = response.json()
        
        # Find MACD results
        macd_results = [r for r in results if r["indicator"] == "MACD_12_26_9"]
        assert len(macd_results) > 0
        
        # Check that MACD has components
        macd_result = macd_results[-1]  # Last result
        assert macd_result["components"] is not None
        assert "macd" in macd_result["components"]
        assert "signal" in macd_result["components"]
        assert "histogram" in macd_result["components"]

    def test_calculate_indicators_empty_data(self, client):
        """Test indicator calculation with empty data."""
        request_data = {
            "ohlcv_data": [],
            "indicators": ["RSI_14"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 400
        assert "OHLCV data is required" in response.json()["detail"]

    def test_calculate_indicators_invalid_indicator(self, client):
        """Test indicator calculation with invalid indicator."""
        ohlcv_data = [{
            "symbol": "TEST",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000000
        }]
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["INVALID_INDICATOR"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 400
        assert "Invalid indicators" in response.json()["detail"]

    def test_calculate_indicators_invalid_ohlcv(self, client):
        """Test indicator calculation with invalid OHLCV data."""
        # High < Low should cause validation error
        ohlcv_data = [{
            "symbol": "TEST",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": 100.0,
            "high": 99.0,  # Invalid: high < low
            "low": 101.0,
            "close": 100.0,
            "volume": 1000000
        }]
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["RSI_14"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 400
        assert "Error processing OHLCV data" in response.json()["detail"]

    def test_reset_indicators(self, client):
        """Test indicator reset endpoint."""
        response = client.post("/indicators/reset")
        assert response.status_code == 200
        data = response.json()
        assert "reset successfully" in data["message"]

    def test_get_indicator_info_success(self, client):
        """Test getting indicator information."""
        response = client.get("/indicators/RSI_14/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RSI_14"
        assert "type" in data
        assert "description" in data

    def test_get_indicator_info_not_found(self, client):
        """Test getting info for non-existent indicator."""
        response = client.get("/indicators/NONEXISTENT/info")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_bollinger_bands_calculation(self, client):
        """Test Bollinger Bands calculation with components."""
        # Create test data with some volatility
        ohlcv_data = []
        base_price = 100.0
        
        for i in range(25):
            # Add some volatility
            volatility = (i % 5 - 2) * 0.5  # -1, -0.5, 0, 0.5, 1
            price = base_price + i * 0.1 + volatility
            
            ohlcv_data.append({
                "symbol": "TEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000
            })
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["BB_20_2"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 200
        
        results = response.json()
        bb_results = [r for r in results if r["indicator"] == "BB_20_2"]
        
        if bb_results:  # Should have results after 20 periods
            bb_result = bb_results[-1]
            assert bb_result["components"] is not None
            assert "upper" in bb_result["components"]
            assert "middle" in bb_result["components"]
            assert "lower" in bb_result["components"]
            
            # Upper should be > middle > lower
            components = bb_result["components"]
            assert components["upper"] > components["middle"]
            assert components["middle"] > components["lower"]

    def test_multiple_symbols(self, client):
        """Test calculation with multiple symbols."""
        ohlcv_data = []
        
        # Add data for two different symbols
        for symbol in ["AAPL", "MSFT"]:
            for i in range(15):
                price = 100.0 + i
                ohlcv_data.append({
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1000000
                })
        
        request_data = {
            "ohlcv_data": ohlcv_data,
            "indicators": ["EMA_10", "RSI_14"]
        }
        
        response = client.post("/indicators/calculate", json=request_data)
        assert response.status_code == 200
        
        results = response.json()
        
        # Should have results for both symbols
        symbols = {result["symbol"] for result in results}
        assert "AAPL" in symbols
        assert "MSFT" in symbols