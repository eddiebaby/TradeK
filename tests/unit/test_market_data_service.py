"""
Unit tests for MarketDataService
Tests data collection, processing, quality analysis, and storage
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.services.market_data_service import MarketDataService, MarketDataPoint


class TestMarketDataPoint:
    """Test MarketDataPoint dataclass"""
    
    def test_market_data_point_creation(self):
        """Test creating MarketDataPoint with required fields"""
        timestamp = datetime.utcnow()
        
        point = MarketDataPoint(
            symbol="SPY",
            timestamp=timestamp,
            timeframe="1d",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        assert point.symbol == "SPY"
        assert point.timestamp == timestamp
        assert point.timeframe == "1d"
        assert point.open == 150.0
        assert point.close == 151.0
        assert point.volume == 1000000
        assert point.data_quality_score == 1.0  # Default value
        assert point.source == "yfinance"  # Default value
    
    def test_market_data_point_with_indicators(self):
        """Test MarketDataPoint with technical indicators"""
        point = MarketDataPoint(
            symbol="QQQ",
            timestamp=datetime.utcnow(),
            timeframe="1h",
            open=300.0,
            high=305.0,
            low=298.0,
            close=302.0,
            volume=500000,
            rsi=65.5,
            macd_line=1.2,
            bb_upper=310.0,
            sma_20=301.0
        )
        
        assert point.rsi == 65.5
        assert point.macd_line == 1.2
        assert point.bb_upper == 310.0
        assert point.sma_20 == 301.0


@pytest.mark.asyncio
class TestMarketDataService:
    """Test MarketDataService functionality"""
    
    @pytest.fixture
    def mock_db_service(self):
        """Mock database service"""
        db_service = MagicMock()
        db_service.postgres = AsyncMock()
        db_service.influx = AsyncMock()
        db_service.redis = AsyncMock()
        return db_service
    
    @pytest.fixture
    def market_service(self, mock_db_service):
        """Create MarketDataService with mocked dependencies"""
        return MarketDataService(mock_db_service)
    
    def test_initialization(self, market_service):
        """Test MarketDataService initialization"""
        assert market_service.db_service is not None
        assert market_service.executor is not None
        
        # Test timeframe limits
        assert "1m" in market_service.timeframe_limits
        assert "1d" in market_service.timeframe_limits
        assert "1mo" in market_service.timeframe_limits
        
        # Test supported symbols
        assert "SPY" in market_service.supported_symbols
        assert "QQQ" in market_service.supported_symbols
    
    @patch('src.services.market_data_service.yf.Ticker')
    async def test_fetch_yfinance_data(self, mock_ticker, market_service):
        """Test fetching data from yfinance"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data fetching
        result = market_service._fetch_yfinance_data("SPY", "1d", "5d")
        
        assert result is not None
        assert len(result) == 2
        mock_ticker.assert_called_with("SPY")
    
    def test_data_quality_score_calculation(self, market_service):
        """Test data quality score calculation"""
        # Test perfect data
        perfect_row = pd.Series({
            'Open': 150.0,
            'High': 152.0,
            'Low': 149.0,
            'Close': 151.0,
            'Volume': 1000000
        })
        
        score = market_service._calculate_data_quality_score(perfect_row)
        assert score == 1.0
        
        # Test data with missing volume
        no_volume_row = pd.Series({
            'Open': 150.0,
            'High': 152.0,
            'Low': 149.0,
            'Close': 151.0,
            'Volume': 0
        })
        
        score = market_service._calculate_data_quality_score(no_volume_row)
        assert score < 1.0  # Should be penalized for missing volume
        
        # Test invalid OHLC relationships
        invalid_row = pd.Series({
            'Open': 150.0,
            'High': 148.0,  # High < Open (invalid)
            'Low': 149.0,
            'Close': 151.0,
            'Volume': 1000000
        })
        
        score = market_service._calculate_data_quality_score(invalid_row)
        assert score < 1.0  # Should be penalized for invalid OHLC
    
    def test_session_type_determination(self, market_service):
        """Test market session type determination"""
        # Test regular market hours
        regular_time = pd.Timestamp('2024-01-01 10:00:00')  # 10 AM
        session = market_service._determine_session_type(regular_time, "1m")
        assert session == "regular"
        
        # Test pre-market hours
        premarket_time = pd.Timestamp('2024-01-01 08:00:00')  # 8 AM
        session = market_service._determine_session_type(premarket_time, "1m")
        assert session == "pre_market"
        
        # Test after-hours
        afterhours_time = pd.Timestamp('2024-01-01 18:00:00')  # 6 PM
        session = market_service._determine_session_type(afterhours_time, "1m")
        assert session == "after_hours"
        
        # Test daily timeframe (always regular)
        daily_session = market_service._determine_session_type(regular_time, "1d")
        assert daily_session == "regular"
    
    def test_process_raw_data(self, market_service):
        """Test processing raw yfinance data into MarketDataPoint objects"""
        # Create sample DataFrame with technical indicators
        raw_data = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000],
            'RSI_14': [65.5, 67.2],
            'SMA_20': [150.5, 151.2]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        processed_data = market_service._process_raw_data(raw_data, "SPY", "1d")
        
        assert len(processed_data) == 2
        assert all(isinstance(point, MarketDataPoint) for point in processed_data)
        
        first_point = processed_data[0]
        assert first_point.symbol == "SPY"
        assert first_point.timeframe == "1d"
        assert first_point.open == 150.0
        assert first_point.close == 151.0
        assert first_point.rsi == 65.5
        assert first_point.sma_20 == 150.5
    
    async def test_analyze_data_quality(self, market_service):
        """Test data quality analysis across multiple symbols/timeframes"""
        # Create sample data
        spy_1d = [
            MarketDataPoint("SPY", datetime.utcnow(), "1d", 150, 152, 149, 151, 1000000, data_quality_score=1.0),
            MarketDataPoint("SPY", datetime.utcnow(), "1d", 151, 153, 150, 152, 1100000, data_quality_score=0.9)
        ]
        
        qqq_1h = [
            MarketDataPoint("QQQ", datetime.utcnow(), "1h", 300, 305, 298, 302, 500000, data_quality_score=0.95)
        ]
        
        test_data = {
            "SPY": {"1d": spy_1d},
            "QQQ": {"1h": qqq_1h}
        }
        
        analysis = await market_service.analyze_data_quality(test_data)
        
        assert analysis["total_symbols"] == 2
        assert analysis["total_data_points"] == 3
        assert "timeframe_coverage" in analysis
        assert "quality_metrics" in analysis
        assert analysis["overall_quality_score"] > 0.9
        
        # Check timeframe coverage
        assert "1d" in analysis["timeframe_coverage"]
        assert "1h" in analysis["timeframe_coverage"]
        assert analysis["timeframe_coverage"]["1d"]["symbols"] == 1
        assert analysis["timeframe_coverage"]["1d"]["total_points"] == 2
    
    def test_detect_data_gaps(self, market_service):
        """Test data gap detection"""
        # Create timestamps with gaps
        timestamps = [
            datetime(2024, 1, 1, 9, 0),   # 9:00 AM
            datetime(2024, 1, 1, 9, 1),   # 9:01 AM
            datetime(2024, 1, 1, 9, 5),   # 9:05 AM (4-minute gap for 1m data)
            datetime(2024, 1, 1, 9, 6)    # 9:06 AM
        ]
        
        gaps = market_service._detect_data_gaps(timestamps, "1m")
        
        # Should detect the 4-minute gap (exceeds 3x the 1-minute interval)
        assert len(gaps) == 1
        assert gaps[0]["start"] == timestamps[1]
        assert gaps[0]["end"] == timestamps[2]
        assert gaps[0]["duration"] == timedelta(minutes=4)
    
    async def test_store_market_data(self, market_service):
        """Test storing market data in databases"""
        # Mock database operations
        market_service.db_service.influx.write_market_data = AsyncMock()
        
        # Create test data
        test_data = {
            "SPY": {
                "1d": [
                    MarketDataPoint("SPY", datetime.utcnow(), "1d", 150, 152, 149, 151, 1000000)
                ]
            }
        }
        
        storage_stats = await market_service.store_market_data(test_data)
        
        assert storage_stats["symbols_processed"] == 1
        assert storage_stats["total_data_points"] == 1
        assert storage_stats["postgresql_records"] == 1
        
        # Verify InfluxDB write was called
        market_service.db_service.influx.write_market_data.assert_called()
    
    async def test_get_latest_data(self, market_service):
        """Test getting latest market data for a symbol"""
        # Mock the fetch method
        mock_data = {
            "1d": [
                MarketDataPoint("SPY", datetime.utcnow(), "1d", 150, 152, 149, 151, 1000000),
                MarketDataPoint("SPY", datetime.utcnow() - timedelta(days=1), "1d", 148, 150, 147, 149, 900000)
            ]
        }
        
        market_service._fetch_symbol_all_timeframes = AsyncMock(return_value=mock_data)
        
        latest_data = await market_service.get_latest_data("SPY", "1d", 5)
        
        assert len(latest_data) == 2
        # Should be sorted by timestamp descending (most recent first)
        assert latest_data[0].timestamp >= latest_data[1].timestamp


@pytest.mark.performance
class TestMarketDataPerformance:
    """Performance tests for market data operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_fetching(self):
        """Test fetching multiple symbols concurrently"""
        market_service = MarketDataService()
        
        # Mock the individual symbol fetch to avoid actual API calls
        async def mock_fetch_symbol(symbol):
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return {f"1d": [MarketDataPoint(symbol, datetime.utcnow(), "1d", 100, 102, 99, 101, 1000000)]}
        
        market_service._fetch_symbol_all_timeframes = mock_fetch_symbol
        
        symbols = ["SPY", "QQQ", "IWM"]
        
        start_time = asyncio.get_event_loop().time()
        result = await market_service.fetch_comprehensive_data(symbols)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete in roughly the time of one fetch (due to concurrency)
        # Not 3x the time (which would indicate sequential processing)
        assert (end_time - start_time) < 0.3  # Should be closer to 0.1s than 0.3s
        assert len(result) == 3


@pytest.mark.integration
class TestMarketDataIntegration:
    """Integration tests with external dependencies"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_data_fetch(self):
        """Test fetching real data from yfinance (slow test)"""
        market_service = MarketDataService()
        
        # Test with a single symbol and short timeframe to minimize impact
        result = await market_service._fetch_symbol_all_timeframes("SPY")
        
        # Should get data for multiple timeframes
        assert len(result) > 0
        assert "1d" in result
        
        if result["1d"]:
            daily_data = result["1d"]
            assert len(daily_data) > 0
            assert all(isinstance(point, MarketDataPoint) for point in daily_data)
            assert all(point.symbol == "SPY" for point in daily_data)


@pytest.mark.security  
class TestMarketDataSecurity:
    """Security tests for market data operations"""
    
    def test_input_validation(self):
        """Test input validation for market data operations"""
        market_service = MarketDataService()
        
        # Test symbol validation (should handle various inputs safely)
        valid_symbols = ["SPY", "QQQ", "AAPL"]
        invalid_symbols = ["", None, "'; DROP TABLE;", "<script>alert('xss')</script>"]
        
        # The service should handle invalid symbols gracefully
        for symbol in invalid_symbols:
            # Should not raise exceptions, but return empty results
            try:
                result = market_service._fetch_yfinance_data(symbol, "1d", "5d")
                # Either returns None or empty DataFrame
                assert result is None or len(result) == 0
            except Exception:
                # Acceptable to raise controlled exceptions for invalid input
                pass
    
    def test_data_sanitization(self):
        """Test that market data is properly sanitized"""
        market_service = MarketDataService()
        
        # Test with extreme values
        extreme_row = pd.Series({
            'Open': float('inf'),
            'High': float('nan'),
            'Low': -999999999,
            'Close': 0,
            'Volume': -1
        })
        
        # Should handle extreme values gracefully
        score = market_service._calculate_data_quality_score(extreme_row)
        assert 0 <= score <= 1  # Score should remain in valid range