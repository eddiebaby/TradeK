"""
Unit Tests for LDES Data Collection Components

Tests market data providers, storage, and the collector orchestrator.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.ldes.core.models import MarketData, LiquiditySignal, SignalType, Side
from src.ldes.core.config import LDESConfig, MarketDataConfig, InfluxDBConfig
from src.ldes.data.market_data_collector import MarketDataCollector
from src.ldes.data.alpaca_client import MockAlpacaDataProvider
from src.ldes.data.binance_client import MockBinanceDataProvider
from src.ldes.data.influxdb_storage import MockInfluxDBStorage


class TestMarketDataCollector:
    """Test MarketDataCollector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config
        self.config = LDESConfig()
        
        # Create mock storage
        self.storage = MockInfluxDBStorage(InfluxDBConfig())
        
        # Create collector
        self.collector = MarketDataCollector(self.config, self.storage)
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        assert self.collector.config == self.config
        assert self.collector.storage == self.storage
        assert len(self.collector.providers) == 0
        assert not self.collector.is_running
        assert self.collector.data_points_processed == 0
    
    def test_add_remove_providers(self):
        """Test adding and removing providers."""
        # Create mock providers
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        binance_provider = MockBinanceDataProvider(MarketDataConfig())
        
        # Add providers
        self.collector.add_provider("alpaca", alpaca_provider)
        self.collector.add_provider("binance", binance_provider)
        
        assert len(self.collector.providers) == 2
        assert "alpaca" in self.collector.providers
        assert "binance" in self.collector.providers
        
        # Remove provider
        self.collector.remove_provider("alpaca")
        assert len(self.collector.providers) == 1
        assert "alpaca" not in self.collector.providers
        assert "binance" in self.collector.providers
    
    @pytest.mark.asyncio
    async def test_connect_all_providers(self):
        """Test connecting to all providers."""
        # Add mock providers
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        binance_provider = MockBinanceDataProvider(MarketDataConfig())
        
        self.collector.add_provider("alpaca", alpaca_provider)
        self.collector.add_provider("binance", binance_provider)
        
        # Connect all
        await self.collector.connect_all()
        
        # Verify connections
        assert alpaca_provider.is_connected
        assert binance_provider.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect_all_providers(self):
        """Test disconnecting from all providers."""
        # Add and connect providers
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        binance_provider = MockBinanceDataProvider(MarketDataConfig())
        
        self.collector.add_provider("alpaca", alpaca_provider)
        self.collector.add_provider("binance", binance_provider)
        
        await self.collector.connect_all()
        assert alpaca_provider.is_connected
        assert binance_provider.is_connected
        
        # Disconnect all
        await self.collector.disconnect_all()
        
        # Verify disconnections
        assert not alpaca_provider.is_connected
        assert not binance_provider.is_connected
    
    @pytest.mark.asyncio
    async def test_subscribe_symbols(self):
        """Test subscribing to symbols."""
        # Add and connect provider
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        self.collector.add_provider("alpaca", alpaca_provider)
        await self.collector.connect_all()
        
        # Subscribe to symbols
        symbols = ["SPY", "QQQ", "IWM"]
        await self.collector.subscribe_symbols(symbols)
        
        # Verify subscription
        assert len(self.collector.subscribed_symbols) == 3
        for symbol in symbols:
            assert symbol in self.collector.subscribed_symbols
    
    @pytest.mark.asyncio
    async def test_data_processing(self):
        """Test data processing and storage."""
        # Set up storage
        await self.storage.connect()
        
        # Create test market data
        test_data = MarketData(
            symbol="SPY",
            timestamp=datetime.now(),
            last_price=Decimal("150.00"),
            volume=1000,
            source="test"
        )
        
        # Process data
        await self.collector._process_market_data(test_data, "test_source")
        
        # Verify processing
        assert self.collector.data_points_processed == 1
        assert self.storage.get_data_count("market_data") == 1
    
    @pytest.mark.asyncio
    async def test_get_latest_data(self):
        """Test getting latest data from providers."""
        # Add mock provider
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        self.collector.add_provider("alpaca", alpaca_provider)
        await self.collector.connect_all()
        
        # Get latest data for a symbol that exists in config
        latest = await self.collector.get_latest_data("SPY")
        
        # Should return mock data
        assert latest is not None
        assert latest.symbol == "SPY"
        assert latest.source == "alpaca_mock"
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self):
        """Test getting historical data."""
        # Add mock provider
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        self.collector.add_provider("alpaca", alpaca_provider)
        await self.collector.connect_all()
        
        # Get historical data for a symbol that exists in config
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        historical_data = await self.collector.get_historical_data(
            "SPY", start_date, end_date
        )
        
        # Should return mock data
        assert len(historical_data) > 0
        assert all(data.symbol == "SPY" for data in historical_data)
    
    def test_get_status(self):
        """Test getting collector status."""
        # Add providers
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        self.collector.add_provider("alpaca", alpaca_provider)
        
        # Get status
        status = self.collector.get_status()
        
        # Verify status structure
        assert "is_running" in status
        assert "providers_count" in status
        assert "connected_providers" in status
        assert "subscribed_symbols_count" in status
        assert "data_points_processed" in status
        assert "errors_encountered" in status
        
        assert status["providers_count"] == 1
        assert status["is_running"] is False
    
    @pytest.mark.asyncio
    async def test_managed_collection_context(self):
        """Test managed collection context manager."""
        # Add mock provider
        alpaca_provider = MockAlpacaDataProvider(MarketDataConfig())
        self.collector.add_provider("alpaca", alpaca_provider)
        
        symbols = ["SPY", "QQQ"]
        
        # Use context manager
        async with self.collector.managed_collection(symbols) as collector:
            assert collector.is_running
            assert alpaca_provider.is_connected
            assert len(collector.subscribed_symbols) == 2
        
        # Verify cleanup
        assert not self.collector.is_running
        assert not alpaca_provider.is_connected
        assert len(self.collector.subscribed_symbols) == 0


class TestMockAlpacaProvider:
    """Test MockAlpacaDataProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MarketDataConfig()
        self.provider = MockAlpacaDataProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test provider connection lifecycle."""
        assert not self.provider.is_connected
        
        await self.provider.connect()
        assert self.provider.is_connected
        
        await self.provider.disconnect()
        assert not self.provider.is_connected
    
    @pytest.mark.asyncio
    async def test_subscription(self):
        """Test symbol subscription."""
        await self.provider.connect()
        
        symbols = ["SPY", "QQQ"]
        await self.provider.subscribe(symbols)
        
        # Mock doesn't track subscriptions, just logs
        # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_data_stream(self):
        """Test mock data stream."""
        await self.provider.connect()
        
        # Collect a few data points
        data_points = []
        async for data in self.provider.get_stream():
            data_points.append(data)
            if len(data_points) >= 5:
                break
        
        # Verify data structure
        assert len(data_points) == 5
        for data in data_points:
            assert isinstance(data, MarketData)
            assert data.source == "alpaca_mock"
            assert data.last_price is not None
    
    @pytest.mark.asyncio
    async def test_historical_data(self):
        """Test historical data generation."""
        start_date = datetime.now() - timedelta(hours=2)
        end_date = datetime.now()
        
        historical_data = await self.provider.get_historical_data(
            "SPY", start_date, end_date
        )
        
        # Verify data
        assert len(historical_data) > 0
        assert all(data.symbol == "SPY" for data in historical_data)
        assert all(data.source == "alpaca_mock" for data in historical_data)
        
        # Verify time ordering
        timestamps = [data.timestamp for data in historical_data]
        assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_latest_quote(self):
        """Test latest quote generation."""
        quote = await self.provider.get_latest_quote("SPY")
        
        assert quote is not None
        assert quote.symbol == "SPY"
        assert quote.source == "alpaca_mock"
        assert quote.bid_price is not None
        assert quote.ask_price is not None


class TestMockBinanceProvider:
    """Test MockBinanceDataProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MarketDataConfig()
        self.provider = MockBinanceDataProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_crypto_data_stream(self):
        """Test crypto mock data stream."""
        await self.provider.connect()
        
        # Collect a few data points
        data_points = []
        async for data in self.provider.get_stream():
            data_points.append(data)
            if len(data_points) >= 3:
                break
        
        # Verify crypto data characteristics
        assert len(data_points) == 3
        for data in data_points:
            assert isinstance(data, MarketData)
            assert data.source == "binance_mock"
            # Crypto prices should be much higher than stock prices
            assert float(data.last_price) > 1000
    
    @pytest.mark.asyncio
    async def test_crypto_historical_data(self):
        """Test crypto historical data."""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now()
        
        historical_data = await self.provider.get_historical_data(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify crypto data
        assert len(historical_data) > 0
        assert all(data.symbol == "BTCUSDT" for data in historical_data)
        assert all(data.source == "binance_mock" for data in historical_data)
        
        # Crypto prices should be higher
        prices = [float(data.last_price) for data in historical_data]
        assert all(price > 10000 for price in prices)


class TestMockInfluxDBStorage:
    """Test MockInfluxDBStorage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InfluxDBConfig()
        self.storage = MockInfluxDBStorage(self.config)
    
    @pytest.mark.asyncio
    async def test_storage_lifecycle(self):
        """Test storage connection lifecycle."""
        assert not self.storage.is_connected
        
        await self.storage.connect()
        assert self.storage.is_connected
        
        await self.storage.disconnect()
        assert not self.storage.is_connected
    
    @pytest.mark.asyncio
    async def test_store_market_data(self):
        """Test storing market data."""
        await self.storage.connect()
        
        # Create test data
        market_data = MarketData(
            symbol="SPY",
            timestamp=datetime.now(),
            last_price=Decimal("150.00"),
            volume=1000,
            source="test"
        )
        
        # Store data
        await self.storage.store_market_data(market_data)
        
        # Verify storage
        assert self.storage.get_data_count("market_data") == 1
    
    @pytest.mark.asyncio
    async def test_store_liquidity_signal(self):
        """Test storing liquidity signal."""
        await self.storage.connect()
        
        # Create test signal
        signal = LiquiditySignal(
            symbol="SPY",
            timestamp=datetime.now(),
            signal_type=SignalType.VOLUME_SPIKE,
            strength=0.8,
            confidence=0.9,
            expected_direction=Side.LONG,
            expected_move_bps=150.0,
            time_horizon_seconds=300
        )
        
        # Store signal
        await self.storage.store_liquidity_signal(signal)
        
        # Verify storage
        assert self.storage.get_data_count("liquidity_signals") == 1
    
    @pytest.mark.asyncio
    async def test_store_multiple_data_types(self):
        """Test storing multiple data types."""
        await self.storage.connect()
        
        # Store different types of data
        market_data = MarketData(
            symbol="SPY", timestamp=datetime.now(),
            last_price=Decimal("150.00"), source="test"
        )
        await self.storage.store_market_data(market_data)
        
        signal = LiquiditySignal(
            symbol="SPY", timestamp=datetime.now(),
            signal_type=SignalType.VOLUME_SPIKE, strength=0.8,
            confidence=0.9, expected_direction=Side.LONG,
            expected_move_bps=150.0, time_horizon_seconds=300
        )
        await self.storage.store_liquidity_signal(signal)
        
        # Verify all stored
        assert self.storage.get_data_count("market_data") == 1
        assert self.storage.get_data_count("liquidity_signals") == 1
        assert self.storage.get_data_count("positions") == 0
        assert self.storage.get_data_count("trade_signals") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])