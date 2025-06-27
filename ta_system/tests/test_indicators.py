"""Tests for technical analysis indicators."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
import numpy as np

from src.indicators import (
    RSICalculator,
    MACDCalculator,
    BollingerBandsCalculator,
    SMApCalculator,
    EMACalculator,
    ATRCalculator,
    IndicatorCalculator,
)
from src.models import OHLCV, Timeframe


class TestSMACalculator:
    """Test Simple Moving Average calculator."""

    def test_sma_calculation(self):
        """Test SMA calculation with known values."""
        calculator = SMApCalculator(period=3)
        
        # Create test data
        prices = [10, 12, 14, 16, 18]
        timestamps = [datetime.now(timezone.utc) for _ in range(5)]
        
        results = []
        for i, (price, ts) in enumerate(zip(prices, timestamps)):
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=ts,
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            if result is not None:
                results.append(result)
        
        # First 2 values should be None (not enough data)
        # Third value should be (10+12+14)/3 = 12
        # Fourth value should be (12+14+16)/3 = 14
        # Fifth value should be (14+16+18)/3 = 16
        assert len(results) == 3
        assert results[0].value == Decimal("12")
        assert results[1].value == Decimal("14")
        assert results[2].value == Decimal("16")

    def test_sma_insufficient_data(self):
        """Test SMA returns None with insufficient data."""
        calculator = SMApCalculator(period=5)
        
        ohlcv = OHLCV(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("10"),
            high=Decimal("11"),
            low=Decimal("9"),
            close=Decimal("10"),
            volume=1000,
        )
        
        result = calculator.calculate(ohlcv)
        assert result is None

    def test_sma_reset(self):
        """Test SMA calculator reset functionality."""
        calculator = SMApCalculator(period=3)
        
        # Add some data
        ohlcv = OHLCV(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("10"),
            high=Decimal("11"),
            low=Decimal("9"),
            close=Decimal("10"),
            volume=1000,
        )
        calculator.calculate(ohlcv)
        
        # Reset and verify state is cleared
        calculator.reset()
        assert len(calculator.prices) == 0


class TestRSICalculator:
    """Test RSI calculator."""

    def test_rsi_calculation(self):
        """Test RSI calculation with known values."""
        calculator = RSICalculator(period=14)
        
        # Create test data that should produce predictable RSI
        # Rising prices should produce RSI > 50
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                 46.03, 46.83, 47.69, 47.54, 47.79, 48.15, 47.64, 46.95, 46.49, 46.26]
        
        timestamps = [datetime.now(timezone.utc) for _ in range(len(prices))]
        
        results = []
        for price, ts in zip(prices, timestamps):
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=ts,
                open=Decimal(str(price)),
                high=Decimal(str(price + 0.5)),
                low=Decimal(str(price - 0.5)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            if result is not None:
                results.append(result)
        
        # Should have results starting from 14th data point
        assert len(results) >= 6
        
        # RSI should be between 0 and 100
        for result in results:
            assert 0 <= result.value <= 100

    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements."""
        calculator = RSICalculator(period=14)
        
        # All rising prices should produce RSI near 100
        base_price = 100
        for i in range(20):
            price = base_price + i * 2  # Consistent gains
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            
            # RSI should be high after enough consistent gains
            if result is not None and i >= 15:
                assert result.value > 80


class TestMACDCalculator:
    """Test MACD calculator."""

    def test_macd_calculation(self):
        """Test MACD calculation."""
        calculator = MACDCalculator(fast=12, slow=26, signal=9)
        
        # Create trending price data
        base_price = 100
        prices = [base_price + i * 0.5 for i in range(50)]
        timestamps = [datetime.now(timezone.utc) for _ in range(50)]
        
        results = []
        for price, ts in zip(prices, timestamps):
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=ts,
                open=Decimal(str(price)),
                high=Decimal(str(price + 0.5)),
                low=Decimal(str(price - 0.5)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            if result is not None:
                results.append(result)
        
        # Should have results after enough data
        assert len(results) > 0
        
        # Check that components exist
        latest = results[-1]
        assert "macd" in latest.components
        assert "signal" in latest.components
        assert "histogram" in latest.components
        
        # Histogram should equal macd - signal
        expected_histogram = latest.components["macd"] - latest.components["signal"]
        assert abs(latest.components["histogram"] - expected_histogram) < Decimal("0.001")


class TestBollingerBandsCalculator:
    """Test Bollinger Bands calculator."""

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        calculator = BollingerBandsCalculator(period=20, std_dev=2)
        
        # Create data with some volatility
        np.random.seed(42)  # For reproducible results
        base_price = 100
        prices = [base_price + np.random.normal(0, 2) for _ in range(30)]
        timestamps = [datetime.now(timezone.utc) for _ in range(30)]
        
        results = []
        for price, ts in zip(prices, timestamps):
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=ts,
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            if result is not None:
                results.append(result)
        
        # Should have results after 20 periods
        assert len(results) > 0
        
        # Check components
        latest = results[-1]
        assert "upper" in latest.components
        assert "middle" in latest.components
        assert "lower" in latest.components
        
        # Upper should be > middle > lower
        assert latest.components["upper"] > latest.components["middle"]
        assert latest.components["middle"] > latest.components["lower"]


class TestATRCalculator:
    """Test Average True Range calculator."""

    def test_atr_calculation(self):
        """Test ATR calculation."""
        calculator = ATRCalculator(period=14)
        
        # Create data with varying ranges
        base_price = 100
        for i in range(20):
            price = base_price + i * 0.1
            range_size = 2 + i * 0.1  # Increasing volatility
            
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal(str(price)),
                high=Decimal(str(price + range_size)),
                low=Decimal(str(price - range_size)),
                close=Decimal(str(price + range_size * 0.5)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            
            # ATR should be positive and increasing
            if result is not None:
                assert result.value > 0
                if i > 15:  # After enough data
                    assert result.value > Decimal("2")  # Should reflect increased volatility


class TestEMACalculator:
    """Test Exponential Moving Average calculator."""

    def test_ema_calculation(self):
        """Test EMA calculation."""
        calculator = EMACalculator(period=10)
        
        # Create trending data
        prices = [100 + i for i in range(20)]
        timestamps = [datetime.now(timezone.utc) for _ in range(20)]
        
        results = []
        for price, ts in zip(prices, timestamps):
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=ts,
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=1000,
            )
            result = calculator.calculate(ohlcv)
            if result is not None:
                results.append(result)
        
        # Should have results starting from first data point
        assert len(results) == 20
        
        # EMA should be trending upward with the prices
        assert results[-1].value > results[0].value
        
        # Last EMA should be close to but less than the last price
        # (since EMA lags the price)
        last_price = Decimal("119")  # 100 + 19
        assert results[-1].value < last_price
        assert results[-1].value > last_price * Decimal("0.9")


class TestIndicatorCalculator:
    """Test main indicator calculator orchestrator."""

    def test_calculate_multiple_indicators(self):
        """Test calculating multiple indicators simultaneously."""
        calculator = IndicatorCalculator()
        
        # Register indicators
        calculator.register("SMA_20", SMApCalculator(period=20))
        calculator.register("RSI_14", RSICalculator(period=14))
        calculator.register("EMA_10", EMACalculator(period=10))
        
        # Create test data
        base_price = 100
        results = []
        
        for i in range(25):
            price = base_price + i * 0.5
            ohlcv = OHLCV(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=1000,
            )
            
            indicator_results = calculator.calculate_all(ohlcv)
            results.append(indicator_results)
        
        # Should have results for all indicators
        latest_results = results[-1]
        assert "SMA_20" in latest_results
        assert "RSI_14" in latest_results
        assert "EMA_10" in latest_results
        
        # Each indicator should have proper values
        assert latest_results["SMA_20"].value > 0
        assert 0 <= latest_results["RSI_14"].value <= 100
        assert latest_results["EMA_10"].value > 0

    def test_reset_all_indicators(self):
        """Test resetting all registered indicators."""
        calculator = IndicatorCalculator()
        
        # Register and use indicators
        sma_calc = SMApCalculator(period=5)
        calculator.register("SMA_5", sma_calc)
        
        # Add some data
        ohlcv = OHLCV(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=1000,
        )
        calculator.calculate_all(ohlcv)
        
        # Reset all
        calculator.reset_all()
        
        # Verify reset worked
        assert len(sma_calc.prices) == 0