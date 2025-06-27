"""Tests for domain models."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.models import (
    Symbol,
    OHLCV,
    Timeframe,
    Indicator,
    Signal,
    SignalType,
    IndicatorValue,
)


class TestSymbol:
    """Test Symbol model."""

    def test_symbol_creation(self):
        """Test creating a symbol."""
        symbol = Symbol(
            ticker="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            market_cap=3000000000000,
        )
        assert symbol.ticker == "AAPL"
        assert symbol.name == "Apple Inc."
        assert symbol.exchange == "NASDAQ"
        assert symbol.sector == "Technology"
        assert symbol.market_cap == 3000000000000

    def test_symbol_validation(self):
        """Test symbol validation."""
        with pytest.raises(ValueError, match="Ticker must be uppercase"):
            Symbol(
                ticker="aapl",
                name="Apple Inc.",
                exchange="NASDAQ",
            )

    def test_symbol_equality(self):
        """Test symbol equality comparison."""
        symbol1 = Symbol(ticker="AAPL", name="Apple Inc.", exchange="NASDAQ")
        symbol2 = Symbol(ticker="AAPL", name="Apple Inc.", exchange="NASDAQ")
        symbol3 = Symbol(ticker="MSFT", name="Microsoft Corp.", exchange="NASDAQ")
        
        assert symbol1 == symbol2
        assert symbol1 != symbol3


class TestOHLCV:
    """Test OHLCV price data model."""

    def test_ohlcv_creation(self):
        """Test creating OHLCV data."""
        timestamp = datetime.now(timezone.utc)
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=timestamp,
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=1000000,
        )
        
        assert ohlcv.symbol == "AAPL"
        assert ohlcv.timestamp == timestamp
        assert ohlcv.open == Decimal("150.00")
        assert ohlcv.high == Decimal("155.00")
        assert ohlcv.low == Decimal("149.00")
        assert ohlcv.close == Decimal("154.00")
        assert ohlcv.volume == 1000000

    def test_ohlcv_validation(self):
        """Test OHLCV data validation."""
        timestamp = datetime.now(timezone.utc)
        
        # High < Low should raise error
        with pytest.raises(ValueError, match="High must be >= Low"):
            OHLCV(
                symbol="AAPL",
                timestamp=timestamp,
                open=Decimal("150.00"),
                high=Decimal("149.00"),
                low=Decimal("155.00"),
                close=Decimal("154.00"),
                volume=1000000,
            )
        
        # Negative volume should raise error
        with pytest.raises(ValueError, match="Volume must be non-negative"):
            OHLCV(
                symbol="AAPL",
                timestamp=timestamp,
                open=Decimal("150.00"),
                high=Decimal("155.00"),
                low=Decimal("149.00"),
                close=Decimal("154.00"),
                volume=-1000,
            )

    def test_ohlcv_typical_price(self):
        """Test typical price calculation."""
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=1000000,
        )
        
        expected = (Decimal("155.00") + Decimal("149.00") + Decimal("154.00")) / 3
        assert ohlcv.typical_price == expected


class TestTimeframe:
    """Test Timeframe enum."""

    def test_timeframe_values(self):
        """Test timeframe enum values."""
        assert Timeframe.M1.value == "1m"
        assert Timeframe.M5.value == "5m"
        assert Timeframe.M15.value == "15m"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.H4.value == "4h"
        assert Timeframe.D1.value == "1d"
        assert Timeframe.W1.value == "1w"

    def test_timeframe_to_seconds(self):
        """Test timeframe to seconds conversion."""
        assert Timeframe.M1.to_seconds() == 60
        assert Timeframe.M5.to_seconds() == 300
        assert Timeframe.M15.to_seconds() == 900
        assert Timeframe.H1.to_seconds() == 3600
        assert Timeframe.H4.to_seconds() == 14400
        assert Timeframe.D1.to_seconds() == 86400
        assert Timeframe.W1.to_seconds() == 604800


class TestIndicator:
    """Test Indicator model."""

    def test_indicator_creation(self):
        """Test creating an indicator."""
        indicator = Indicator(
            name="RSI",
            full_name="Relative Strength Index",
            category="momentum",
            parameters={"period": 14},
            description="Measures momentum",
        )
        
        assert indicator.name == "RSI"
        assert indicator.full_name == "Relative Strength Index"
        assert indicator.category == "momentum"
        assert indicator.parameters == {"period": 14}
        assert indicator.description == "Measures momentum"

    def test_indicator_with_multiple_parameters(self):
        """Test indicator with multiple parameters."""
        indicator = Indicator(
            name="MACD",
            full_name="Moving Average Convergence Divergence",
            category="trend",
            parameters={"fast": 12, "slow": 26, "signal": 9},
        )
        
        assert indicator.parameters["fast"] == 12
        assert indicator.parameters["slow"] == 26
        assert indicator.parameters["signal"] == 9


class TestIndicatorValue:
    """Test IndicatorValue model."""

    def test_indicator_value_creation(self):
        """Test creating an indicator value."""
        timestamp = datetime.now(timezone.utc)
        value = IndicatorValue(
            symbol="AAPL",
            timestamp=timestamp,
            indicator="RSI",
            timeframe=Timeframe.H1,
            value=Decimal("65.5"),
            parameters={"period": 14},
        )
        
        assert value.symbol == "AAPL"
        assert value.timestamp == timestamp
        assert value.indicator == "RSI"
        assert value.timeframe == Timeframe.H1
        assert value.value == Decimal("65.5")
        assert value.parameters == {"period": 14}

    def test_indicator_value_with_components(self):
        """Test indicator value with multiple components."""
        timestamp = datetime.now(timezone.utc)
        value = IndicatorValue(
            symbol="AAPL",
            timestamp=timestamp,
            indicator="MACD",
            timeframe=Timeframe.H1,
            value=Decimal("2.5"),
            parameters={"fast": 12, "slow": 26, "signal": 9},
            components={
                "macd": Decimal("2.5"),
                "signal": Decimal("2.1"),
                "histogram": Decimal("0.4"),
            },
        )
        
        assert value.components["macd"] == Decimal("2.5")
        assert value.components["signal"] == Decimal("2.1")
        assert value.components["histogram"] == Decimal("0.4")


class TestSignal:
    """Test Signal model."""

    def test_signal_creation(self):
        """Test creating a trading signal."""
        timestamp = datetime.now(timezone.utc)
        signal = Signal(
            symbol="AAPL",
            timestamp=timestamp,
            signal_type=SignalType.BUY,
            strategy="RSI_Oversold",
            confidence=Decimal("0.85"),
            price=Decimal("150.00"),
            indicators={
                "RSI": Decimal("28.5"),
                "MACD_histogram": Decimal("0.5"),
            },
        )
        
        assert signal.symbol == "AAPL"
        assert signal.timestamp == timestamp
        assert signal.signal_type == SignalType.BUY
        assert signal.strategy == "RSI_Oversold"
        assert signal.confidence == Decimal("0.85")
        assert signal.price == Decimal("150.00")
        assert signal.indicators["RSI"] == Decimal("28.5")

    def test_signal_validation(self):
        """Test signal validation."""
        timestamp = datetime.now(timezone.utc)
        
        # Confidence must be between 0 and 1
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Signal(
                symbol="AAPL",
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                strategy="Test",
                confidence=Decimal("1.5"),
                price=Decimal("150.00"),
            )

    def test_signal_types(self):
        """Test signal type enum."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
        assert SignalType.CLOSE.value == "CLOSE"