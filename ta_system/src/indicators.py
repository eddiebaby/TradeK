"""Technical analysis indicators implementation."""

from abc import ABC, abstractmethod
from collections import deque
from decimal import Decimal
from typing import Deque, Dict, List, Optional

from .models import OHLCV, IndicatorValue, Timeframe


class BaseIndicator(ABC):
    """Base class for technical indicators."""

    def __init__(self, name: str):
        """Initialize indicator."""
        self.name = name
        self.reset()

    @abstractmethod
    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate indicator value for given OHLCV data."""

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""


class SMApCalculator(BaseIndicator):
    """Simple Moving Average calculator."""

    def __init__(self, period: int):
        """Initialize SMA calculator."""
        self.period = period
        super().__init__(f"SMA_{period}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.prices: Deque[Decimal] = deque(maxlen=self.period)

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate SMA value."""
        self.prices.append(ohlcv.close)

        if len(self.prices) < self.period:
            return None

        sma_value = sum(self.prices) / len(self.prices)

        return IndicatorValue(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            indicator=self.name,
            timeframe=Timeframe.H1,  # Default timeframe
            value=sma_value,
            parameters={"period": self.period},
        )


class EMACalculator(BaseIndicator):
    """Exponential Moving Average calculator."""

    def __init__(self, period: int):
        """Initialize EMA calculator."""
        self.period = period
        self.multiplier = Decimal(2) / (period + 1)
        super().__init__(f"EMA_{period}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.ema_value: Optional[Decimal] = None
        self.is_first = True

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate EMA value."""
        if self.is_first:
            self.ema_value = ohlcv.close
            self.is_first = False
        else:
            self.ema_value = (
                ohlcv.close * self.multiplier + self.ema_value * (1 - self.multiplier)
            )

        return IndicatorValue(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            indicator=self.name,
            timeframe=Timeframe.H1,
            value=self.ema_value,
            parameters={"period": self.period},
        )


class RSICalculator(BaseIndicator):
    """Relative Strength Index calculator."""

    def __init__(self, period: int = 14):
        """Initialize RSI calculator."""
        self.period = period
        super().__init__(f"RSI_{period}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.prices: List[Decimal] = []
        self.gains: Deque[Decimal] = deque(maxlen=self.period)
        self.losses: Deque[Decimal] = deque(maxlen=self.period)
        self.avg_gain: Optional[Decimal] = None
        self.avg_loss: Optional[Decimal] = None

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate RSI value."""
        if len(self.prices) > 0:
            change = ohlcv.close - self.prices[-1]
            gain = max(change, Decimal(0))
            loss = max(-change, Decimal(0))

            self.gains.append(gain)
            self.losses.append(loss)

        self.prices.append(ohlcv.close)

        if len(self.gains) < self.period:
            return None

        if self.avg_gain is None:
            # First calculation - simple average
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            # Subsequent calculations - Wilder's smoothing
            latest_gain = self.gains[-1]
            latest_loss = self.losses[-1]
            
            self.avg_gain = (self.avg_gain * (self.period - 1) + latest_gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + latest_loss) / self.period

        if self.avg_loss == 0:
            rsi = Decimal(100)
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))

        return IndicatorValue(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            indicator=self.name,
            timeframe=Timeframe.H1,
            value=rsi,
            parameters={"period": self.period},
        )


class MACDCalculator(BaseIndicator):
    """MACD (Moving Average Convergence Divergence) calculator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD calculator."""
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        super().__init__(f"MACD_{fast}_{slow}_{signal}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.fast_ema = EMACalculator(self.fast)
        self.slow_ema = EMACalculator(self.slow)
        self.signal_ema = EMACalculator(self.signal_period)
        self.macd_values: List[Decimal] = []

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate MACD value."""
        fast_value = self.fast_ema.calculate(ohlcv)
        slow_value = self.slow_ema.calculate(ohlcv)

        if fast_value is None or slow_value is None:
            return None

        macd_line = fast_value.value - slow_value.value
        self.macd_values.append(macd_line)

        # Calculate signal line (EMA of MACD)
        if len(self.macd_values) >= self.signal_period:
            # Create dummy OHLCV for signal EMA calculation
            signal_ohlcv = OHLCV(
                symbol=ohlcv.symbol,
                timestamp=ohlcv.timestamp,
                open=macd_line,
                high=macd_line,
                low=macd_line,
                close=macd_line,
                volume=0,
            )
            signal_value = self.signal_ema.calculate(signal_ohlcv)
            
            if signal_value is not None:
                histogram = macd_line - signal_value.value

                return IndicatorValue(
                    symbol=ohlcv.symbol,
                    timestamp=ohlcv.timestamp,
                    indicator=self.name,
                    timeframe=Timeframe.H1,
                    value=macd_line,
                    parameters={
                        "fast": self.fast,
                        "slow": self.slow,
                        "signal": self.signal_period,
                    },
                    components={
                        "macd": macd_line,
                        "signal": signal_value.value,
                        "histogram": histogram,
                    },
                )

        return None


class BollingerBandsCalculator(BaseIndicator):
    """Bollinger Bands calculator."""

    def __init__(self, period: int = 20, std_dev: float = 2):
        """Initialize Bollinger Bands calculator."""
        self.period = period
        self.std_dev = Decimal(str(std_dev))
        super().__init__(f"BB_{period}_{std_dev}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.prices: Deque[Decimal] = deque(maxlen=self.period)

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate Bollinger Bands."""
        self.prices.append(ohlcv.close)

        if len(self.prices) < self.period:
            return None

        # Calculate middle band (SMA)
        middle = sum(self.prices) / len(self.prices)

        # Calculate standard deviation
        variance = sum((price - middle) ** 2 for price in self.prices) / len(self.prices)
        std_dev_value = variance.sqrt()

        # Calculate bands
        upper = middle + (self.std_dev * std_dev_value)
        lower = middle - (self.std_dev * std_dev_value)

        # Return position within bands as main value (0-1 scale)
        if upper == lower:
            position = Decimal("0.5")
        else:
            position = (ohlcv.close - lower) / (upper - lower)

        return IndicatorValue(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            indicator=self.name,
            timeframe=Timeframe.H1,
            value=position,
            parameters={"period": self.period, "std_dev": float(self.std_dev)},
            components={
                "upper": upper,
                "middle": middle,
                "lower": lower,
            },
        )


class ATRCalculator(BaseIndicator):
    """Average True Range calculator."""

    def __init__(self, period: int = 14):
        """Initialize ATR calculator."""
        self.period = period
        super().__init__(f"ATR_{period}")

    def reset(self) -> None:
        """Reset calculator state."""
        self.true_ranges: Deque[Decimal] = deque(maxlen=self.period)
        self.previous_close: Optional[Decimal] = None

    def calculate(self, ohlcv: OHLCV) -> Optional[IndicatorValue]:
        """Calculate ATR value."""
        if self.previous_close is not None:
            # Calculate True Range
            hl = ohlcv.high - ohlcv.low
            hc = abs(ohlcv.high - self.previous_close)
            lc = abs(ohlcv.low - self.previous_close)
            
            true_range = max(hl, hc, lc)
            self.true_ranges.append(true_range)

        self.previous_close = ohlcv.close

        if len(self.true_ranges) < self.period:
            return None

        atr = sum(self.true_ranges) / len(self.true_ranges)

        return IndicatorValue(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            indicator=self.name,
            timeframe=Timeframe.H1,
            value=atr,
            parameters={"period": self.period},
        )


class IndicatorCalculator:
    """Main calculator for orchestrating multiple indicators."""

    def __init__(self):
        """Initialize indicator calculator."""
        self.indicators: Dict[str, BaseIndicator] = {}

    def register(self, name: str, indicator: BaseIndicator) -> None:
        """Register an indicator."""
        self.indicators[name] = indicator

    def calculate_all(self, ohlcv: OHLCV) -> Dict[str, IndicatorValue]:
        """Calculate all registered indicators."""
        results = {}
        
        for name, indicator in self.indicators.items():
            result = indicator.calculate(ohlcv)
            if result is not None:
                results[name] = result
        
        return results

    def reset_all(self) -> None:
        """Reset all registered indicators."""
        for indicator in self.indicators.values():
            indicator.reset()

    def get_indicator(self, name: str) -> Optional[BaseIndicator]:
        """Get registered indicator by name."""
        return self.indicators.get(name)