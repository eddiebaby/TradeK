"""
LDES Core Interfaces

Defines abstract base classes and protocols for the LDES system components.
This enables dependency injection, testing with mocks, and clean architecture.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Protocol

from .models import (
    BacktestResult,
    LiquiditySignal,
    MarketData,
    Position,
    Side,
    TradeSignal,
)


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for given symbols."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for given symbols."""
        pass

    @abstractmethod
    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Get real-time market data stream."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> list[MarketData]:
        """Get historical market data."""
        pass

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Get latest quote for a symbol."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        pass

    @property
    @abstractmethod
    def supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        pass


class LiquidityDetector(ABC):
    """Abstract base class for liquidity detection algorithms."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detector (load models, etc.)."""
        pass

    @abstractmethod
    async def process_market_data(self, data: MarketData) -> list[LiquiditySignal]:
        """Process market data and detect liquidity events."""
        pass

    @abstractmethod
    async def update_models(self) -> None:
        """Update/retrain detection models."""
        pass

    @abstractmethod
    async def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from ML models."""
        pass

    @abstractmethod
    def get_detection_latency_ms(self) -> float:
        """Get average detection latency in milliseconds."""
        pass

    @abstractmethod
    def get_accuracy_metrics(self) -> dict[str, float]:
        """Get accuracy metrics (precision, recall, etc.)."""
        pass


class RiskManager(ABC):
    """Abstract base class for risk management."""

    @abstractmethod
    async def check_pre_trade_risk(self, signal: TradeSignal) -> bool:
        """Check if trade passes pre-trade risk checks."""
        pass

    @abstractmethod
    async def calculate_position_size(
        self,
        signal: LiquiditySignal,
        portfolio_value: float,
        existing_positions: list[Position],
    ) -> float:
        """Calculate optimal position size using Kelly criterion."""
        pass

    @abstractmethod
    async def calculate_stop_loss(
        self, signal: LiquiditySignal, entry_price: float
    ) -> float:
        """Calculate stop loss price."""
        pass

    @abstractmethod
    async def calculate_profit_target(
        self, signal: LiquiditySignal, entry_price: float
    ) -> float:
        """Calculate profit target price."""
        pass

    @abstractmethod
    async def check_portfolio_limits(
        self, positions: list[Position], portfolio_value: float
    ) -> dict[str, bool]:
        """Check various portfolio-level risk limits."""
        pass

    @abstractmethod
    async def calculate_var(
        self, positions: list[Position], confidence_level: float = 0.95
    ) -> float:
        """Calculate portfolio Value at Risk."""
        pass

    @abstractmethod
    async def should_halt_trading(self) -> bool:
        """Determine if trading should be halted due to risk conditions."""
        pass


class ExecutionEngine(ABC):
    """Abstract base class for trade execution."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize execution engine."""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: str = "limit",
        price: float | None = None,
        time_in_force: str = "day",
    ) -> str:
        """Place a trading order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> str:
        """Get order execution status."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def close_position(self, position_id: str) -> bool:
        """Close an existing position."""
        pass

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information (buying power, etc.)."""
        pass

    @abstractmethod
    async def get_execution_metrics(self) -> dict[str, float]:
        """Get execution quality metrics."""
        pass


class BacktestEngine(ABC):
    """Abstract base class for backtesting."""

    @abstractmethod
    async def initialize(
        self, start_date: datetime, end_date: datetime, initial_capital: float
    ) -> None:
        """Initialize backtest with parameters."""
        pass

    @abstractmethod
    async def add_strategy(self, strategy: "TradingStrategy") -> None:
        """Add a trading strategy to backtest."""
        pass

    @abstractmethod
    async def run_backtest(self) -> BacktestResult:
        """Execute the backtest."""
        pass

    @abstractmethod
    async def get_performance_metrics(self) -> dict[str, float]:
        """Get detailed performance metrics."""
        pass

    @abstractmethod
    async def export_results(self, filepath: str) -> None:
        """Export backtest results to file."""
        pass


class DataStorage(ABC):
    """Abstract base class for data storage."""

    @abstractmethod
    async def store_market_data(self, data: MarketData) -> None:
        """Store market data point."""
        pass

    @abstractmethod
    async def store_signal(self, signal: LiquiditySignal) -> None:
        """Store liquidity signal."""
        pass

    @abstractmethod
    async def store_position(self, position: Position) -> None:
        """Store trading position."""
        pass

    @abstractmethod
    async def get_historical_signals(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[LiquiditySignal]:
        """Retrieve historical signals."""
        pass

    @abstractmethod
    async def get_position_history(
        self, start_date: datetime, end_date: datetime
    ) -> list[Position]:
        """Retrieve position history."""
        pass

    @abstractmethod
    async def cleanup_old_data(self, retention_days: int) -> int:
        """Clean up old data and return number of records deleted."""
        pass


class TradingStrategy(Protocol):
    """Protocol for trading strategies."""

    async def generate_signals(
        self, market_data: MarketData, positions: list[Position]
    ) -> list[TradeSignal]:
        """Generate trading signals based on market data."""
        ...

    async def update_positions(
        self, positions: list[Position], market_data: MarketData
    ) -> list[TradeSignal]:
        """Update existing positions (stops, targets, etc.)."""
        ...

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        ...

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        ...


class EventHandler(Protocol):
    """Protocol for event handlers."""

    async def on_market_data(self, data: MarketData) -> None:
        """Handle market data event."""
        ...

    async def on_signal_generated(self, signal: LiquiditySignal) -> None:
        """Handle signal generation event."""
        ...

    async def on_order_filled(self, position: Position) -> None:
        """Handle order fill event."""
        ...

    async def on_position_closed(self, position: Position) -> None:
        """Handle position close event."""
        ...

    async def on_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle error event."""
        ...


class PerformanceMonitor(ABC):
    """Abstract base class for performance monitoring."""

    @abstractmethod
    async def record_metric(
        self, name: str, value: float, tags: dict[str, str] = None
    ) -> None:
        """Record a performance metric."""
        pass

    @abstractmethod
    async def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record operation latency."""
        pass

    @abstractmethod
    async def record_trade(self, position: Position) -> None:
        """Record trade for performance analysis."""
        pass

    @abstractmethod
    async def get_system_health(self) -> dict[str, Any]:
        """Get system health metrics."""
        pass

    @abstractmethod
    async def generate_report(
        self, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate performance report."""
        pass


class NotificationService(ABC):
    """Abstract base class for notifications."""

    @abstractmethod
    async def send_alert(
        self, level: str, message: str, context: dict[str, Any] = None
    ) -> None:
        """Send alert notification."""
        pass

    @abstractmethod
    async def send_trade_notification(self, position: Position) -> None:
        """Send trade notification."""
        pass

    @abstractmethod
    async def send_performance_summary(self, metrics: dict[str, float]) -> None:
        """Send performance summary."""
        pass
