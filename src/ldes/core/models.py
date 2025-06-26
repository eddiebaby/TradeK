"""
LDES Core Data Models

Defines the fundamental data structures for the Liquidity Detection & Execution System.
All models use Pydantic for validation and serialization, following the existing
TradeKnowledge architecture patterns.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    """Types of liquidity signals that can be detected."""

    FORCED_LIQUIDATION = "forced_liquidation"
    MARGIN_CALL = "margin_call"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    VOLUME_SPIKE = "volume_spike"
    SPREAD_EXPANSION = "spread_expansion"
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"


class Side(str, Enum):
    """Trading side/direction."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Position lifecycle status."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderBookLevel(BaseModel):
    """Single level in order book."""

    price: Decimal = Field(..., description="Price level")
    size: int = Field(..., description="Aggregate size at this level")
    orders: int = Field(default=1, description="Number of orders at this level")


class OrderBook(BaseModel):
    """Market order book snapshot."""

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Snapshot timestamp")
    bids: list[OrderBookLevel] = Field(default_factory=list, description="Bid levels")
    asks: list[OrderBookLevel] = Field(default_factory=list, description="Ask levels")

    @field_validator("bids", "asks")
    @classmethod
    def validate_sorted_levels(cls, v):
        """Ensure price levels are properly sorted."""
        if not v:
            return v
        # Bids should be sorted high to low, asks low to high
        return v

    @property
    def best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> float | None:
        """Get spread in basis points."""
        if self.spread and self.best_bid:
            return float(self.spread / self.best_bid * 10000)
        return None


class MarketData(BaseModel):
    """Normalized market data point."""

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    bid_price: Decimal | None = Field(None, description="Best bid price")
    bid_size: int | None = Field(None, description="Best bid size")
    ask_price: Decimal | None = Field(None, description="Best ask price")
    ask_size: int | None = Field(None, description="Best ask size")
    last_price: Decimal | None = Field(None, description="Last trade price")
    last_size: int | None = Field(None, description="Last trade size")
    volume: int = Field(default=0, description="Cumulative volume")
    vwap: Decimal | None = Field(None, description="Volume weighted average price")
    order_book: OrderBook | None = Field(None, description="Full order book")
    source: str = Field(..., description="Data source (alpaca, binance, etc.)")

    @field_validator("bid_price", "ask_price", "last_price")
    @classmethod
    def validate_positive_prices(cls, v):
        """Ensure prices are positive."""
        if v is not None and v <= 0:
            raise ValueError("Prices must be positive")
        return v

    @field_validator("bid_size", "ask_size", "last_size", "volume")
    @classmethod
    def validate_non_negative_sizes(cls, v):
        """Ensure sizes and volumes are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Sizes and volumes must be non-negative")
        return v

    @property
    def spread(self) -> Decimal | None:
        """Get bid-ask spread."""
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return None

    @property
    def mid_price(self) -> Decimal | None:
        """Get mid-market price."""
        if self.bid_price and self.ask_price:
            return (self.bid_price + self.ask_price) / 2
        return None


class LiquiditySignal(BaseModel):
    """Detected liquidity event signal."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique signal ID"
    )
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Signal generation time")
    signal_type: SignalType = Field(..., description="Type of liquidity event")
    strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength (0-1)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    expected_direction: Side = Field(..., description="Expected profitable direction")
    expected_move_bps: float = Field(
        ..., description="Expected price move in basis points"
    )
    time_horizon_seconds: int = Field(
        ..., description="Expected duration of opportunity"
    )
    volume_surge_factor: float = Field(
        default=1.0, description="Volume increase factor"
    )
    price_velocity: float = Field(default=0.0, description="Price change velocity")
    order_book_imbalance: float = Field(
        default=0.0, description="Buy/sell imbalance ratio"
    )
    features: dict[str, float] = Field(
        default_factory=dict, description="Additional features"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    knowledge_context: dict[str, Any] | None = Field(
        None, description="TradeKnowledge insights"
    )

    @field_validator("strength", "confidence")
    @classmethod
    def validate_probabilities(cls, v):
        """Ensure probability values are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probabilities must be between 0 and 1")
        return v


class Position(BaseModel):
    """Active trading position."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique position ID"
    )
    symbol: str = Field(..., description="Trading symbol")
    side: Side = Field(..., description="Position side")
    quantity: float = Field(..., description="Position quantity")
    entry_price: Decimal = Field(..., description="Average entry price")
    entry_time: datetime = Field(..., description="Position entry time")
    current_price: Decimal | None = Field(None, description="Current market price")
    target_price: Decimal | None = Field(None, description="Target exit price")
    stop_price: Decimal | None = Field(None, description="Stop loss price")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized P&L")
    commission: Decimal = Field(default=Decimal("0"), description="Commission paid")
    status: PositionStatus = Field(
        default=PositionStatus.OPEN, description="Position status"
    )
    signal_id: str | None = Field(None, description="Originating signal ID")
    risk_metrics: dict[str, float] = Field(
        default_factory=dict, description="Risk calculations"
    )

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v):
        """Ensure quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @property
    def market_value(self) -> Decimal | None:
        """Calculate current market value."""
        if self.current_price:
            return self.current_price * Decimal(str(self.quantity))
        return None

    @property
    def pnl_percentage(self) -> float | None:
        """Calculate P&L percentage."""
        if self.current_price and self.entry_price:
            pnl_factor = float(self.current_price / self.entry_price)
            if self.side == Side.LONG:
                return (pnl_factor - 1.0) * 100
            else:  # SHORT
                return (1.0 - pnl_factor) * 100
        return None


class TradeSignal(BaseModel):
    """Actionable trading signal with position sizing."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique trade signal ID"
    )
    liquidity_signal_id: str = Field(..., description="Source liquidity signal ID")
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Signal generation time")
    side: Side = Field(..., description="Recommended trade side")
    quantity: float = Field(..., description="Recommended position size")
    entry_price: Decimal = Field(..., description="Target entry price")
    target_price: Decimal | None = Field(None, description="Target exit price")
    stop_price: Decimal | None = Field(None, description="Stop loss price")
    expected_return: float = Field(..., description="Expected return in %")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk assessment (0-1)")
    kelly_fraction: float = Field(..., description="Kelly criterion position size")
    portfolio_allocation: float = Field(..., description="Recommended % of portfolio")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    time_horizon_seconds: int = Field(..., description="Expected holding period")
    status: OrderStatus = Field(
        default=OrderStatus.PENDING, description="Execution status"
    )

    @field_validator("quantity", "portfolio_allocation")
    @classmethod
    def validate_positive_values(cls, v):
        """Ensure positive values."""
        if v <= 0:
            raise ValueError("Quantity and allocation must be positive")
        return v


class BacktestResult(BaseModel):
    """Comprehensive backtest performance results."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique backtest ID"
    )
    strategy_name: str = Field(..., description="Strategy name")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(..., description="Starting capital")
    final_capital: Decimal = Field(..., description="Ending capital")
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return metric")
    sortino_ratio: float = Field(..., description="Downside risk-adjusted return")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    max_drawdown_duration_days: int = Field(..., description="Longest drawdown period")
    win_rate: float = Field(..., description="Percentage of winning trades")
    profit_factor: float = Field(..., description="Gross profit / gross loss")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: float = Field(..., description="Average winning trade %")
    avg_loss: float = Field(..., description="Average losing trade %")
    largest_win: float = Field(..., description="Largest winning trade %")
    largest_loss: float = Field(..., description="Largest losing trade %")
    avg_trade_duration_minutes: float = Field(..., description="Average holding period")
    avg_daily_return: float = Field(..., description="Average daily return %")
    volatility: float = Field(..., description="Strategy volatility %")
    var_95: float = Field(..., description="Value at Risk 95%")
    cvar_95: float = Field(..., description="Conditional Value at Risk 95%")
    calmar_ratio: float = Field(..., description="Return / max drawdown")
    commission_paid: Decimal = Field(..., description="Total commission costs")
    slippage_cost: Decimal = Field(..., description="Total slippage costs")

    # Additional metrics
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional metrics"
    )
    trades: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual trade results"
    )
    daily_returns: list[float] = Field(
        default_factory=list, description="Daily return series"
    )
    equity_curve: list[dict[str, Any]] = Field(
        default_factory=list, description="Portfolio value over time"
    )

    @property
    def roi(self) -> float:
        """Return on investment percentage."""
        return float(
            (self.final_capital - self.initial_capital) / self.initial_capital * 100
        )

    @property
    def profit_factor_ratio(self) -> float:
        """Profit factor calculation."""
        if self.losing_trades > 0:
            gross_profit = self.winning_trades * self.avg_win
            gross_loss = abs(self.losing_trades * self.avg_loss)
            return gross_profit / gross_loss if gross_loss > 0 else float("inf")
        return float("inf") if self.winning_trades > 0 else 0.0
