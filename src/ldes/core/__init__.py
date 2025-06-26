"""
LDES Core Module

Contains fundamental data models, interfaces, and configuration
for the Liquidity Detection & Execution System.
"""

from .config import LDESConfig
from .interfaces import (
    ExecutionEngine,
    LiquidityDetector,
    MarketDataProvider,
    RiskManager,
)
from .models import (
    BacktestResult,
    LiquiditySignal,
    MarketData,
    OrderBook,
    OrderBookLevel,
    OrderStatus,
    Position,
    PositionStatus,
    Side,
    SignalType,
    TradeSignal,
)

__all__ = [
    "LiquiditySignal",
    "MarketData",
    "Position",
    "TradeSignal",
    "BacktestResult",
    "OrderBook",
    "OrderBookLevel",
    "SignalType",
    "Side",
    "PositionStatus",
    "OrderStatus",
    "LDESConfig",
    "MarketDataProvider",
    "LiquidityDetector",
    "ExecutionEngine",
    "RiskManager",
]
