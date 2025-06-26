"""
LDES (Liquidity Detection & Execution System) Module

An institutional-grade system for detecting forced liquidation events
and executing mean reversion strategies based on market microstructure analysis.

This module integrates with the TradeKnowledge system to provide:
- Real-time market data processing
- Liquidity event detection algorithms
- Risk-managed execution strategies
- Comprehensive backtesting framework
- Knowledge-enhanced trading decisions
"""

from .core.config import LDESConfig
from .core.models import (
    BacktestResult,
    LiquiditySignal,
    MarketData,
    Position,
    TradeSignal,
)

__version__ = "1.0.0"
__author__ = "TradeKnowledge LDES Team"

__all__ = [
    "LiquiditySignal",
    "MarketData",
    "Position",
    "TradeSignal",
    "BacktestResult",
    "LDESConfig",
]
