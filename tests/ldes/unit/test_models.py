"""
Unit Tests for LDES Core Models

Comprehensive test coverage for all data models following TDD principles.
Tests validation, calculations, and edge cases.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from src.ldes.core.models import (
    MarketData,
    LiquiditySignal,
    Position,
    TradeSignal,
    BacktestResult,
    OrderBook,
    OrderBookLevel,
    SignalType,
    Side,
    PositionStatus,
    OrderStatus
)


class TestOrderBookLevel:
    """Test OrderBookLevel model."""
    
    def test_create_valid_order_book_level(self):
        """Test creating a valid order book level."""
        level = OrderBookLevel(
            price=Decimal('100.50'),
            size=1000,
            orders=5
        )
        
        assert level.price == Decimal('100.50')
        assert level.size == 1000
        assert level.orders == 5
    
    def test_order_book_level_defaults(self):
        """Test default values for order book level."""
        level = OrderBookLevel(
            price=Decimal('100.50'),
            size=1000
        )
        
        assert level.orders == 1  # Default value


class TestOrderBook:
    """Test OrderBook model."""
    
    def test_create_empty_order_book(self):
        """Test creating an empty order book."""
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now()
        )
        
        assert book.symbol == "AAPL"
        assert len(book.bids) == 0
        assert len(book.asks) == 0
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.spread_bps is None
    
    def test_order_book_with_levels(self):
        """Test order book with bid/ask levels."""
        bids = [
            OrderBookLevel(price=Decimal('100.00'), size=500),
            OrderBookLevel(price=Decimal('99.50'), size=1000)
        ]
        asks = [
            OrderBookLevel(price=Decimal('100.25'), size=300),
            OrderBookLevel(price=Decimal('100.50'), size=800)
        ]
        
        book = OrderBook(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
        
        assert book.best_bid == Decimal('100.00')
        assert book.best_ask == Decimal('100.25')
        assert book.spread == Decimal('0.25')
        assert book.spread_bps == pytest.approx(25.0, rel=1e-2)


class TestMarketData:
    """Test MarketData model."""
    
    def test_create_valid_market_data(self):
        """Test creating valid market data."""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=Decimal('150.00'),
            bid_size=100,
            ask_price=Decimal('150.25'),
            ask_size=200,
            last_price=Decimal('150.10'),
            last_size=50,
            volume=10000,
            source="alpaca"
        )
        
        assert data.symbol == "AAPL"
        assert data.bid_price == Decimal('150.00')
        assert data.ask_price == Decimal('150.25')
        assert data.spread == Decimal('0.25')
        assert data.mid_price == Decimal('150.125')
        assert data.source == "alpaca"
    
    def test_market_data_price_validation(self):
        """Test price validation in market data."""
        with pytest.raises(ValueError, match="Prices must be positive"):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                bid_price=Decimal('-150.00'),  # Invalid negative price
                source="alpaca"
            )
    
    def test_market_data_size_validation(self):
        """Test size validation in market data."""
        with pytest.raises(ValueError, match="Sizes and volumes must be non-negative"):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                bid_size=-100,  # Invalid negative size
                source="alpaca"
            )
    
    def test_market_data_spread_calculation(self):
        """Test spread calculation."""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=Decimal('100.00'),
            ask_price=Decimal('100.50'),
            source="alpaca"
        )
        
        assert data.spread == Decimal('0.50')
        assert data.mid_price == Decimal('100.25')
    
    def test_market_data_missing_prices(self):
        """Test behavior with missing bid/ask prices."""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            last_price=Decimal('150.00'),
            source="alpaca"
        )
        
        assert data.spread is None
        assert data.mid_price is None


class TestLiquiditySignal:
    """Test LiquiditySignal model."""
    
    def test_create_valid_liquidity_signal(self):
        """Test creating a valid liquidity signal."""
        signal = LiquiditySignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.FORCED_LIQUIDATION,
            strength=0.8,
            confidence=0.9,
            expected_direction=Side.LONG,
            expected_move_bps=150.0,
            time_horizon_seconds=300
        )
        
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.FORCED_LIQUIDATION
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.expected_direction == Side.LONG
        assert signal.expected_move_bps == 150.0
        assert signal.time_horizon_seconds == 300
        assert len(signal.id) > 0  # UUID generated
    
    def test_signal_strength_validation(self):
        """Test signal strength validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LiquiditySignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type=SignalType.FORCED_LIQUIDATION,
                strength=1.5,  # Invalid > 1.0
                confidence=0.9,
                expected_direction=Side.LONG,
                expected_move_bps=150.0,
                time_horizon_seconds=300
            )
    
    def test_signal_confidence_validation(self):
        """Test signal confidence validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LiquiditySignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type=SignalType.FORCED_LIQUIDATION,
                strength=0.8,
                confidence=-0.1,  # Invalid < 0.0
                expected_direction=Side.LONG,
                expected_move_bps=150.0,
                time_horizon_seconds=300
            )
    
    def test_signal_with_features(self):
        """Test signal with additional features."""
        features = {
            "volume_spike": 3.5,
            "price_velocity": 25.0,
            "spread_expansion": 2.1
        }
        
        signal = LiquiditySignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.VOLUME_SPIKE,
            strength=0.7,
            confidence=0.8,
            expected_direction=Side.SHORT,
            expected_move_bps=75.0,
            time_horizon_seconds=180,
            features=features
        )
        
        assert signal.features == features
        assert signal.features["volume_spike"] == 3.5


class TestPosition:
    """Test Position model."""
    
    def test_create_valid_position(self):
        """Test creating a valid position."""
        position = Position(
            symbol="AAPL",
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            entry_time=datetime.now(),
            current_price=Decimal('152.50'),
            target_price=Decimal('165.00'),
            stop_price=Decimal('142.50')
        )
        
        assert position.symbol == "AAPL"
        assert position.side == Side.LONG
        assert position.quantity == 100.0
        assert position.entry_price == Decimal('150.00')
        assert position.current_price == Decimal('152.50')
        assert position.status == PositionStatus.OPEN  # Default
        assert len(position.id) > 0  # UUID generated
    
    def test_position_quantity_validation(self):
        """Test position quantity validation."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Position(
                symbol="AAPL",
                side=Side.LONG,
                quantity=-100.0,  # Invalid negative quantity
                entry_price=Decimal('150.00'),
                entry_time=datetime.now()
            )
    
    def test_position_market_value_calculation(self):
        """Test market value calculation."""
        position = Position(
            symbol="AAPL",
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            entry_time=datetime.now(),
            current_price=Decimal('152.50')
        )
        
        assert position.market_value == Decimal('15250.00')
    
    def test_position_pnl_percentage_long(self):
        """Test P&L percentage calculation for long position."""
        position = Position(
            symbol="AAPL",
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            entry_time=datetime.now(),
            current_price=Decimal('165.00')  # 10% gain
        )
        
        assert position.pnl_percentage == pytest.approx(10.0, rel=1e-2)
    
    def test_position_pnl_percentage_short(self):
        """Test P&L percentage calculation for short position."""
        position = Position(
            symbol="AAPL",
            side=Side.SHORT,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            entry_time=datetime.now(),
            current_price=Decimal('135.00')  # 10% gain for short
        )
        
        assert position.pnl_percentage == pytest.approx(10.0, rel=1e-2)
    
    def test_position_without_current_price(self):
        """Test position behavior without current price."""
        position = Position(
            symbol="AAPL",
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            entry_time=datetime.now()
        )
        
        assert position.market_value is None
        assert position.pnl_percentage is None


class TestTradeSignal:
    """Test TradeSignal model."""
    
    def test_create_valid_trade_signal(self):
        """Test creating a valid trade signal."""
        signal = TradeSignal(
            liquidity_signal_id=str(uuid4()),
            symbol="AAPL",
            timestamp=datetime.now(),
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            target_price=Decimal('165.00'),
            stop_price=Decimal('142.50'),
            expected_return=10.0,
            risk_score=0.3,
            kelly_fraction=0.15,
            portfolio_allocation=5.0,
            confidence=0.8,
            time_horizon_seconds=1800
        )
        
        assert signal.symbol == "AAPL"
        assert signal.side == Side.LONG
        assert signal.quantity == 100.0
        assert signal.expected_return == 10.0
        assert signal.risk_score == 0.3
        assert signal.status == OrderStatus.PENDING  # Default
    
    def test_trade_signal_quantity_validation(self):
        """Test trade signal quantity validation."""
        with pytest.raises(ValueError, match="Quantity and allocation must be positive"):
            TradeSignal(
                liquidity_signal_id=str(uuid4()),
                symbol="AAPL",
                timestamp=datetime.now(),
                side=Side.LONG,
                quantity=0.0,  # Invalid zero quantity
                entry_price=Decimal('150.00'),
                expected_return=10.0,
                risk_score=0.3,
                kelly_fraction=0.15,
                portfolio_allocation=5.0,
                confidence=0.8,
                time_horizon_seconds=1800
            )
    
    def test_trade_signal_allocation_validation(self):
        """Test trade signal allocation validation."""
        with pytest.raises(ValueError, match="Quantity and allocation must be positive"):
            TradeSignal(
                liquidity_signal_id=str(uuid4()),
                symbol="AAPL",
                timestamp=datetime.now(),
                side=Side.LONG,
                quantity=100.0,
                entry_price=Decimal('150.00'),
                expected_return=10.0,
                risk_score=0.3,
                kelly_fraction=0.15,
                portfolio_allocation=-5.0,  # Invalid negative allocation
                confidence=0.8,
                time_horizon_seconds=1800
            )


class TestBacktestResult:
    """Test BacktestResult model."""
    
    def test_create_valid_backtest_result(self):
        """Test creating a valid backtest result."""
        result = BacktestResult(
            strategy_name="LDES Strategy",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal('100000'),
            final_capital=Decimal('125000'),
            total_return=25.0,
            annualized_return=25.0,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            max_drawdown=5.0,
            max_drawdown_duration_days=15,
            win_rate=65.0,
            profit_factor=2.1,
            total_trades=150,
            winning_trades=98,
            losing_trades=52,
            avg_win=3.5,
            avg_loss=-1.8,
            largest_win=15.2,
            largest_loss=-4.5,
            avg_trade_duration_minutes=120.0,
            avg_daily_return=0.08,
            volatility=12.0,
            var_95=2.5,
            cvar_95=3.8,
            calmar_ratio=5.0,
            commission_paid=Decimal('150'),
            slippage_cost=Decimal('75')
        )
        
        assert result.strategy_name == "LDES Strategy"
        assert result.total_return == 25.0
        assert result.sharpe_ratio == 2.5
        assert result.win_rate == 65.0
        assert result.total_trades == 150
        assert result.winning_trades + result.losing_trades == result.total_trades
    
    def test_backtest_roi_calculation(self):
        """Test ROI calculation."""
        result = BacktestResult(
            strategy_name="Test Strategy",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal('100000'),
            final_capital=Decimal('120000'),
            total_return=20.0,
            annualized_return=20.0,
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            max_drawdown=3.0,
            max_drawdown_duration_days=10,
            win_rate=60.0,
            profit_factor=1.8,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            avg_win=2.5,
            avg_loss=-1.5,
            largest_win=8.0,
            largest_loss=-3.0,
            avg_trade_duration_minutes=90.0,
            avg_daily_return=0.06,
            volatility=10.0,
            var_95=2.0,
            cvar_95=3.0,
            calmar_ratio=6.67,
            commission_paid=Decimal('100'),
            slippage_cost=Decimal('50')
        )
        
        assert result.roi == pytest.approx(20.0, rel=1e-2)
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        result = BacktestResult(
            strategy_name="Test Strategy",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=Decimal('100000'),
            final_capital=Decimal('110000'),
            total_return=10.0,
            annualized_return=10.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=2.0,
            max_drawdown_duration_days=5,
            win_rate=50.0,
            profit_factor=2.0,
            total_trades=100,
            winning_trades=50,
            losing_trades=50,
            avg_win=4.0,  # 50 trades * 4.0% = 200% gross profit
            avg_loss=-2.0,  # 50 trades * -2.0% = -100% gross loss
            largest_win=10.0,
            largest_loss=-5.0,
            avg_trade_duration_minutes=60.0,
            avg_daily_return=0.04,
            volatility=8.0,
            var_95=1.5,
            cvar_95=2.5,
            calmar_ratio=5.0,
            commission_paid=Decimal('75'),
            slippage_cost=Decimal('25')
        )
        
        # Profit factor = gross profit / gross loss = 200 / 100 = 2.0
        assert result.profit_factor_ratio == pytest.approx(2.0, rel=1e-2)


class TestModelIntegration:
    """Test integration between models."""
    
    def test_signal_to_trade_workflow(self):
        """Test the workflow from liquidity signal to trade signal."""
        # Create liquidity signal
        liquidity_signal = LiquiditySignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type=SignalType.FORCED_LIQUIDATION,
            strength=0.9,
            confidence=0.85,
            expected_direction=Side.LONG,
            expected_move_bps=200.0,
            time_horizon_seconds=600
        )
        
        # Create trade signal based on liquidity signal
        trade_signal = TradeSignal(
            liquidity_signal_id=liquidity_signal.id,
            symbol=liquidity_signal.symbol,
            timestamp=liquidity_signal.timestamp,
            side=liquidity_signal.expected_direction,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            target_price=Decimal('153.00'),  # 2% gain
            stop_price=Decimal('147.00'),    # 2% loss
            expected_return=2.0,
            risk_score=0.2,
            kelly_fraction=0.1,
            portfolio_allocation=5.0,
            confidence=liquidity_signal.confidence,
            time_horizon_seconds=liquidity_signal.time_horizon_seconds
        )
        
        # Verify connection
        assert trade_signal.liquidity_signal_id == liquidity_signal.id
        assert trade_signal.symbol == liquidity_signal.symbol
        assert trade_signal.side == liquidity_signal.expected_direction
        assert trade_signal.confidence == liquidity_signal.confidence
    
    def test_trade_to_position_workflow(self):
        """Test the workflow from trade signal to position."""
        # Create trade signal
        trade_signal = TradeSignal(
            liquidity_signal_id=str(uuid4()),
            symbol="AAPL",
            timestamp=datetime.now(),
            side=Side.LONG,
            quantity=100.0,
            entry_price=Decimal('150.00'),
            target_price=Decimal('165.00'),
            stop_price=Decimal('142.50'),
            expected_return=10.0,
            risk_score=0.3,
            kelly_fraction=0.15,
            portfolio_allocation=5.0,
            confidence=0.8,
            time_horizon_seconds=1800
        )
        
        # Create position from trade signal
        position = Position(
            symbol=trade_signal.symbol,
            side=trade_signal.side,
            quantity=trade_signal.quantity,
            entry_price=trade_signal.entry_price,
            entry_time=trade_signal.timestamp,
            target_price=trade_signal.target_price,
            stop_price=trade_signal.stop_price,
            signal_id=trade_signal.id
        )
        
        # Verify connection
        assert position.signal_id == trade_signal.id
        assert position.symbol == trade_signal.symbol
        assert position.side == trade_signal.side
        assert position.quantity == trade_signal.quantity
        assert position.entry_price == trade_signal.entry_price
        assert position.target_price == trade_signal.target_price
        assert position.stop_price == trade_signal.stop_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])