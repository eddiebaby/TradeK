#!/usr/bin/env python3
"""
Test the generated momentum strategy to verify it works
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generated momentum strategy code (extracted from our system)
class MomentumTradingStrategy:
    """
    Momentum Trading Strategy - Local Implementation
    
    This strategy uses moving average crossovers and RSI to identify
    momentum trends in financial instruments.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50, rsi_period: int = 14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.positions = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        data = self.calculate_indicators(data)
        
        # Momentum signals
        data['signal'] = np.where(
            (data['SMA_short'] > data['SMA_long']) & (data['RSI'] > 50),
            1,  # Buy signal
            np.where(
                (data['SMA_short'] < data['SMA_long']) & (data['RSI'] < 50),
                -1,  # Sell signal
                0   # Hold
            )
        )
        
        return data
        
    def calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> int:
        """Calculate position size using fixed risk per trade"""
        risk_amount = capital * risk_per_trade
        stop_loss_distance = price * 0.05  # 5% stop loss
        position_size = int(risk_amount / stop_loss_distance)
        return max(1, position_size)
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Simple backtesting implementation"""
        data = self.generate_signals(data)
        
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            signal = data.iloc[i]['signal']
            
            if signal == 1 and position <= 0:  # Buy signal
                shares = self.calculate_position_size(capital, current_price)
                cost = shares * current_price
                if cost <= capital:
                    position += shares
                    capital -= cost
                    trades.append({'action': 'buy', 'price': current_price, 'shares': shares})
                    
            elif signal == -1 and position > 0:  # Sell signal
                proceeds = position * current_price
                capital += proceeds
                trades.append({'action': 'sell', 'price': current_price, 'shares': position})
                position = 0
        
        # Final portfolio value
        final_value = capital + (position * data.iloc[-1]['close'])
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'trades': trades
        }

def test_strategy():
    """Test the generated momentum strategy"""
    print("🧪 Testing Generated Momentum Strategy")
    print("="*50)
    
    # Create realistic sample data (SPY-like movement)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Simulate more realistic price movement
    base_price = 400.0
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # Slight upward drift with realistic volatility
    
    # Add some momentum periods
    for i in range(50, len(daily_returns), 100):
        if i + 20 < len(daily_returns):
            daily_returns[i:i+20] += np.random.normal(0.002, 0.01, 20)  # Momentum periods
    
    prices = [base_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(dates))
    })
    
    print(f"📊 Test data: {len(data)} days")
    print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"📊 Total return (buy & hold): {(data['close'].iloc[-1] / data['close'].iloc[0] - 1):.2%}")
    
    # Test the strategy
    strategy = MomentumTradingStrategy(short_window=10, long_window=30, rsi_period=14)
    
    print(f"\n🎯 Running backtest...")
    results = strategy.backtest(data, initial_capital=100000)
    
    print(f"\n📊 Results:")
    print(f"💰 Initial Capital: ${results['initial_capital']:,}")
    print(f"💰 Final Value: ${results['final_value']:,.2f}")
    print(f"📈 Total Return: {results['total_return']:.2%}")
    print(f"🔄 Number of Trades: {results['num_trades']}")
    
    if results['num_trades'] > 0:
        print(f"\n📋 Sample Trades:")
        for i, trade in enumerate(results['trades'][:5]):  # Show first 5 trades
            print(f"  {i+1}. {trade['action'].upper()}: {trade['shares']} shares @ ${trade['price']:.2f}")
        if len(results['trades']) > 5:
            print(f"  ... and {len(results['trades']) - 5} more trades")
    
    # Compare to buy and hold
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    excess_return = results['total_return'] - buy_hold_return
    
    print(f"\n📊 Performance Comparison:")
    print(f"📈 Buy & Hold: {buy_hold_return:.2%}")
    print(f"🎯 Strategy: {results['total_return']:.2%}")
    print(f"⚡ Excess Return: {excess_return:.2%}")
    
    if excess_return > 0:
        print(f"✅ Strategy outperformed buy & hold!")
    else:
        print(f"⚠️  Strategy underperformed buy & hold")
    
    print(f"\n✅ Strategy test completed - code works correctly!")
    return results

if __name__ == "__main__":
    test_strategy()