#!/usr/bin/env python3
"""
📊 S&P 500 (SPX) - Entry Points & Stop Loss Analysis
Tactical trading levels with risk management framework
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

async def generate_spx_entry_stops():
    """Generate detailed entry points and stop loss analysis for SPX"""
    
    # Get real market data
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(period="6mo")
    
    # Calculate technical indicators
    close_prices = hist['Close']
    high_prices = hist['High']
    low_prices = hist['Low']
    
    # Moving averages
    sma_20 = close_prices.rolling(window=20).mean()
    sma_50 = close_prices.rolling(window=50).mean()
    sma_200 = close_prices.rolling(window=200).mean()
    
    # Bollinger Bands
    bb_middle = close_prices.rolling(window=20).mean()
    bb_std = close_prices.rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for stop placement
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift(1))
    tr3 = abs(low_prices - close_prices.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    current_price = close_prices.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_atr = atr.iloc[-1]
    
    # Calculate support and resistance levels
    recent_high = high_prices.tail(60).max()
    recent_low = low_prices.tail(60).min()
    
    # Fibonacci retracement levels
    fib_diff = recent_high - recent_low
    fib_23_6 = recent_high - (fib_diff * 0.236)
    fib_38_2 = recent_high - (fib_diff * 0.382)
    fib_50_0 = recent_high - (fib_diff * 0.500)
    fib_61_8 = recent_high - (fib_diff * 0.618)
    
    print("📊 S&P 500 (SPX) - ENTRY POINTS & STOP LOSS ANALYSIS")
    print("=" * 65)
    print()
    
    print("🎯 CURRENT MARKET POSITION")
    print("-" * 28)
    print()
    print(f"Current Level: {current_price:,.2f}")
    print(f"RSI: {current_rsi:.1f}")
    print(f"ATR (14): {current_atr:.2f}")
    print(f"20-Day SMA: {sma_20.iloc[-1]:,.2f}")
    print(f"50-Day SMA: {sma_50.iloc[-1]:,.2f}")
    print(f"200-Day SMA: {sma_200.iloc[-1]:,.2f}")
    print()
    
    print("📈 BULLISH ENTRY STRATEGIES")
    print("-" * 29)
    print()
    
    print("STRATEGY 1: Pullback to Moving Average Support")
    print("=" * 50)
    print(f"Entry Zone: {sma_20.iloc[-1]:,.0f} - {sma_50.iloc[-1]:,.0f}")
    print(f"Best Entry: {sma_20.iloc[-1]:,.0f} (20-day SMA)")
    print(f"Stop Loss: {sma_50.iloc[-1] - (current_atr * 1.5):,.0f}")
    print(f"Risk: {((sma_20.iloc[-1] - (sma_50.iloc[-1] - current_atr * 1.5)) / sma_20.iloc[-1]) * 100:.1f}%")
    print(f"Target 1: {current_price + (current_atr * 2):,.0f} (2:1 R/R)")
    print(f"Target 2: {current_price + (current_atr * 3):,.0f} (3:1 R/R)")
    print()
    print("Rationale: Wait for pullback to key moving average support")
    print("Trigger: RSI below 40 + price at/near 20-day SMA")
    print("Timeframe: 3-7 trading days")
    print()
    
    print("STRATEGY 2: Breakout Above Resistance")
    print("=" * 38)
    breakout_level = 6090
    print(f"Entry: {breakout_level:,} (break above key resistance)")
    print(f"Confirmation: Volume > 1.5x average")
    print(f"Stop Loss: {breakout_level - (current_atr * 2):,.0f}")
    print(f"Risk: {((breakout_level - (breakout_level - current_atr * 2)) / breakout_level) * 100:.1f}%")
    print(f"Target 1: {breakout_level + 100:,} (+100 points)")
    print(f"Target 2: {breakout_level + 200:,} (+200 points)")
    print()
    print("Rationale: Momentum play on technical breakout")
    print("Trigger: Clean break above 6,090 with volume")
    print("Timeframe: 1-3 trading days")
    print()
    
    print("STRATEGY 3: Fibonacci Retracement Buy")
    print("=" * 39)
    print(f"Entry Zone: {fib_38_2:,.0f} - {fib_50_0:,.0f}")
    print(f"Best Entry: {fib_38_2:,.0f} (38.2% retracement)")
    print(f"Stop Loss: {fib_61_8 - 50:,.0f}")
    print(f"Risk: {((fib_38_2 - (fib_61_8 - 50)) / fib_38_2) * 100:.1f}%")
    print(f"Target 1: {recent_high - 50:,.0f}")
    print(f"Target 2: {recent_high + 100:,.0f}")
    print()
    print("Rationale: Mean reversion to Fibonacci support levels")
    print("Trigger: RSI oversold + price at fib level")
    print("Timeframe: 5-10 trading days")
    print()
    
    print("📉 BEARISH ENTRY STRATEGIES")
    print("-" * 29)
    print()
    
    print("STRATEGY 4: Failed Breakout Short")
    print("=" * 35)
    failed_breakout = 6090
    print(f"Entry: {failed_breakout - 30:,} (failed break above {failed_breakout:,})")
    print(f"Confirmation: Close below {failed_breakout:,} after initial break")
    print(f"Stop Loss: {failed_breakout + 50:,}")
    print(f"Risk: {(((failed_breakout + 50) - (failed_breakout - 30)) / (failed_breakout - 30)) * 100:.1f}%")
    print(f"Target 1: {failed_breakout - 150:,}")
    print(f"Target 2: {failed_breakout - 250:,}")
    print()
    print("Rationale: Fade failed breakout attempts")
    print("Trigger: Rejection at resistance + volume decline")
    print("Timeframe: 2-5 trading days")
    print()
    
    print("STRATEGY 5: Moving Average Breakdown")
    print("=" * 37)
    print(f"Entry: {sma_50.iloc[-1] - 20:,.0f} (break below 50-day SMA)")
    print(f"Confirmation: Close below 50-day SMA for 2+ days")
    print(f"Stop Loss: {sma_20.iloc[-1] + 30:,.0f}")
    print(f"Risk: {(((sma_20.iloc[-1] + 30) - (sma_50.iloc[-1] - 20)) / (sma_50.iloc[-1] - 20)) * 100:.1f}%")
    print(f"Target 1: {sma_200.iloc[-1]:,.0f} (200-day SMA)")
    print(f"Target 2: {sma_200.iloc[-1] - 100:,.0f}")
    print()
    print("Rationale: Trend breakdown trade")
    print("Trigger: Loss of key moving average support")
    print("Timeframe: 1-2 weeks")
    print()
    
    print("⚠️ RISK MANAGEMENT FRAMEWORK")
    print("-" * 31)
    print()
    
    print("POSITION SIZING RULES")
    print("=" * 21)
    print("• Maximum risk per trade: 1-2% of portfolio")
    print("• Position size = (Portfolio Value × Risk%) ÷ (Entry - Stop)")
    print("• Scale in on larger positions (1/3, 1/3, 1/3)")
    print("• Never risk more than 5% on correlated trades")
    print()
    
    print("STOP LOSS GUIDELINES")
    print("=" * 20)
    print("• Technical Stops: Below key support/resistance levels")
    print("• ATR Stops: 1.5-2.0x ATR for swing trades")
    print("• Time Stops: Exit if no progress in 5-7 days")
    print("• Fundamental Stops: Major policy/economic changes")
    print()
    
    print("PROFIT TAKING STRATEGY")
    print("=" * 22)
    print("• Take 1/3 profit at 1:1 risk/reward")
    print("• Take 1/3 profit at 2:1 risk/reward")
    print("• Trail final 1/3 with 20-day SMA or ATR stop")
    print("• Book full profits if RSI > 80 (overbought)")
    print()
    
    print("🎯 TACTICAL ENTRY SETUPS")
    print("-" * 26)
    print()
    
    print("IMMEDIATE OPPORTUNITIES")
    print("=" * 23)
    
    # Current market analysis
    if current_rsi < 40:
        print("🟢 OVERSOLD BOUNCE SETUP")
        print(f"• Current RSI: {current_rsi:.1f} (oversold)")
        print(f"• Entry: {current_price - 20:,.0f} - {current_price + 10:,.0f}")
        print(f"• Stop: {current_price - (current_atr * 2):,.0f}")
        print(f"• Target: {current_price + (current_atr * 2):,.0f}")
    elif current_rsi > 70:
        print("🔴 OVERBOUGHT FADE SETUP")
        print(f"• Current RSI: {current_rsi:.1f} (overbought)")
        print(f"• Entry: {current_price - 10:,.0f} - {current_price + 20:,.0f}")
        print(f"• Stop: {current_price + (current_atr * 1.5):,.0f}")
        print(f"• Target: {current_price - (current_atr * 2):,.0f}")
    else:
        print("🟡 NEUTRAL - WAIT FOR SETUP")
        print(f"• Current RSI: {current_rsi:.1f} (neutral zone)")
        print("• Wait for oversold (<40) or overbought (>70)")
        print("• Monitor key levels: 6,090 resistance, 5,875 support")
    print()
    
    print("WEEKLY LEVELS TO WATCH")
    print("=" * 22)
    print(f"🔴 RESISTANCE: 6,090 | 6,150 | 6,290")
    print(f"🟢 SUPPORT: 5,875 | 5,750 | 5,670")
    print(f"📊 NEUTRAL ZONE: 5,950 - 6,050")
    print()
    
    print("OPTIONS STRATEGIES")
    print("=" * 17)
    print("• BULLISH: Buy calls at support levels (5,875-5,950)")
    print("• BEARISH: Buy puts at resistance levels (6,090-6,150)")
    print("• NEUTRAL: Iron condors in 5,950-6,050 range")
    print("• VOLATILITY: Straddles before Fed meetings/earnings")
    print()
    
    print("=" * 65)
    print("✅ ENTRY & STOP ANALYSIS COMPLETE")
    print("⚠️  Always use proper position sizing and risk management")
    print("📊 Adjust levels based on changing market conditions")
    print("🎯 Focus on high probability setups with defined risk")

if __name__ == "__main__":
    asyncio.run(generate_spx_entry_stops())